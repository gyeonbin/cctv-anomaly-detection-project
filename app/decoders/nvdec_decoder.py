# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, time, ctypes, threading
from contextlib import nullcontext
from typing import Optional, Dict

import numpy as np
import cv2
import torch

from core.types import FrameMeta
from core.ring_gpu import CudaFrameRing
from core.telemetry import Telemetry
from app.stores.frame_store import FrameStore
from app.wiring.bootstrap import RING_SLOTS, DEFAULT_REALTIME_FPS, SAFE_DETACH

class NVDECDecoderThread(threading.Thread):
    def __init__(self, path: str, meta_q, frame_store: FrameStore, stats: Dict, telemetry: Telemetry, gpuid: int = 0):
        super().__init__(daemon=True)
        self.path = path
        self.meta_q = meta_q
        self.frame_store = frame_store
        self.stats = stats
        self.telemetry = telemetry
        self.stop_flag = False
        self.gpuid = gpuid
        self.ring: Optional[CudaFrameRing] = None

    def run(self):
        try:
            import PyNvVideoCodec as nvc
        except Exception as e:
            print("[NVDEC] PyNvVideoCodec import failed → fall back to PyAV:", e)
            self.meta_q.put("FALLBACK_TO_PYAV")
            return

        # CUDA primary context + stream via Driver API (ctypes)
        if sys.platform.startswith("win"):
            _cuda = ctypes.windll.LoadLibrary("nvcuda.dll")
        else:
            _cuda = ctypes.cdll.LoadLibrary("libcuda.so")
        CUDA_SUCCESS = 0
        CUdevice = ctypes.c_int
        CUcontext = ctypes.c_void_p
        CUstream = ctypes.c_void_p
        def _ck(res, what):
            if res != CUDA_SUCCESS:
                raise RuntimeError(f"{what} failed ({res})")
        cuInit = _cuda.cuInit; cuInit.argtypes=[ctypes.c_uint]; cuInit.restype=ctypes.c_int
        cuDeviceGet = _cuda.cuDeviceGet; cuDeviceGet.argtypes=[ctypes.POINTER(CUdevice), ctypes.c_int]; cuDeviceGet.restype=ctypes.c_int
        cuDevicePrimaryCtxRetain = _cuda.cuDevicePrimaryCtxRetain; cuDevicePrimaryCtxRetain.argtypes=[ctypes.POINTER(CUcontext), CUdevice]; cuDevicePrimaryCtxRetain.restype=ctypes.c_int
        cuDevicePrimaryCtxRelease = _cuda.cuDevicePrimaryCtxRelease; cuDevicePrimaryCtxRelease.argtypes=[CUdevice]; cuDevicePrimaryCtxRelease.restype=ctypes.c_int
        cuCtxSetCurrent = _cuda.cuCtxSetCurrent; cuCtxSetCurrent.argtypes=[CUcontext]; cuCtxSetCurrent.restype=ctypes.c_int
        _cuStreamDestroy_sym = "cuStreamDestroy_v2" if hasattr(_cuda, "cuStreamDestroy_v2") else "cuStreamDestroy"
        cuStreamDestroy = getattr(_cuda, _cuStreamDestroy_sym); cuStreamDestroy.argtypes=[CUstream]; cuStreamDestroy.restype=ctypes.c_int
        cuStreamCreate = _cuda.cuStreamCreate; cuStreamCreate.argtypes=[ctypes.POINTER(CUstream), ctypes.c_uint]; cuStreamCreate.restype=ctypes.c_int

        def _make_ctx_stream(gpuid: int):
            _ck(cuInit(0), "cuInit")
            dev = CUdevice(); _ck(cuDeviceGet(ctypes.byref(dev), gpuid), "cuDeviceGet")
            ctx = CUcontext(); _ck(cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
            _ck(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
            stream = CUstream(); _ck(cuStreamCreate(ctypes.byref(stream), 0), "cuStreamCreate")
            return dev, ctx, stream

        def _destroy(dev, stream, ctx):
            try:
                if stream: cuStreamDestroy(stream)
            except Exception: pass
            try:
                if ctx: cuCtxSetCurrent(ctx)
                if dev is not None: cuDevicePrimaryCtxRelease(dev)
            except Exception: pass

        OCT = nvc.OutputColorType
        LTY = nvc.DisplayDecodeLatencyType

        dev=None; ctx=None; stream=None
        dmx=None; dec=None
        fid=0; cnt=0; t0=time.perf_counter()
        self.stats['src_fps'] = float(self.stats.get('src_fps', DEFAULT_REALTIME_FPS))
        try:
            with threading.Lock():
                dmx = nvc.CreateDemuxer(self.path)
                codec = dmx.GetNvCodecId() if hasattr(dmx, "GetNvCodecId") else getattr(dmx, "GetVideoCodec", lambda: nvc.cudaVideoCodec.H264)()
                dev, ctx, stream = _make_ctx_stream(0)
                ctx_handle = int(ctx.value) if ctx else 0
                stream_handle = int(stream.value) if stream else 0
                dec = nvc.CreateDecoder(0, codec, ctx_handle, stream_handle, True, True, 0, 0, OCT.NATIVE, False, LTY.LOW)
            while not self.stop_flag:
                with self.telemetry.meter("decode").span() if self.telemetry else nullcontext():
                    bs = dmx.Demux()
                if not bs: break
                surfaces = dec.Decode(bs)
                if surfaces is None: continue
                if not isinstance(surfaces, (list, tuple)):
                    surfaces=[surfaces]
                for surf in surfaces:
                    t = torch.utils.dlpack.from_dlpack(surf)  # NV12 2D (H*3/2, W) uint8 cuda
                    if SAFE_DETACH: t = t.clone()
                    h12, w = t.shape; h = (h12*2)//3

                    if self.ring is None:
                        self.ring = CudaFrameRing(slots=RING_SLOTS, h=h, w=w)
                        self.frame_store.set_shape((h,w,3))

                    slot = self.ring.write(t, handle=surf)

                    # GUI preview: NV12 → CPU → RGB (preview only)
                    try:
                        with self.telemetry.meter("copy").span() if self.telemetry else nullcontext():
                            nv12_cpu = t.detach().to("cpu", non_blocking=False).contiguous().numpy()
                        with self.telemetry.meter("colorspace").span() if self.telemetry else nullcontext():
                            rgb = cv2.cvtColor(nv12_cpu[:(h+h//2), :], cv2.COLOR_YUV2RGB_NV12)
                    except Exception:
                        rgb = np.zeros((h,w,3), np.uint8)

                    self.frame_store.put(fid, rgb)
                    pts = 0.0
                    try: pts = float(getattr(dmx, "GetLastPTS", lambda: 0.0)())
                    except: pass
                    fm = FrameMeta(stream_id="cam0", fid=fid, pts=pts, slot=slot, shape=(h,w,1), fmt="NV12", device="cuda")
                    self.meta_q.put(fm)
                    fid += 1; cnt += 1
                    if cnt % 120 == 0:
                        el = time.perf_counter() - t0
                        fps = cnt/el if el>0 else 0.0
                        self.stats['dec_fps'] = fps
        except Exception as e:
            print("[NVDEC Decoder] error:", e)
        finally:
            self.meta_q.put(None)
            try: del dec
            except Exception: pass
            try: del dmx
            except Exception: pass
            _destroy(dev, stream, ctx)
            print("[Decoder] end")

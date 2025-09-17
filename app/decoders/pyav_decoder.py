# -*- coding: utf-8 -*-
from __future__ import annotations
import time, threading
from contextlib import nullcontext
from typing import Optional, Dict

import numpy as np
import cv2
import av

from core.types import FrameMeta
from core.ring_gpu import CudaFrameRing
from core.telemetry import Telemetry
from app.stores.frame_store import FrameStore
from app.util.ema import EMA
from app.util.rotation import apply_rotation_metadata
from app.wiring.bootstrap import RING_SLOTS, DEFAULT_REALTIME_FPS

class DecoderThread(threading.Thread):
    def __init__(self, path: str, meta_q, frame_store: FrameStore, stats: Dict, telemetry: Telemetry):
        super().__init__(daemon=True)
        self.path = path
        self.meta_q = meta_q
        self.frame_store = frame_store
        self.stats = stats
        self.telemetry = telemetry
        self.stop_flag = False
        self.ring: Optional[CudaFrameRing] = None

    def run(self):
        fid=0; cnt=0; t0=time.time(); ema=EMA(0.8)
        container=None
        try:
            container = av.open(self.path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            src_fps=None
            try:
                if stream.average_rate is not None:
                    src_fps=float(stream.average_rate)
            except: pass
            self.stats['src_fps'] = src_fps if (src_fps and src_fps>0) else DEFAULT_REALTIME_FPS

            for frame in container.decode(stream):
                if self.stop_flag: break
                with self.telemetry.meter("decode").span() if self.telemetry else nullcontext():
                    img = frame.to_ndarray(format="rgb24")
                img = apply_rotation_metadata(frame, img)
                h,w = img.shape[:2]
                if self.ring is None:
                    self.ring = CudaFrameRing(slots=RING_SLOTS, h=h, w=w)
                    self.frame_store.set_shape((h,w,3))
                self.frame_store.put(fid, img)
                slot = self.ring.write_from_cpu(img)
                fm = FrameMeta(stream_id="cam0", fid=fid, pts=time.time(), slot=slot, shape=(h,w,3), fmt="rgb24", device="cuda")
                self.meta_q.put(fm)
                fid += 1; cnt += 1
                now=time.time()
                if now-t0>=0.5:
                    ema.update(cnt/(now-t0))
                    self.stats['dec_fps']=ema.get()
                    t0=now; cnt=0
        except Exception as e:
            print("[PyAV Decoder] error:", e)
        finally:
            self.meta_q.put(None)
            try:
                if container: container.close()
            except: pass
            print("[Decoder] end")

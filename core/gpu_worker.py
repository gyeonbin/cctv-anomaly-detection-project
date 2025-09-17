# core/gpu_worker.py — NV12→RGB on CUDA, full-GPU YOLO input
from __future__ import annotations

import os
import time
import queue
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.utils import LOGGER as _ULTRA_LOGGER
import logging as _ul_logging
from contextlib import nullcontext

from core.types import FrameMeta, InferOut
from core.ring_gpu import CudaFrameRing
from core.telemetry import Telemetry

_perf = time.perf_counter_ns

# silence Ultralytics noisy warnings (e.g., tensor normalization hints)
try:
    _ULTRA_LOGGER.setLevel(_ul_logging.ERROR)
except Exception:
    pass

# ==== knobs ====
BATCH       = int(os.getenv("BATCH", "16"))
CONF_THRES  = float(os.getenv("CONF_THRES", "0.50"))
IOU_THRES   = float(os.getenv("IOU_THRES",  "0.50"))
PERSON_ONLY = os.getenv("PERSON_ONLY", "1") != "0"
IMG_SIZE    = int(os.getenv("YOLO_IMGSZ", "1280"))

OSNET_SIZE  = (256, 128)
FEAT_DIM    = 256


# --------------------------------------------------
# Telemetry helper
# --------------------------------------------------
def _meter_ctx(telemetry: Optional[Telemetry], name: str):
    """
    Return a context manager for a telemetry meter, accommodating either
    .start() or .span() APIs. Falls back to nullcontext() if unavailable.
    """
    if telemetry is None:
        return nullcontext()
    try:
        m = telemetry.meter(name)
    except Exception:
        return nullcontext()
    # prefer .start()
    if hasattr(m, "start") and callable(getattr(m, "start")):
        try:
            return m.start()
        except Exception:
            return nullcontext()
    # fallback .span()
    if hasattr(m, "span") and callable(getattr(m, "span")):
        try:
            return m.span()
        except Exception:
            return nullcontext()
    return nullcontext()


# --------------------------------------------------
# Utilities
# --------------------------------------------------
@torch.no_grad()
def _letterbox_chw_cuda(
    img_chw: torch.Tensor,
    new_size: int = 640,
    stride: int = 32,
    pad_value: float = 0.0,
):
    """Letterbox a single CHW float image on CUDA to square new_size, stride-aligned.
    Returns (out_chw, meta) where meta=(sx, sy, px, py, H, W, S) with
      - sx, sy: effective scale factors W→nw, H→nh
      - px, py: left, top padding in pixels (on output space S=new_size)
      - H, W: original size; S: output size
    """
    assert img_chw.is_cuda and img_chw.ndim == 3 and img_chw.shape[0] == 3
    _, h, w = img_chw.shape
    # scale preserving aspect
    r = min(new_size / float(h), new_size / float(w))
    nh, nw = int(round(h * r)), int(round(w * r))
    # ensure stride alignment (floor to stride)
    nh = max(1, (nh // stride) * stride)
    nw = max(1, (nw // stride) * stride)
    # resize
    resized = F.interpolate(img_chw.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False).squeeze(0)
    # pad to square
    out = torch.full((3, new_size, new_size), pad_value, device=img_chw.device, dtype=img_chw.dtype)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    out[:, top:top+nh, left:left+nw] = resized
    sx = nw / float(w)
    sy = nh / float(h)
    meta = (sx, sy, left, top, h, w, new_size)
    return out, meta


@torch.no_grad()
def _prep_batch_for_yolo(imgs_chw: List[torch.Tensor], size: int, stride: int = 32):
    """Return (BCHW float, metas list). metas[i] corresponds to imgs_chw[i]."""
    imgs: List[torch.Tensor] = []
    metas: List[tuple] = []
    for x in imgs_chw:
        y, m = _letterbox_chw_cuda(x, size, stride)
        imgs.append(y)
        metas.append(m)
    if len(imgs) == 0:
        # Make an empty, device-sane tensor
        dev = imgs_chw[0].device if len(imgs_chw) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.empty((0, 3, size, size), device=dev), metas
    return torch.stack(imgs, dim=0), metas


@torch.no_grad()
def _unletterbox_boxes_xyxy(boxes: torch.Tensor, meta) -> torch.Tensor:
    """Map boxes from letterboxed space (SxS) back to original HxW using meta from _letterbox_chw_cuda.
    meta=(sx, sy, px, py, H, W, S)
    """
    if boxes.numel() == 0:
        return boxes
    sx, sy, px, py, H, W, S = meta
    # Inverse: x = (x_l - px) / sx ; y = (y_l - py) / sy
    x1 = (boxes[:, 0] - px) / max(sx, 1e-6)
    y1 = (boxes[:, 1] - py) / max(sy, 1e-6)
    x2 = (boxes[:, 2] - px) / max(sx, 1e-6)
    y2 = (boxes[:, 3] - py) / max(sy, 1e-6)
    # clamp to original image bounds
    x1 = x1.clamp(0, W-1); x2 = x2.clamp(0, W-1)
    y1 = y1.clamp(0, H-1); y2 = y2.clamp(0, H-1)
    return torch.stack([x1, y1, x2, y2], dim=1)


@torch.no_grad()
def _nv12_to_rgb_cuda(nv12_2d: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """NV12 (H+H/2,W) uint8 CUDA → (CHW float in [0,1], HWC uint8 CUDA).
    - Upsample UV by nearest to H×W
    - BT.601 (limited range) YUV→RGB
    """
    assert nv12_2d.is_cuda and nv12_2d.dtype == torch.uint8 and nv12_2d.ndim == 2
    y = nv12_2d[:h, :].to(dtype=torch.float32)
    uv = nv12_2d[h:h + h//2, :].to(dtype=torch.float32)
    # UV interleaved at half res: (H/2,W) → (H/2,W/2,2)
    uv = uv.view(h//2, w//2, 2)
    u = uv[..., 0].unsqueeze(0).unsqueeze(0)  # 1×1×H/2×W/2
    v = uv[..., 1].unsqueeze(0).unsqueeze(0)
    u = F.interpolate(u, size=(h, w), mode='nearest').squeeze(0).squeeze(0)
    v = F.interpolate(v, size=(h, w), mode='nearest').squeeze(0).squeeze(0)

    c = y - 16.0
    d = u - 128.0
    e = v - 128.0
    # BT.601 limited-range
    r = 1.164 * c + 1.596 * e
    g = 1.164 * c - 0.392 * d - 0.813 * e
    b = 1.164 * c + 2.017 * d
    rgb = torch.stack([r, g, b], dim=0)  # CHW, 0..255
    rgb = rgb.clamp_(0.0, 255.0)
    rgb_f = (rgb / 255.0).contiguous()                 # CHW float in [0,1]
    rgb_u8_hwc = rgb.permute(1, 2, 0).round().to(torch.uint8)  # HWC uint8
    return rgb_f, rgb_u8_hwc


@torch.no_grad()
def _only_person_and_nms(xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter by class=person (0) and apply NMS on a single image.
    Returns (boxes[N,4], scores[N]) on the same device as inputs.
    """
    if xyxy.numel() == 0:
        return xyxy, conf
    if PERSON_ONLY:
        keep = (cls == 0)
        xyxy = xyxy[keep]; conf = conf[keep]
        if xyxy.numel() == 0:
            return xyxy, conf
    keep = nms(xyxy, conf, IOU_THRES)
    return xyxy[keep], conf[keep]


@torch.no_grad()
def _parse_yolo_results(results, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    per_boxes: List[torch.Tensor] = []
    per_scores: List[torch.Tensor] = []
    for r in results:
        if getattr(r, 'boxes', None) is None:
            per_boxes.append(torch.empty((0, 4), device=device))
            per_scores.append(torch.empty((0,), device=device))
            continue
        b = r.boxes
        xyxy = b.xyxy.to(device=device, dtype=torch.float32)
        sc = b.conf.to(device=device, dtype=torch.float32)
        cl = b.cls.to(device=device, dtype=torch.long) if getattr(b, 'cls', None) is not None else torch.zeros_like(sc, dtype=torch.long)
        xyxy, sc = _only_person_and_nms(xyxy, sc, cl)
        per_boxes.append(xyxy); per_scores.append(sc)
    return per_boxes, per_scores


@torch.no_grad()
def _map_boxes_back_to_original(per_boxes: List[torch.Tensor], metas_b: List[tuple]) -> List[torch.Tensor]:
    mapped: List[torch.Tensor] = []
    for bx, meta in zip(per_boxes, metas_b):
        if bx is None or bx.numel() == 0:
            mapped.append(bx if bx is not None else torch.empty((0, 4), device=per_boxes[0].device if per_boxes else "cuda"))
        else:
            mapped.append(_unletterbox_boxes_xyxy(bx, meta))
    return mapped


@torch.no_grad()
def _norm_for_osnet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def _simple_crops(
    frames_hwc_cuda: List[torch.Tensor],
    per_boxes: List[torch.Tensor],
    out_size=OSNET_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[H,W,3] uint8 cuda → [M,3,H,W] float on cuda; also return offsets for ragged split (CPU long)."""
    device = frames_hwc_cuda[0].device if frames_hwc_cuda else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crops: List[torch.Tensor] = []
    counts: List[int] = []
    for img, boxes in zip(frames_hwc_cuda, per_boxes):
        if boxes is None or boxes.numel() == 0:
            counts.append(0); continue
        H, W = img.shape[0], img.shape[1]
        b = boxes.round().to(dtype=torch.int64)
        b[:,0] = b[:,0].clamp(0, W-1); b[:,2] = b[:,2].clamp(0, W-1)
        b[:,1] = b[:,1].clamp(0, H-1); b[:,3] = b[:,3].clamp(0, H-1)
        kept = 0
        for (x1,y1,x2,y2) in b.tolist():
            if x2 <= x1 or y2 <= y1:
                continue
            patch = img[y1:y2, x1:x2, :].permute(2,0,1).float() / 255.0
            patch = F.interpolate(patch.unsqueeze(0), size=out_size, mode='bilinear', align_corners=False).squeeze(0)
            crops.append(patch)
            kept += 1
        counts.append(kept)
    offs = [0]
    acc = 0
    for k in counts:
        acc += int(k); offs.append(acc)
    offs_t = torch.tensor(offs, dtype=torch.long, device='cpu')
    if len(crops) == 0:
        return torch.empty((0,3,*out_size), device=device), offs_t
    return torch.stack(crops, dim=0).to(device), offs_t


# --------------------------------------------------
# Main worker
# --------------------------------------------------
@torch.inference_mode()
def gpu_worker_loop(
    meta_q,
    ring: CudaFrameRing,
    out_q,
    device: str = 'cuda',
    yolo_weights: str = 'yolov8n.pt',
    osnet_model: Optional[torch.nn.Module] = None,
    telemetry: Optional[Telemetry] = None,
    MAX_WAIT_MS: int = 8,
):
    dev = torch.device(device)
    yolo = YOLO(yolo_weights)
    # put model on device explicitly (avoid implicit transfers)
    try:
        yolo.to(dev)
    except Exception:
        pass

    # enforce overrides
    try:
        yolo.overrides = getattr(yolo, 'overrides', {})
        yolo.overrides['imgsz'] = IMG_SIZE
        yolo.overrides['conf']  = CONF_THRES
        yolo.overrides['iou']   = IOU_THRES
        if PERSON_ONLY:
            yolo.overrides['classes'] = [0]
    except Exception:
        pass

    osnet = osnet_model  # if None → skip embeddings

    s_yolo = torch.cuda.Stream(dev) if dev.type == 'cuda' else None
    s_feat = torch.cuda.Stream(dev) if (dev.type == 'cuda' and osnet is not None) else None

    last_stream_id = None
    prev_for_osnet = None  # (frames_hwc_u8, per_boxes, fids, pts, slots, per_scores)

    def _flush_prev():
        nonlocal prev_for_osnet
        if prev_for_osnet is None:
            return
        frames_prev, per_boxes_prev, fids_prev, pts_prev, slots_prev, per_scores_prev = prev_for_osnet
        if osnet is not None:
            crops_prev, offsets_prev = _simple_crops(frames_prev, per_boxes_prev, out_size=OSNET_SIZE)
            if s_feat is not None:
                with torch.cuda.stream(s_feat):
                    feats_prev = osnet(_norm_for_osnet(crops_prev)) if crops_prev.numel() else torch.empty((0, FEAT_DIM), device=dev)
                torch.cuda.current_stream().wait_stream(s_feat)
            else:
                feats_prev = osnet(_norm_for_osnet(crops_prev)) if crops_prev.numel() else torch.empty((0, FEAT_DIM), device=dev)
        else:
            feats_prev = torch.empty((0, FEAT_DIM), device=dev)
            offsets_prev = torch.tensor([0]*(len(per_boxes_prev)+1), dtype=torch.long, device='cpu')

        out_q.put(InferOut(
            stream_id=last_stream_id,
            fids=fids_prev,
            pts=pts_prev,
            boxes=per_boxes_prev,
            scores=per_scores_prev,
            classes=[None]*len(per_boxes_prev),  # keep shape per-frame
            feat_flat=feats_prev,
            feat_offsets=offsets_prev
        ))
        for s in slots_prev:
            ring.release(s)
        prev_for_osnet = None

    while True:
        # ---------------- batch collect ----------------
        batch_imgs_chw: List[torch.Tensor] = []      # YOLO input (CHW float[0,1], CUDA)
        batch_frames_hwc_u8: List[torch.Tensor] = [] # OSNet crops (HWC uint8, CUDA)
        batch_fids: List[int] = []
        batch_pts:  List[float] = []
        batch_slots: List[int] = []

        meta: FrameMeta = meta_q.get()
        if meta is None:
            _flush_prev()
            out_q.put(None)
            break
        if getattr(meta, 'stream_id', None) is not None:
            last_stream_id = meta.stream_id

        def _push_one(m: FrameMeta):
            t = ring.get(m.slot)  # NV12 2D (H*3/2,W) uint8 cuda OR RGB HWC cuda
            fmt = str(getattr(m, 'fmt', 'rgb24')).upper()
            if t.ndim == 2 or fmt == 'NV12' or (hasattr(m, 'shape') and m.shape[2] == 1):
                h, w, _ = m.shape
                with _meter_ctx(telemetry, 'colorspace_gpu'):
                    chw_f, hwc_u8 = _nv12_to_rgb_cuda(t, h, w)
                img_chw = chw_f  # [3,H,W] float in [0,1]
                img_hwc = hwc_u8  # [H,W,3] uint8
            else:
                # already RGB HWC CUDA (uint8 assumed)
                img_hwc = t
                img_chw = (t.permute(2,0,1).float() / 255.0)
            batch_imgs_chw.append(img_chw)
            batch_frames_hwc_u8.append(img_hwc)
            batch_fids.append(m.fid)
            batch_pts.append(m.pts)
            batch_slots.append(m.slot)

        _push_one(meta)
        t0 = _perf()
        while len(batch_imgs_chw) < BATCH:
            waited = (_perf() - t0) / 1e6
            remain = MAX_WAIT_MS - waited
            if remain <= 0:
                break
            try:
                m2: FrameMeta = meta_q.get(timeout=max(0.0, remain) * 1e-3)
            except queue.Empty:
                break
            if m2 is None:
                break
            _push_one(m2)

        # 빈 배치면 다음 루프로
        if len(batch_imgs_chw) == 0:
            continue

        # ---------------- YOLO (current) ----------------
        # Make BCHW, square, stride-aligned (Ultralytics LoadTensor requirement)
        imgs_b, metas_b = _prep_batch_for_yolo(batch_imgs_chw, IMG_SIZE, 32)
        if imgs_b.numel() == 0:
            # 안전 가드
            continue

        # Feed float32 0..255 to satisfy Ultralytics' loader (it will divide by 255 internally)
        imgs_b = (imgs_b * 255.0).round().to(torch.float32)

        kwargs = dict(conf=CONF_THRES, iou=IOU_THRES, imgsz=IMG_SIZE, verbose=False, device=str(dev))
        if PERSON_ONLY:
            kwargs['classes'] = [0]

        if s_yolo is not None:
            with torch.cuda.stream(s_yolo):
                yolo_results = yolo.predict(imgs_b, **kwargs)
            torch.cuda.current_stream().wait_stream(s_yolo)
        else:
            yolo_results = yolo.predict(imgs_b, **kwargs)

        # parse
        per_boxes, per_scores = _parse_yolo_results(yolo_results, device=dev)

        # ---------------- OSNet (previous) overlap ----------------
        _flush_prev()

        # Map YOLO boxes back to original HxW for overlay & crops
        per_boxes_orig = _map_boxes_back_to_original(per_boxes, metas_b)

        # hold current batch for OSNet next round (so we can release ring after OSNet done)
        prev_for_osnet = (batch_frames_hwc_u8, per_boxes_orig, batch_fids, batch_pts, batch_slots, per_scores)

    # end while

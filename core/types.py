# core/types.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch

Tensor = torch.Tensor

@dataclass(frozen=True)
class FrameMeta:
    stream_id: str
    fid: int
    pts: float
    slot: int                 # VRAM 링버퍼 슬롯 번호
    shape: Tuple[int, int, int]
    fmt: str                  # "rgb24" or "nv12" 등
    device: str = "cuda"      # 고정: cuda

@dataclass(frozen=True)
class InferOut:
    stream_id: str
    fids: List[int]
    pts: List[float]
    boxes: List[Tensor]       # per-frame [K_i, 4] (xyxy, float32, cuda)
    scores: List[Tensor]      # per-frame [K_i]
    classes: List[Tensor]     # per-frame [K_i]
    # 방법 A: 프레임별 리스트
    #features: Optional[List[Tensor]] = None  # per-frame [M_i, C] (cuda)
    # 방법 B (선호): 하나의 큰 텐서 + offsets (성능/전송에 유리)
    feat_flat: Optional[Tensor] = None       # [sum M_i, C] (cuda)
    feat_offsets: Optional[Tensor] = None    # [len(fids)+1] (cpu int64)

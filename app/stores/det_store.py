# -*- coding: utf-8 -*-
import threading
from typing import Optional, Dict
import numpy as np
from core.gpu_worker import FEAT_DIM  # 기존 상수 재사용

class DetStore:
    """fid -> (boxes[N,4], feats[N,C], scores[N] or None)"""
    def __init__(self, cap: int = 1024):
        self._boxes: Dict[int, np.ndarray] = {}
        self._feats: Dict[int, np.ndarray] = {}
        self._scores: Dict[int, Optional[np.ndarray]] = {}
        self._cap = cap
        self._lock = threading.Lock()

    def put(self, fid: int, boxes: np.ndarray, feats: Optional[np.ndarray], scores: Optional[np.ndarray]):
        with self._lock:
            self._boxes[fid] = boxes
            self._feats[fid] = feats if feats is not None else np.zeros((0, FEAT_DIM), dtype=np.float32)
            self._scores[fid] = scores
            if len(self._boxes) > self._cap:
                first = next(iter(self._boxes))
                self._boxes.pop(first, None)
                self._feats.pop(first, None)
                self._scores.pop(first, None)

    def get(self, fid: int):
        """Peek without removing; returns (boxes, feats, scores) or None if not ready."""
        with self._lock:
            if fid not in self._boxes:
                return None
            return (self._boxes[fid], self._feats.get(fid, None), self._scores.get(fid, None))

    def pop(self, fid: int):
        with self._lock:
            return self._boxes.pop(fid, None), self._feats.pop(fid, None), self._scores.pop(fid, None)

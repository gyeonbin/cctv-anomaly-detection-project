# -*- coding: utf-8 -*-
import threading
from typing import Optional, Tuple, Dict
import numpy as np

class FrameStore:
    def __init__(self, cap: int = 512):
        self._buf: Dict[int, np.ndarray] = {}
        self._cap = cap
        self._lock = threading.Lock()
        self.shape: Optional[Tuple[int, int, 3]] = None

    def set_shape(self, shape):
        self.shape = shape

    def put(self, fid: int, img: np.ndarray):
        with self._lock:
            self._buf[fid] = img
            if len(self._buf) > self._cap:
                first = next(iter(self._buf))
                self._buf.pop(first, None)

    def pop(self, fid: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._buf.pop(fid, None)

    def get(self, fid: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._buf.get(fid, None)

# core/ring_gpu.py
import threading
import torch
import numpy as np
from typing import Optional, Tuple

class CudaFrameRing:
    """
    VRAM 링 버퍼 (포맷 불문 torch.Tensor 보관).
    - write_from_cpu: CPU numpy(uint8) → torch.cuda Tensor 업로드 후 저장
    - write: 이미 GPU에 있는 torch.Tensor (NV12 2D, RGB 3D 등) 그대로 저장
    - get(slot): 저장된 torch.Tensor 반환
    - release(slot): 소비자가 사용 끝낸 슬롯 반환
    동일 프로세스/동일 CUDA 컨텍스트 사용 가정.
    """
    def __init__(self, slots: int, h: int, w: int):
        self.slots = int(slots)
        self.h, self.w = int(h), int(w)
        # 포맷 고정 사전할당을 없애고, 포맷-불가지론 저장소로 변경
        self._buf = [None] * self.slots           # type: list[Optional[torch.Tensor]]
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._free = list(range(self.slots))      # 가용 슬롯 스택

    @property
    def shape(self) -> Tuple[int, int, int]:
        # 이전 코드 호환: 기본적으로 RGB 예상 형태를 돌려주지만,
        # 실제 저장 텐서의 shape는 get(slot).shape를 신뢰해야 함.
        return (self.h, self.w, 3)

    def write_from_cpu(self, img_cpu_uint8: np.ndarray) -> int:
        """
        CPU Numpy(uint8) 이미지를 GPU로 업로드하여 저장.
        img_cpu_uint8: (H,W,3) 또는 (H*3//2, W) 등 어떤 형태든 허용.
        """
        if not isinstance(img_cpu_uint8, np.ndarray) or img_cpu_uint8.dtype != np.uint8:
            raise TypeError("img_cpu_uint8 must be numpy.ndarray with dtype=uint8")
        with self._cv:
            while not self._free:
                self._cv.wait()
            slot = self._free.pop()
        # 업로드 (non_blocking=False: 안정 우선)
        t = torch.from_numpy(img_cpu_uint8).to(device="cuda", non_blocking=False)
        self._buf[slot] = t
        return slot

    def write(self, t: torch.Tensor, handle=None) -> int:
        """
        이미 GPU 상에 존재하는 텐서를 그대로 저장.
        NV12 2D 텐서((H*3/2, W) uint8)나 RGB 3D 텐서((H,W,3) uint8) 모두 허용.
        handle은 필요한 경우 외부에서 참조용으로만 쓰이며, 여기서는 보관하지 않음.
        """
        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a torch.Tensor")
        if t.device.type != "cuda":
            raise ValueError("t must be on CUDA device")
        if t.dtype != torch.uint8:
            raise ValueError("t must be dtype=torch.uint8")
        with self._cv:
            while not self._free:
                self._cv.wait()
            slot = self._free.pop()
        self._buf[slot] = t
        return slot

    def get(self, slot: int) -> torch.Tensor:
        t = self._buf[slot]
        if t is None:
            raise RuntimeError(f"slot {slot} is empty")
        return t

    def release(self, slot: int):
        with self._cv:
            # 소비가 끝났으니 참조 제거
            self._buf[slot] = None
            self._free.append(slot)
            self._cv.notify()

    def capacity(self) -> int:
        return self.slots

    def free_count(self) -> int:
        # 잠금 없이도 지표 용도로 충분
        return len(self._free)

    def used(self) -> int:
        # 사용 중 슬롯 개수
        return self.slots - len(self._free)

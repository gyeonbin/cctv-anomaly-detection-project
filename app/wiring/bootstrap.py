# -*- coding: utf-8 -*-
"""
환경변수/상수 모음 + 백엔드 선택 유틸.
원본 ccc.py의 모듈 전역 변수를 그대로 가져와서 노출한다.
"""
from __future__ import annotations
import os

# ---- app knobs (from env) ----
BATCH = int(os.getenv("BATCH", "16"))
PERSON_ONLY = True
PLAYBACK_MODE = os.getenv("PLAYBACK_MODE", "realtime")  # or "max"
DEFAULT_REALTIME_FPS = 30.0
RING_SLOTS = int(os.getenv("RING_SLOTS", "64"))
LONG_EDGE = int(os.getenv("LONG_EDGE", "0"))  # 0 = keep 4K

UPLOAD_METHOD = os.getenv("WZC_UPLOAD_METHOD", "pbo").lower()  # subdata|pbo|persist
VSYNC = int(os.getenv("WZC_VSYNC", "0"))
GUI_FPS = int(os.getenv("GUI_FPS", "30"))
DECODE_BACKEND = os.getenv("DECODE_BACKEND", "pyav").lower() #cpu 디코딩: pyav gpu 디코딩: nvdec
SAFE_DETACH = True  # detach NVDEC surface tensor defensively

SIM_THRESH = 0.45

# ID/렌더 유지 계열
MAX_MISSES = 120          # ← 기존 30에서 ↑ (4초@30fps 가정)
KEEP_RENDER_MISSES = 30   # 검출 끊겨도 예측박스를 최대 30프레임 표시

EMB_EMA = 0.7
MAX_TRACKS = 512

IOU_THRESH_NEAR = 0.3
ALPHA_APP = 0.7

def pick_backend() -> str:
    """원본의 _pick_backend와 동일: nvdec 우선, 실패 시 pyav"""
    if DECODE_BACKEND == "pyav":
        return "pyav"
    try:
        import PyNvVideoCodec as _  # noqa:F401
        return "nvdec"
    except Exception:
        return "pyav"

# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
from .roi_manager import ROIManager

# 이벤트는 "enter"만 로그(침입). 필요시 "exit"도 확장 가능.
class IntrusionDetector:
    def __init__(self, roi: ROIManager):
        self.roi = roi
        self._prev_inside: Dict[int, bool] = {}  # tid -> inside?

    @staticmethod
    def bottom_center(bbox) -> Tuple[int,int]:
        x1,y1,x2,y2 = bbox
        return (int((x1+x2)*0.5), int(y2))

    def update(self, ids: List[int], boxes) -> List[Tuple[int, float]]:
        """ids[i]와 boxes[i] 기준으로 ROI 진입(enter) 발생시 [(tid, pts_sec), ...] 반환 (pts_sec은 호출측에서 채움)"""
        events = []
        if not self.roi.is_defined():  # ROI 미정의면 아무것도 안함
            return events
        poly = self.roi.polygon()

        for i, tid in enumerate(ids):
            bc = self.bottom_center(boxes[i])
            inside = ROIManager.point_in_poly(bc, poly)
            was = self._prev_inside.get(tid, None)
            if was is None:
                self._prev_inside[tid] = inside
            else:
                if not was and inside:
                    # enter 이벤트 (시간은 호출측에서 부여)
                    events.append((tid, -1.0))
                    self._prev_inside[tid] = True
                elif was and not inside:
                    # 필요시 exit 처리 가능
                    self._prev_inside[tid] = False
        return events

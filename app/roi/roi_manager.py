# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import List, Tuple

Point = Tuple[int, int]

class ROIManager:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self._poly: List[Point] = []
        self._load_if_exists()

    def polygon(self) -> List[Point]:
        return list(self._poly)

    def set_polygon(self, pts: List[Point]):
        self._poly = list(pts)

    def add_point(self, x: int, y: int):
        self._poly.append((int(x), int(y)))

    def pop_point(self):
        if self._poly:
            self._poly.pop()

    def clear(self):
        self._poly.clear()

    def is_defined(self) -> bool:
        return len(self._poly) >= 3

    def save(self):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"polygon": self._poly}, f, ensure_ascii=False, indent=2)

    def _load_if_exists(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._poly = [tuple(map(int, p)) for p in data.get("polygon", [])]
            except Exception:
                self._poly = []

    @staticmethod
    def point_in_poly(pt: Point, poly: List[Point]) -> bool:
        if len(poly) < 3: return False
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
                inside = not inside
        return inside

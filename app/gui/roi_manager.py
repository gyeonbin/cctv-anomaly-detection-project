from __future__ import annotations
import json
from typing import List, Tuple, Optional

Point = Tuple[float, float]

class ROIManager:
    def __init__(self):
        self._points: List[Point] = []

    # --- edit ---
    def add(self, x: float, y: float) -> None:
        self._points.append((float(x), float(y)))

    def undo(self) -> None:
        if self._points:
            self._points.pop()

    def clear(self) -> None:
        self._points.clear()

    # --- query ---
    def is_empty(self) -> bool:
        return len(self._points) == 0

    def n(self) -> int:
        return len(self._points)

    def polygon(self) -> List[Point]:
        return list(self._points)

    def set_polygon(self, pts: List[Point]) -> None:
        self._points = [(float(x), float(y)) for x, y in pts]

    # --- io ---
    def to_json_dict(self) -> dict:
        return {"polygon": [[x, y] for x, y in self._points]}

    def from_json_dict(self, d: dict) -> None:
        pts = d.get("polygon", [])
        self.set_polygon([(p[0], p[1]) for p in pts])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.from_json_dict(d)

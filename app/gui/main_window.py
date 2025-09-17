# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import time
from collections import deque
from typing import Optional, Dict, List, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QElapsedTimer, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QMessageBox
)

from core.types import InferOut
from core.gpu_worker import FEAT_DIM
from core.telemetry import Telemetry

from app.gui.gl_video_widget import GLVideoWidget
from app.stores.frame_store import FrameStore
from app.stores.det_store import DetStore
from app.tracking.reid_tracker import ReIDTracker
from app.wiring.bootstrap import DEFAULT_REALTIME_FPS, GUI_FPS, PLAYBACK_MODE
from app.roi.roi_manager import ROIManager
from app.roi.intrusion import IntrusionDetector


class MainWindow(QMainWindow):
    def __init__(self, res_q, frame_store: FrameStore, det_store: DetStore, stats: Dict, video_path: str, telemetry: Telemetry):
        super().__init__()
        self.setWindowTitle("NVDEC/PyAV → YOLO → OSNet → Triple PBO + GPU overlay")
        # 중앙 위젯: 상단바 + 좌(비디오) / 우(로그)
        self._central = QWidget(self)
        self.setCentralWidget(self._central)
        outer = QVBoxLayout(self._central); outer.setContentsMargins(6,6,6,6); outer.setSpacing(6)

        # 상단바
        top_bar = QHBoxLayout(); top_bar.setSpacing(8)
        self.btn_roi = QPushButton("ROI 설정")
        self.btn_save = QPushButton("ROI 저장")
        self.lbl_hint = QLabel("좌클릭: 점 추가 / 우클릭: 되돌리기")
        top_bar.addWidget(self.btn_roi); top_bar.addWidget(self.btn_save); top_bar.addWidget(self.lbl_hint); top_bar.addStretch(1)
        outer.addLayout(top_bar)

        # 본문: 좌측 GL, 우측 로그
        body = QHBoxLayout(); body.setSpacing(6)
        self.view = GLVideoWidget(telemetry, flip_y=True)
        body.addWidget(self.view, stretch=4)
        self.log_list = QListWidget()
        self.log_list.setMinimumWidth(260)
        body.addWidget(self.log_list, stretch=0)
        outer.addLayout(body, stretch=1)

        # 상태/큐
        self.res_q=res_q; self.frame_store=frame_store; self.det_store=det_store
        self.stats=stats; self.video_path=video_path; self.telemetry=telemetry
        self._pts_map: Dict[int, float] = {}                 # fid -> pts(sec)
        self._intrusion_logs: List[Tuple[int, float]] = []   # (tid, pts_sec)
        self._next_fid = 0

        # 타이머/폴링
        self._timer = QElapsedTimer(); self._timer.start()
        self._pts0=None; self._t0=None
        self._target_dt_ms = int(1000.0/max(float(stats.get('src_fps', DEFAULT_REALTIME_FPS)),1e-6))
        self._poll = QTimer(self); self._poll.setTimerType(Qt.PreciseTimer)
        if GUI_FPS>0:
            self._poll_interval_ms = max(1, int(1000.0/GUI_FPS))
        else:
            self._poll_interval_ms = 0 if PLAYBACK_MODE=="max" else max(1, self._target_dt_ms//2)
        self._poll.timeout.connect(self._on_tick)
        self._poll.start(self._poll_interval_ms)

        # 트래커
        self.tracker = ReIDTracker(
            FEAT_DIM,
            dt=1.0 / max(float(stats.get('src_fps', DEFAULT_REALTIME_FPS)), 1e-6),
            w_app=0.5, w_iou=0.3, w_mot=0.2
        )

        # ROI / 침입감지
        base = os.path.splitext(os.path.basename(str(video_path)))[0]
        folder = os.path.dirname(os.path.abspath(str(video_path)))
        roi_json = os.path.join(folder, f"{base}_roi.json")
        self.roi = ROIManager(roi_json)
        self.intrusion = IntrusionDetector(self.roi)

        try:
            if os.path.exists(roi_json):
                self.roi.load()
        except Exception as e:
            # 필요시 로그만 쌓고 넘어가기
            self.log_list.addItem(QListWidgetItem(f"[ROI] 로드 실패: {e}"))

        self.view.set_roi_points(self.roi.polygon())

        # 버튼 연결
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self._on_toggle_roi)
        self.btn_save.clicked.connect(self._on_save_roi)

    def _on_toggle_roi(self, on: bool):
        self.view.toggle_roi_edit(on)
        self.lbl_hint.setText("편집중: 좌클릭=점 추가 / 우클릭=되돌리기" if on else "좌클릭: 점 추가 / 우클릭: 되돌리기")

    def _on_save_roi(self):
        # GL 위젯의 현재 ROI 포인트를 저장
        pts = self.view.get_roi_points()
        if not pts:
            QMessageBox.warning(self, "ROI 저장", "ROI가 비어 있습니다. 점을 먼저 찍어주세요.")
            return

        self.roi.set_polygon(pts)
        try:
            self.roi.save()  # 내부에서 경로(생성자 인자)로 저장
            # 저장 성공 알림
            QMessageBox.information(self, "ROI 저장", "저장되었습니다!")
            # 상태바가 있다면 짧게 노출
            self.statusBar().showMessage("ROI 저장되었습니다!", 2000) if self.statusBar() else None
        except Exception as e:
            QMessageBox.critical(self, "ROI 저장 실패", str(e))

    # 결과 드레인 (fid→pts 맵 저장 포함)
    def _drain_results_to_buffer(self, budget_items:int=4):
        drained=0
        while drained<budget_items:
            try:
                out: Optional[InferOut] = self.res_q.get_nowait()
            except Exception:
                break
            if out is None:
                self.close(); return False
            drained+=1
            if self._pts0 is None and len(out.pts)>0:
                self._pts0 = out.pts[0]; self._t0 = self._timer.elapsed()

            offs = None
            if getattr(out, 'feat_offsets', None) is not None:
                offs = np.asarray(out.feat_offsets.cpu().numpy() if hasattr(out.feat_offsets, 'cpu') else out.feat_offsets, dtype=np.int64)
            feats_flat_np = None
            if getattr(out, 'feat_flat', None) is not None:
                ff = out.feat_flat
                feats_flat_np = ff.detach().float().cpu().numpy()

            # per frame split + pts 저장
            for i, fid in enumerate(out.fids):
                boxes_i = out.boxes[i]
                boxes_np = boxes_i.detach().float().cpu().numpy() if hasattr(boxes_i, 'detach') else np.asarray(boxes_i, dtype=np.float32)
                scores_np = None
                if getattr(out, 'scores', None) is not None:
                    si = out.scores[i]
                    scores_np = si.detach().float().cpu().numpy() if hasattr(si,'detach') else np.asarray(si, dtype=np.float32)
                feats_i = None
                if feats_flat_np is not None and offs is not None:
                    a = int(offs[i]); b = int(offs[i+1])
                    feats_i = feats_flat_np[a:b]
                self.det_store.put(int(fid), boxes_np, feats_i, scores_np)

                # pts 저장
                try:
                    pts = float(out.pts[i])
                except Exception:
                    pts = 0.0
                self._pts_map[int(fid)] = pts
        return True

    def _append_intrusion_log(self, tid: int, pts_sec: float):
        self._intrusion_logs.append((tid, pts_sec))
        item = QListWidgetItem(f"ID {tid}  •  {pts_sec:0.3f}s")
        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    def _on_tick(self):
        if not self._drain_results_to_buffer(8):
            return

        fid = self._next_fid

        det_ready = self.det_store.get(fid) is not None
        frm_ready = self.frame_store.get(fid) is not None
        if not (det_ready and frm_ready):
            return

        img = self.frame_store.pop(fid)
        boxes, feats, scores = self.det_store.pop(fid)
        if img is None:
            return
        if boxes is None:
            boxes = np.zeros((0,4), dtype=np.float32)

        ids, smooth_boxes = self.tracker.update(0.0, boxes, feats)

        # 침입 감지
        events = self.intrusion.update(ids, smooth_boxes)
        if events:
            pts_sec = float(self._pts_map.get(fid, 0.0))
            for (tid, _) in events:
                self._append_intrusion_log(tid, pts_sec)

        # ROI 동기화 & 렌더
        # 편집 중일 땐 덮어쓰지 않는다 (점들이 사라지는 원인 제거)
        if not self.btn_roi.isChecked():
            roi_poly = self.roi.polygon()
            # 저장된 ROI가 "비어있지" 않을 때만 반영 (빈 값으로 덮어쓰기 방지)
            if roi_poly:
                self.view.set_roi_points(roi_poly)

        self.view.set_frame(img)
        self.view.set_boxes(smooth_boxes, ids)

        self._next_fid += 1

# -*- coding: utf-8 -*-
"""
GLVideoWidget
- CPU RGB 프레임을 OpenGL 텍스처로 업로드(PBO/persist 지원)하여 렌더
- 박스(ID 색상) 오버레이를 GPU 라인으로 그림
- ROI 편집(토글): 좌클릭으로 점 추가, 우클릭으로 마지막 점 되돌리기
- ROI 폴리곤/포인트를 GPU 라인/포인트 + 반투명 채움으로 렌더
- ROI 저장/로드: <영상파일경로>_roi.json 형식으로 {"polygon": [[x,y], ...]}

외부에서 쓰는 주요 API:
- set_frame(img_rgb: np.ndarray)
- set_boxes(boxes_xyxy: np.ndarray, ids: List[int])
- set_roi_points(pts: List[Tuple[int,int]])     # ROI 갱신(로드/저장 후 동기화)
- get_roi_points() -> List[Tuple[int,int]]      # 현재 위젯에 표시 중인 ROI 가져오기
- toggle_roi_edit(on: bool)                     # ROI 편집 모드 토글
- set_video_path(path: str)                     # 저장/로드 기본 경로를 위해 필수
- save_roi(path: Optional[str] = None) -> bool  # 저장 버튼에서 호출
- load_roi(path: Optional[str] = None, auto: bool=False) -> bool
"""
from __future__ import annotations
import os
import json
import ctypes
from contextlib import nullcontext
from typing import Optional, Tuple, List

import numpy as np
from OpenGL import GL
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox  # 저장/로드 알림용

from core.telemetry import Telemetry
from app.wiring.bootstrap import UPLOAD_METHOD, VSYNC


class GLVideoWidget(QGLWidget):
    def __init__(self, telemetry: Telemetry, flip_y: bool = True):
        super().__init__()
        self.setAutoBufferSwap(True)
        self._tex = None
        self._pbo = [0, 0, 0]
        self._pbo_ptr = [None, None, None]
        self._cur = 0
        self._w = 0
        self._h = 0
        self._pbo_size = 0
        self._has_frame = False
        self._latest: Optional[np.ndarray] = None
        self.flip_y = flip_y
        self.telemetry = telemetry
        self._upload_method = UPLOAD_METHOD
        self._pbo_supported = False
        self._maprange_supported = False
        self._persist_supported = False
        self._boxes: Optional[np.ndarray] = None
        self._ids: Optional[list] = None
        self._line_thick = 2.0

        # ROI 편집/표시용
        self._roi_points: List[Tuple[int, int]] = []
        self._roi_edit_mode: bool = False
        self._roi_line_width = 2.0

        # ROI 저장/로드 기본 경로 생성을 위한 비디오 경로
        self._video_path: Optional[str] = None

    # ======================= External API =======================

    def set_video_path(self, path: str):
        """저장/로드 기본 경로(<path>_roi.json) 생성을 위해 비디오 경로 설정."""
        self._video_path = path

    def set_frame(self, img_rgb: np.ndarray):
        """새 프레임 설정 (CPU RGB)"""
        self._latest = img_rgb
        self._has_frame = True
        self.update()

    def set_boxes(self, boxes_xyxy, ids):
        """검출/트래킹 박스와 ID(동일 인덱스)를 설정"""
        self._boxes = None if boxes_xyxy is None else boxes_xyxy.astype(np.float32, copy=False)
        self._ids = ids
        self.update()

    def set_roi_points(self, pts: List[Tuple[int, int]]):
        """외부(저장/로드 등)에서 갱신한 ROI를 위젯에 반영"""
        self._roi_points = [(int(x), int(y)) for x, y in pts]
        self.update()

    def get_roi_points(self) -> List[Tuple[int, int]]:
        """현재 위젯에 표시 중인 ROI 포인트 목록 반환"""
        return list(self._roi_points)

    def toggle_roi_edit(self, on: bool):
        """ROI 편집 모드 토글 (True면 좌클릭: 점 추가, 우클릭: 되돌리기)"""
        self._roi_edit_mode = bool(on)
        self.update()

    # ======================= ROI Save/Load =======================

    def _default_roi_path(self) -> Optional[str]:
        if not self._video_path:
            return None
        root, _ = os.path.splitext(self._video_path)
        return f"{root}_roi.json"

    def save_roi(self, path: Optional[str] = None) -> bool:
        """ROI를 JSON으로 저장. 성공 시 True."""
        if len(self._roi_points) == 0:
            QMessageBox.warning(self, "ROI 저장", "ROI가 비어 있습니다. 점을 먼저 찍어주세요.")
            return False
        p = path or self._default_roi_path()
        if not p:
            QMessageBox.warning(self, "ROI 저장", "영상 경로를 알 수 없습니다. set_video_path()를 먼저 호출하세요.")
            return False
        try:
            data = {"polygon": [[int(x), int(y)] for (x, y) in self._roi_points]}
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "ROI 저장", "저장되었습니다!")
            return True
        except Exception as e:
            QMessageBox.critical(self, "ROI 저장 실패", str(e))
            return False

    def load_roi(self, path: Optional[str] = None, auto: bool = False) -> bool:
        """ROI를 JSON에서 로드. 성공 시 True. auto=True면 파일 없을 때 메시지 생략."""
        p = path or self._default_roi_path()
        if not p:
            if not auto:
                QMessageBox.warning(self, "ROI 로드", "영상 경로를 알 수 없습니다. set_video_path()를 먼저 호출하세요.")
            return False
        if not os.path.exists(p):
            if not auto:
                QMessageBox.information(self, "ROI 로드", "저장된 ROI 파일이 없습니다.")
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            pts = d.get("polygon", [])
            self._roi_points = [(int(px), int(py)) for (px, py) in pts]
            self.update()
            if not auto:
                QMessageBox.information(self, "ROI 로드", "불러왔습니다.")
            return True
        except Exception as e:
            QMessageBox.critical(self, "ROI 로드 실패", str(e))
            return False

    # ======================= GL / Upload Probes =======================

    def _gl_clear_error(self):
        try:
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass
        except Exception:
            pass

    def _probe_pbo(self) -> bool:
        self._gl_clear_error()
        try:
            tmp = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, tmp)
            GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, 16, None, GL.GL_STREAM_DRAW)
            err = GL.glGetError()
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            GL.glDeleteBuffers(int(tmp))
            return err == GL.GL_NO_ERROR
        except Exception:
            return False

    def _probe_maprange(self) -> bool:
        if not hasattr(GL, "glMapBufferRange"):
            return False
        self._gl_clear_error()
        try:
            tmp = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, tmp)
            GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, 32, None, GL.GL_STREAM_DRAW)
            ptr = GL.glMapBufferRange(
                GL.GL_PIXEL_UNPACK_BUFFER, 0, 32, GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT
            )
            if ptr:
                GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)
            err = GL.glGetError()
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            GL.glDeleteBuffers(int(tmp))
            return err == GL.GL_NO_ERROR and bool(ptr)
        except Exception:
            return False

    def _probe_persistent(self) -> bool:
        if not hasattr(GL, "glBufferStorage") or not hasattr(GL, "glMapBufferRange"):
            return False
        self._gl_clear_error()
        try:
            tmp = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, tmp)
            storage_flags = GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT
            map_flags = GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT
            GL.glBufferStorage(GL.GL_PIXEL_UNPACK_BUFFER, 32, None, storage_flags)
            ptr = GL.glMapBufferRange(GL.GL_PIXEL_UNPACK_BUFFER, 0, 32, map_flags)
            ok = (GL.glGetError() == GL.GL_NO_ERROR) and bool(ptr)
            if ptr:
                GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            GL.glDeleteBuffers(int(tmp))
            return ok
        except Exception:
            return False

    # ======================= GL Lifecycle =======================

    def initializeGL(self):
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glClearColor(0.05, 0.05, 0.05, 1.0)

        self._pbo_supported = self._probe_pbo()
        self._maprange_supported = self._probe_maprange() if self._pbo_supported else False
        self._persist_supported = self._probe_persistent() if (self._pbo_supported and self._upload_method == "persist") else False
        if self._upload_method == "persist" and not self._persist_supported:
            self._upload_method = "pbo"
        if self._upload_method == "pbo" and not self._pbo_supported:
            self._upload_method = "subdata"

        self._tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        if self._upload_method in ("pbo", "persist"):
            self._pbo = GL.glGenBuffers(3)

        # try turn VSync off
        if VSYNC == 0:
            try:
                from OpenGL.WGL.EXT.swap_control import wglSwapIntervalEXT
                wglSwapIntervalEXT(0)
            except Exception:
                try:
                    from OpenGL.GLX.SGI.swap_control import glXSwapIntervalSGI
                    glXSwapIntervalSGI(0)
                except Exception:
                    pass

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)

    # ======================= Upload & Draw Helpers =======================

    def _ensure_tex(self, w, h):
        if w == self._w and h == self._h:
            return
        self._w, self._h = w, h
        self._pbo_size = w * h * 3
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        if self._upload_method == "subdata":
            return
        # unmap old persistent maps
        try:
            for i in range(3):
                if self._pbo_ptr[i]:
                    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo[i])
                    try:
                        GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)
                    except Exception:
                        pass
                    self._pbo_ptr[i] = None
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        except Exception:
            pass
        # recreate PBOs
        try:
            for buf in self._pbo:
                if buf:
                    GL.glDeleteBuffers(int(buf))
        except Exception:
            pass
        self._pbo = GL.glGenBuffers(3)

        if self._upload_method == "persist" and self._persist_supported:
            try:
                storage_flags = GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT
                map_flags = GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT
                for i in range(3):
                    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo[i])
                    GL.glBufferStorage(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo_size, None, storage_flags)
                    ptr = GL.glMapBufferRange(GL.GL_PIXEL_UNPACK_BUFFER, 0, self._pbo_size, map_flags)
                    if not ptr:
                        raise RuntimeError("persistent map failed")
                    self._pbo_ptr[i] = ptr
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            except Exception:
                self._upload_method = "pbo"
                self._persist_supported = False

        if self._upload_method == "pbo":
            for i in range(3):
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo[i])
                GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo_size, None, GL.GL_STREAM_DRAW)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _id_color(self, tid: Optional[int]) -> Tuple[float, float, float]:
        if tid is None:
            return (0.0, 1.0, 0.0)
        i = int(tid)
        r = (i * 123457) & 255
        g = (i * 35467) & 255
        b = (i * 76493) & 255
        return (r / 255.0, g / 255.0, b / 255.0)

    def _draw_boxes_gpu(self, w: int, h: int):
        boxes = self._boxes
        if boxes is None or boxes.size == 0:
            return
        ctx = self.telemetry.meter("overlay").span() if self.telemetry else nullcontext()
        with ctx:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glLineWidth(max(1.0, float(self._line_thick)))
            for idx in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[idx].tolist()
                tid = None
                if self._ids is not None and idx < len(self._ids):
                    tid = self._ids[idx]
                r, g, b = self._id_color(tid)
                GL.glColor4f(r, g, b, 1.0)

                def ndc(px, py):
                    nx = (px / float(w)) * 2.0 - 1.0
                    ny = 1.0 - (py / float(h)) * 2.0
                    return nx, ny

                x1n, y1n = ndc(x1, y1)
                x2n, y2n = ndc(x2, y2)
                GL.glBegin(GL.GL_LINE_LOOP)
                GL.glVertex2f(x1n, y1n)
                GL.glVertex2f(x2n, y1n)
                GL.glVertex2f(x2n, y2n)
                GL.glVertex2f(x1n, y2n)
                GL.glEnd()
            GL.glColor4f(1.0, 1.0, 1.0, 1.0)
            GL.glDisable(GL.GL_BLEND)

    # ----------------------- ROI Drawing/Editing -----------------------

    def _draw_roi_gpu(self, w: int, h: int):
        pts = self._roi_points
        if len(pts) == 0:
            return

        def ndc(px, py):
            nx = (px / float(w)) * 2.0 - 1.0
            ny = 1.0 - (py / float(h)) * 2.0
            return nx, ny

        ndc_pts = [ndc(px, py) for (px, py) in pts]

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # ① 채움 (3점 이상일 때)
        if len(ndc_pts) >= 3:
            GL.glColor4f(1.0, 1.0, 0.0, 0.18)  # 반투명 노랑
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            for (nx, ny) in ndc_pts:
                GL.glVertex2f(nx, ny)
            GL.glEnd()

        # ② 외곽선
        GL.glColor4f(1.0, 1.0, 0.0, 0.95)
        GL.glLineWidth(max(1.0, float(self._roi_line_width)))
        GL.glBegin(GL.GL_LINE_LOOP if len(ndc_pts) >= 3 else GL.GL_LINE_STRIP)
        for (nx, ny) in ndc_pts:
            GL.glVertex2f(nx, ny)
        GL.glEnd()

        # ③ 점 표시
        GL.glPointSize(5.0)
        GL.glBegin(GL.GL_POINTS)
        for (nx, ny) in ndc_pts:
            GL.glVertex2f(nx, ny)
        GL.glEnd()

        GL.glDisable(GL.GL_BLEND)
        GL.glColor4f(1.0, 1.0, 1.0, 1.0)  # 상태 클리어

    def mousePressEvent(self, e):
        # ROI 편집 모드에서만 처리 (좌클릭 추가 / 우클릭 되돌리기)
        if not self._roi_edit_mode or self._latest is None:
            return super().mousePressEvent(e)

        # 위젯 좌표 → 이미지 픽셀 좌표 (현재는 영상이 위젯을 꽉 채워 있다고 가정)
        w = max(1, self.width())
        h = max(1, self.height())
        img_h, img_w = self._latest.shape[:2]

        x_widget = e.x()
        y_widget = e.y()

        # 간단 비율 매핑 (레터박스 없음 가정)
        x_img = int(x_widget * img_w / w)
        y_img = int(y_widget * img_h / h)

        if e.button() == Qt.LeftButton:
            self._roi_points.append((x_img, y_img))
        elif e.button() == Qt.RightButton:
            if self._roi_points:
                self._roi_points.pop()

        self.update()

    # ======================= Paint =======================

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if not self._has_frame or self._latest is None:
            return

        img = self._latest
        h, w = img.shape[:2]
        self._ensure_tex(w, h)

        ctx = self.telemetry.meter("render").span() if self.telemetry else nullcontext()
        with ctx:
            if self._upload_method == "subdata":
                with self.telemetry.meter("upload").span() if self.telemetry else nullcontext():
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
                    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            else:
                idx = self._cur
                try:
                    with self.telemetry.meter("upload").span() if self.telemetry else nullcontext():
                        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo[idx])
                        if self._persist_supported and self._pbo_ptr[idx]:
                            ctypes.memmove(int(self._pbo_ptr[idx]), img.ctypes.data, self._pbo_size)
                        else:
                            if self._maprange_supported and hasattr(GL, "glMapBufferRange"):
                                GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo_size, None, GL.GL_STREAM_DRAW)
                                ptr = GL.glMapBufferRange(
                                    GL.GL_PIXEL_UNPACK_BUFFER, 0, self._pbo_size,
                                    GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT
                                )
                                if ptr:
                                    ctypes.memmove(int(ptr), img.ctypes.data, self._pbo_size)
                                    GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)
                                else:
                                    GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo_size, None, GL.GL_STREAM_DRAW)
                                    GL.glBufferSubData(GL.GL_PIXEL_UNPACK_BUFFER, 0, img)
                            else:
                                if hasattr(GL, "glMapBuffer"):
                                    GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo_size, None, GL.GL_STREAM_DRAW)
                                    ptr = GL.glMapBuffer(GL.GL_PIXEL_UNPACK_BUFFER, GL.GL_WRITE_ONLY)
                                    if ptr:
                                        ctypes.memmove(int(ptr), img.ctypes.data, self._pbo_size)
                                        GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)
                                    else:
                                        GL.glBufferSubData(GL.GL_PIXEL_UNPACK_BUFFER, 0, img)
                                else:
                                    raise RuntimeError("No map buffer support")
                        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
                        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
                        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                except Exception:
                    try:
                        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
                    except Exception:
                        pass
                    self._upload_method = "subdata"
                    with self.telemetry.meter("upload").span() if self.telemetry else nullcontext():
                        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
                        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img)
                        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            # textured quad (orthographic)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glOrtho(-1, 1, -1, 1, -1, 1)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

            # 텍스처가 색상에 물들지 않도록 흰색으로 리셋
            GL.glColor4f(1.0, 1.0, 1.0, 1.0)

            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
            GL.glBegin(GL.GL_QUADS)
            # flip-Y 고정 매핑
            GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(-1.0, -1.0)
            GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f( 1.0, -1.0)
            GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f( 1.0,  1.0)
            GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(-1.0,  1.0)
            GL.glEnd()
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glDisable(GL.GL_TEXTURE_2D)

            # draw overlays
            self._draw_boxes_gpu(w, h)
            self._draw_roi_gpu(w, h)

        if self._upload_method in ("pbo", "persist"):
            self._cur = (self._cur + 1) % 3

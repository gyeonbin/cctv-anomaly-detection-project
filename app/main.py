# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, time, queue, threading, argparse

from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QApplication

from core.types import FrameMeta, InferOut
from core.ring_gpu import CudaFrameRing
from core.gpu_worker import gpu_worker_loop
from core.telemetry import Telemetry

from app.wiring.bootstrap import pick_backend, RING_SLOTS, VSYNC
from app.stores.frame_store import FrameStore
from app.stores.det_store import DetStore
from app.gui.main_window import MainWindow
from app.decoders.nvdec_decoder import NVDECDecoderThread
from app.decoders.pyav_decoder import DecoderThread

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('video', help='input video file path')
    args = p.parse_args(argv)

    telemetry = Telemetry(csv_path=os.getenv('TELEM_CSV','telemetry.csv'), period_sec=1.0)
    telemetry.start()

    meta_q: "queue.Queue[FrameMeta]" = queue.Queue(maxsize=RING_SLOTS)
    res_q:  "queue.Queue[InferOut]"  = queue.Queue(maxsize=RING_SLOTS)
    frame_store = FrameStore(cap=1024)
    det_store   = DetStore(cap=2048)
    stats: dict = {}

    backend = pick_backend()
    if backend=="nvdec":
        dec = NVDECDecoderThread(args.video, meta_q, frame_store, stats, telemetry)
    else:
        dec = DecoderThread(args.video, meta_q, frame_store, stats, telemetry)
    dec.start()

    # wait for ring
    ring: CudaFrameRing | None = None
    for _ in range(500):  # ~5s
        if getattr(dec, 'ring', None) is not None:
            ring = dec.ring; break
        time.sleep(0.01)
    if ring is None:
        print("[Main] decoder failed to create ring; falling back to PyAV")
        try: dec.stop_flag=True
        except Exception: pass
        dec = DecoderThread(args.video, meta_q, frame_store, stats, telemetry)
        dec.start()
        for _ in range(500):
            if getattr(dec, 'ring', None) is not None:
                ring = dec.ring; break
            time.sleep(0.01)
        if ring is None:
            print("[Main] no ring; exiting")
            return 1

    telemetry.gauge('q_meta', lambda: meta_q.qsize())
    telemetry.gauge('q_infer', lambda: res_q.qsize())
    telemetry.gauge('ring_used', lambda: ring.used() if ring else 0)
    telemetry.gauge('ring_cap',  lambda: ring.capacity() if ring else 0)

    worker_thr = threading.Thread(
        target=gpu_worker_loop,
        kwargs=dict(meta_q=meta_q, ring=ring, out_q=res_q, device='cuda',
                    yolo_weights=os.getenv('YOLO_WEIGHTS','yolov8n.pt'),
                    osnet_model=None, telemetry=telemetry, MAX_WAIT_MS=40),
        daemon=True,
    )
    worker_thr.start()

    fmt = QSurfaceFormat(); fmt.setSwapInterval(1 if VSYNC else 0); QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    win = MainWindow(res_q, frame_store, det_store, stats, args.video, telemetry)
    win.resize(1280, 720)
    win.show()
    code = app.exec_()

    # teardown
    try: dec.stop_flag=True
    except Exception: pass
    meta_q.put(None)
    res_q.put(None)
    worker_thr.join(timeout=1.0)
    telemetry.stop()
    return code

if __name__ == '__main__':
    raise SystemExit(main())

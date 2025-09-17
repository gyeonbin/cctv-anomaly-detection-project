# core/telemetry.py
import time, csv, threading, collections
from contextlib import contextmanager

_perf = time.perf_counter_ns

class EMA:
    def __init__(self, alpha=0.2): self.v=None; self.a=alpha
    def add(self, x): self.v = x if self.v is None else (self.a*x + (1-self.a)*self.v)
    def value(self): return 0.0 if self.v is None else float(self.v)

class PWin:
    def __init__(self, cap=512): self.d=collections.deque(maxlen=cap)
    def add(self, x): self.d.append(x)
    def p(self, q):
        if not self.d: return 0.0
        arr = sorted(self.d); k = max(0, min(len(arr)-1, int((q/100.0)*(len(arr)-1))))
        return float(arr[k])

class Counter:
    def __init__(self): self.n=0
    def inc(self, k=1): self.n+=k
    def get(self): return self.n

class Meter:
    def __init__(self):
        self.ema = EMA(0.2); self.win = PWin(512); self.counter = Counter()
    @contextmanager
    def span(self):
        t0=_perf()
        try: yield
        finally:
            dt_ms=( _perf()-t0 )/1e6
            self.ema.add(dt_ms); self.win.add(dt_ms); self.counter.inc()
    def ema_ms(self): return round(self.ema.value(),3)
    def p95_ms(self): return round(self.win.p(95),3)
    def count(self):  return self.counter.get()

class Telemetry:
    def __init__(self, csv_path="telemetry.csv", period_sec=1.0):
        self.meters = collections.defaultdict(Meter)
        self.counters = collections.defaultdict(Counter)
        self.gauges = {}
        self._period = period_sec
        self._csv = csv_path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._fixed_cols = [
            "t",
            "decode_fps","decode_ms",
            "batch_fill_ms","batch_gap_ms","batch_size_eff","batch_timeout_hits",
            "yolo_fps","yolo_ms","post_ms",
            "embed_fps","embed_ms",
            "render_fps","render_ms","upload_ms","draw_ms",
            "q_meta","q_infer","ring_used","ring_cap",
        ]
        # 메모리 버퍼 (실행 중만 기록, 종료 시 flush)
        self._rows = []

    def start(self): self._thread.start()
    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
        # 종료 시 CSV로 flush
        try:
            with open(self._csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(self._fixed_cols)
                w.writerows(self._rows)
            print(f"[Telemetry] saved {len(self._rows)} rows -> {self._csv}")
        except Exception as e:
            print("[Telemetry] save failed:", e)

    def meter(self, name): return self.meters[name]
    def inc(self, name, k=1): self.counters[name].inc(k)
    def gauge(self, name, fn): self.gauges[name]=fn

    def _fps(self, ms): return round(1000.0/ms,2) if ms>0 else 0.0
    def _get(self, m, stat="ema"):
        mm=self.meters[m]
        return mm.ema_ms() if stat=="ema" else (mm.p95_ms() if stat=="p95" else mm.count())

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self._period)
            row=[]; row.append(round(time.time(),3))
            d_ms=self._get("decode","ema"); row += [self._fps(d_ms), d_ms]
            row += [ self._get("batch_fill","ema"), self._get("batch_gap","ema"),
                     self._get("batch_size","ema"), self.counters["batch_timeout_hits"].get() ]
            y_ms=self._get("yolo","ema"); row += [ self._fps(y_ms), y_ms, self._get("post","ema") ]
            e_ms=self._get("embed","ema"); row += [ self._fps(e_ms), e_ms ]
            r_ms=self._get("render","ema"); row += [ self._fps(r_ms), r_ms,
                                                     self._get("upload","ema"), self._get("draw","ema") ]
            q_meta = int(self.gauges.get("q_meta", lambda:0)())
            q_inf  = int(self.gauges.get("q_infer",lambda:0)())
            ring_u = int(self.gauges.get("ring_used",lambda:0)())
            ring_c = int(self.gauges.get("ring_cap", lambda:0)())
            row += [q_meta, q_inf, ring_u, ring_c]

            # 디스크 대신 메모리에 쌓아두기
            self._rows.append(row)

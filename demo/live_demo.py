#!/usr/bin/env python3
"""Live demo: NVIDIA's DeepStream YOLO CPU parser, three versions, one camera.

Pipeline per frame, the same stages DeepStream's nvinfer runs:

  camera/video -> letterbox 640x640 -> YOLO ONNX (onnxruntime, stands in for
  TensorRT) -> NvDsInferParseCustomYoloV7 (NVIDIA's C++ parser, unchanged, from
  one of three dylibs) -> cluster stage (pre-cluster threshold + per-class NMS,
  as nvinfer's cluster-mode=2 with the sample's config) -> boxes on screen.

Keys: 1 upstream, 2 agent's edit, 3 fix, A auto-cycle, SPACE pause,
      B binary view (compiled aarch64 before/after; [ ] scroll, N/P next/prev change),
      F fullscreen, Q / ESC quit.

All timings on screen are wall clock on this Mac. The LOCI card shows LOCI's
figures for the aarch64 object (Neoverse V2 model at 2601 MHz, Jetson Thor
class) from results/, judged against the drafted contract in CONTRACT.md.
"""
import argparse
import collections
import ctypes
import math
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

from binary_view import BinaryView

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NET = 640
PRECLUSTER_THRESHOLD = 0.25  # pre-cluster-threshold in config_infer_primary_yoloV7.txt
NMS_IOU = 0.65               # nms-iou-threshold
TOPK = 300                   # topk

W, H = 1600, 900
PANEL_X = 1120
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
BG = (24, 22, 20)
CARD = (40, 37, 34)
LINE = (70, 66, 62)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
GREEN = (0, 185, 118)   # NVIDIA green #76B900
RED = (70, 70, 235)
AMBER = (0, 175, 255)

VARIANTS = [
    {
        "key": "upstream", "hotkey": "1", "tab": "1 upstream",
        "title": "NVIDIA upstream",
        "what": ["no threshold in the parser: every row is",
                 "handed to nvinfer's cluster stage"],
        "color": (200, 200, 200),
        # LOCI, aarch64 object, neoverse-v2 @ 2601 MHz (results/baseline/)
        "loci": [("worst path", ">= 172 ns", "<= 250 ns", True),
                 ("hot path vs before", "84 ns", "<= +10 %", True),
                 ("stack frame", "224 B", "<= 288 B", True),
                 ("recursion cycles", "0", "= 0", True),
                 ("rows loop in rows loop", "no", "forbidden", True)],
    },
    {
        "key": "innocent", "hotkey": "2", "tab": "2 agent edit",
        "title": "AI agent's edit",
        "what": ["fixes 2 bugs + adds NMS over all 25,200",
                 "rows inside the parser: O(n^2)"],
        "color": RED,
        "loci": [("worst path", ">= 333 ns", "<= 250 ns", False),
                 ("hot path vs before", "+116 %", "<= +10 %", False),
                 ("stack frame", "320 B", "<= 288 B", False),
                 ("recursion cycles", "1 (std::sort)", "= 0", False),
                 ("rows loop in rows loop", "yes :259/:260", "forbidden", False)],
    },
    {
        "key": "fix", "hotkey": "3", "tab": "3 fix",
        "title": "Acceptable fix",
        "what": ["threshold per row; NMS left to",
                 "nvinfer cluster-mode=2"],
        "color": GREEN,
        "loci": [("worst path", ">= 180 ns", "<= 250 ns", True),
                 ("hot path vs before", "-32 %", "<= +10 %", True),
                 ("stack frame", "240 B", "<= 288 B", True),
                 ("recursion cycles", "0", "= 0", True),
                 ("rows loop in rows loop", "no", "forbidden", True)],
    },
]


class Parser:
    """One parser variant: NVIDIA's .cpp + parser_shim.cpp in a dylib."""

    def __init__(self, path, max_rows):
        self.lib = ctypes.CDLL(path)
        f = self.lib.demo_parse
        f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
        f.restype = ctypes.c_int
        self.fn = f
        self.out = np.zeros((max_rows, 6), np.float32)

    def __call__(self, tensor):
        rows, dims = tensor.shape
        total = ctypes.c_int()
        ms = ctypes.c_double()
        n = self.fn(tensor.ctypes.data, rows, dims, NET, NET, PRECLUSTER_THRESHOLD,
                    self.out.ctypes.data, len(self.out), ctypes.byref(total), ctypes.byref(ms))
        return self.out[:max(n, 0)], total.value, ms.value


def letterbox(frame):
    h, w = frame.shape[:2]
    s = min(NET / w, NET / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    px, py = (NET - nw) // 2, (NET - nh) // 2
    img = np.full((NET, NET, 3), 114, np.uint8)
    img[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)
    return blob, s, px, py


def cluster(objs):
    """nvinfer's post-parse stage: per-class pre-cluster threshold, NMS, top-k."""
    if len(objs) == 0:
        return objs
    objs = objs[objs[:, 4] >= PRECLUSTER_THRESHOLD]
    if len(objs) == 0:
        return objs
    idx = cv2.dnn.NMSBoxesBatched(objs[:, :4].tolist(), objs[:, 4].tolist(),
                                  objs[:, 5].astype(int).tolist(),
                                  PRECLUSTER_THRESHOLD, NMS_IOU, top_k=TOPK)
    return objs[np.array(idx, dtype=int).reshape(-1)]


def class_color(c):
    h = (int(c) * 47) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def text(img, s, x, y, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def text_w(s, scale=0.5, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def fmt_ms(ms):
    return f"{ms:.2f} ms" if ms < 10 else f"{ms:.0f} ms" if ms >= 100 else f"{ms:.1f} ms"


class Demo:
    def __init__(self, args):
        self.args = args
        self.labels = [l.strip() for l in open(os.path.join(
            ROOT, "third_party/yolo_deepstream/deepstream_yolo/labels.txt")) if l.strip()]
        self.session = self._session(args.ep)
        out_shape = self.session.get_outputs()[0].shape
        self.parsers = {}
        for v in VARIANTS:
            p = os.path.join(ROOT, "build/demo", f"libparser_{v['key']}.dylib")
            if not os.path.exists(p):
                sys.exit(f"live_demo: {p} missing; run demo/setup.sh")
            self.parsers[v["key"]] = Parser(p, int(out_shape[1]) if isinstance(out_shape[1], int) else 25200)
        self.active = 0
        self.auto = False
        self.auto_t = time.time()
        self.paused = False
        self.fullscreen = args.fullscreen
        self.hist = {v["key"]: collections.deque(maxlen=60) for v in VARIANTS}
        self.timeline = collections.deque(maxlen=200)  # (variant index, parser ms), all variants
        self.last = {v["key"]: None for v in VARIANTS}
        self.fps = 0.0
        self.t_prev = None
        self.binview = BinaryView(ROOT)
        self.show_binary = args.binary

    def _session(self, ep):
        opts = ort.SessionOptions()
        model = os.path.join(HERE, "assets/yolo.onnx")
        if ep == "coreml":
            try:
                s = ort.InferenceSession(model, opts, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
                self.ep_name = "CoreML"
                return s
            except Exception as e:  # fall back rather than fail on stage
                print(f"live_demo: CoreML unavailable ({e}); using CPU", file=sys.stderr)
        self.ep_name = "CPU"
        return ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])

    def open_source(self):
        a = self.args
        if a.image:
            self.still = cv2.imread(a.image)
            if self.still is None:
                sys.exit(f"live_demo: cannot read {a.image}")
            self.source_name = os.path.basename(a.image)
            return None
        self.still = None
        if a.video:
            cap = cv2.VideoCapture(a.video)
            self.source_name = os.path.basename(a.video)
        else:
            cap = cv2.VideoCapture(a.camera, cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.source_name = f"camera {a.camera}"
        if not cap.isOpened():
            sys.exit("live_demo: cannot open the camera. Grant your terminal camera access in "
                     "System Settings > Privacy & Security > Camera, or run with "
                     "--image demo/assets/sample.jpg / --video <file>.")
        return cap

    def read(self, cap):
        if self.still is not None:
            return self.still.copy()
        ok, frame = cap.read()
        if not ok and self.args.video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return frame if ok else None

    def process(self, frame):
        v = VARIANTS[self.active]
        blob, s, px, py = letterbox(frame)
        t0 = time.perf_counter()
        inp = self.session.get_inputs()[0]
        if inp.type == "tensor(float16)":  # the v7.0 yolov5s.onnx release is an FP16 export
            blob = blob.astype(np.float16)
        tensor = self.session.run(None, {inp.name: blob})[0][0]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        t1 = time.perf_counter()
        objs, total, parse_ms = self.parsers[v["key"]](tensor)
        t2 = time.perf_counter()
        dets = cluster(objs)
        t3 = time.perf_counter()

        dets = dets.copy()
        dets[:, 0] = (dets[:, 0] - px) / s
        dets[:, 1] = (dets[:, 1] - py) / s
        dets[:, 2] /= s
        dets[:, 3] /= s
        stats = {
            "rows": tensor.shape[0], "parser_out": total, "dets": len(dets),
            "infer_ms": (t1 - t0) * 1e3, "parse_ms": parse_ms, "cluster_ms": (t3 - t2) * 1e3,
        }
        self.hist[v["key"]].append(parse_ms)
        self.timeline.append((self.active, parse_ms))
        self.last[v["key"]] = stats
        return dets, stats

    # ---------- drawing ----------

    def draw_video(self, canvas, frame, dets):
        fh, fw = frame.shape[:2]
        s = min(PANEL_X / fw, H / fh)
        nw, nh = int(fw * s), int(fh * s)
        ox, oy = (PANEL_X - nw) // 2, (H - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh))
        for x, y, w, h, conf, c in dets:
            col = class_color(c)
            p1 = (int(ox + x * s), int(oy + y * s))
            p2 = (int(ox + (x + w) * s), int(oy + (y + h) * s))
            cv2.rectangle(canvas, p1, p2, col, 2, cv2.LINE_AA)
            name = self.labels[int(c)] if int(c) < len(self.labels) else str(int(c))
            lab = f"{name} {conf:.2f}"
            tw = text_w(lab, 0.5)
            cv2.rectangle(canvas, (p1[0], p1[1] - 20), (p1[0] + tw + 8, p1[1]), col, -1)
            text(canvas, lab, p1[0] + 4, p1[1] - 6, 0.5, (0, 0, 0))
        # banner over the video
        v = VARIANTS[self.active]
        st = self.last[v["key"]]
        cv2.rectangle(canvas, (0, 0), (PANEL_X, 46), (0, 0, 0), -1)
        text(canvas, f"{v['title']}", 16, 31, 0.8, v["color"], 2)
        if st:
            msg = f"parser {fmt_ms(st['parse_ms'])}   |   {self.fps:4.1f} FPS end to end"
            text(canvas, msg, PANEL_X - text_w(msg, 0.7, 2) - 16, 31, 0.7, TEXT, 2)
        if self.paused:
            text(canvas, "PAUSED", 16, H - 20, 0.8, AMBER, 2)

    def draw_panel(self, canvas):
        x0 = PANEL_X + 20
        xr = W - 20
        cv2.rectangle(canvas, (PANEL_X, 0), (W, H), BG, -1)
        cv2.line(canvas, (PANEL_X, 0), (PANEL_X, H), LINE, 1)
        text(canvas, "DeepStream YOLO parser", x0, 36, 0.75, TEXT, 2)
        text(canvas, "NvDsInferParseCustomYoloV7  .  NVIDIA-AI-IOT/yolo_deepstream", x0, 58, 0.4, MUTED)

        # tabs
        y = 78
        tw = (xr - x0 - 16) // 3
        for i, v in enumerate(VARIANTS):
            tx = x0 + i * (tw + 8)
            active = i == self.active
            cv2.rectangle(canvas, (tx, y), (tx + tw, y + 52), CARD if not active else (60, 56, 52), -1)
            cv2.rectangle(canvas, (tx, y), (tx + tw, y + 52), v["color"] if active else LINE, 2 if active else 1)
            text(canvas, v["tab"], tx + 10, y + 21, 0.5, v["color"] if active else TEXT, 1)
            h = self.hist[v["key"]]
            if h:
                text(canvas, fmt_ms(float(np.median(h))), tx + 10, y + 42, 0.55, TEXT, 1)
            else:
                text(canvas, "not run yet", tx + 10, y + 42, 0.42, MUTED)
        v = VARIANTS[self.active]
        y += 74
        for line in v["what"]:
            text(canvas, line, x0, y, 0.46, MUTED)
            y += 19

        # pipeline breakdown
        y += 14
        text(canvas, f"THIS MAC, PER FRAME ({self.ep_name} inference)", x0, y, 0.42, MUTED)
        st = self.last[v["key"]]
        y += 14
        bars = [("inference", st["infer_ms"] if st else 0, (180, 140, 90)),
                ("parser", st["parse_ms"] if st else 0, v["color"]),
                ("cluster", st["cluster_ms"] if st else 0, (150, 120, 160))]
        scale_max = max(50.0, max(b[1] for b in bars))
        bw = xr - x0 - 170
        for name, ms, col in bars:
            y += 26
            text(canvas, name, x0, y, 0.48, TEXT)
            bx = x0 + 85
            cv2.rectangle(canvas, (bx, y - 13), (bx + bw, y + 3), CARD, -1)
            cv2.rectangle(canvas, (bx, y - 13), (bx + max(2, int(bw * ms / scale_max)), y + 3), col, -1)
            text(canvas, fmt_ms(ms), bx + bw + 10, y, 0.48, TEXT)
        if st:
            y += 28
            text(canvas, f"{st['rows']:,} rows in  ->  parser out {st['parser_out']:,}  ->  {st['dets']} boxes",
                 x0, y, 0.45, TEXT)

        # parser time history, log scale
        y += 22
        self.draw_chart(canvas, x0, y, xr - x0, 170)
        y += 170 + 22

        # LOCI card
        self.draw_loci(canvas, x0, y, xr - x0, H - y - 36)

        keys = "1/2/3 parser  B binary  A auto  SPACE pause  F full  Q quit"
        text(canvas, keys, x0, H - 14, 0.4, MUTED)

    def draw_chart(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD, -1)
        text(canvas, "parser time per frame (log, last 200 frames)", x + 8, y + 16, 0.4, MUTED)
        lo, hi = math.log10(0.05), math.log10(1000)
        top, bot = y + 24, y + h - 6

        def ty(ms):
            v = (math.log10(max(ms, 0.05)) - lo) / (hi - lo)
            return int(bot - v * (bot - top))

        for ms, lab in ((0.1, "0.1"), (1, "1"), (10, "10"), (100, "100 ms")):
            yy = ty(ms)
            cv2.line(canvas, (x + 40, yy), (x + w - 4, yy), LINE, 1)
            text(canvas, lab, x + 4, yy + 4, 0.35, MUTED)
        # budget: one frame at NVIDIA's 120 FPS (Orin, FP16, cpu post-process)
        yy = ty(8.33)
        cv2.line(canvas, (x + 40, yy), (x + w - 4, yy), AMBER, 1, cv2.LINE_AA)
        text(canvas, "8.3 ms = whole frame at 120 FPS", x + 46, yy - 5, 0.38, AMBER)
        # one shared timeline, newest at the right; each segment in the colour
        # of the parser that produced it, so a switch shows as a step
        pts = list(self.timeline)
        n = self.timeline.maxlen
        step = (w - 46) / (n - 1)
        x_end = x + w - 4
        for k in range(1, len(pts)):
            (i0, m0), (i1, m1) = pts[k - 1], pts[k]
            p0 = (int(x_end - (len(pts) - k) * step), ty(m0))
            p1 = (int(x_end - (len(pts) - 1 - k) * step), ty(m1))
            cv2.line(canvas, p0, p1, VARIANTS[i1]["color"], 2, cv2.LINE_AA)

    def draw_loci(self, canvas, x, y, w, h):
        v = VARIANTS[self.active]
        ok = all(r[3] for r in v["loci"])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD, -1)
        cv2.rectangle(canvas, (x, y), (x + 6, y + h), GREEN if ok else RED, -1)
        text(canvas, "LOCI  .  compiled aarch64, Jetson Thor class", x + 16, y + 22, 0.48, TEXT, 1)
        text(canvas, "Neoverse V2 model @ 2601 MHz, no device needed", x + 16, y + 40, 0.4, MUTED)
        verdict = "PASS" if ok else "FAIL"
        text(canvas, verdict, x + w - text_w(verdict, 0.9, 2) - 12, y + 34, 0.9, GREEN if ok else RED, 2)
        yy = y + 66
        text(canvas, "contract", x + 16, yy, 0.38, MUTED)
        text(canvas, "measured", x + 200, yy, 0.38, MUTED)
        text(canvas, "bound", x + 320, yy, 0.38, MUTED)
        for name, val, bound, passed in v["loci"]:
            yy += 23
            text(canvas, name, x + 16, yy, 0.44, TEXT)
            text(canvas, val, x + 200, yy, 0.44, GREEN if passed else RED, 1)
            text(canvas, bound, x + 320, yy, 0.44, MUTED)
            mark = "ok" if passed else "X"
            text(canvas, mark, x + w - 30, yy, 0.48, GREEN if passed else RED, 2)
        # what said "go" for every variant
        yy += 40
        cv2.line(canvas, (x + 16, yy - 22), (x + w - 12, yy - 22), LINE, 1)
        text(canvas, "unit test (tests/test_parser.cpp, 5 rows)", x + 16, yy, 0.44, TEXT)
        text(canvas, "pass", x + w - 50, yy, 0.48, GREEN, 2)
        text(canvas, "a real frame has 25,200 rows", x + 16, yy + 20, 0.4, MUTED)
        text(canvas, "figures: results/  .  bounds: CONTRACT.md (draft)", x + 16, y + h - 12, 0.38, MUTED)

    # ---------- loop ----------

    def run(self):
        cap = self.open_source()
        win = "DeepStream YOLO parser - live"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, W * 4 // 5, H * 4 // 5)
        if self.fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        frame, dets = None, np.zeros((0, 6), np.float32)
        while True:
            if not self.paused or frame is None:
                f = self.read(cap)
                if f is None:
                    break
                frame = f
                dets, _ = self.process(frame)
                now = time.perf_counter()
                if self.t_prev is not None:
                    inst = 1.0 / max(now - self.t_prev, 1e-6)
                    self.fps = inst if self.fps == 0 else 0.85 * self.fps + 0.15 * inst
                self.t_prev = now
            canvas = np.full((H, W, 3), BG, np.uint8)
            self.draw_video(canvas, frame, dets)
            if self.show_binary:
                self.binview.draw(canvas, 0, 46, PANEL_X, H - 46, VARIANTS[self.active])
            self.draw_panel(canvas)
            cv2.imshow(win, canvas)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k in (ord("1"), ord("2"), ord("3")):
                self.select(k - ord("1"))
                self.auto = False
            elif k in (ord("a"), ord("A")):
                self.auto = not self.auto
                self.auto_t = time.time()
            elif k in (ord("b"), ord("B")):
                self.show_binary = not self.show_binary
            elif self.show_binary and k in (ord("["), ord("]")):
                self.binview.scroll_by(VARIANTS[self.active]["key"], -20 if k == ord("[") else 20)
            elif self.show_binary and k in (ord("n"), ord("N"), ord("p"), ord("P")):
                self.binview.jump_next_change(VARIANTS[self.active]["key"], back=k in (ord("p"), ord("P")))
            elif k == ord(" "):
                self.paused = not self.paused
                self.t_prev = None
            elif k in (ord("f"), ord("F")):
                self.fullscreen = not self.fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            if self.auto and time.time() - self.auto_t > self.args.auto_seconds:
                self.select((self.active + 1) % len(VARIANTS))
                self.auto_t = time.time()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    def select(self, i):
        self.active = i
        self.fps = 0.0
        self.t_prev = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    src.add_argument("--video", help="video file, looped")
    src.add_argument("--image", help="still image, re-run every frame")
    ap.add_argument("--ep", choices=["coreml", "cpu"], default="coreml", help="onnxruntime provider")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--binary", action="store_true", help="start in the binary view (key B)")
    ap.add_argument("--auto-seconds", type=float, default=8.0, help="dwell per parser in auto mode")
    ap.add_argument("--selftest", type=int, metavar="N",
                    help="no window: run N frames per parser on the source and print timings")
    args = ap.parse_args()
    demo = Demo(args)
    if args.selftest:
        cap = demo.open_source()
        for i, v in enumerate(VARIANTS):
            demo.select(i)
            for _ in range(args.selftest):
                frame = demo.read(cap)
                dets, st = demo.process(frame)
            names = sorted({demo.labels[int(c)] for c in dets[:, 5]})
            print(f"{v['key']:9s} infer {st['infer_ms']:6.1f} ms  parser median {np.median(demo.hist[v['key']]):8.2f} ms"
                  f"  cluster {st['cluster_ms']:5.2f} ms  parser out {st['parser_out']:6d}  boxes {st['dets']:3d}  {names}")
        return
    demo.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Side by side, live: the same frames through two shipped parsers.

  left   WITHOUT LOCI  the agent's edit shipped (its 5-row unit test passed)
  right  WITH LOCI     LOCI flagged the agent's edit, the fix shipped

One capture thread publishes frames in real time (camera, or a video paced at
its own frame rate). Each side has its own worker thread running the full
pipeline -- YOLO ONNX -> NVIDIA's NvDsInferParseCustomYoloV7 -> cluster stage
-- and always takes the newest frame when it is free, like DeepStream on a
live source. A side shows a new picture only when its pipeline finishes one,
so a slow parser shows up as a low frame rate, high latency and dropped frames.
Every number is measured wall clock on this Mac; nothing is simulated.

  demo/compare.sh                                   # window, sample video
  demo/compare.sh --camera 0                        # window, webcam
  demo/compare.sh --record demo/out/compare.mp4 --seconds 30
  demo/compare.sh --record demo/out/compare.mp4 --seconds 30 --no-window

Keys: Q / ESC quit, F fullscreen.
"""
import argparse
import collections
import math
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

import live_demo as L

W, H = 1920, 1080
PANE = W // 2
VID_Y, VID_H = 96, 640
FPS_OUT = 30

SIDES = [
    {"variant": "innocent", "head": "WITHOUT LOCI", "color": L.RED,
     "sub": "agent's edit shipped: its 5-row unit test passed",
     "badge": ["LOCI not run. Nothing measured the compiled code.",
               "NMS over all 25,200 rows inside the parser: O(n^2)."]},
    {"variant": "fix", "head": "WITH LOCI", "color": L.GREEN,
     "sub": "LOCI flagged the agent's edit -> the fix shipped",
     "badge": ["Agent's edit: FAIL  worst path >= 333 ns vs 250 ns, hot path +116 %,",
               "rows x rows loop, std::sort recursion.  Fix: PASS  >= 180 ns, -32 %."]},
]


class Source(threading.Thread):
    """Publishes the newest frame; a video file is paced at its own fps and looped."""

    def __init__(self, args):
        super().__init__(daemon=True)
        if args.camera is not None:
            self.cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.paced = False
            self.name = f"camera {args.camera}"
        else:
            self.cap = cv2.VideoCapture(args.video)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.paced = True
            self.name = os.path.basename(args.video)
        if not self.cap.isOpened():
            sys.exit("compare: cannot open the source (camera permission for your terminal, or --video FILE)")
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.k = -1
        self.frame = None
        self.t_capture = 0.0
        self.t0 = None
        self.stop = False

    def run(self):
        while not self.stop:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            now = time.perf_counter()
            if self.t0 is None:
                self.t0 = now
            if self.paced:
                due = self.t0 + (self.k + 1) / self.fps
                if due > now:
                    time.sleep(due - now)
                now = time.perf_counter()
            with self.cond:
                self.k += 1
                self.frame = frame
                self.t_capture = now
                self.cond.notify_all()

    def wait_newer(self, k_seen):
        with self.cond:
            while self.k <= k_seen and not self.stop:
                self.cond.wait(0.1)
            return self.k, self.frame, self.t_capture


class Worker(threading.Thread):
    """One side's pipeline: inference -> NVIDIA parser -> cluster, newest frame first."""

    def __init__(self, side, source, session, labels):
        super().__init__(daemon=True)
        self.side = side
        self.source = source
        self.session = session
        self.labels = labels
        self.parser = L.Parser(os.path.join(L.ROOT, "build/demo", f"libparser_{side['variant']}.dylib"),
                               int(session.get_outputs()[0].shape[1]))
        self.lock = threading.Lock()
        self.result = None           # (k, frame, dets, stats, t_capture, t_done)
        self.done_times = collections.deque(maxlen=400)
        self.parse_hist = collections.deque(maxlen=120)
        self.processed = 0
        self.first_k = None

    def run(self):
        inp = self.session.get_inputs()[0]
        k_seen = -1
        while not self.source.stop:
            k, frame, t_cap = self.source.wait_newer(k_seen)
            if frame is None:
                continue
            if self.first_k is None:
                self.first_k = k
            k_seen = k
            blob, s, px, py = L.letterbox(frame)
            if inp.type == "tensor(float16)":
                blob = blob.astype(np.float16)
            t0 = time.perf_counter()
            tensor = np.ascontiguousarray(self.session.run(None, {inp.name: blob})[0][0], dtype=np.float32)
            t1 = time.perf_counter()
            objs, total, parse_ms = self.parser(tensor)
            dets = L.cluster(objs).copy()
            t2 = time.perf_counter()
            dets[:, 0] = (dets[:, 0] - px) / s
            dets[:, 1] = (dets[:, 1] - py) / s
            dets[:, 2:4] /= s
            stats = {"infer_ms": (t1 - t0) * 1e3, "parse_ms": parse_ms,
                     "cluster_ms": max(0.0, (t2 - t1) * 1e3 - parse_ms), "dets": len(dets)}
            with self.lock:
                self.result = (k, frame, dets, stats, t_cap, t2)
                self.done_times.append(t2)
                self.parse_hist.append(parse_ms)
                self.processed += 1

    def snapshot(self):
        with self.lock:
            return self.result, list(self.done_times), list(self.parse_hist), self.processed, self.first_k


def big(img, s, x, y, scale, color, thick=2):
    cv2.putText(img, s, (int(x), int(y)), L.FONT, scale, color, thick, cv2.LINE_AA)


def draw_pane(canvas, x0, side, worker, source, labels, now):
    res, done, hist, processed, first_k = worker.snapshot()
    col = side["color"]
    cv2.rectangle(canvas, (x0, 0), (x0 + PANE, H), L.BG, -1)
    # title
    cv2.rectangle(canvas, (x0, 0), (x0 + PANE, VID_Y), (0, 0, 0), -1)
    cv2.rectangle(canvas, (x0, 0), (x0 + 10, VID_Y), col, -1)
    big(canvas, side["head"], x0 + 30, 50, 1.3, col, 3)
    big(canvas, side["sub"], x0 + 30, 82, 0.62, L.TEXT, 1)
    # video
    if res is None:
        big(canvas, "starting...", x0 + 40, VID_Y + 60, 0.9, L.MUTED)
        return
    k, frame, dets, stats, t_cap, t_done = res
    fh, fw = frame.shape[:2]
    s = min(PANE / fw, VID_H / fh)
    nw, nh = int(fw * s), int(fh * s)
    ox, oy = x0 + (PANE - nw) // 2, VID_Y + (VID_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh))
    for bx, by, bw, bh, conf, c in dets:
        cc = L.class_color(c)
        p1 = (int(ox + bx * s), int(oy + by * s))
        p2 = (int(ox + (bx + bw) * s), int(oy + (by + bh) * s))
        cv2.rectangle(canvas, p1, p2, cc, 2, cv2.LINE_AA)
        name = labels[int(c)] if int(c) < len(labels) else str(int(c))
        cv2.putText(canvas, f"{name} {conf:.2f}", (p1[0] + 2, p1[1] - 5), L.FONT, 0.45, cc, 1, cv2.LINE_AA)
    # age of what is on screen: capture -> now
    age_ms = (now - t_cap) * 1e3
    cv2.rectangle(canvas, (ox, oy), (ox + 250, oy + 34), (0, 0, 0), -1)
    big(canvas, f"frame #{k}   age {age_ms:5.0f} ms", ox + 10, oy + 24, 0.6, L.TEXT, 1)
    # the parser timing, large, on the picture
    ptxt = L.fmt_ms(stats["parse_ms"])
    pw = L.text_w(ptxt, 1.9, 4)
    bx1 = x0 + PANE - 20
    cv2.rectangle(canvas, (bx1 - pw - 40, oy + 12), (bx1, oy + 118), (0, 0, 0), -1)
    cv2.rectangle(canvas, (bx1 - pw - 40, oy + 12), (bx1, oy + 118), col, 2)
    big(canvas, "NVIDIA PARSER, THIS FRAME", bx1 - pw - 28, oy + 38, 0.5, L.TEXT, 1)
    big(canvas, ptxt, bx1 - pw - 20, oy + 100, 1.9, col, 4)

    # per-frame time, one shared linear scale on both sides
    total = stats["infer_ms"] + stats["parse_ms"] + stats["cluster_ms"]
    scale_ms = 400.0
    gx, gy, gw, gh = x0 + 30, VID_Y + VID_H + 34, PANE - 60, 26
    big(canvas, f"PER FRAME ON THIS MAC  {total:.0f} ms", gx, gy - 10, 0.5, L.MUTED, 1)
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), L.CARD, -1)
    segs = [("inference", stats["infer_ms"], (180, 140, 90)), ("parser", stats["parse_ms"], col),
            ("cluster", stats["cluster_ms"], (150, 120, 160))]
    cx = gx
    for name, ms, c in segs:
        wpx = int(gw * min(ms, scale_ms) / scale_ms)
        cv2.rectangle(canvas, (cx, gy), (cx + max(wpx, 1), gy + gh), c, -1)
        cx += wpx
    legend = (f"inference {stats['infer_ms']:.0f} ms   parser {L.fmt_ms(stats['parse_ms'])}   "
              f"cluster {L.fmt_ms(stats['cluster_ms'])}")
    big(canvas, legend, gx + gw - L.text_w(legend, 0.45), gy - 10, 0.45, L.TEXT, 1)
    # the source delivers a frame every 1/fps: past this line frames are dropped
    fx = gx + int(gw * (1000.0 / source.fps) / scale_ms)
    cv2.line(canvas, (fx, gy - 4), (fx, gy + gh + 4), L.AMBER, 2)
    big(canvas, f"new frame every {1000.0 / source.fps:.0f} ms", fx + 6, gy + gh + 18, 0.42, L.AMBER, 1)
    big(canvas, f"0 - {scale_ms:.0f} ms, same scale both sides", gx, gy + gh + 18, 0.42, L.MUTED, 1)

    # metrics
    window = 2.0
    fps = sum(1 for t in done if t > now - window) / window
    captured = source.k - (first_k or 0) + 1
    dropped = max(0, captured - processed)
    drop_pct = 100.0 * dropped / max(captured, 1)
    latency = (t_done - t_cap) * 1e3
    share = 100.0 * stats["parse_ms"] / max(total, 1e-6)
    med = float(np.median(hist)) if hist else 0.0
    my = gy + gh + 52
    metrics = [("PARSER MEDIAN", L.fmt_ms(med)),
               ("PARSER SHARE", f"{share:.0f} %"),
               ("FPS", f"{fps:.1f}"),
               ("LATENCY", f"{latency:.0f} ms"),
               ("DROPPED", f"{drop_pct:.0f} %")]
    mw = (PANE - 60) // len(metrics)
    for i, (name, val) in enumerate(metrics):
        mx = x0 + 30 + i * mw
        big(canvas, name, mx, my, 0.45, L.MUTED, 1)
        big(canvas, val, mx, my + 38, 1.05, col, 2)
    # parser-time history (log, 0.1 ms .. 1 s)
    sx, sy, sw, sh = x0 + 30, my + 54, PANE - 60, 62
    cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), L.CARD, -1)
    lo, hi = math.log10(0.1), math.log10(1000)

    def ty(ms):
        return int(sy + sh - 4 - (math.log10(max(ms, 0.1)) - lo) / (hi - lo) * (sh - 8))

    for ms, lab in ((1, "1 ms"), (10, "10 ms"), (100, "100 ms")):
        cv2.line(canvas, (sx + 52, ty(ms)), (sx + sw, ty(ms)), L.LINE, 1)
        big(canvas, lab, sx + 4, ty(ms) + 4, 0.35, L.MUTED, 1)
    if len(hist) > 1:
        n = len(hist)
        pts = np.array([[sx + sw - (n - 1 - i) * (sw - 40) / 119.0, ty(m)] for i, m in enumerate(hist)], np.int32)
        cv2.polylines(canvas, [pts], False, col, 2, cv2.LINE_AA)
    # LOCI badge
    by0 = sy + sh + 22
    for i, line in enumerate(side["badge"]):
        big(canvas, line, x0 + 30, by0 + i * 20, 0.5, col if i == 0 else L.TEXT, 1)


class Recorder:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W}x{H}", "-r", str(FPS_OUT), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def write(self, canvas):
        self.p.stdin.write(canvas.tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        base, ext = os.path.splitext(self.path)
        # the two halves as their own videos too
        for name, x in (("without-loci", 0), ("with-loci", PANE)):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.path,
                            "-vf", f"crop={PANE}:{H}:{x}:0", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            f"{base}-{name}{ext}"], check=True)
        print(f"wrote {self.path}\n      {base}-without-loci{ext}\n      {base}-with-loci{ext}")


def make_session(ep, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    model = os.path.join(L.HERE, "assets/yolo.onnx")
    providers = ["CPUExecutionProvider"]
    if ep == "coreml":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model, opts, providers=providers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=os.path.join(L.HERE, "assets/vtest.avi"))
    ap.add_argument("--camera", type=int, help="use a webcam instead of --video")
    ap.add_argument("--ep", choices=["cpu", "coreml"], default="cpu",
                    help="onnxruntime provider for both sides (default cpu: two sessions share it evenly)")
    ap.add_argument("--threads", type=int, default=4, help="inference threads per side")
    ap.add_argument("--record", help="write the side-by-side to this .mp4 (plus -with-loci / -without-loci halves)")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    labels = [l.strip() for l in open(os.path.join(
        L.ROOT, "third_party/yolo_deepstream/deepstream_yolo/labels.txt")) if l.strip()]
    source = Source(args)
    workers = [Worker(s, source, make_session(args.ep, args.threads), labels) for s in SIDES]
    source.start()
    for w in workers:
        w.start()

    rec = Recorder(args.record) if args.record else None
    win = "Without LOCI | With LOCI"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, W * 2 // 3, H * 2 // 3)
        if args.fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    t_start = time.perf_counter()
    next_tick = t_start
    try:
        while True:
            now = time.perf_counter()
            if args.seconds and now - t_start > args.seconds:
                break
            canvas = np.full((H, W, 3), L.BG, np.uint8)
            for i, (side, w) in enumerate(zip(SIDES, workers)):
                draw_pane(canvas, i * PANE, side, w, source, labels, now)
            cv2.line(canvas, (PANE, 0), (PANE, H), (0, 0, 0), 4)
            note = (f"same {source.name} frames, same YOLO model, both live on this Mac; "
                    "only NVIDIA's parser differs. Timings: wall clock here, not a Jetson.")
            big(canvas, note, (W - L.text_w(note, 0.45)) // 2, H - 10, 0.45, L.MUTED, 1)
            if rec:
                rec.write(canvas)
            if not args.no_window:
                cv2.imshow(win, canvas)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k in (ord("f"), ord("F")):
                    args.fullscreen = not args.fullscreen
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if args.fullscreen else cv2.WINDOW_NORMAL)
            # the recording runs on a fixed 30 fps clock so its timeline is real time
            next_tick += 1.0 / FPS_OUT
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        source.stop = True
        if rec:
            rec.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        for side, w in zip(SIDES, workers):
            _, done, hist, processed, first_k = w.snapshot()
            captured = source.k - (first_k or 0) + 1
            if hist:
                print(f"{side['head']:13s} {side['variant']:9s} processed {processed:4d} of {captured:4d} frames, "
                      f"parser median {np.median(hist):8.2f} ms")


if __name__ == "__main__":
    main()

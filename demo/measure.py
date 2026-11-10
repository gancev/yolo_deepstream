#!/usr/bin/env python3
"""Measure each parser version alone through the live pipeline, same clip, same conditions.

For each of upstream (before), innocent (agent's change) and fix, runs the
compare.py pipeline with a single worker for --seconds on the paced video
source, then records parser time (median, p95), parser share of the frame,
inference time, delivered FPS, latency and frames processed. One version at a
time, so no version shares the CPU with another.

  demo/.venv/bin/python demo/measure.py --seconds 30 --out demo/out/measurements.json
"""
import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np

import compare as C
import live_demo as L

VERSIONS = [("upstream", "NVIDIA original (before)"),
            ("innocent", "Agent's change"),
            ("fix", "Fix")]


def cpu_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return platform.processor()


def measure(variant, args, labels):
    source = C.Source(args)
    side = {"variant": variant}
    worker = C.Worker(side, source, C.make_session(args.ep, args.threads), labels)
    parse, infer, total, latency = [], [], [], []
    orig_run = worker.run

    source.start()
    worker.start()
    t0 = time.perf_counter()
    last = None
    while time.perf_counter() - t0 < args.seconds:
        res, _, _, _, _ = worker.snapshot()
        if res is not None and res is not last:
            _, _, _, st, t_cap, t_done = res
            parse.append(st["parse_ms"])
            infer.append(st["infer_ms"])
            total.append(st["infer_ms"] + st["parse_ms"] + st["cluster_ms"])
            latency.append((t_done - t_cap) * 1e3)
            last = res
        time.sleep(0.002)
    source.stop = True
    _, done, _, processed, first_k = worker.snapshot()
    captured = source.k - (first_k or 0) + 1
    worker.join(timeout=5)
    source.join(timeout=5)
    # skip the first two results (session warm-up)
    p = np.array(parse[2:] or parse)
    return {
        "variant": variant,
        "frames_captured": int(captured),
        "frames_processed": int(processed),
        "dropped_pct": round(100.0 * max(0, captured - processed) / max(captured, 1), 1),
        "fps": round(processed / args.seconds, 2),
        "parser_ms_median": round(float(np.median(p)), 3),
        "parser_ms_p95": round(float(np.percentile(p, 95)), 3),
        "parser_ms_min": round(float(np.min(p)), 3),
        "inference_ms_median": round(float(np.median(infer[2:] or infer)), 2),
        "frame_ms_median": round(float(np.median(total[2:] or total)), 2),
        "parser_share_pct": round(100.0 * float(np.median(p)) / float(np.median(total[2:] or total)), 1),
        "latency_ms_median": round(float(np.median(latency[2:] or latency)), 1),
        "samples": len(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(L.HERE, "assets/vtest.avi"))
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--ep", default="cpu", choices=["cpu", "coreml"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(L.HERE, "out/measurements.json"))
    args = ap.parse_args()
    args.camera = None
    labels = [l.strip() for l in open(os.path.join(
        L.ROOT, "third_party/yolo_deepstream/deepstream_yolo/labels.txt")) if l.strip()]
    rows = []
    for v, name in VERSIONS:
        r = measure(v, args, labels)
        r["name"] = name
        rows.append(r)
        print(f"{name:26s} parser median {r['parser_ms_median']:9.3f} ms  p95 {r['parser_ms_p95']:9.3f} ms  "
              f"share {r['parser_share_pct']:5.1f} %  fps {r['fps']:5.2f}  latency {r['latency_ms_median']:6.1f} ms  "
              f"processed {r['frames_processed']}/{r['frames_captured']}")
    out = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cpu": cpu_name(),
           "clip": os.path.basename(args.video), "seconds_per_version": args.seconds,
           "inference": f"onnxruntime {args.ep}, {args.threads} threads, YOLOv5s FP16 640x640",
           "versions": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

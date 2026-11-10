#!/usr/bin/env bash
# One-time setup for the live demo (macOS, Apple silicon):
#   - a Python venv in demo/.venv with onnxruntime, opencv and numpy (via uv)
#   - a YOLO ONNX model whose output is the YOLOv7 head layout the parser
#     expects: [1, 25200, 85] = cx, cy, w, h, objectness, 80 class scores
#     (Ultralytics YOLOv5s v7.0 export; same tensor as YOLOv7 at 640x640)
#   - a sample image for when no camera is available
#   - the three parser dylibs (build_parsers.sh)
# Needs: ./bootstrap.sh done (headers), uv, clang++, curl.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

command -v uv >/dev/null || { echo "setup.sh: uv not found (brew install uv)" >&2; exit 1; }

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -q "numpy<3" "opencv-python>=4.8" "onnxruntime>=1.18"

mkdir -p assets
MODEL_URL="${MODEL_URL:-https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx}"
[ -f assets/yolo.onnx ] || curl -fL --progress-bar -o assets/yolo.onnx "$MODEL_URL"
[ -f assets/sample.jpg ] || curl -fsSL -o assets/sample.jpg \
  https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg

"$HERE/build_parsers.sh"
"$HERE/build_binaries.sh"   # aarch64 before/after objects + LOCI extracts (key B)
echo "setup done. Run: demo/run.sh   (webcam)   or   demo/run.sh --video <file.mp4>"

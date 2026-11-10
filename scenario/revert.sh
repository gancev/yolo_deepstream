#!/usr/bin/env bash
# Restore src/nvdsparsebbox_Yolo.cpp to the pristine upstream file (pinned
# commit in third_party/YOLO_DEEPSTREAM_COMMIT).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
UP="third_party/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp"
[ -f "$UP" ] || { echo "revert.sh: $UP missing; run ./bootstrap.sh" >&2; exit 1; }
cp "$UP" src/nvdsparsebbox_Yolo.cpp
echo "src/nvdsparsebbox_Yolo.cpp restored to upstream $(cat third_party/YOLO_DEEPSTREAM_COMMIT | cut -c1-12)"

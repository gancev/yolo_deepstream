#!/usr/bin/env bash
# Side-by-side live comparison: without LOCI (agent's edit) | with LOCI (fix).
# Arguments pass through to compare.py (--camera N, --video FILE, --record OUT.mp4,
# --seconds S, --no-window, --fullscreen).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$HERE/.venv/bin/python" ] || { echo "compare.sh: run demo/setup.sh first" >&2; exit 1; }
[ -f "$HERE/assets/vtest.avi" ] || curl -fsSL -o "$HERE/assets/vtest.avi" \
  https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi
cd "$HERE"
exec "$HERE/.venv/bin/python" "$HERE/compare.py" "$@"

#!/usr/bin/env bash
# Start the live demo. Arguments pass through to live_demo.py
# (--video FILE, --image FILE, --camera N, --ep cpu|coreml, --fullscreen).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$HERE/.venv/bin/python" ] || { echo "run.sh: run demo/setup.sh first" >&2; exit 1; }
exec "$HERE/.venv/bin/python" "$HERE/live_demo.py" "$@"

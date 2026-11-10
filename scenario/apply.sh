#!/usr/bin/env bash
# Apply one scenario variant to src/nvdsparsebbox_Yolo.cpp.
#
#   scenario/apply.sh innocent   # the agent's edit: bug fixes + NMS over all 25,200 rows
#   scenario/apply.sh fix        # the acceptable fix: threshold first, NMS left to cluster-mode
#
# Both start from the pristine upstream file (scenario/revert.sh), so the
# patches are independent of each other. Use scenario/revert.sh to go back.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
case "${1:-}" in
  innocent) PATCH=scenario/innocent_edit.patch ;;
  fix)      PATCH=scenario/fix.patch ;;
  *) echo "usage: scenario/apply.sh {innocent|fix}" >&2; exit 2 ;;
esac
"$HERE/scenario/revert.sh" >/dev/null
patch -p1 --silent < "$PATCH"
echo "applied $PATCH to src/nvdsparsebbox_Yolo.cpp"

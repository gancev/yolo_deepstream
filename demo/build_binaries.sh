#!/usr/bin/env bash
# Build the before/after aarch64 binaries for the demo's binary view (key B).
#
# Each variant's nvdsparsebbox_Yolo.cpp is compiled with the LOCI recipe's
# flags (-O2 -g -mcpu=neoverse-v2 -std=c++11 -fPIC, aarch64-linux-gnu-g++),
# i.e. the same object LOCI measures, then LOCI extracts what the demo shows:
#   build/demo/aarch64/<variant>.o           the object
#   build/demo/aarch64/<variant>-asm/        loci elf asm  (per-function assembly)
#   build/demo/aarch64/diff-<variant>.json   loci elf diff (upstream -> variant)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
CROSS_CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
FLAGS="-O2 -g -mcpu=neoverse-v2 -std=c++11 -fPIC -Wall -Wno-error=deprecated-declarations -Wno-deprecated-declarations"
INCS="-Ithird_party/deepstream-includes -Ithird_party/cuda-includes -Ithird_party/tensorrt-includes"
FUNCS="NvDsInferParseCustomYoloV7"
OUT=build/demo/aarch64
mkdir -p "$OUT"

src_of() {
  case "$1" in
    upstream) echo third_party/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp ;;
    *)        echo "scenario/variants/$1/nvdsparsebbox_Yolo.cpp" ;;
  esac
}

for v in upstream innocent fix; do
  "$CROSS_CXX" $FLAGS $INCS -c -o "$OUT/$v.o" "$(src_of $v)"
  rm -rf "$OUT/$v-asm"
  loci elf asm --elf "$OUT/$v.o" --arch neoverse-v2 --functions "$FUNCS" \
    --out-dir "$OUT/$v-asm" --project-root "$ROOT" > "$OUT/$v-asm.json"
  echo "built $OUT/$v.o"
done
for v in innocent fix; do
  loci elf diff --elf "$OUT/upstream.o" --comparing-elf "$OUT/$v.o" --arch neoverse-v2 \
    --out-dir "$OUT/diff-$v" --project-root "$ROOT" > "$OUT/diff-$v.json"
done
echo "binaries + LOCI extracts in $OUT"

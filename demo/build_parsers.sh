#!/usr/bin/env bash
# Build one host dylib per parser variant for the live demo:
#   upstream  - NVIDIA's file at the pinned commit (third_party/yolo_deepstream)
#   innocent  - the agent's edit (bug fixes + NMS over all 25,200 rows)
#   fix       - threshold per row, NMS left to nvinfer's cluster stage
# Each dylib is that variant's nvdsparsebbox_Yolo.cpp, unchanged, plus
# parser_shim.cpp. Host Apple clang -O2: these are wall-clock demo numbers on
# this Mac, not the aarch64 object LOCI measures.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CXX="${HOST_CXX:-clang++}"
INCS="-I$ROOT/third_party/deepstream-includes -I$ROOT/third_party/cuda-includes -I$ROOT/third_party/tensorrt-includes"
[ -f "$ROOT/third_party/deepstream-includes/nvdsinfer_custom_impl.h" ] || {
  echo "build_parsers.sh: DeepStream headers missing; run ./bootstrap.sh" >&2; exit 1; }

OUT="$ROOT/build/demo"
mkdir -p "$OUT"
build() {
  "$CXX" -O2 -std=c++11 -dynamiclib -fPIC -Wno-deprecated-declarations \
    -Wno-unneeded-internal-declaration -Wno-non-c-typedef-for-linkage $INCS \
    -o "$OUT/libparser_$1.dylib" "$HERE/parser_shim.cpp" "$2"
  echo "built build/demo/libparser_$1.dylib"
}
build upstream "$ROOT/third_party/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp"
build innocent "$ROOT/scenario/variants/innocent/nvdsparsebbox_Yolo.cpp"
build fix      "$ROOT/scenario/variants/fix/nvdsparsebbox_Yolo.cpp"

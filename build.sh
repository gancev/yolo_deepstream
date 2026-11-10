#!/usr/bin/env bash
# Build the aarch64 Neoverse V2-tuned object (what LOCI measures; Jetson Thor
# Neoverse V3AE modeled by LOCI's neoverse-v2 target), the shared
# library, and the unit-test / benchmark executables (cross aarch64 static, and
# native host). Run bootstrap.sh first so the headers are in place.
#
#   ./build.sh            # everything
#   ./build.sh object     # only the CMake-driven .o + .so (regenerates compile_commands.json)
#   ./build.sh tests      # only the test and bench binaries (cross + host)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

DS_INC="$HERE/third_party/deepstream-includes"
CUDA_INC="$HERE/third_party/cuda-includes"
TRT_INC="$HERE/third_party/tensorrt-includes"
SRC="$HERE/src/nvdsparsebbox_Yolo.cpp"
CROSS_CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
HOST_CXX="${HOST_CXX:-clang++}"
CROSS_FLAGS="-O2 -g -mcpu=neoverse-v2 -std=c++11 -fPIC -Wall -Wno-error=deprecated-declarations"
# The test/bench executables are run inside an arm64 container on an Apple
# silicon host, which has no SVE: they are built with SVE off so they execute
# there. The object LOCI measures (CMake, above) keeps plain -mcpu=neoverse-v2.
# On a Jetson Thor / Grace box set TEST_CPU=neoverse-v2.
TEST_CPU="${TEST_CPU:-neoverse-v2+nosve}"
TEST_FLAGS="-O2 -g -mcpu=$TEST_CPU -std=c++11 -fPIC -Wall -Wno-error=deprecated-declarations"

[ -f "$DS_INC/nvdsinfer_custom_impl.h" ] || { echo "build.sh: $DS_INC/nvdsinfer_custom_impl.h missing; run ./bootstrap.sh" >&2; exit 1; }
[ -f "$CUDA_INC/cuda_runtime_api.h" ] || { echo "build.sh: $CUDA_INC/cuda_runtime_api.h missing; run ./bootstrap.sh" >&2; exit 1; }
[ -f "$TRT_INC/NvInfer.h" ] || { echo "build.sh: $TRT_INC/NvInfer.h missing; run ./bootstrap.sh" >&2; exit 1; }
INCS="-I$DS_INC -I$CUDA_INC -I$TRT_INC"
command -v "$CROSS_CXX" >/dev/null || { echo "build.sh: $CROSS_CXX not on PATH (brew install aarch64-linux-gnu-gcc or aarch64-unknown-linux-gnu)" >&2; exit 1; }

what="${1:-all}"

build_object() {
  # CMake drives the one translation unit so a generated compile_commands.json
  # exists for LOCI (the recipe records it); the flags are in CMakeLists.txt.
  if [ ! -f build/CMakeCache.txt ]; then
    cmake -S "$HERE" -B "$HERE/build" \
      -DCMAKE_TOOLCHAIN_FILE="$HERE/cmake/aarch64-neoverse-v2.cmake" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null
  fi
  cmake --build "$HERE/build" --target nvdsinfer_custom_impl_Yolo
  echo "object: $(ls build/CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/src/*.o)"
  echo "shared: build/libnvdsinfer_custom_impl_Yolo.so"
  echo "compdb: build/compile_commands.json"
}

build_tests() {
  mkdir -p build/tests
  # Cross-compiled, statically linked aarch64 test binaries: run them natively
  # on a Jetson or in an arm64 container (test.sh does the latter).
  "$CROSS_CXX" $TEST_FLAGS $INCS -static \
    -o build/tests/test_parser_aarch64 tests/test_parser.cpp "$SRC"
  "$CROSS_CXX" $TEST_FLAGS $INCS -static \
    -o build/tests/bench_parser_aarch64 tests/bench_parser.cpp "$SRC"
  echo "cross tests: build/tests/test_parser_aarch64 build/tests/bench_parser_aarch64"
  # Native host builds (Apple clang on macOS) for a quick local run.
  if command -v "$HOST_CXX" >/dev/null; then
    "$HOST_CXX" -O2 -std=c++11 -Wall -Wno-deprecated-declarations -Wno-unneeded-internal-declaration $INCS \
      -o build/tests/test_parser_host tests/test_parser.cpp "$SRC"
    "$HOST_CXX" -O2 -std=c++11 -Wall -Wno-deprecated-declarations -Wno-unneeded-internal-declaration $INCS \
      -o build/tests/bench_parser_host tests/bench_parser.cpp "$SRC"
    echo "host tests:  build/tests/test_parser_host build/tests/bench_parser_host"
  fi
}

case "$what" in
  all)    build_object; build_tests ;;
  object) build_object ;;
  tests)  build_tests ;;
  *) echo "usage: build.sh [all|object|tests]" >&2; exit 2 ;;
esac

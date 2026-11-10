#!/usr/bin/env bash
# One-time setup for the DeepStream YOLO parser LOCI demo on a macOS arm64 host
# (also works on Linux with docker). Re-runnable: each step is skipped when its
# output already exists.
#
#  1. checks the host tools (aarch64 cross GCC, cmake, docker, loci)
#  2. clones NVIDIA-AI-IOT/yolo_deepstream at the pinned commit (Apache-2.0,
#     the parser file itself is MIT) and copies the parser to src/
#  3. pulls the DeepStream multiarch image for linux/arm64 and copies
#     /opt/nvidia/deepstream/deepstream/sources/includes out of it
#        -> third_party/deepstream-includes   (nvdsinfer.h, nvdsinfer_custom_impl.h, ...)
#  4. downloads the matching TensorRT and CUDA *header* packages for arm64 from
#     NVIDIA's public apt repository (ubuntu2404/sbsa) inside an arm64 container
#     and extracts them (no install, no GPU, no NGC login needed):
#        -> third_party/tensorrt-includes     (NvInfer.h, NvInferRuntime*.h, NvOnnxParser.h)
#        -> third_party/cuda-includes         (cuda_runtime_api.h and what it includes)
#     nvdsinfer_custom_impl.h includes <NvInfer.h> and <NvOnnxParser.h>, and
#     NvInfer.h includes <cuda_runtime_api.h>; the CPU parser uses none of them
#     at run time, but the translation unit does not compile without them.
#
# All of that is NVIDIA SDK/EULA material and is NOT redistributed in this
# repository (git-ignored); see third_party/NOTICE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PIN="$(cat third_party/YOLO_DEEPSTREAM_COMMIT)"
DS_IMAGE="${DS_IMAGE:-nvcr.io/nvidia/deepstream:8.0-samples-multiarch}"
# TensorRT and CUDA versions that match the DeepStream 8.0 image's runtime
# libraries (dpkg -l inside the image: libnvinfer10 10.13.2.6-1+cuda13.0,
# cuda-cudart-13-0 13.0.48-1).
TRT_VER="${TRT_VER:-10.13.2.6-1+cuda13.0}"
CUDA_PKGS="${CUDA_PKGS:-cuda-cudart-dev-13-0 cuda-crt-13-0}"

echo "== host tools"
MISSING=
for t in aarch64-linux-gnu-g++ cmake docker git loci; do
  if command -v "$t" >/dev/null; then echo "   ok      $t -> $(command -v "$t")"; else echo "   MISSING $t" >&2; MISSING=1; fi
done
if [ -n "$MISSING" ]; then
  cat >&2 <<'EOT'
Install the missing tools and re-run:
  brew install aarch64-unknown-linux-gnu   # provides aarch64-linux-gnu-g++ (GCC 15, glibc sysroot)
  brew install cmake
  brew install colima docker && colima start --arch aarch64
  LOCI: the Claude Code plugin's /loci:setup installs the loci CLI; then `loci login`
EOT
  exit 1
fi
docker info >/dev/null 2>&1 || { echo "docker daemon not reachable (colima start?)" >&2; exit 1; }

echo "== upstream source (pinned $PIN)"
if [ ! -d third_party/yolo_deepstream/.git ]; then
  git clone -q https://github.com/NVIDIA-AI-IOT/yolo_deepstream.git third_party/yolo_deepstream
  git -C third_party/yolo_deepstream checkout -q "$PIN"
fi
echo "   at $(git -C third_party/yolo_deepstream rev-parse HEAD)"
UP="third_party/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp"
if [ ! -f src/nvdsparsebbox_Yolo.cpp ]; then
  cp "$UP" src/nvdsparsebbox_Yolo.cpp; echo "   copied parser to src/"
elif ! cmp -s "$UP" src/nvdsparsebbox_Yolo.cpp; then
  echo "   note: src/nvdsparsebbox_Yolo.cpp differs from upstream (a scenario patch is applied; scenario/revert.sh restores it)"
fi

echo "== DeepStream headers ($DS_IMAGE)"
if [ ! -f third_party/deepstream-includes/nvdsinfer_custom_impl.h ]; then
  docker pull --platform linux/arm64 "$DS_IMAGE"
  cid="$(docker create --platform linux/arm64 "$DS_IMAGE")"
  mkdir -p third_party/deepstream-includes
  docker cp "$cid:/opt/nvidia/deepstream/deepstream/sources/includes/." third_party/deepstream-includes/
  docker cp "$cid:/opt/nvidia/deepstream/deepstream/LICENSE.txt" third_party/deepstream-includes/LICENSE.txt 2>/dev/null || true
  docker rm "$cid" >/dev/null
  docker image inspect --format '{{.Id}} {{.RepoDigests}}' "$DS_IMAGE" > third_party/deepstream-includes/IMAGE_DIGEST
fi
echo "   $(ls third_party/deepstream-includes | wc -l | tr -d ' ') files in third_party/deepstream-includes"

echo "== TensorRT + CUDA headers (NVIDIA apt repo, arm64 sbsa, download + extract only)"
if [ ! -f third_party/tensorrt-includes/NvInfer.h ] || [ ! -f third_party/cuda-includes/cuda_runtime_api.h ]; then
  mkdir -p third_party/_debs
  docker run --rm --platform linux/arm64 -v "$HERE/third_party/_debs:/out" ubuntu:24.04 bash -c "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null
    apt-get install -y -qq curl ca-certificates >/dev/null
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -o /tmp/k.deb
    dpkg -i /tmp/k.deb >/dev/null
    apt-get update -qq >/dev/null
    cd /out
    apt-get download libnvinfer-headers-dev=$TRT_VER libnvonnxparsers-dev=$TRT_VER $CUDA_PKGS
    mkdir -p /out/root
    for d in /out/*.deb; do dpkg -x \"\$d\" /out/root; done
  "
  mkdir -p third_party/tensorrt-includes third_party/cuda-includes
  cp third_party/_debs/root/usr/include/aarch64-linux-gnu/*.h third_party/tensorrt-includes/
  cp -R third_party/_debs/root/usr/local/cuda-13.0/targets/sbsa-linux/include/. third_party/cuda-includes/
  cp third_party/_debs/root/usr/share/doc/libnvinfer-headers-dev/copyright third_party/tensorrt-includes/COPYRIGHT 2>/dev/null || true
  cp third_party/_debs/root/usr/share/doc/cuda-cudart-dev-13-0/copyright third_party/cuda-includes/COPYRIGHT 2>/dev/null || true
  ls third_party/_debs/*.deb | xargs -n1 basename > third_party/DEB_SOURCES
fi
echo "   $(ls third_party/tensorrt-includes | wc -l | tr -d ' ') files in third_party/tensorrt-includes; $(ls third_party/cuda-includes | wc -l | tr -d ' ') in third_party/cuda-includes"

echo "== done. Next: ./build.sh, then INSTRUCTIONS.md step 3 (loci init)."

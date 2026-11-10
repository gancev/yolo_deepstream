cmake_minimum_required(VERSION 3.16)
project(deepstream_yolo_parser CXX)

# One translation unit: NVIDIA's CPU YOLO output parser for DeepStream nvinfer
# (yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp),
# compiled exactly as the upstream Makefile does (-Wall -std=c++11 -fPIC
# -Wno-error=deprecated-declarations against the DeepStream, TensorRT and CUDA
# include directories) plus -O2 -mcpu=neoverse-v2 (the toolchain file sets it: LOCI's
# Neoverse V2 model stands in for the Jetson Thor's Neoverse V3AE cores; the AGX
# Orin's Cortex-A78AE variant is cmake/aarch64-cortex-a78ae.cmake, for reference)
# and -g so LOCI can map blocks back to source lines. The .cu twin and the
# TensorRT/CUDA link libraries are not needed by the CPU path.
#
# Configure with the aarch64 toolchain file (bootstrap.sh / build.sh do this):
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-neoverse-v2.cmake \
#         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEEPSTREAM_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/deepstream-includes"
    CACHE PATH "DeepStream sources/includes (nvdsinfer.h, nvdsinfer_custom_impl.h)")
set(CUDA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/cuda-includes"
    CACHE PATH "CUDA runtime headers (cuda_runtime_api.h)")
set(TENSORRT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/tensorrt-includes"
    CACHE PATH "TensorRT headers (NvInfer.h, NvOnnxParser.h); on a Jetson these live in /usr/include/aarch64-linux-gnu")

add_library(nvdsinfer_custom_impl_Yolo SHARED src/nvdsparsebbox_Yolo.cpp)
target_include_directories(nvdsinfer_custom_impl_Yolo PRIVATE
    "${DEEPSTREAM_INCLUDE_DIR}" "${CUDA_INCLUDE_DIR}" "${TENSORRT_INCLUDE_DIR}")
target_compile_options(nvdsinfer_custom_impl_Yolo PRIVATE
    -Wall -Wno-error=deprecated-declarations -fPIC -O2 -g)
set_target_properties(nvdsinfer_custom_impl_Yolo PROPERTIES
    PREFIX "lib" OUTPUT_NAME "nvdsinfer_custom_impl_Yolo")

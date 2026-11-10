# REFERENCE ONLY (unused by build.sh, which uses aarch64-neoverse-v2.cmake):
# cross toolchain tuned for the Cortex-A78AE cluster of the Jetson AGX Orin (the same -mcpu DPDK prescribes
# for that core). CMAKE_BUILD_TYPE is left empty so the flags in CMakeLists.txt
# are the whole flag set (no hidden -DNDEBUG or extra -O level).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

find_program(AARCH64_GXX NAMES aarch64-linux-gnu-g++ PATHS /opt/homebrew/bin /usr/local/bin /usr/bin)
find_program(AARCH64_GCC NAMES aarch64-linux-gnu-gcc PATHS /opt/homebrew/bin /usr/local/bin /usr/bin)
if(NOT AARCH64_GXX)
  message(FATAL_ERROR "aarch64-linux-gnu-g++ not found; see bootstrap.sh")
endif()
set(CMAKE_C_COMPILER   "${AARCH64_GCC}")
set(CMAKE_CXX_COMPILER "${AARCH64_GXX}")

set(CMAKE_CXX_FLAGS_INIT "-mcpu=cortex-a78ae")
set(CMAKE_C_FLAGS_INIT   "-mcpu=cortex-a78ae")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

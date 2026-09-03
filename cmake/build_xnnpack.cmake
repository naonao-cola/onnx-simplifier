# Fetch and build Google's XNNPACK (a library of optimized neural-network
# operator kernels for Arm/x86/WebAssembly CPUs) as an additional, opt-in
# ModelExecutor backend for constant folding -- see onnxsim/xnnpack_executor.h
# and docs/dlpack-executor.md.
#
# Unlike cmake/build_ort.cmake, this does not need ExternalProject's full
# out-of-tree isolation: XNNPACK does not define an `onnx`/`onnx_proto` CMake
# target, so it cannot collide with onnxsim's own third_party/onnx fork the
# way an in-tree ONNX Runtime build would (see build_ort.cmake's own comment
# for that collision). A plain FetchContent + add_subdirectory is therefore
# sufficient here, and leaves onnxsim linking a normal CMake target (XNNPACK)
# instead of an imported prebuilt library.
#
# XNNPACK has no tagged releases -- it is a rolling `master` -- so it is
# pinned by commit hash rather than a version tag. Bump
# ONNXSIM_XNNPACK_GIT_TAG to pick up newer kernels/ops.
include(FetchContent)

set(ONNXSIM_XNNPACK_GIT_TAG "a5acbbec8f21a1903bbe8ef711f4fc309970ee6d" CACHE STRING
    "XNNPACK commit to build against (XNNPACK has no tagged releases)")

# XNNPACK's own CMakeLists.txt defaults both of these ON (it is normally
# built as part of a larger project's own test suite too); onnxsim only ever
# links the library itself, so skip compiling XNNPACK's own tests and
# benchmarks -- by far the largest share of its build time.
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_LIBRARY ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  xnnpack
  GIT_REPOSITORY https://github.com/google/XNNPACK.git
  GIT_TAG "${ONNXSIM_XNNPACK_GIT_TAG}")
# XNNPACK's own CMakeLists.txt calls project(XNNPACK C CXX ASM) and, when
# neither CPUINFO_SOURCE_DIR nor PTHREADPOOL_SOURCE_DIR is predefined,
# transitively fetches its own `cpuinfo` and `pthreadpool` dependencies the
# same way (see its cmake/DownloadCpuinfo.cmake / DownloadPThreadPool.cmake) --
# nothing further is needed here for those.
FetchContent_MakeAvailable(xnnpack)

set(XNNPACK_INCLUDE_DIR "${xnnpack_SOURCE_DIR}/include")

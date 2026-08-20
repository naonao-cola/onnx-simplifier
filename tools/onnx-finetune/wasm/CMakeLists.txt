cmake_minimum_required(VERSION 3.22)
project(onnx-finetune-wasm CXX)

if (NOT EMSCRIPTEN)
  message(FATAL_ERROR "configure this with emcmake, e.g.: emcmake cmake -B build ...")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Mirrors onnxsim's own wasm build (../../CMakeLists.txt's EMSCRIPTEN branch,
# see also ../../build_wasm.sh) rather than reinventing it: ONNX Runtime is
# built as a static wasm archive and absorbed straight into this module via
# add_subdirectory, instead of linking a prebuilt libonnxruntime as the
# native ../CMakeLists.txt does (there's no shared-library concept in a wasm
# build the way there is on native).
#
# The one thing that ISN'T inherited from onnxsim's existing recipe:
# onnxruntime_ENABLE_TRAINING_APIS=ON. onnxsim's own wasm build has never
# turned that on, and upstream's own wasm+training JS bindings were removed
# from ONNX Runtime's source tree after v1.19.2 (see ../README.md's context
# in the parent conversation) -- so whether ORT's CMake still produces a
# working training-enabled onnxruntime_webassembly target is UNVERIFIED. This
# file is a structurally-faithful port of a build that's known to work
# (onnxsim's own), with training turned on; treat the training+wasm
# combination itself as the experimental part.
if (NOT ORT_SOURCE_DIR)
  message(FATAL_ERROR "ORT_SOURCE_DIR is not set -- point it at an onnxruntime source checkout. See README.md.")
endif()
if (NOT ONNX_CUSTOM_PROTOC_EXECUTABLE)
  message(FATAL_ERROR "ONNX_CUSTOM_PROTOC_EXECUTABLE is not set -- point it at a HOST protoc binary "
                       "(protobuf's codegen step runs on the host, not the wasm target). See README.md.")
endif()

option(onnxruntime_BUILD_WEBASSEMBLY "" ON)
option(onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB "" ON)
option(onnxruntime_ENABLE_WEBASSEMBLY_SIMD "" ON)
option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING "" ON)
option(onnxruntime_ENABLE_WEBASSEMBLY_THREADS "" OFF)
option(onnxruntime_BUILD_UNIT_TESTS "" OFF)
option(onnxruntime_ENABLE_TRAINING_APIS "" ON)
set(onnxruntime_EMSCRIPTEN_SETTINGS "MALLOC=dlmalloc" CACHE STRING "" FORCE)
set(ONNXRUNTIME_INCLUDE_DIR ${ORT_SOURCE_DIR}/include)

add_subdirectory(${ORT_SOURCE_DIR}/cmake onnxruntime-build)

add_executable(onnx-finetune-wasm src/onnx_finetune_wasm.cpp)
target_include_directories(onnx-finetune-wasm PRIVATE
  ${ORT_SOURCE_DIR}/include/onnxruntime/core/session
  ${ORT_SOURCE_DIR}/orttraining/orttraining/training_api/include
)
target_link_libraries(onnx-finetune-wasm PRIVATE embind onnxruntime_webassembly)

set_target_properties(onnx-finetune-wasm PROPERTIES
  LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4294967296 -s EXIT_RUNTIME=0 \
-s FORCE_FILESYSTEM=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"createOnnxFinetune\"' \
-s ERROR_ON_UNDEFINED_SYMBOLS=0"
)

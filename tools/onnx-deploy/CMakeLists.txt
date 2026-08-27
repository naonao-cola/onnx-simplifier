cmake_minimum_required(VERSION 3.18)
project(onnx-deploy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Unlike tools/onnx-finetune (which needs a training-enabled from-source ORT
# build), this project never links against libonnxruntime at build time at
# all -- see include/onnx_deploy/kv_cache_pipeline.h's ORT_API_MANUAL_INIT
# usage and src/onnx_deploy_c_api.cpp's dlopen-based onnx_deploy_load_ort().
# ORT_HOME only needs to point at *some* ONNX Runtime distribution's headers
# (any recent version works -- the ORT C API is stable/backward-compatible);
# the actual libonnxruntime to run against is chosen at *runtime* via
# onnx_deploy_load_ort()/--libort, and can be swapped without rebuilding.
#   cmake -B build -DORT_HOME=/path/to/onnxruntime-linux-x64-1.19.2
if(NOT ORT_HOME)
  message(FATAL_ERROR "ORT_HOME is not set -- point it at any extracted ONNX Runtime release (only its include/ is used at build time). See README.md.")
endif()

# The swappable-libort C ABI: builds and links with zero dependency on any
# specific libonnxruntime -- only ${CMAKE_DL_LIBS} (dlopen/dlsym) plus the
# headers from ORT_HOME. This is the one artifact every other consumer
# (the CLI below, a Python/Rust/Go/etc. binding, a future WASM build) is
# meant to link against or dlopen itself.
add_library(onnx_deploy_c SHARED src/onnx_deploy_c_api.cpp)
target_include_directories(onnx_deploy_c PRIVATE include ${ORT_HOME}/include)
target_link_libraries(onnx_deploy_c PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(onnx_deploy_c PROPERTIES
  PUBLIC_HEADER "include/onnx_deploy/onnx_deploy_c_api.h"
)

# CLI demo: a thin consumer of the C ABI above (see src/main.cpp), included
# to double as an executable smoke test of the ABI itself, not just of the
# C++ core it wraps.
add_executable(onnx-deploy src/main.cpp)
target_include_directories(onnx-deploy PRIVATE include)
target_link_libraries(onnx-deploy PRIVATE onnx_deploy_c)
set_target_properties(onnx-deploy PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

# Optional nanobind Python extension (tools/onnx-deploy/python/), following
# the same nanobind convention as onnxsim's own onnxsim/cpp2py_export.cc.
# Off by default since it needs Python dev headers + `pip install nanobind`;
# turn on explicitly:
#   cmake -B build -DORT_HOME=... -DONNX_DEPLOY_PYTHON=ON
option(ONNX_DEPLOY_PYTHON "Build the onnx_deploy_py nanobind extension" OFF)
if(ONNX_DEPLOY_PYTHON)
  add_subdirectory(python)
endif()

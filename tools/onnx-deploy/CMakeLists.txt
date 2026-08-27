cmake_minimum_required(VERSION 3.18)
project(onnx-deploy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Unlike tools/onnx-finetune (which needs a training-enabled from-source ORT
# build), this tool only calls the ordinary inference C++ API, so any plain
# ONNX Runtime distribution works: the official prebuilt release tarball
# (github.com/microsoft/onnxruntime/releases,
# onnxruntime-<platform>-<version>.tgz) or a `pip install onnxruntime`'s
# bundled headers/lib. Point ORT_HOME at a prefix containing
# include/onnxruntime_cxx_api.h and lib/libonnxruntime.{so,dylib}:
#   cmake -B build -DORT_HOME=/path/to/onnxruntime-linux-x64-1.19.2
if(NOT ORT_HOME)
  message(FATAL_ERROR "ORT_HOME is not set -- point it at an extracted ONNX Runtime release (contains include/onnxruntime_cxx_api.h and lib/). See README.md.")
endif()

add_executable(onnx-deploy src/main.cpp)

target_include_directories(onnx-deploy PRIVATE
  include
  ${ORT_HOME}/include
)

find_library(ORT_LIBRARY onnxruntime PATHS ${ORT_HOME}/lib REQUIRED NO_DEFAULT_PATH)
target_link_libraries(onnx-deploy PRIVATE ${ORT_LIBRARY})
set_target_properties(onnx-deploy PROPERTIES
  BUILD_RPATH ${ORT_HOME}/lib
  INSTALL_RPATH ${ORT_HOME}/lib
)

cmake_minimum_required(VERSION 3.22)

# For std::filesystem in onnx optimizer
# Must be a cache variable and be set before project()
# Reference: https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_DEPLOYMENT_TARGET.html
# It can be a normal variable if policy CMP0126 is set to NEW.
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15 CACHE STRING "Minimum OS X deployment version")

project(onnxsim CXX)

set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

option(ONNXSIM_PYTHON "" OFF)
option(ONNXSIM_BUILTIN_ORT "" ON)
option(ONNXSIM_WASM_NODE "For node (enable NODERAWFS etc.)" OFF)

if (ONNXSIM_PYTHON AND EMSCRIPTEN)
  message(STATUS "python and emscripten cannot be built at the same time")
endif()

if (NOT ONNXSIM_BUILTIN_ORT AND EMSCRIPTEN)
  message(STATUS "emscripten needs builtin ort")
endif()

add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>>
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>>
  $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:Clang>:-fcolor-diagnostics>>)
if (WIN32)
  add_compile_definitions(NOMINMAX)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (ONNXSIM_BUILTIN_ORT)
  include(cmake/build_ort.cmake)
  if (EMSCRIPTEN)
    set(ORT_NAME onnxruntime_webassembly)
  else()
    set(ORT_NAME onnxruntime)
  endif()
endif()

# configure onnx-optimizer after onnxruntime, because they both depend on onnx and onnxruntime has its own flags for onnx
add_subdirectory(third_party/onnx-optimizer EXCLUDE_FROM_ALL)

add_library(onnxsim onnxsim/onnxsim.cpp)
if (ONNXSIM_BUILTIN_ORT)
  target_include_directories(onnxsim PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
endif()
target_include_directories(onnxsim PUBLIC onnxsim)
if (NOT ONNXSIM_BUILTIN_ORT)
  target_compile_definitions(onnxsim PUBLIC NO_BUILTIN_ORT)
endif()
if (EMSCRIPTEN)
  target_link_libraries(onnxsim ${ORT_NAME} onnx_optimizer)
else()
  target_link_libraries(onnxsim ${ORT_NAME} onnx_optimizer onnx)
endif()

if(EMSCRIPTEN)
  list(APPEND ONNXSIM_WASM_INTERFACE scripts/convertmodel/interface.cpp)
endif()

add_executable(
  onnxsim_bin
  onnxsim/bin/onnxsim_bin.cpp
  onnxsim/bin/onnxsim_option.cpp
  ${ONNXSIM_WASM_INTERFACE})
set_target_properties(onnxsim_bin PROPERTIES OUTPUT_NAME onnxsim)
if (EMSCRIPTEN)
  set_target_properties(onnxsim_bin PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s EXIT_RUNTIME=0 -s FORCE_FILESYSTEM=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_onnxsim\"' -s 
  'EXPORTED_RUNTIME_METHODS=[ENV]' -Wl,--threads=8 -Wl,--lto-partitions=8 -Wl,--lto-O0 -Wl,--lto-CGO0 -s ERROR_ON_UNDEFINED_SYMBOLS=0 -sASSERTIONS=2 -sINVOKE_RUN=0")
  target_link_libraries(onnxsim_bin embind "-Wl,--whole-archive" onnxsim "-Wl,--no-whole-archive")
else()
  target_link_libraries(onnxsim_bin onnxsim)
endif()

if (ONNXSIM_PYTHON)
  find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module REQUIRED
    OPTIONAL_COMPONENTS Development.SABIModule)
  nanobind_add_module(onnxsim_cpp2py_export onnxsim/cpp2py_export.cc STABLE_ABI)
  target_link_libraries(onnxsim_cpp2py_export PRIVATE onnxsim)
endif()

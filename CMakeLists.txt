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
set(CMAKE_CXX_STANDARD 20)

option(ONNXSIM_PYTHON "" OFF)
option(ONNXSIM_BUILTIN_ORT "" ON)
option(ONNXSIM_WASM_NODE "For node (enable NODERAWFS etc.)" OFF)
option(ONNXSIM_C_API "Build the C ABI shared library used by the Rust wrapper and other FFI consumers" OFF)
option(ONNXSIM_COVERAGE "Instrument onnxsim's own C++ targets for gcov/llvm-cov coverage" OFF)
option(ONNXSIM_TESTS "Build the dependency-free C++ unit tests (e.g. sym_expr)" OFF)

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
  # The top-level project only enables CXX, but ONNX Runtime (added below as a
  # subdirectory) compiles C and assembly sources. Enable the languages it needs
  # before configuring it. Emscripten's toolchain handles this on its own.
  if (NOT EMSCRIPTEN)
    enable_language(C)
    enable_language(ASM)
  endif()
  include(cmake/build_ort.cmake)
  if (EMSCRIPTEN)
    set(ORT_NAME onnxruntime_webassembly)
  else()
    set(ORT_NAME onnxruntime)
  endif()
endif()

# configure onnx-optimizer after onnxruntime, because they both depend on onnx and onnxruntime has its own flags for onnx
add_subdirectory(third_party/onnx-optimizer EXCLUDE_FROM_ALL)

add_library(onnxsim onnxsim/onnxsim.cpp onnxsim/contrib_schemas.cpp onnxsim/function_rewriter.cpp onnxsim/profiler.cpp onnxsim/model_info.cpp onnxsim/sym_expr.cpp onnxsim/sym_value_eval.cpp onnxsim/model_metrics.cpp)
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
  # The profiler samples RSS on a background std::thread, so link the platform
  # threading library explicitly rather than relying on it arriving transitively
  # through ONNX Runtime. Emscripten provides threading through its own toolchain
  # flags, so skip it there.
  find_package(Threads REQUIRED)
  target_link_libraries(onnxsim ${ORT_NAME} onnx_optimizer onnx Threads::Threads)
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

if (ONNXSIM_C_API)
  if (NOT ONNXSIM_BUILTIN_ORT)
    message(FATAL_ERROR "ONNXSIM_C_API requires ONNXSIM_BUILTIN_ORT=ON")
  endif()
  # In the builtin-ORT path onnx_optimizer is built as a shared library
  # (build_ort.cmake sets BUILD_SHARED_LIBS ON), but it inherits the
  # project-wide "hidden" visibility preset, so it exports none of its symbols.
  # onnxsim/onnxsim_c call into it across the shared-library boundary (e.g.
  # OptimizeFixed, GetFuseAndEliminationPass), which then fail to resolve at
  # runtime with "undefined symbol". Restore default visibility so those symbols
  # are exported. (The Python build avoids this by linking it statically.)
  if (TARGET onnx_optimizer)
    set_target_properties(onnx_optimizer PROPERTIES CXX_VISIBILITY_PRESET default
                                                    VISIBILITY_INLINES_HIDDEN FALSE)
  endif()
  add_library(onnxsim_c SHARED onnxsim/capi/onnxsim_c_api.cpp)
  target_link_libraries(onnxsim_c PRIVATE onnxsim onnx_optimizer onnx)
  target_include_directories(onnxsim_c PUBLIC onnxsim/capi)
  # The exported symbols opt back into default visibility via ONNXSIM_C_API in
  # the header, so the project-wide "hidden" preset keeps everything else local.
  set_target_properties(onnxsim_c PROPERTIES C_VISIBILITY_PRESET hidden
                                             CXX_VISIBILITY_PRESET hidden)
endif()

if (ONNXSIM_PYTHON)
  find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module REQUIRED
    OPTIONAL_COMPONENTS Development.SABIModule)
  # FREE_THREADED marks the module Py_MOD_GIL_NOT_USED so free-threaded
  # interpreters (e.g. the cp314t wheel) keep the GIL disabled instead of
  # re-enabling it at import. nanobind ignores STABLE_ABI on free-threaded
  # builds (the limited API is unavailable there), so the two keywords coexist:
  # STABLE_ABI applies on regular interpreters, FREE_THREADED on cp314t.
  nanobind_add_module(onnxsim_cpp2py_export onnxsim/cpp2py_export.cc STABLE_ABI FREE_THREADED)
  target_link_libraries(onnxsim_cpp2py_export PRIVATE onnxsim)
endif()

if (ONNXSIM_COVERAGE)
  # Instrument onnxsim's own C++ sources (onnxsim.cpp, contrib_schemas.cpp,
  # cpp2py_export.cc) with gcov/llvm-cov counters. When the Python extension is
  # imported and called from pytest, the exercised C++ writes .gcda profiles
  # next to the .gcno files in the build directory, which gcovr then turns into
  # a C++ coverage report to sit alongside coverage.py's Python report.
  #
  # STABLE_ABI is irrelevant here: coverage builds are Debug/-O0 dev builds, not
  # distributable wheels. Only GCC and Clang support --coverage; MSVC does not.
  if (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
    message(FATAL_ERROR
      "ONNXSIM_COVERAGE requires a GCC- or Clang-compatible compiler that "
      "understands --coverage (got '${CMAKE_CXX_COMPILER_ID}').")
  endif()

  # --coverage expands to -fprofile-arcs -ftest-coverage at compile time and
  # links the coverage runtime (libgcov / clang's profile runtime) at link time.
  # Instrument the static library and every consumer that links it so the
  # runtime is pulled into the final Python module / executable.
  set(_onnxsim_cov_targets onnxsim)
  if (TARGET onnxsim_cpp2py_export)
    list(APPEND _onnxsim_cov_targets onnxsim_cpp2py_export)
  endif()
  if (TARGET onnxsim_bin)
    list(APPEND _onnxsim_cov_targets onnxsim_bin)
  endif()
  if (TARGET onnxsim_c)
    list(APPEND _onnxsim_cov_targets onnxsim_c)
  endif()
  foreach(_t IN LISTS _onnxsim_cov_targets)
    target_compile_options(${_t} PRIVATE --coverage -O0 -g)
    target_link_options(${_t} PRIVATE --coverage)
  endforeach()
endif()

if (ONNXSIM_TESTS)
  # sym_expr is a self-contained, dependency-free unit (no ONNX / ORT), so its
  # test compiles and runs on its own -- including under Emscripten -- without
  # configuring the rest of the project.
  enable_testing()
  add_executable(sym_expr_test onnxsim/sym_expr_test.cpp onnxsim/sym_expr.cpp)
  target_include_directories(sym_expr_test PRIVATE onnxsim)
  add_test(NAME sym_expr_test COMMAND sym_expr_test)

  add_executable(model_metrics_test onnxsim/model_metrics_test.cpp
                                    onnxsim/model_metrics.cpp onnxsim/sym_expr.cpp)
  target_include_directories(model_metrics_test PRIVATE onnxsim)
  add_test(NAME model_metrics_test COMMAND model_metrics_test)

  # The M1 symbolic value evaluator (issue #532) is likewise dependency-free
  # (only SymExpr), so it builds and runs standalone -- including under
  # Emscripten -- without configuring ONNX / onnxruntime.
  add_executable(sym_value_eval_test onnxsim/sym_value_eval_test.cpp
                                     onnxsim/sym_value_eval.cpp onnxsim/sym_expr.cpp)
  target_include_directories(sym_value_eval_test PRIVATE onnxsim)
  add_test(NAME sym_value_eval_test COMMAND sym_value_eval_test)
endif()

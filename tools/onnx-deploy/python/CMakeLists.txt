# Builds onnx_deploy_py, a compiled nanobind extension over the onnx_deploy_c
# ABI (see ../include/onnx_deploy/onnx_deploy_c_api.h and onnx_deploy_py.cc).
# Only configured when the parent CMakeLists is run with -DONNX_DEPLOY_PYTHON=ON;
# see ../README.md for why this exists (a compiled alternative to
# optimum.onnxruntime's pure-Python generate() loop) and its scope.
#
# Follows the same nanobind CMake integration pattern as onnxsim's own
# top-level CMakeLists.txt (onnxsim_cpp2py_export): nanobind ships its CMake
# package config inside its own pip-installed wheel, so it must be located
# via `python -m nanobind --cmake_dir` before find_package(nanobind) works.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE ONNX_DEPLOY_NANOBIND_CMAKE_DIR)
set(nanobind_DIR "${ONNX_DEPLOY_NANOBIND_CMAKE_DIR}" CACHE PATH "" FORCE)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(onnx_deploy_py onnx_deploy_py.cc)
target_include_directories(onnx_deploy_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(onnx_deploy_py PRIVATE onnx_deploy_c)
set_target_properties(onnx_deploy_py PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

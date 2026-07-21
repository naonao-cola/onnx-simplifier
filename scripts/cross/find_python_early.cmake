# Injected via CMAKE_PROJECT_TOP_LEVEL_INCLUDES for the cross-build.
#
# When cross-compiling, onnx's CMakeLists takes a branch that finds the `Python`
# package with only the Interpreter component (the target dev libraries go to the
# `Python3` namespace instead). But nanobind-config.cmake requires the
# `Python::Module` target and errors out otherwise. onnx configures its
# subdirectory before onnxsim's own find_package(Python) runs, so we create the
# targets here -- at the top-level project() call, before any add_subdirectory --
# using the target dev-lib hints passed on the CMake command line
# (Python_EXECUTABLE / Python_INCLUDE_DIR / Python_LIBRARY / Python_SABI_LIBRARY).
#
# Development.SABIModule is included because onnx's cross branch requires it
# (via the Python3 package) and onnxsim uses nanobind's STABLE_ABI. Note the
# produced wheels are still version-specific: python-build-standalone's
# python3.lib imports the versioned pythonXY.dll rather than the stable-ABI
# python3.dll, so each wheel is built and tagged per CPython version.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

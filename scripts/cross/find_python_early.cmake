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
# Development.SABIModule is REQUIRED (not optional): nanobind silently disables
# the stable ABI and links the version-specific Python::Module (-> pythonXY.dll)
# whenever the Python::SABIModule target is absent, which produces a wheel that
# loads only on the exact build-time CPython. Forcing the target to exist makes
# nanobind emit a genuine abi3 module depending only on python3.dll.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

if(TARGET Python::SABIModule)
  message(STATUS "XDIAG@early SABIModule=YES Python_VERSION=${Python_VERSION} INTERPRETER_ID=${Python_INTERPRETER_ID}")
else()
  message(STATUS "XDIAG@early SABIModule=NO Python_VERSION=${Python_VERSION} INTERPRETER_ID=${Python_INTERPRETER_ID}")
endif()

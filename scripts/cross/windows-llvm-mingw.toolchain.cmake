# CMake toolchain for cross-compiling Windows (x86_64) wheels on Linux with an
# llvm-mingw toolchain (clang + lld + a self-contained UCRT/libc++ sysroot):
#   https://github.com/mstorsjo/llvm-mingw
#
# Unlike the clang-cl/xwin path this needs no Microsoft SDK download -- the
# mingw sysroot ships the Windows headers and CRT import libraries.  The
# produced .pyd targets the UCRT, matching modern python.org CPython, and
# statically links the C++ runtime so no extra DLLs need to be shipped.
#
# The C++ ABI differs from MSVC, but that is invisible to CPython: only the
# extern-"C" PyInit_* entry point crosses the boundary, and every dependency
# (onnx, protobuf, abseil, nanobind) is built with this same toolchain.
#
# Compilers default to plain clang/clang++ invoked with an explicit mingw
# target triple; llvm-mingw's clang auto-locates its bundled sysroot from that.
# Override with -DMINGW_C_COMPILER / -DMINGW_CXX_COMPILER (and optionally
# -DMINGW_SYSROOT) when using a system clang against a separate mingw sysroot.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

if(NOT DEFINED MINGW_TARGET)
  set(MINGW_TARGET "x86_64-w64-mingw32")
endif()

if(NOT DEFINED MINGW_C_COMPILER)
  find_program(MINGW_C_COMPILER NAMES clang REQUIRED)
endif()
if(NOT DEFINED MINGW_CXX_COMPILER)
  find_program(MINGW_CXX_COMPILER NAMES clang++ REQUIRED)
endif()
find_program(MINGW_RC NAMES llvm-rc ${MINGW_TARGET}-windres windres)
find_program(MINGW_AR NAMES llvm-ar ${MINGW_TARGET}-ar)
find_program(MINGW_RANLIB NAMES llvm-ranlib ${MINGW_TARGET}-ranlib)

set(CMAKE_C_COMPILER   "${MINGW_C_COMPILER}")
set(CMAKE_CXX_COMPILER "${MINGW_CXX_COMPILER}")
set(CMAKE_C_COMPILER_TARGET   "${MINGW_TARGET}")
set(CMAKE_CXX_COMPILER_TARGET "${MINGW_TARGET}")
if(MINGW_RC)
  set(CMAKE_RC_COMPILER "${MINGW_RC}")
endif()
if(MINGW_AR)
  set(CMAKE_AR "${MINGW_AR}")
endif()
if(MINGW_RANLIB)
  set(CMAKE_RANLIB "${MINGW_RANLIB}")
endif()

set(_flags "--target=${MINGW_TARGET} -Wno-unused-command-line-argument")
if(DEFINED MINGW_SYSROOT)
  string(APPEND _flags " --sysroot=${MINGW_SYSROOT}")
endif()
set(CMAKE_C_FLAGS_INIT   "${_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_flags}")

# Use lld, and statically link the C++/support runtimes (libc++/libunwind or
# libstdc++/libgcc, plus libwinpthread) so the wheel is self-contained.
set(_link "-fuse-ld=lld -static")
foreach(_t EXE SHARED MODULE)
  set(CMAKE_${_t}_LINKER_FLAGS_INIT "${_link}")
endforeach()

# std::thread pulls pthread symbols from the static C++ runtime, so winpthread
# must appear after the objects on the link line (i.e. among the standard
# libraries, not the link flags). The platform module overwrites the *_INIT
# hook, so FORCE the cache entry here (the toolchain is read before the platform
# module, whose non-FORCE cache set then becomes a no-op). clang's mingw driver
# still adds the base Win32 import libraries automatically.
set(CMAKE_C_STANDARD_LIBRARIES   "-lwinpthread" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_LIBRARIES "-lwinpthread" CACHE STRING "" FORCE)

# The compiler-detection try_compile sub-projects re-include this file but do
# not inherit arbitrary cache variables; propagate the ones used here.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  MINGW_TARGET MINGW_C_COMPILER MINGW_CXX_COMPILER MINGW_RC MINGW_AR
  MINGW_RANLIB MINGW_SYSROOT)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

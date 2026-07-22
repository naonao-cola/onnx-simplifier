# CMake toolchain file for cross-compiling Windows (x86_64, MSVC ABI) wheels on Linux.
#
# It drives clang in MSVC mode (clang-cl) and lld-link, pointed at an MSVC CRT +
# Windows SDK that has been downloaded with `xwin` (https://github.com/Jake-Shadle/xwin).
#
# Required cache variable (pass with -D on the CMake command line):
#   XWIN_DIR   Directory produced by `xwin splat --output <XWIN_DIR>`; must
#              contain crt/ and sdk/ subdirectories.
# Optional overrides: CLANG_CL, LLD_LINK, LLVM_RC (else found on PATH).
#
# The resulting binaries use the Microsoft C/C++ ABI, so a produced .pyd is
# binary-compatible with the official python.org CPython on Windows.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

if(NOT DEFINED XWIN_DIR)
  message(FATAL_ERROR "XWIN_DIR must be set to the directory produced by 'xwin splat'")
endif()

# CMake re-includes this toolchain file for the compiler-detection try_compile
# sub-projects, which do NOT inherit arbitrary cache variables. Propagate the
# ones this toolchain relies on so those sub-builds configure correctly.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  XWIN_DIR CLANG_CL LLD_LINK LLVM_RC LLVM_LIB LLVM_MT)

# --- locate the LLVM tools -----------------------------------------------------
if(NOT DEFINED CLANG_CL)
  find_program(CLANG_CL NAMES clang-cl clang-cl-18 REQUIRED)
endif()
if(NOT DEFINED LLD_LINK)
  find_program(LLD_LINK NAMES lld-link lld-link-18 REQUIRED)
endif()
if(NOT DEFINED LLVM_RC)
  find_program(LLVM_RC NAMES llvm-rc llvm-rc-18 REQUIRED)
endif()
find_program(LLVM_LIB NAMES llvm-lib llvm-lib-18)
find_program(LLVM_MT  NAMES llvm-mt  llvm-mt-18)

set(CMAKE_C_COMPILER   "${CLANG_CL}")
set(CMAKE_CXX_COMPILER "${CLANG_CL}")
# clang-cl produces MSVC-style objects; CMake links them by invoking the linker
# directly (link.exe semantics), so point that at lld-link.
set(CMAKE_LINKER       "${LLD_LINK}")
set(CMAKE_RC_COMPILER  "${LLVM_RC}")
if(LLVM_LIB)
  set(CMAKE_AR "${LLVM_LIB}")
endif()
if(LLVM_MT)
  set(CMAKE_MT "${LLVM_MT}")
endif()

# clang-cl needs an explicit target triple on a non-Windows host.
set(_triple "x86_64-pc-windows-msvc")

# --- MSVC CRT + Windows SDK include paths (clang-cl /imsvc, attached form) ------
set(_inc_flags
  "/imsvc${XWIN_DIR}/crt/include"
  "/imsvc${XWIN_DIR}/sdk/include/ucrt"
  "/imsvc${XWIN_DIR}/sdk/include/um"
  "/imsvc${XWIN_DIR}/sdk/include/shared")
string(JOIN " " _inc_flags_str ${_inc_flags})

set(_common_flags "--target=${_triple} -Wno-unused-command-line-argument ${_inc_flags_str}")
set(CMAKE_C_FLAGS_INIT   "${_common_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_common_flags}")

# --- library search paths (link.exe / lld-link style) --------------------------
set(_lib_flags
  "-libpath:${XWIN_DIR}/crt/lib/x86_64"
  "-libpath:${XWIN_DIR}/sdk/lib/um/x86_64"
  "-libpath:${XWIN_DIR}/sdk/lib/ucrt/x86_64")
# /manifest:no avoids needing mt.exe for a side-by-side manifest.
string(JOIN " " _lib_flags_str "/manifest:no" ${_lib_flags})

foreach(_t EXE SHARED MODULE)
  set(CMAKE_${_t}_LINKER_FLAGS_INIT "${_lib_flags_str}")
endforeach()

# Find programs (e.g. the host protoc) on the host PATH, but restrict library and
# header discovery to the sysroots we control.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

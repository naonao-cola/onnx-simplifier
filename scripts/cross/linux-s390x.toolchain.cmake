# CMake toolchain for cross-building onnxsim from an x86_64 Linux host to
# s390x Linux (big endian).
#
# s390x is the only mainstream big-endian Linux target still shipped by a major
# distro, so it is what onnxsim's endianness coverage is built and run against
# (under qemu-user). The sysroot is an Ubuntu s390x rootfs produced by
# `debootstrap --arch=s390x`; see scripts/cross/README.md.
#
# Point ONNXSIM_S390X_SYSROOT at that rootfs (default /rootfs-s390x).

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR s390x)

if(NOT DEFINED ONNXSIM_S390X_SYSROOT)
  set(ONNXSIM_S390X_SYSROOT "/rootfs-s390x")
endif()

set(CMAKE_C_COMPILER   s390x-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER s390x-linux-gnu-g++)

set(CMAKE_SYSROOT "${ONNXSIM_S390X_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${ONNXSIM_S390X_SYSROOT}")

# Programs (protoc, the host interpreter) must come from the host: an s390x
# binary cannot run during the build. Everything else resolves in the sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# The rootfs' -dev symlinks (e.g. libpython3.12.so -> libpython3.12.so.1) are
# relative, but its runtime linker paths are absolute; -rpath-link lets the
# linker follow the transitive DT_NEEDED of sysroot shared objects.
set(_s390x_libdirs
  "${ONNXSIM_S390X_SYSROOT}/usr/lib/s390x-linux-gnu"
  "${ONNXSIM_S390X_SYSROOT}/lib/s390x-linux-gnu")
foreach(_dir IN LISTS _s390x_libdirs)
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " -Wl,-rpath-link,${_dir}")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " -Wl,-rpath-link,${_dir}")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " -Wl,-rpath-link,${_dir}")
endforeach()
unset(_s390x_libdirs)

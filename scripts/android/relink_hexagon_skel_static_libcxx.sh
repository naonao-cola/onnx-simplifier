#!/bin/sh
# Relink TVM's libhexagon_rpc_skel.so against the static SDK libc++/libc++abi.
#
# The SDK 6.4 dynamic libc++abi.so.1 needs libc symbols (aligned_alloc,
# __cxa_thread_atexit_impl) that the phone's built-in DSP libc lacks, so the
# stock skel fails dlopen_ex. Statically linking libc++ plus two tiny shims fixes
# it. Kernel .so files must also avoid libc++ (bench script links with -nostdlib++).
#
# Usage: HEXAGON_TOOLCHAIN=... SKEL_BUILD_DIR=<hexagon_tvm_runtime_rpc-build dir> \
#        HEXAGON_RPC_LIB_DIR=<hexagon_api_output> [HEX_ARCH=v68] sh relink_hexagon_skel_static_libcxx.sh
set -eu
arch=${HEX_ARCH:-v68}
lib=$HEXAGON_TOOLCHAIN/target/hexagon/lib/$arch/G0/pic
cd "$SKEL_BUILD_DIR"
cat > shim.c <<'SHIM'
#include <stdlib.h>
void* aligned_alloc(size_t a, size_t n) { void* p = 0; return posix_memalign(&p, a < sizeof(void*) ? sizeof(void*) : a, n) ? 0 : p; }
int __cxa_thread_atexit_impl(void (*d)(void*), void* o, void* dso) { return 0; }
SHIM
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -m$arch -fPIC -O2 -c shim.c -o shim.o
cmd=$(sed 's|-o libhexagon_rpc_skel.so|-nostdlib++ -o libhexagon_rpc_skel_static.so shim.o|' \
  CMakeFiles/hexagon_rpc_skel.dir/link.txt)
sh -c "$cmd -Wl,--whole-archive $lib/libc++.a $lib/libc++abi.a -Wl,--no-whole-archive"
cp -n "$HEXAGON_RPC_LIB_DIR/libhexagon_rpc_skel.so" "$HEXAGON_RPC_LIB_DIR/libhexagon_rpc_skel.so.dyn-libcxx.bak" || true
cp libhexagon_rpc_skel_static.so "$HEXAGON_RPC_LIB_DIR/libhexagon_rpc_skel.so"

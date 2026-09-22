/* From-scratch native ARM64 FastRPC client: proves TVM's Python/C++ RPC stack is NOT required
 * to drive a custom Hexagon skel. Talks to mini_rpc.so (our own, non-TVM skel, built by
 * ourselves via qaic) purely through libcdsprpc.so's remote_handle64_open/invoke/close, using
 * the qaic-generated stub (mini_rpc_stub.c) for marshaling. No tvm.rpc, no tvm.contrib.hexagon,
 * no MinRPC, no libhexagon_rpc_skel.so anywhere in this binary or its dependency graph. */
#include <stdio.h>
#include <string.h>
#include "mini_rpc.h"

int main(int argc, char** argv) {
  const char* uri = argv[1];  /* e.g. "file:///data/local/tmp/native_transport/mini_rpc.so?mini_rpc_skel_handle_invoke&_dom=cdsp" */

  /* The missing piece: unsigned .so loading on the CDSP is refused (AEE_ECONNREFUSED) unless
   * this process explicitly opts in per-session first. TVM's own launcher/session code
   * (apps/hexagon_launcher/launcher_android.cc, src/runtime/hexagon/rpc/android/session.cc)
   * does exactly this before its first hexagon_rpc_open() call. */
  struct remote_rpc_control_unsigned_module unsigned_pd;
  unsigned_pd.domain = CDSP_DOMAIN_ID;
  unsigned_pd.enable = 1;
  int urc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_pd, sizeof(unsigned_pd));
  printf("enable_unsigned_pd rc=%d\n", urc);

  remote_handle64 h = 0;
  int rc = mini_rpc_open(uri, &h);
  if (rc != 0) {
    printf("open failed rc=%d\n", rc);
    return 1;
  }
  printf("open OK, handle=0x%llx\n", (unsigned long long)h);

  int32_t result = -1;
  rc = mini_rpc_run_add(h, 40, 2, &result);
  printf("run_add(40,2) rc=%d result=%d\n", rc, result);

  unsigned char a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  unsigned char b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  unsigned char c[8] = {0};
  rc = mini_rpc_run_kernel(h, a, 8, b, 8, c, 8);
  printf("run_kernel rc=%d out=", rc);
  for (int i = 0; i < 8; i++) printf("%d ", c[i]);
  printf("\n");

  mini_rpc_close(h);
  printf("closed OK\n");
  return (result == 42 && rc == 0) ? 0 : 2;
}

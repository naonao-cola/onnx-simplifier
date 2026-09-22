#include "mini_rpc.h"
#include <stdlib.h>

int mini_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int mini_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}
int mini_rpc_run_add(remote_handle64 h, int32 a, int32 b, int32* result) {
  *result = a + b;
  return 0;
}
int mini_rpc_run_kernel(remote_handle64 h, const unsigned char* a, int aLen,
                         const unsigned char* b, int bLen, unsigned char* c, int cLen) {
  // placeholder: real kernel would go here; for the transport PoC just sum bytes.
  int n = aLen < cLen ? aLen : cLen;
  for (int i = 0; i < n; i++) c[i] = a[i] + (i < bLen ? b[i] : 0);
  return 0;
}

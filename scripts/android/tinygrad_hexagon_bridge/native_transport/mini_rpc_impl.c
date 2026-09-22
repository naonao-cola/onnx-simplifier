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
/* Real hex_gemm_kernel.py-generated vrmpy GEMM, cin=64,cout=256,m=54400 (the real Mask R-CNN
 * backbone shape, hex_gemm_kernel.py's own default) -- verified bit-exact under qemu before
 * being pasted here (`hex_gemm_kernel.py --cin 64 --cout 256 --m 54400`). */
__attribute__((noinline)) void hex_gemm(int* restrict __attribute__((align_value(128))) data0_13926400, unsigned char* restrict __attribute__((align_value(128))) data1_3481600, unsigned char* restrict __attribute__((align_value(128))) data2_16384) {
  int buf0[32];
  for (int Lidx0 = 0; Lidx0 < 54400; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 8; Lidx1++) {
      *(buf0+0) = 0;
      *(buf0+1) = 0;
      *(buf0+2) = 0;
      *(buf0+3) = 0;
      *(buf0+4) = 0;
      *(buf0+5) = 0;
      *(buf0+6) = 0;
      *(buf0+7) = 0;
      *(buf0+8) = 0;
      *(buf0+9) = 0;
      *(buf0+10) = 0;
      *(buf0+11) = 0;
      *(buf0+12) = 0;
      *(buf0+13) = 0;
      *(buf0+14) = 0;
      *(buf0+15) = 0;
      *(buf0+16) = 0;
      *(buf0+17) = 0;
      *(buf0+18) = 0;
      *(buf0+19) = 0;
      *(buf0+20) = 0;
      *(buf0+21) = 0;
      *(buf0+22) = 0;
      *(buf0+23) = 0;
      *(buf0+24) = 0;
      *(buf0+25) = 0;
      *(buf0+26) = 0;
      *(buf0+27) = 0;
      *(buf0+28) = 0;
      *(buf0+29) = 0;
      *(buf0+30) = 0;
      *(buf0+31) = 0;
      for (int Ridx2 = 0; Ridx2 < 16; Ridx2++) {
        *(int __attribute__((vector_size(128)))*)(buf0+0) = __builtin_HEXAGON_V6_vrmpyub_acc_128B(*(int __attribute__((vector_size(128)))*)(buf0+0), *(unsigned char __attribute__((vector_size(128)))*)(data2_16384+((Lidx1<<11)+(Ridx2<<7))), *(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))));
      }
      *(int __attribute__((vector_size(128)))*)(data0_13926400+((Lidx0<<8)+(Lidx1<<5))) = *(int __attribute__((vector_size(128)))*)(buf0+0);
    }
  }
}

#define HEX_GEMM_A_LEN 3481600
#define HEX_GEMM_B_LEN 16384
#define HEX_GEMM_C_LEN 55705600

int mini_rpc_run_kernel(remote_handle64 h, const unsigned char* a, int aLen,
                         const unsigned char* b, int bLen, unsigned char* c, int cLen) {
  /* Dispatch by buffer size: the original transport-PoC test (8/8/8-byte buffers, see
   * ../README.md) still gets the placeholder byte-add below; a call sized for the real
   * cin=64,cout=256,m=54400 GEMM (hex_gemm_kernel.py's own default shape) runs the real kernel.
   * This keeps the original PoC test working unchanged while adding real capability, without an
   * IDL change. */
  if (aLen == HEX_GEMM_A_LEN && bLen == HEX_GEMM_B_LEN && cLen == HEX_GEMM_C_LEN) {
    hex_gemm((int*)(void*)c, (unsigned char*)(void*)a, (unsigned char*)(void*)b);
    return 0;
  }
  int n = aLen < cLen ? aLen : cLen;
  for (int i = 0; i < n; i++) c[i] = a[i] + (i < bLen ? b[i] : 0);
  return 0;
}

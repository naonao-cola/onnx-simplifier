/* Host run of attn_kernel.h's scalar body (pack + rows) on emulate.py case directories.
 *   cc -O2 -o attn_host_check attn_host_check.c && ./attn_host_check <case dir>... */
#include "attn_io.h"

/* mingw's msvcrt has no aligned_alloc; the scalar body needs no alignment */
static void* al(long n) { return malloc((size_t)(n < 128 ? 128 : n)); }

int main(int argc, char** argv) {
  int rc = 0;
  for (int i = 1; i < argc; i++) {
    attn_case_t c;
    if (attn_load(argv[i], &c, al)) { fprintf(stderr, "load %s failed\n", argv[i]); return 2; }
    const attn_args_t* a = &c.a;
    int32_t* srow = malloc((size_t)a->LK * 4);
    uint8_t* prow = malloc((size_t)a->LK);
    for (int h = 0; h < ATTN_H; h++) attn_pack(a, h, 0, a->LK);
    for (int h = 0; h < ATTN_H; h++) attn_rows_scalar(a, h, 0, a->LQ, srow, prow);
    rc |= attn_compare(argv[i], a->out, c.ref, (long)a->LQ * ATTN_C);
  }
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}

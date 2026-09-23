/* Host run of msda_kernel.h's scalar body on msda_ref.py case directories, vs torch.
 *   cc -O2 -o msda_host_check msda_host_check.c -lm && ./msda_host_check <case dir>... */
#include <time.h>

#include "msda_io.h"

static void* al(long n) { return aligned_alloc(128, (size_t)((n + 127) / 128 * 128)); }

int main(int argc, char** argv) {
  int rc = 0;
  for (int i = 1; i < argc; i++) {
    msda_case_t c;
    if (msda_load(argv[i], &c, al)) { fprintf(stderr, "load %s failed\n", argv[i]); return 2; }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    msda_run_scalar(&c.a, 0, c.a.Q);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[%.1f ms host] ", (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6);
    rc |= msda_compare(argv[i], c.a.out, c.ref_out, msda_n_out(&c.a), 5e-5);
  }
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}

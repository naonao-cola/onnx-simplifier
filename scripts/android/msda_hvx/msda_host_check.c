/* Host run of msda_kernel.h's scalar body on msda_ref.py case directories, vs torch.
 *   cc -O2 -o msda_host_check msda_host_check.c -lm && ./msda_host_check <case dir>... */
#include <time.h>

#include "msda_io.h"

/* mingw's msvcrt has no aligned_alloc (and no reliable clock_gettime): use the Windows
 * equivalents there. Buffers live until exit, so _aligned_malloc's separate free is moot. */
#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
static void* al(long n) { return _aligned_malloc((size_t)((n + 127) / 128 * 128), 128); }
static double now_ms(void) {
  LARGE_INTEGER f, t;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&t);
  return (double)t.QuadPart * 1e3 / (double)f.QuadPart;
}
#else
static void* al(long n) { return aligned_alloc(128, (size_t)((n + 127) / 128 * 128)); }
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
#endif

int main(int argc, char** argv) {
  int rc = 0;
  for (int i = 1; i < argc; i++) {
    msda_case_t c;
    if (msda_load(argv[i], &c, al)) { fprintf(stderr, "load %s failed\n", argv[i]); return 2; }
    double t0 = now_ms();
    msda_run_scalar(&c.a, 0, c.a.Q);
    printf("[%.1f ms host] ", now_ms() - t0);
    rc |= msda_compare(argv[i], c.a.out, c.ref_out, msda_n_out(&c.a), msda_tol(&c.a, 0));
  }
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}

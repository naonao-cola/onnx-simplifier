/* hexagon-sim (standalone, HVX 128B, --timing) run of msda_kernel.h's HVX body on a split.py dump
 * case: the first NQ queries (argv[2], default all), output vs torch, cycles per query. Phone-free:
 * qemu can't decode HVX float, the simulator can.
 *   hexagon-clang -mv69 -mhvx -mhvx-length=128B -O2 msda_sim.c -o msda_sim.elf -lm
 *   hexagon-sim -mv69 --timing --simulated_returnval msda_sim.elf -- <case dir> [NQ] */
#include <stdlib.h>

#include <hexagon_sim_timer.h>

#include "msda_io.h"

static void* al(long n) { return memalign(128, (size_t)((n + 127) / 128 * 128)); }
static unsigned long long pcycles(void) { return hexagon_sim_read_pcycles(); }

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: msda_sim <case dir> [NQ]\n"); return 2; }
  msda_case_t c;
  if (msda_load(argv[1], &c, al)) { printf("FAIL load %s\n", argv[1]); return 2; }
  const int nq = argc > 2 ? atoi(argv[2]) : c.a.Q;
  unsigned long long t0 = pcycles();
  msda_run(&c.a, 0, nq);
  unsigned long long t1 = pcycles();
  long taps = 0;
  for (int q = 0; q < nq; q++)
    for (int v = 0; v < c.a.NV; v++) taps += c.a.vis[(long)v * c.a.Q + q] ? MSDA_M * c.a.P : 0;
  printf("%s: %d queries, %llu pcycles, %.0f per query, %.1f per point\n", argv[1], nq, t1 - t0,
         (double)(t1 - t0) / nq, (double)(t1 - t0) / (taps ? taps : 1));
  int rc = msda_compare(argv[1], c.a.out, c.ref_out, (long)nq * MSDA_C, 1e-5);
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}

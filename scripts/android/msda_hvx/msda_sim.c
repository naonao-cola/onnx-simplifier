/* hexagon-sim (standalone, HVX 128B) run of msda_kernel.h on an msda_ref.py case directory: the
 * first NQ queries (default all), output vs torch, cycles per query/point. Phone-free, and the
 * HVX check: qemu 8.2 can't decode HVX float, the simulator can.
 *   hexagon-clang -mv69 -mhvx -mhvx-length=128B -O2 msda_sim.c -o msda_sim.elf -lm -lhexagon
 *   hexagon-sim -mv69 [--timing] --simulated_returnval msda_sim.elf -- <case dir> [NQ] */
#include <hexagon_sim_timer.h>

#include "msda_io.h"

static void* al(long n) { return memalign(128, (size_t)((n + 127) / 128 * 128)); }

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: msda_sim <case dir> [NQ]\n"); return 2; }
  msda_case_t c;
  if (msda_load(argv[1], &c, al)) { printf("FAIL load %s\n", argv[1]); return 2; }
  const int nq = argc > 2 && atoi(argv[2]) > 0 && atoi(argv[2]) < c.a.Q ? atoi(argv[2]) : c.a.Q;
#ifdef MSDA_HVX
  printf("body: %s\n", msda_hvx_ok(&c.a) ? "hvx" : "scalar");
#endif
  unsigned long long t0 = hexagon_sim_read_pcycles();
  msda_run(&c.a, 0, nq);
  unsigned long long t1 = hexagon_sim_read_pcycles();
  long pts = 0;
  for (int q = 0; q < nq; q++)
    for (int v = 0; v < c.a.NV; v++) pts += (!c.a.vis || c.a.vis[(long)v * c.a.Q + q]) ? (long)c.a.M * c.a.L * c.a.P : 0;
  printf("%s: %d queries, %llu pcycles, %.0f per query, %.1f per point\n", argv[1], nq, t1 - t0, (double)(t1 - t0) / nq,
         (double)(t1 - t0) / (pts ? pts : 1));
#ifdef MSDA_HVX
  const int hvx = msda_hvx_ok(&c.a);
#else
  const int hvx = 0;
#endif
  int rc = msda_compare(argv[1], c.a.out, c.ref_out, (long)nq * c.a.M * c.a.D, msda_tol(&c.a, hvx));
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}

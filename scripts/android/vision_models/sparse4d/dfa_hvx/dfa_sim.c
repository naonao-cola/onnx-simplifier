/* hexagon-sim run of dfa_core.h (the HVX body of msda_kernel.h, 24 calls per layer) on a
 * dfa_case.py case: output vs torch, cycles. Phone-free HVX check (qemu can't decode HVX float).
 *   hexagon-clang -mv69 -mhvx -mhvx-length=128B -O2 -I ../../../msda_hvx dfa_sim.c -o dfa_sim.elf -lm -lhexagon
 *   hexagon-sim -mv69 --simulated_returnval dfa_sim.elf -- <case dir> */
#include <hexagon_sim_timer.h>

#include "dfa_io.h"

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: dfa_sim <case dir>\n"); return 2; }
  dfa_args_t a;
  dfa_case_load(argv[1], &a);
  if (dfa_check(&a)) { printf("FAIL shape\n"); return 2; }
  msda_args_t m;
  dfa_msda_args(&a, 0, 0, &m);
#ifdef MSDA_HVX
  printf("body: %s\n", msda_hvx_ok(&m) ? "hvx" : "scalar");
#endif
  unsigned long long t0 = hexagon_sim_read_pcycles();
  dfa_run(&a, 0, a.Q);
  unsigned long long t1 = hexagon_sim_read_pcycles();
  printf("%s: Q %d, %llu pcycles (%.0f per anchor)\n", argv[1], a.Q, t1 - t0, (double)(t1 - t0) / a.Q);
  dfa_compare(argv[1], "ref_u8.f32", &a);
  dfa_compare(argv[1], "ref_f32.f32", &a);
  return 0;
}

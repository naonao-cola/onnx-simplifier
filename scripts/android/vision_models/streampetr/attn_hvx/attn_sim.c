/* hexagon-sim (standalone, HVX 128B) run of attn_kernel.h's HVX body on an emulate.py case: pack +
 * the first NR rows of every head (default all), bit-exact vs the contract, pcycles per row-head.
 *   hexagon-clang -mv69 -mhvx -mhvx-length=128B -O2 attn_sim.c -o attn_sim.elf -lhexagon
 *   hexagon-sim -mv69 [--timing] --simulated_returnval attn_sim.elf -- <case dir> [NR] */
#include <hexagon_sim_timer.h>

#include "attn_io.h"

static void* al(long n) { return memalign(128, (size_t)((n + 127) / 128 * 128)); }

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: attn_sim <case dir> [NR]\n"); return 2; }
  attn_case_t c;
  if (attn_load(argv[1], &c, al)) { printf("FAIL load %s\n", argv[1]); return 2; }
  attn_args_t* a = &c.a;
  const int nr = argc > 2 && atoi(argv[2]) > 0 && atoi(argv[2]) < a->LQ ? atoi(argv[2]) : a->LQ;
  int32_t* srow = al((long)ATTN_RB * a->LK * 4);
  uint8_t* prow = al((long)ATTN_RB * a->LK);
  unsigned long long t0 = hexagon_sim_read_pcycles();
  for (int h = 0; h < ATTN_H; h++) attn_pack(a, h, 0, a->LK);
  unsigned long long t1 = hexagon_sim_read_pcycles();
  for (int h = 0; h < ATTN_H; h++) attn_rows_hvx(a, h, 0, nr, srow, prow);
  unsigned long long t2 = hexagon_sim_read_pcycles();
  printf("%s: pack %llu pcycles, %d rows x %d heads %llu pcycles (%.0f per row-head)\n", argv[1], t1 - t0, nr, ATTN_H,
         t2 - t1, (double)(t2 - t1) / nr / ATTN_H);
#ifdef ATTN_PROF
  printf("QK %llu, exp %llu, AV %llu pcycles\n", attn_prof[0], attn_prof[1], attn_prof[2]);
#endif
  int rc = 0;
  for (int i = 0; i < nr; i++)
    rc |= memcmp(a->out + (long)i * ATTN_C, c.ref + (long)i * ATTN_C, ATTN_C) != 0;
  if (rc) attn_compare(argv[1], a->out, c.ref, (long)nr * ATTN_C);
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}

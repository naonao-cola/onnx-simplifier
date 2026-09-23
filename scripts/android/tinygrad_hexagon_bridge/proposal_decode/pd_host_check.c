/* Host check: runs pd_kernel.h for every captured level, both delta sources, and compares against
 * ONNX Runtime's real output (grid-quantized boxes), with the reference (division) and fast (LUT)
 * paths.
 *   cc -O2 -ffp-contract=off -o pd_host_check pd_host_check.c && ./pd_host_check DATA_DIR */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pd_kernel.h"

static void* load(const char* dir, const char* name, int lvl, long bytes) {
  char p[512]; snprintf(p, sizeof p, "%s/l%d_%s.bin", dir, lvl, name);
  void* b = malloc(bytes); FILE* f = fopen(p, "rb");
  if (!f || fread(b, 1, bytes, f) != (size_t)bytes) { fprintf(stderr, "load %s\n", p); exit(1); }
  fclose(f); return b;
}

int main(int argc, char** argv) {
  const char* dir = argv[1];
  char p[512]; snprintf(p, sizeof p, "%s/levels.txt", dir);
  FILE* lf = fopen(p, "r");
  int A, k, H, W, lvl = 0, bad = 0; pd_params P;
  while (fscanf(lf, "%d %d %d %d %f %d %f %d %f %f %f %f %d %f %d", &A, &k, &H, &W, &P.s1, &P.z1, &P.s2,
                &P.z2, &P.exp_clip, &P.clip_x, &P.clip_y, &P.box_s, &P.box_z, &P.bb_s, &P.bb_z) == 15) {
    float* an = load(dir, "anchors", lvl, 16L * A);
    int32_t* idx = load(dir, "idx", lvl, 4L * k);
    float* dl = load(dir, "deltas", lvl, 16L * A);
    uint8_t* nq = load(dir, "nchw_q", lvl, 12L * H * W);
    float* ref = load(dir, "ref", lvl, 16L * k);
    float* out = malloc(16L * k);
    snprintf(p, sizeof p, "%s/l%d_params.bin", dir, lvl); /* raw pd_params for the qemu / phone harnesses */
    FILE* pf = fopen(p, "wb"); fwrite(&P, sizeof P, 1, pf); fclose(pf);
    pd_prep Qt, Q; pd_prepare(&P, &Qt); Q = Qt; pd_prepare_anchors(an, H, W, &Q);
    printf("level%d anchors are an exact base+stride grid: %s (stride %g)\n", lvl, Q.grid ? "yes" : "NO", Q.stride);
    bad |= !Q.grid;
    const char* pname[3] = {"reference", "fast/table-anchors", "fast/grid-anchors"};
    for (int v = 0; v < 6; v++) {
      int src = v & 1, fast = v >> 1;
      pd_decode(an, idx, k, src ? NULL : dl, src ? nq : NULL, H, W, &P, fast == 0 ? NULL : fast == 1 ? &Qt : &Q, out);
      int nd = 0; float mx = 0;
      for (int j = 0; j < 4 * k; j++) { float e = out[j] - ref[j]; if (e < 0) e = -e; if (e > mx) mx = e; nd += out[j] != ref[j]; }
      printf("level%d A=%d k=%d src=%-10s path=%-18s diff_elems=%d/%d max_abs_diff=%g\n", lvl, A, k, src ? "nchw_u8" : "deltas_f32", pname[fast], nd, 4 * k, mx);
      bad |= nd != 0;
    }
    { /* off-grid fp32 deltas never occur in this graph; nudge every 7th one by an ulp so the fast
       * path's exact fallback runs, and require it to still match the reference path */
      float* pert = malloc(16L * A); float* o2 = malloc(16L * k);
      memcpy(pert, dl, 16L * A);
      for (long e = 0; e < 4L * A; e += 7) { union { float f; int32_t i; } u = {pert[e]}; u.i += 1; pert[e] = u.f; }
      pd_decode(an, idx, k, pert, NULL, H, W, &P, NULL, out);
      pd_decode(an, idx, k, pert, NULL, H, W, &P, &Q, o2);
      int nd = 0; for (int j = 0; j < 4 * k; j++) nd += out[j] != o2[j];
      printf("level%d off-grid fp32 deltas: fast vs reference diff_elems=%d/%d\n", lvl, nd, 4 * k);
      bad |= nd != 0; free(pert); free(o2);
    }
    free(an); free(idx); free(dl); free(nq); free(ref); free(out);
    lvl++;
  }
  puts(bad ? "FAIL" : "PASS (bit-exact vs ORT: all levels, both delta sources, reference and both fast paths)");
  return bad;
}

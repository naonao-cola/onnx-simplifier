/* CI self-consistency check for ../proposal_decode/pd_kernel.h on synthetic data (no captured model).
 * The phone/host checks compare against ONNX Runtime's real rest.onnx outputs; CI has no model, so this
 * checks the invariants that don't need one, on every path the kernel has:
 *   - both delta sources (fp32 per-anchor deltas on the backbone's uint8 grid, raw backbone NCHW uint8),
 *   - the reference (per-box, real division) path vs the fast blocked path with table anchors and with
 *     computed grid anchors,
 *   - off-grid fp32 deltas (the fast path's exact fallback) vs the reference path.
 * All five must agree bit for bit. Also writes lN_*.bin + levels.txt in ../proposal_decode's format,
 * with lN_ref.bin = the host reference path, so pd_qemu.c can check the hexagon build against it.
 *   cc -O2 -ffp-contract=off -I ../proposal_decode -o pd_selfcheck pd_selfcheck.c && ./pd_selfcheck OUT_DIR */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pd_kernel.h"

static uint32_t seed = 12345u;
static uint32_t rnd(void) { seed = seed * 1103515245u + 12345u; return seed >> 8; }

static void save(const char* dir, int lvl, const char* name, const void* p, long bytes) {
  char path[512];
  snprintf(path, sizeof path, "%s/l%d_%s.bin", dir, lvl, name);
  FILE* f = fopen(path, "wb");
  if (!f || fwrite(p, 1, bytes, f) != (size_t)bytes) { fprintf(stderr, "write %s\n", path); exit(1); }
  fclose(f);
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  /* H, W, k per level: small maps, plus one with W=1 and one with k = all anchors */
  const int dims[4][3] = {{20, 27, 300}, {9, 13, 150}, {5, 1, 15}, {3, 4, 36}};
  char p[512];
  snprintf(p, sizeof p, "%s/levels.txt", dir);
  FILE* lf = fopen(p, "w");
  int bad = 0;
  for (int lvl = 0; lvl < 4; lvl++) {
    const int H = dims[lvl][0], W = dims[lvl][1], k = dims[lvl][2], A = 3 * H * W, HW = H * W;
    /* plausible constants in the real graph's ranges (rest.onnx: small delta scales, uint8 zero points,
     * dw/dh clip log(1000/16), image clip 1279x959, a box Q/DQ grid) */
    pd_params P = {.s1 = 0.0078125f * (1 + lvl), .z1 = 120 + lvl, .s2 = 0.00390625f * (2 + lvl), .z2 = 128,
                   .exp_clip = 4.135166556742356f, .clip_x = 1279.0f, .clip_y = 959.0f,
                   .box_s = 5.0f + lvl, .box_z = 0, .bb_s = 0.0137f * (1 + lvl), .bb_z = 131 - 3 * lvl};
    const float stride = (float)(4 << lvl);
    float* an = malloc(16L * A);
    for (int i = 0; i < A; i++) { /* anchor i = base[i % 3] + (w, h, w, h) * stride, like the real tables */
      int a = i % 3, hw = i / 3, h = hw / W, w = hw % W;
      float bw = stride * (a + 1), bh = stride * (3 - a);
      float base[4] = {-bw, -bh, bw, bh};
      an[4 * i + 0] = base[0] + w * stride; an[4 * i + 1] = base[1] + h * stride;
      an[4 * i + 2] = base[2] + w * stride; an[4 * i + 3] = base[3] + h * stride;
    }
    uint8_t* nq = malloc(12L * HW);
    for (long j = 0; j < 12L * HW; j++) nq[j] = (uint8_t)(rnd() & 0xff);
    float* dl = malloc(16L * A); /* the same deltas as fp32 on the backbone grid: (q - bb_z) * bb_s */
    for (int i = 0; i < A; i++)
      for (int c = 0; c < 4; c++)
        dl[4 * i + c] = ((float)nq[((i % 3) * 4 + c) * HW + i / 3] - (float)P.bb_z) * P.bb_s;
    int32_t* idx = malloc(4L * k);
    { /* k distinct anchor indices in random order, like a TopK's output */
      int* perm = malloc(4L * A);
      for (int i = 0; i < A; i++) perm[i] = i;
      for (int i = A - 1; i > 0; i--) { int j = rnd() % (i + 1), t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
      for (int j = 0; j < k; j++) idx[j] = perm[j];
      free(perm);
    }
    pd_prep Qt, Q;
    pd_prepare(&P, &Qt);
    Q = Qt;
    pd_prepare_anchors(an, H, W, &Q);
    if (!Q.grid) { printf("level%d: synthetic anchors not detected as a grid\n", lvl); bad = 1; }
    float* ref = malloc(16L * k);
    float* out = malloc(16L * k);
    pd_decode(an, idx, k, dl, NULL, H, W, &P, NULL, ref);
    const char* pname[3] = {"reference", "fast/table-anchors", "fast/grid-anchors"};
    for (int v = 0; v < 6; v++) {
      int src = v & 1, fast = v >> 1;
      pd_decode(an, idx, k, src ? NULL : dl, src ? nq : NULL, H, W, &P, fast == 0 ? NULL : fast == 1 ? &Qt : &Q, out);
      int nd = 0;
      for (int j = 0; j < 4 * k; j++) nd += memcmp(&out[j], &ref[j], 4) != 0;
      printf("level%d H=%d W=%d k=%d src=%-10s path=%-18s diff_elems=%d/%d\n", lvl, H, W, k, src ? "nchw_u8" : "deltas_f32",
             pname[fast], nd, 4 * k);
      bad |= nd != 0;
    }
    { /* off-grid fp32 deltas: nudge every 7th by one ulp, fast path must still match the reference path */
      float* pert = malloc(16L * A);
      float* o2 = malloc(16L * k);
      memcpy(pert, dl, 16L * A);
      for (long e = 0; e < 4L * A; e += 7) { union { float f; int32_t i; } u = {pert[e]}; u.i += 1; pert[e] = u.f; }
      pd_decode(an, idx, k, pert, NULL, H, W, &P, NULL, out);
      pd_decode(an, idx, k, pert, NULL, H, W, &P, &Q, o2);
      int nd = 0;
      for (int j = 0; j < 4 * k; j++) nd += memcmp(&out[j], &o2[j], 4) != 0;
      printf("level%d off-grid fp32 deltas: fast vs reference diff_elems=%d/%d\n", lvl, nd, 4 * k);
      bad |= nd != 0;
      free(pert); free(o2);
    }
    fprintf(lf, "%d %d %d %d %.9g %d %.9g %d %.9g %.9g %.9g %.9g %d %.9g %d\n", A, k, H, W, P.s1, P.z1, P.s2, P.z2,
            P.exp_clip, P.clip_x, P.clip_y, P.box_s, P.box_z, P.bb_s, P.bb_z);
    save(dir, lvl, "params", &P, sizeof P);
    save(dir, lvl, "anchors", an, 16L * A);
    save(dir, lvl, "idx", idx, 4L * k);
    save(dir, lvl, "deltas", dl, 16L * A);
    save(dir, lvl, "nchw_q", nq, 12L * HW);
    save(dir, lvl, "ref", ref, 16L * k);
    free(an); free(nq); free(dl); free(idx); free(ref); free(out);
  }
  fclose(lf);
  puts(bad ? "FAIL" : "PASS (self-consistent: both delta sources, reference and both fast paths, off-grid fallback)");
  return bad;
}

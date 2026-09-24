/* Host (and qemu-hexagon) check: run topk_kernel.h on every real TopK call in calls.txt and compare
 * values AND int64 indices byte-for-byte against ONNX Runtime's captured outputs, for both the
 * vector-collect and the scalar-collect variant. Build with clang (uses __builtin_reduce_or). */
#include <stdio.h>
#include <stdlib.h>
#include "topk_kernel.h"

static void* slurp(const char* p, long* n) {
  FILE* f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void* b = malloc(*n > 0 ? *n : 1);
  if (fread(b, 1, *n, f) != (size_t)*n) exit(1);
  fclose(f);
  return b;
}

/* Reference ordering (the one ORT was observed to use on all 7 real calls): value desc, index asc,
 * with -0 == +0 (plain float compare). */
static const float* g_x;
static int cmp_ref(const void* a, const void* b) {
  int i = *(const int*)a, j = *(const int*)b;
  if (g_x[i] > g_x[j]) return -1;
  if (g_x[i] < g_x[j]) return 1;
  return i < j ? -1 : i > j;
}

/* Randomized stress: few distinct values (like dequantized int8), +/-0, negatives, adversarial
 * layouts (big values only at the end, all-equal) that force the threshold retry/fallback path. */
static int stress(int iters) {
  unsigned seed = 12345;
  int bad = 0;
  for (int it = 0; it < iters; it++) {
    seed = seed * 1103515245u + 12345u;
    int n = 1 + (int)((seed >> 8) % 200000);
    seed = seed * 1103515245u + 12345u;
    int k = (int)((seed >> 8) % (n + 1));
    int mode = it % 4, distinct = 1 + it % 300;
    float* x = malloc(4 * n);
    for (int i = 0; i < n; i++) {
      seed = seed * 1103515245u + 12345u;
      int q = (int)((seed >> 8) % distinct) - distinct / 2;
      float v = q * 0.0625f;
      if (q == 0 && (seed & 1)) v = -0.0f;
      if (mode == 1 && i > n - n / 50) v += 100.0f;   /* all the top values packed at the end */
      if (mode == 2) v = 1.5f;                         /* every value tied */
      if (mode == 3 && n > 4 * TK_SAMPLE) {                   /* sample sees only big values: forces the retry/fallback */
        int st = (n / TK_BLOCKS) & ~(TK_BLOCK - 1);
        if (i / st < TK_BLOCKS && i % st < TK_BLOCK) v = 50.0f + q * 0.0625f;
      }
      if (mode == 0 && it % 8 == 0) v = (float)(seed % 1000003) * 1e-3f - 500.0f; /* many distinct values: radix fallback */
      x[i] = v;
    }
    int* ord = malloc(4 * n);
    for (int i = 0; i < n; i++) ord[i] = i;
    g_x = x;
    qsort(ord, n, 4, cmp_ref);
    uint32_t* scr = malloc(4 * TK_SCRATCH_WORDS(n));
    float* ov = malloc(4 * k + 4);
    int64_t* oi = malloc(8 * k + 8);
    for (int hvx = 3; hvx >= 0; hvx--) {
      topk_desc(x, n, k, ov, oi, scr, hvx);
      for (int j = 0; j < k; j++) {
        if (oi[j] != ord[j] || memcmp(&ov[j], &x[ord[j]], 4)) { printf("stress it=%d n=%d k=%d mode=%d hvx=%d MISMATCH at %d\n", it, n, k, mode, hvx, j); bad = 1; break; }
      }
    }
    free(x); free(ord); free(scr); free(ov); free(oi);
  }
  printf("stress %d cases %s\n", iters, bad ? "FAIL" : "PASS");
  return bad;
}

int main(int argc, char** argv) {
  if (argc > 1) return stress(atoi(argv[1]));
  FILE* cf = fopen("calls.txt", "r");
  long n, k, sz;
  int axis, i = 0, bad = 0;
  while (fscanf(cf, "%ld %ld %d", &n, &k, &axis) == 3) {
    char p[64];
    snprintf(p, sizeof p, "call%d_x.bin", i); float* x = slurp(p, &sz);
    snprintf(p, sizeof p, "call%d_vals.bin", i); float* rv = slurp(p, &sz);
    snprintf(p, sizeof p, "call%d_idx.bin", i); int64_t* ri = slurp(p, &sz);
    uint32_t* scr = malloc(4 * TK_SCRATCH_WORDS(n));
    float* ov = malloc(4 * k + 4);
    int64_t* oi = malloc(8 * k + 8);
    for (int hvx = 3; hvx >= 0; hvx--) {
      int c = topk_desc(x, (int)n, (int)k, ov, oi, scr, hvx);
      int ok = memcmp(ov, rv, 4 * k) == 0 && memcmp(oi, ri, 8 * k) == 0;
      if (!ok) bad = 1;
      printf("call%d n=%ld k=%ld %s survivors=%d passes=%d %s\n", i, n, k, hvx == 3 ? "vec-reduce_or-mask" : hvx == 2 ? "vec-rot" : hvx == 1 ? "vec-reduce_or" : "scalar", c, tk_passes, ok ? "EXACT" : "MISMATCH");
    }
    free(x); free(rv); free(ri); free(scr); free(ov); free(oi);
    i++;
  }
  puts(bad ? "FAIL" : "PASS");
  return bad;
}

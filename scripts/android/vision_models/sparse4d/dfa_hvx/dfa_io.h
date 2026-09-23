/* Loads a dfa_case.py case into dfa_args_t and compares the result with its references. Shared
 * by dfa_host_check.c (scalar body, host) and dfa_sim.c (HVX body, hexagon-sim). */
#ifndef DFA_IO_H
#define DFA_IO_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dfa_core.h"

static void* dfa_load(const char* dir, const char* name, long bytes) {
  char p[1024];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
  void* b = NULL;
  if (posix_memalign(&b, 128, bytes < 128 ? 128 : bytes)) exit(1);
  if (fread(b, 1, bytes, f) != (size_t)bytes) { fprintf(stderr, "short %s\n", p); exit(1); }
  fclose(f);
  return b;
}

static void* dfa_zalloc(long bytes) {
  void* b = NULL;
  if (posix_memalign(&b, 128, bytes)) exit(1);
  memset(b, 0, bytes);
  return b;
}

static void dfa_case_load(const char* dir, dfa_args_t* a) {
  char p[1024];
  snprintf(p, sizeof p, "%s/meta.txt", dir);
  FILE* f = fopen(p, "r");
  if (!f || fscanf(f, "%d", &a->Q) != 1) { fprintf(stderr, "bad %s\n", p); exit(1); }
  for (int l = 0; l < DFA_LEVELS; l++) {
    int zp = 0;
    if (fscanf(f, "%d %d %f %d", &a->H[l], &a->W[l], &a->scale[l], &zp) != 4) exit(1);
    a->zp[l] = zp;
    char n[16];
    snprintf(n, sizeof n, "val%d.u8", l);
    a->val[l] = (const uint8_t*)dfa_load(dir, n, (long)DFA_CAMS * a->H[l] * a->W[l] * DFA_C);
  }
  fclose(f);
  a->pts = (const float*)dfa_load(dir, "pts.f32", (long)DFA_CAMS * a->Q * DFA_P * 2 * 4);
  a->w = (const float*)dfa_load(dir, "w.f32", (long)DFA_CAMS * DFA_LEVELS * a->Q * DFA_M * DFA_P * 4);
  a->zeros = (const float*)dfa_zalloc((long)a->Q * DFA_M * DFA_P * 2 * 4);
  a->vis = (uint8_t*)dfa_zalloc((long)DFA_CAMS * a->Q);
  a->tmp = (float*)dfa_zalloc((long)a->Q * DFA_C * 4);
  a->out = (float*)dfa_zalloc((long)a->Q * DFA_C * 4);
}

/* max abs diff, relative to max |ref|, and cosine vs one reference */
static void dfa_compare(const char* dir, const char* ref_name, const dfa_args_t* a) {
  const float* r = (const float*)dfa_load(dir, ref_name, (long)a->Q * DFA_C * 4);
  double md = 0, mr = 0, dot = 0, na = 0, nb = 0;
  for (long i = 0; i < (long)a->Q * DFA_C; i++) {
    double x = a->out[i], y = r[i];
    md = fmax(md, fabs(x - y));
    mr = fmax(mr, fabs(y));
    dot += x * y; na += x * x; nb += y * y;
  }
  printf("  vs %-12s max|diff| %.3g (%.2e of max|ref| %.3g)  cos %.9f\n", ref_name, md, md / mr, mr,
         dot / sqrt(na * nb + 1e-300));
}
#endif

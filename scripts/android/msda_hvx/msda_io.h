/* Loads one msda_ref.py case directory and compares an output with its torch reference. Shared by
 * msda_host_check.c, msda_sim.c and msda_client.c. A case directory holds
 *   meta.txt     "NV L S M D P Q NO mode NVR RL R RD has_vis" then one "H W start" line per level
 *   value.f32 loc.f32 attw.f32 [ref.f32 unless mode 0] [vis.u8 if has_vis] ref_out.f32
 * `alloc` returns 128-byte aligned memory (rpcmem on the phone). */
#ifndef MSDA_IO_H
#define MSDA_IO_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "msda_shape.h"

typedef struct {
  msda_args_t a;
  float* ref_out;
  int has_vis;
} msda_case_t;

static int msda_read(const char* dir, const char* name, void* dst, long bytes) {
  char p[1024];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) { fprintf(stderr, "open %s\n", p); return -1; }
  long got = (long)fread(dst, 1, (size_t)bytes, f);
  fclose(f);
  if (got != bytes) { fprintf(stderr, "%s: %ld of %ld bytes\n", p, got, bytes); return -1; }
  return 0;
}

static int msda_load(const char* dir, msda_case_t* c, void* (*alloc)(long)) {
  char p[1024];
  snprintf(p, sizeof p, "%s/meta.txt", dir);
  FILE* f = fopen(p, "r");
  if (!f) { fprintf(stderr, "open %s\n", p); return -1; }
  msda_args_t* a = &c->a;
  memset(a, 0, sizeof *a);
  int ok = fscanf(f, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d", &a->NV, &a->L, &a->S, &a->M, &a->D, &a->P, &a->Q, &a->NO,
                  &a->mode, &a->NVR, &a->RL, &a->R, &a->RD, &c->has_vis) == 14;
  for (int l = 0; ok && l < a->L && l < MSDA_MAX_L; l++) ok = fscanf(f, "%d %d %d", &a->H[l], &a->W[l], &a->start[l]) == 3;
  fclose(f);
  if (!ok || msda_check(a)) { fprintf(stderr, "%s: bad meta\n", p); return -1; }
  a->value = alloc(msda_n_value(a) * 4);
  a->loc = alloc(msda_n_loc(a) * 4);
  a->ref = a->mode == MSDA_LOC ? 0 : alloc(msda_n_ref(a) * 4);
  a->attw = alloc(msda_n_attw(a) * 4);
  a->vis = c->has_vis ? alloc(msda_n_vis(a)) : 0;
  a->out = alloc(msda_n_out(a) * 4);
  c->ref_out = alloc(msda_n_out(a) * 4);
  if (!a->value || !a->loc || (a->mode != MSDA_LOC && !a->ref) || !a->attw || (c->has_vis && !a->vis) || !a->out ||
      !c->ref_out)
    return -1;
  return msda_read(dir, "value.f32", (void*)a->value, msda_n_value(a) * 4) ||
                 msda_read(dir, "loc.f32", (void*)a->loc, msda_n_loc(a) * 4) ||
                 (a->ref && msda_read(dir, "ref.f32", (void*)a->ref, msda_n_ref(a) * 4)) ||
                 msda_read(dir, "attw.f32", (void*)a->attw, msda_n_attw(a) * 4) ||
                 (a->vis && msda_read(dir, "vis.u8", (void*)a->vis, msda_n_vis(a))) ||
                 msda_read(dir, "ref_out.f32", c->ref_out, msda_n_out(a) * 4)
             ? -1
             : 0;
}

/* max abs diff, max |ref|, cosine; 0 when abs <= tol * max|ref| and cos >= 0.99999 */
static int msda_compare(const char* label, const float* got, const float* ref, long n, double tol) {
  double md = 0, mr = 0, dot = 0, ng = 0, nr = 0;
  long bad = 0;
  for (long i = 0; i < n; i++) {
    double g = got[i], r = ref[i], d = fabs(g - r);
    if (!(d <= md)) md = d; /* NaN -> md NaN */
    if (fabs(r) > mr) mr = fabs(r);
    dot += g * r; ng += g * g; nr += r * r;
  }
  double cs = dot / (sqrt(ng * nr) + 1e-30);
  for (long i = 0; i < n; i++)
    if (!(fabs((double)got[i] - ref[i]) <= tol * mr)) bad++;
  printf("%s: max abs %.3g (max |ref| %.3g, rel %.3g), cos %.9f, %ld of %ld beyond %.0e rel\n", label, md, mr,
         md / (mr > 0 ? mr : 1), cs, bad, n, tol);
  return (md == md && md <= tol * (mr > 0 ? mr : 1) && (cs >= 0.99999 || mr == 0)) ? 0 : 1;
}

#endif

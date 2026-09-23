/* Loads one split.py dump case (<dir>/meta.txt + value/ref/off/attw/vis/ref_out) and compares an
 * output with its torch reference. Shared by msda_host_check.c, msda_sim.c and msda_client.c.
 * `alloc` returns 128-byte aligned memory (rpcmem on the phone). */
#ifndef MSDA_IO_H
#define MSDA_IO_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "msda_kernel.h"

typedef struct {
  msda_args_t a;
  float* ref_out;
  long n_value, n_ref, n_off, n_attw, n_vis, n_out; /* element counts */
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
  int ok = fscanf(f, "%d %d %d %d %d %d %d %d", &a->NV, &a->H, &a->W, &a->Q, &a->R, &a->NO, &a->P, &c->has_vis) == 8;
  fclose(f);
  if (!ok || a->NV > MSDA_MAX_NV) return -1;
  c->n_value = (long)a->NV * a->H * a->W * MSDA_C;
  c->n_ref = (long)a->NV * a->Q * a->R * 2;
  c->n_off = (long)a->Q * MSDA_M * a->NO * a->P * 2;
  c->n_attw = (long)a->Q * MSDA_M * a->NO * a->P;
  c->n_vis = (long)a->NV * a->Q;
  c->n_out = (long)a->Q * MSDA_C;
  a->value = alloc(c->n_value * 4);
  a->ref = alloc(c->n_ref * 4);
  a->off = alloc(c->n_off * 4);
  a->attw = alloc(c->n_attw * 4);
  a->vis = alloc(c->n_vis);
  a->out = alloc(c->n_out * 4);
  c->ref_out = alloc(c->n_out * 4);
  if (!a->value || !a->ref || !a->off || !a->attw || !a->vis || !a->out || !c->ref_out) return -1;
  return msda_read(dir, "value.f32", (void*)a->value, c->n_value * 4) || msda_read(dir, "ref.f32", (void*)a->ref, c->n_ref * 4) ||
                 msda_read(dir, "off.f32", (void*)a->off, c->n_off * 4) || msda_read(dir, "attw.f32", (void*)a->attw, c->n_attw * 4) ||
                 msda_read(dir, "vis.u8", (void*)a->vis, c->n_vis) || msda_read(dir, "ref_out.f32", c->ref_out, c->n_out * 4)
             ? -1
             : 0;
}

/* max abs diff, max |ref|, cosine; returns 0 when within tol (abs <= tol * max|ref|, cos >= 0.99999). */
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
    if (fabs((double)got[i] - ref[i]) > tol * mr) bad++;
  printf("%s: max abs %.3g (max |ref| %.3g, rel %.3g), cos %.9f, %ld of %ld beyond %.0e rel\n", label, md, mr, md / mr,
         cs, bad, n, tol);
  return (md == md && md <= tol * mr && cs >= 0.99999) ? 0 : 1;
}

#endif

/* Case directories written by emulate.py case: q.bin k.bin v.bin (uint8), params.txt
 * "LQ LK zq m16 dcl", ref_out.bin (uint8, the bit-exact contract). Host, sim and phone share this. */
#pragma once
#include <stdio.h>
#include <stdlib.h>

#include "attn_kernel.h"

typedef void* (*attn_alloc_fn)(long n);

typedef struct {
  attn_args_t a;
  uint8_t *qb, *kbuf, *vb, *ref;
} attn_case_t;

static int attn_read(const char* dir, const char* name, void* dst, long n) {
  char p[1024];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) return -1;
  long got = (long)fread(dst, 1, (size_t)n, f);
  fclose(f);
  return got == n ? 0 : -1;
}

static int attn_load(const char* dir, attn_case_t* c, attn_alloc_fn al) {
  char p[1024];
  snprintf(p, sizeof p, "%s/params.txt", dir);
  FILE* f = fopen(p, "r");
  if (!f) return -1;
  attn_args_t* a = &c->a;
  memset(c, 0, sizeof *c);
  int ok = fscanf(f, "%d %d %d %d %d", &a->LQ, &a->LK, &a->zq, &a->m16, &a->dcl) == 5;
  fclose(f);
  if (!ok || a->LK % 128) return -1;
  const long nq = (long)a->LQ * ATTN_C, nk = (long)a->LK * ATTN_C;
  c->qb = al(nq); c->kbuf = al(nk); c->vb = al(nk); c->ref = al(nq);
  a->out = al(nq);
  a->kp = al((long)ATTN_H * a->LK * 32); a->vp = al((long)ATTN_H * a->LK * ATTN_D); a->kb = al((long)ATTN_H * a->LK * 4);
  if (!c->qb || !c->kbuf || !c->vb || !c->ref || !a->out || !a->kp || !a->vp || !a->kb) return -1;
  a->q = c->qb; a->k = c->kbuf; a->v = c->vb;
  return attn_read(dir, "q.bin", c->qb, nq) || attn_read(dir, "k.bin", c->kbuf, nk) || attn_read(dir, "v.bin", c->vb, nk) ||
         attn_read(dir, "ref_out.bin", c->ref, nq);
}

/* bit-exact contract: every byte must match */
static int attn_compare(const char* label, const uint8_t* out, const uint8_t* ref, long n) {
  long bad = 0, first = -1;
  for (long i = 0; i < n; i++)
    if (out[i] != ref[i]) { if (first < 0) first = i; bad++; }
  if (bad) printf("%s: %ld of %ld bytes differ (first at %ld: %d vs %d)\n", label, bad, n, first, out[first], ref[first]);
  else printf("%s: %ld bytes exact\n", label, n);
  return bad != 0;
}

/* hmx_probe client: hmx_client uri mode variant limit iters act wei bias outlen [outfile]
 * act/wei/bias: a file path (raw bytes) or "=<byte>x<len>" (constant fill) or "-" (empty).
 * Prints the step return codes, DSP time, and a summary of the output region (0xCD = untouched). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hmx_rpc.h"
#include "remote.h"
#include "rpcmem.h"

static unsigned char* load(const char* spec, int* n) {
  *n = 0;
  if (!strcmp(spec, "-")) return NULL;
  if (spec[0] == '=') {
    int val = 0, len = 0;
    sscanf(spec + 1, "%ix%i", &val, &len);
    unsigned char* p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, len < 128 ? 128 : len);
    memset(p, val, len);
    *n = len;
    return p;
  }
  FILE* f = fopen(spec, "rb");
  if (!f) { fprintf(stderr, "open %s\n", spec); exit(2); }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char* p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, len < 128 ? 128 : len);
  if (fread(p, 1, len, f) != (size_t)len) exit(2);
  fclose(f);
  *n = (int)len;
  return p;
}

int main(int argc, char** argv) {
  if (argc < 10) { fprintf(stderr, "usage: %s uri mode variant limit iters act wei bias outlen [outfile]\n", argv[0]); return 2; }
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  int urc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  int orc = hmx_rpc_open(argv[1], &h);
  printf("unsigned_rc %d open_rc %d\n", urc, orc);
  if (orc) return 1;
  int prc = 0;
  /* HMX_POWER=1: also vote HMX power-up (perf_vote bit 1) */
  const char* hp = getenv("HMX_POWER");
  hmx_rpc_perf_vote(h, 1 | ((hp && atoi(hp)) ? 2 : 0), &prc);
  int na, nw, nb;
  unsigned char* a = load(argv[6], &na);
  unsigned char* w = load(argv[7], &nw);
  unsigned char* b = load(argv[8], &nb);
  int nout = atoi(argv[9]);
  unsigned char* o = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, nout < 128 ? 128 : nout);
  int32 codes[12];
  uint64 us = 0;
  int rc = hmx_rpc_probe(h, atoi(argv[2]), strtol(argv[3], 0, 0), strtol(argv[4], 0, 0), atoi(argv[5]), a, na, w, nw, b, nb,
                         o, nout, codes, 12, &us);
  printf("probe_rc %d perf_rc %d dsp_us %llu\ncodes:", rc, prc, (unsigned long long)us);
  const char* names[12] = {"attr_init", "vtcm_param", "hmx_param", "ctx", "hvx_lock", "hmx_lock",
                           "ran",       "hmx_unlock", "vtcm_ptr_rc", "vtcm_size", "vtcm_addr", "thread"};
  for (int i = 0; i < 12; i++) printf(" %s=%d", names[i], codes[i]);
  printf("\n");
  if (rc == 0 && nout) {
    int touched = 0, first = -1, last = -1;
    for (int i = 0; i < nout; i++)
      if (o[i] != 0xCD) { touched++; if (first < 0) first = i; last = i; }
    printf("out: %d/%d bytes written (first %d last %d)\nhead:", touched, nout, first, last);
    for (int i = 0; i < 64 && i < nout; i++) printf(" %02x", o[i]);
    printf("\n");
    if (argc > 10) { FILE* f = fopen(argv[10], "wb"); fwrite(o, 1, nout, f); fclose(f); }
  }
  hmx_rpc_close(h);
  return rc;
}

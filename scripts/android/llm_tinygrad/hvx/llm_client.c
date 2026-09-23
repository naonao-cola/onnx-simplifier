/* ARM64 FastRPC client for the LLM GEMV skel: for each decode-step GEMV shape, runs the tinygrad-generated kernels
 * (tc / int32 variants) and the hand vrmpy kernel on the CDSP, checks each against the numpy reference, and prints
 * the DSP-side time (min / median of reps) and the weight-streaming bandwidth it implies (K*N bytes per call).
 * Buffers are rpcmem (ION, mapped, not copied). Usage: llm_client <uri> [reps] [turbo] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "remote.h"
#include "rpcmem.h"
#include "llm_rpc.h"

static long fsize(const char* p) { struct stat s; return stat(p, &s) ? -1 : (long)s.st_size; }
static unsigned char* load(const char* p, long n) {
  unsigned char* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n);
  FILE* f = fopen(p, "rb");
  if (!b || !f || fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "load %s failed\n", p); exit(1); }
  fclose(f);
  return b;
}

int main(int argc, char** argv) {
  const char* uri = argv[1];
  int reps = argc > 2 ? atoi(argv[2]) : 15, turbo = argc > 3 ? atoi(argv[3]) : 0;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (llm_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  int vrc = -1;
  llm_rpc_perf_vote(h, turbo, &vrc);
  printf("perf_vote(turbo=%d) rc=%d\n", turbo, vrc);
  struct { const char* op; int K, N, kerns[3]; const char* names[3]; } ops[] = {
      {"up", 576, 1536, {0, 1, 4}, {"tinygrad-tc", "tinygrad-int32", "hand-vrmpy"}},
      {"down", 1536, 576, {2, 3, 4}, {"tinygrad-tc", "tinygrad-int32", "hand-vrmpy"}},
  };
  int bad = 0;
  printf("%-5s %-15s %10s %10s %9s  %s\n", "op", "kernel", "min_us", "med_us", "GB/s", "exact");
  for (int o = 0; o < 2; o++) {
    char px[64], pw[64], pp[64], pr[64];
    snprintf(px, sizeof px, "%s_x.bin", ops[o].op); snprintf(pw, sizeof pw, "%s_w.bin", ops[o].op);
    snprintf(pp, sizeof pp, "%s_wp.bin", ops[o].op); snprintf(pr, sizeof pr, "%s_ref.bin", ops[o].op);
    long nx = fsize(px), nw = fsize(pw), nr = fsize(pr);
    unsigned char *x = load(px, nx), *w = load(pw, nw), *wp = load(pp, nw), *ref = load(pr, nr);
    unsigned char* out = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, nr);
    for (int v = 0; v < 3; v++) {
      memset(out, 0xa5, nr);
      uint64 mn = 0, md = 0;
      int rc = llm_rpc_run(h, ops[o].kerns[v], reps, ops[o].K, ops[o].N, x, (int)nx, v == 2 ? wp : w, (int)nw, out,
                           (int)nr, &mn, &md);
      int ok = rc == 0 && memcmp(out, ref, nr) == 0;
      bad += !ok;
      printf("%-5s %-15s %10llu %10llu %9.2f  %s%s\n", ops[o].op, ops[o].names[v], (unsigned long long)mn,
             (unsigned long long)md, md ? (double)nw / md / 1e3 : 0.0, ok ? "yes" : "NO", rc ? " (rc!=0)" : "");
    }
    rpcmem_free(x); rpcmem_free(w); rpcmem_free(wp); rpcmem_free(ref); rpcmem_free(out);
  }
  llm_rpc_close(h);
  return bad ? 2 : 0;
}

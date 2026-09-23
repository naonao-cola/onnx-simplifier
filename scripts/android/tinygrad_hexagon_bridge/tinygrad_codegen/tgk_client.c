/* Native ARM64 FastRPC client (no TVM): for each op in gen_kernels.py's output, runs the plain-tinygrad-generated
 * kernel and the hand custom_kernel on the CDSP, checks both outputs byte-for-byte against the numpy reference, and
 * reports DSP-side kernel time (min / median of `reps`) plus the client-side round trip of one call. Buffers come from
 * rpcmem (ION-backed, mapped rather than copied). Usage: tgk_client <uri> [reps] [turbo] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "remote.h"
#include "rpcmem.h"
#include "tgk_rpc.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static long fsize(const char* p) { struct stat s; return stat(p, &s) ? -1 : (long)s.st_size; }
static unsigned char* load(const char* p, long n) {
  unsigned char* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n > 0 ? n : 128);
  if (!b) { fprintf(stderr, "rpcmem_alloc %ld failed\n", n); exit(1); }
  if (n <= 0) return b;
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  if (fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p); exit(1); }
  fclose(f); return b;
}

int main(int argc, char** argv) {
  const char* uri = argv[1];
  int reps = argc > 2 ? atoi(argv[2]) : 7, turbo = argc > 3 ? atoi(argv[3]) : 0;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (tgk_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  int vrc = -1; tgk_rpc_perf_vote(h, turbo, &vrc); printf("perf_vote(turbo=%d) rc=%d\n", turbo, vrc);
  const char* ops[] = {"add", "maxpool", "requant"};
  int bad = 0;
  printf("%-8s %-5s %12s %12s %12s  %s\n", "op", "kind", "dsp_min_us", "dsp_med_us", "rpc_rt_us", "exact");
  for (int o = 0; o < 3; o++) {
    char p0[64], p1[64], pr[64];
    snprintf(p0, sizeof p0, "%s_in0.bin", ops[o]); snprintf(p1, sizeof p1, "%s_in1.bin", ops[o]);
    snprintf(pr, sizeof pr, "%s_ref.bin", ops[o]);
    long n0 = fsize(p0), n1 = fsize(p1), nr = fsize(pr);
    if (n0 < 0 || nr < 0) { printf("%-8s (no data, skipped)\n", ops[o]); continue; }
    unsigned char *a = load(p0, n0), *b = load(p1, n1 > 0 ? n1 : 0), *ref = load(pr, nr);
    unsigned char* out = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, nr);
    for (int v = 0; v < 2; v++) {
      memset(out, 0xa5, nr);
      uint64 mn = 0, md = 0;
      double t0 = now_us();
      int rc = tgk_rpc_run(h, o * 2 + v, reps, a, (int)n0, b, n1 > 0 ? (int)n1 : 0, out, (int)nr, &mn, &md);
      double rt = (now_us() - t0) / reps;  /* per-call share of the whole round trip incl. the reps */
      int ok = rc == 0 && memcmp(out, ref, nr) == 0;
      if (!ok) bad++;
      printf("%-8s %-5s %12llu %12llu %12.0f  %s%s\n", ops[o], v ? "hand" : "gen", (unsigned long long)mn,
             (unsigned long long)md, rt, ok ? "yes" : "NO", rc ? " (rc!=0)" : "");
    }
    rpcmem_free(a); rpcmem_free(b); rpcmem_free(ref); rpcmem_free(out);
  }
  tgk_rpc_close(h);
  return bad ? 2 : 0;
}

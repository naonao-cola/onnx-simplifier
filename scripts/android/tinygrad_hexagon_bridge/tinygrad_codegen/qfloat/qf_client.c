/* ARM64 FastRPC client: for each line "<kern> <tag> <datatag> <nbufs> <size0> <size1> ..." of calls.txt, loads <datatag>_buf1..N.bin
 * into rpcmem, runs the kernel on the CDSP, writes <tag>_out.bin (checked host-side by analyze.py) and prints DSP-side
 * min/median time. Usage: qf_client <uri> [reps] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "remote.h"
#include "rpcmem.h"
#include "qf_rpc.h"
static unsigned char* load(const char* p, long n) {
  unsigned char* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n > 0 ? n : 128);
  if (!b) { fprintf(stderr, "rpcmem_alloc %ld failed\n", n); exit(1); }
  if (n <= 0) return b;
  FILE* f = fopen(p, "rb"); if (!f || fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "read %s failed\n", p); exit(1); }
  fclose(f); return b;
}
int main(int argc, char** argv) {
  int reps = argc > 2 ? atoi(argv[2]) : 9;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h; if (qf_rpc_open(argv[1], &h)) { puts("open failed"); return 1; }
  int vrc; qf_rpc_perf_vote(h, 0, &vrc);
  FILE* cf = fopen("calls.txt", "r"); char tag[64], dtag[64]; int kern, nb;
  printf("%-16s %10s %10s\n", "kernel", "dsp_min_us", "dsp_med_us");
  while (fscanf(cf, "%d %63s %63s %d", &kern, tag, dtag, &nb) == 4) {
    long sz[6] = {0}; unsigned char* b[6] = {0}; char p[128];
    for (int i = 0; i < nb; i++) fscanf(cf, "%ld", &sz[i]);
    for (int i = 1; i < 6; i++) { snprintf(p, sizeof p, "%s_buf%d.bin", dtag, i); b[i] = load(p, i < nb ? sz[i] : 0); }
    unsigned char* out = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, sz[0]);
    uint64 mn = 0, md = 0;
    int rc = qf_rpc_run(h, kern, reps, b[1], nb > 1 ? sz[1] : 0, b[2], nb > 2 ? sz[2] : 0, b[3], nb > 3 ? sz[3] : 0,
                        b[4], nb > 4 ? sz[4] : 0, b[5], nb > 5 ? sz[5] : 0, out, sz[0], &mn, &md);
    printf("%-16s %10llu %10llu%s\n", tag, (unsigned long long)mn, (unsigned long long)md, rc ? "  rc!=0" : "");
    snprintf(p, sizeof p, "%s_out.bin", tag); FILE* o = fopen(p, "wb"); fwrite(out, 1, sz[0], o); fclose(o);
    for (int i = 1; i < 6; i++) rpcmem_free(b[i]); rpcmem_free(out);
  }
  qf_rpc_close(h); return 0;
}

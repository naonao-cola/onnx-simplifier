/* Native ARM64 FastRPC client for the merged uint8 RoiAlign skel. Reads one capture_merged_io.py
 * directory (cwd), and for box and mask runs the whole span as ONE RPC per head (and, for A/B, as
 * one RPC per FPN level) over a grid of thread counts / prefetch / locality sort, reporting the
 * DSP-side time (HAP_perf_get_time_us), the client round trip, and the output's exact-byte fraction
 * and max |diff| vs QuantizeLinear(ORT fp32 RoiAlign) merged (the *_ref_u8.bin bytes).
 *   ./roialign_u8_client URI [reps] [turbo] [configs: comma list of flags, see roialign_u8_rpc.idl] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "roialign_u8_rpc.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void* load(const char* p, long* n) {
  FILE* f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, *n + 128);
  if (!b || fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "load %s failed\n", p); exit(1); }
  fclose(f);
  return b;
}
static int cmp_d(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  const char* uri = argv[1];
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  int turbo = argc > 3 ? atoi(argv[3]) : 0;
  const char* cfgs = argc > 4 ? argv[4] : "1,4,6,260,262,516,518,772,774";
  if (reps > 64) reps = 64;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (roialign_u8_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  if (turbo) { int vrc = -1; roialign_u8_rpc_perf_vote(h, 1, &vrc); printf("perf_vote(TURBO) rc=%d\n", vrc); }

  FILE* mf = fopen("meta.txt", "r");
  if (!mf) { perror("meta.txt"); return 1; }
  int geom[12], nlev[2][4] = {{0}}, N[2] = {0}, OH[2] = {0}, sr[2] = {0}, zo[2] = {0}, C = 0;
  float fp[8], so[2] = {0};
  char line[256];
  while (fgets(line, sizeof line, mf)) {
    int k, a, b, c, d; float x, y; char tag[16];
    if (sscanf(line, "map %d %d %d %d %f %d %f", &k, &a, &b, &c, &x, &d, &y) == 7) {
      geom[3 * k] = a; geom[3 * k + 1] = b; geom[3 * k + 2] = d; fp[2 * k] = x; fp[2 * k + 1] = y; C = c;
    } else if (sscanf(line, "%15s %d %d", tag, &k, &a) == 3 && strstr(tag, "_level")) {
      nlev[tag[0] == 'm'][k] = a;
    } else if (sscanf(line, "%15s %d %d %d %d %f %d", tag, &a, &b, &c, &d, &x, &k) == 7) {
      int t = tag[0] == 'm';
      N[t] = a; OH[t] = b; sr[t] = d; so[t] = x; zo[t] = k;
    }
  }
  uint8_t* maps[4]; long mlen[4];
  for (int k = 0; k < 4; k++) { char p[32]; snprintf(p, sizeof p, "l%d_u8.bin", k); maps[k] = load(p, &mlen[k]); }

  int bad = 0;
  for (int t = 0; t < 2; t++) {
    const char* tag = t ? "mask" : "box";
    int n = 0;
    for (int k = 0; k < 4; k++) n += nlev[t][k];
    float* rois = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n * 16 + 16);
    int32_t* rows = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n * 4 + 16);
    int off[5] = {0};
    for (int k = 0; k < 4; k++) {
      off[k + 1] = off[k] + nlev[t][k];
      if (!nlev[t][k]) continue;
      char p[64]; long ln;
      snprintf(p, sizeof p, "%s_rois%d.bin", tag, k); float* r = load(p, &ln);
      memcpy(rois + 4 * off[k], r, ln); rpcmem_free(r);
      snprintf(p, sizeof p, "%s_rows%d.bin", tag, k); int32_t* w = load(p, &ln);
      memcpy(rows + off[k], w, ln); rpcmem_free(w);
    }
    char p[64]; long rlen;
    snprintf(p, sizeof p, "%s_ref_u8.bin", tag);
    uint8_t* ref = load(p, &rlen);
    uint8_t* out = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, rlen + 128);
    char cbuf[256]; strncpy(cbuf, cfgs, sizeof cbuf - 1); cbuf[sizeof cbuf - 1] = 0;
    for (char* tok = strtok(cbuf, ","); tok; tok = strtok(NULL, ",")) {
      const int flags = atoi(tok);
      for (int per_level = 0; per_level < 2; per_level++) {
        double d[64], rt[64];
        for (int r = 0; r < reps; r++) {
          memset(out, 0xAA, rlen);
          double t0 = now_us(), dsp = 0;
          for (int k = per_level ? 0 : -1; k < (per_level ? 4 : 0); k++) {
            int cnt[4] = {nlev[t][0], nlev[t][1], nlev[t][2], nlev[t][3]}, o = 0;
            if (per_level) {
              if (!nlev[t][k]) continue;
              for (int q = 0; q < 4; q++) cnt[q] = q == k ? nlev[t][k] : 0;
              o = off[k];
            }
            uint64_t us = 0;
            int rc = roialign_u8_rpc_run(h, maps[0], mlen[0], maps[1], mlen[1], maps[2], mlen[2], maps[3], mlen[3],
                                         geom, 12, fp, 8, cnt, 4, rois + 4 * o, 4 * (per_level ? cnt[k] : n),
                                         rows + o, per_level ? cnt[k] : n, C, OH[t], OH[t], sr[t], so[t], zo[t],
                                         flags, out, rlen, &us);
            if (rc) { printf("%s rc=%d\n", tag, rc); return 1; }
            dsp += us;
          }
          rt[r] = now_us() - t0; d[r] = dsp;
        }
        long exact = 0; int maxd = 0;
        for (long i = 0; i < rlen; i++) {
          int df = abs((int)out[i] - (int)ref[i]);
          exact += !df; if (df > maxd) maxd = df;
        }
        if (maxd > 1) bad = 1;
        qsort(d, reps, sizeof d[0], cmp_d); qsort(rt, reps, sizeof rt[0], cmp_d);
        printf("%-4s N=%d rois=%d %-9s threads=%d prefetch=%d sort=%d dsp_ms(med)=%.2f rpc_ms(med)=%.2f rpc_ms(min)=%.2f exact=%.4f%% maxdiff=%d\n",
               tag, N[t], n, per_level ? "per-level" : "merged", flags & 0xff, (flags >> 8) & 1, (flags >> 9) & 1,
               d[reps / 2] / 1e3, rt[reps / 2] / 1e3, rt[0] / 1e3, 100.0 * exact / rlen, maxd);
        fflush(stdout);
      }
    }
    rpcmem_free(rois); rpcmem_free(rows); rpcmem_free(ref); rpcmem_free(out);
  }
  roialign_u8_rpc_close(h);
  puts(bad ? "FAIL" : "PASS");
  return bad;
}

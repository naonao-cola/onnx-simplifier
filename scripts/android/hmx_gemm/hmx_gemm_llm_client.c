/* hmx_gemm_llm_client <uri> <dir> [iters]: run every GEMM of <dir>/manifest.txt ("name M K N", files
 * name.a = M x K fp16, name.wp = prepacked fp16 weights) on the HMX skel in one session; writes name.c
 * and times.txt ("name dsp_us wall_us"). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hmx_gemm_rpc.h"
#include "remote.h"
static void* slurp(const char* path, size_t n) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  void* p = malloc(n);
  size_t r = fread(p, 1, n, f);
  fclose(f);
  return r == n ? p : NULL;
}
static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const char* dir = argv[2];
  int iters = argc > 3 ? atoi(argv[3]) : 1;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  if (hmx_gemm_rpc_open(argv[1], &h)) { printf("open failed\n"); return 1; }
  int prc = 0;
  hmx_gemm_rpc_perf_vote(h, 3, &prc);
  char path[512], name[128];
  snprintf(path, sizeof path, "%s/manifest.txt", dir);
  FILE* mf = fopen(path, "r");
  snprintf(path, sizeof path, "%s/out/times.txt", dir);
  FILE* tf = fopen(path, "w");
  if (!mf || !tf) { printf("manifest/out missing\n"); return 1; }
  int M, K, N, n = 0, fails = 0;
  double dsp = 0, wall = 0;
  while (fscanf(mf, "%127s %d %d %d", name, &M, &K, &N) == 4) {
    snprintf(path, sizeof path, "%s/%s.a", dir, name);
    uint16_t* A = slurp(path, (size_t)M * K * 2);
    snprintf(path, sizeof path, "%s/%s.wp", dir, name);
    uint16_t* W = slurp(path, (size_t)K * N * 2);
    uint16_t* C = malloc((size_t)M * N * 2);
    if (!A || !W) { printf("missing %s\n", name); return 1; }
    uint64 t[4];
    int codes[8];
    double t0 = now_us();
    int rc = hmx_gemm_rpc_gemm_f16(h, 0, M, K, N, iters, A, M * K, W, K * N, NULL, 0, C, M * N, t, 4, codes, 8);
    double w = (now_us() - t0) / iters;
    if (rc || codes[3]) { printf("%s rc %d gemm %d\n", name, rc, codes[3]); fails++; }
    snprintf(path, sizeof path, "%s/out/%s.c", dir, name);
    FILE* cf = fopen(path, "wb");
    fwrite(C, 2, (size_t)M * N, cf);
    fclose(cf);
    fprintf(tf, "%s %.1f %.1f\n", name, (double)t[0] / iters, w);
    dsp += (double)t[0] / iters, wall += w, n++;
    free(A), free(W), free(C);
  }
  fclose(tf);
  printf("%d GEMMs, %d failed: %.2f ms on the DSP, %.2f ms wall\n", n, fails, dsp / 1e3, wall / 1e3);
  hmx_gemm_rpc_close(h);
  return fails ? 3 : 0;
}

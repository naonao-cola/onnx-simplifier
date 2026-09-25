/* hmx_qconv_client <uri> <case dir> [iters]: QDQ 1x1 conv (hmx_qconv.h) on the DSP for a qnn_parity/export_case.py
 * case, both requantization modes, vs ORT CPU's output; prints mismatches and per-conv time / TMAC/s. */
#include <stdio.h>
#include "hmx_gemm_rpc.h"
#include "remote.h"
#include "qc_case.h"
int main(int argc, char** argv) {
  if (argc < 3) return 2;
  int iters = argc > 3 ? atoi(argv[3]) : 20;
  qc_case_t c;
  if (qc_load_case(argv[2], &c)) { printf("bad case %s\n", argv[2]); return 2; }
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  int rc = hmx_gemm_rpc_open(argv[1], &h), prc = 0;
  if (rc) { printf("open failed %d\n", rc); return 1; }
  hmx_gemm_rpc_perf_vote(h, getenv("HMX_FLAGS") ? atoi(getenv("HMX_FLAGS")) : 3, &prc);
  size_t plen = sizeof(qc_blk_t) * (c.N / 32) + sizeof(qc_hdr_t);
  uint8_t* prm = malloc(plen);
  memcpy(prm, c.blk, sizeof(qc_blk_t) * (c.N / 32));
  memcpy(prm + sizeof(qc_blk_t) * (c.N / 32), &c.hdr, sizeof(qc_hdr_t));
  uint8_t* y = malloc((size_t)c.M * c.N);
  int fails = 0;
  for (int mode = 0; mode < 2; mode++) {
    uint64 t[4];
    int codes[8];
    memset(y, 0, (size_t)c.M * c.N);
    rc = hmx_gemm_rpc_qconv(h, mode, c.M, c.K, c.N, iters, c.x, c.M * c.K, c.wp, c.K * c.N, prm, (int)plen, y, c.M * c.N, t, 4,
                            codes, 8);
    int hist[3], bad = qc_compare(&c, y, hist);
    double us = (double)t[0] / iters, macs = (double)c.M * c.K * c.N;
    printf("%s %dx%dx%d relu %d: rc %d ctx %d hmx %d vtcm %d; %d mismatches of %d vs ORT (%.3f%%; -1: %d, +1: %d, other: %d); "
           "scalar-fixed groups %llu; %.1f us/conv, %.2f TMAC/s (pack A %llu us)\n",
           mode ? "exact" : "fast ", c.M, c.K, c.N, c.relu, rc, codes[0] != 0, codes[2], codes[4], bad, c.M * c.N,
           100.0 * bad / (c.M * c.N), hist[0], hist[1], hist[2], (unsigned long long)t[2], us, macs / us / 1e6,
           (unsigned long long)t[1]);
    if (mode && bad) fails++;
  }
  hmx_gemm_rpc_close(h);
  return fails ? 3 : 0;
}

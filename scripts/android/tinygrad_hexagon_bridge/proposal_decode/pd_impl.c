/* DSP side of the proposal-decode FastRPC skel: pd_kernel.h over all 5 FPN levels in one call,
 * optionally split across QuRT threads by box. Model constants (anchors, per-level params and
 * their LUTs) are uploaded once and stay resident. Timed on the DSP with HAP_perf_get_time_us, so
 * RPC/cache overhead is reported separately by the client. Same skel pattern as
 * ../roialign_fast/roialign_impl.c. */
#include <stdlib.h>
#include <string.h>
#include "pd_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "pd_kernel.h"

extern unsigned long long HAP_perf_get_time_us(void);

#define NLVL 5
static float* g_anchors;
static int32_t g_meta[8 * NLVL];
static pd_params g_params[NLVL];
static pd_prep g_prep[NLVL];
static int g_ready;

AEEResult pd_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
  void* ctx = (void*)(uintptr_t)h;
  HAP_power_request_t d = {0};
  d.type = HAP_power_set_DCVS_v2;
  d.dcvs_v2.dcvs_enable = turbo ? FALSE : TRUE;
  d.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  d.dcvs_v2.set_latency = TRUE;
  d.dcvs_v2.latency = 40;
  d.dcvs_v2.set_dcvs_params = turbo ? TRUE : FALSE;
  d.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
  d.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_TURBO;
  d.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
  *rc = HAP_power_set(ctx, &d);
  return 0;
}

int pd_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int pd_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

AEEResult pd_rpc_set_model(remote_handle64 h, const float* anchors, int anchorsLen, const int32* meta,
                           int metaLen, const unsigned char* params, int paramsLen) {
  if (metaLen != 8 * NLVL || paramsLen != (int)sizeof g_params) return -1;
  free(g_anchors);
  g_anchors = malloc(sizeof(float) * anchorsLen);
  if (!g_anchors) return -1;
  memcpy(g_anchors, anchors, sizeof(float) * anchorsLen);
  memcpy(g_meta, meta, sizeof g_meta);
  memcpy(g_params, params, sizeof g_params);
  for (int l = 0; l < NLVL; l++) {
    pd_prepare(&g_params[l], &g_prep[l]);
    pd_prepare_anchors(g_anchors + g_meta[8 * l + 4], g_meta[8 * l + 2], g_meta[8 * l + 3], &g_prep[l]);
  }
  g_ready = 1;
  return 0;
}

#define MAX_THREADS 8
#define STACK_SIZE 16384
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8)));

typedef struct {
  int kind;   /* 0 decode, 1 bb_layout */
  int b0, b1; /* global box range (decode) or global row range (bb_layout) */
  const int32_t* idx; const float* deltas; const uint8_t* nchw; int src, fast; float* out;
} job_t;

static void run_decode(const job_t* j) {
  int base = 0; /* level l's boxes (and top-k indices) are global rows [base, base + k) */
  for (int l = 0; l < NLVL; l++) {
    const int32_t* m = g_meta + 8 * l;
    int k = m[1], H = m[2], W = m[3];
    int lo = j->b0 > base ? j->b0 : base, hi = j->b1 < base + k ? j->b1 : base + k;
    if (lo < hi)
      pd_decode(g_anchors + m[4], j->idx + lo, hi - lo, j->src ? 0 : j->deltas + m[5],
                j->src ? j->nchw + m[6] : 0, H, W, &g_params[l], j->fast ? &g_prep[l] : 0, j->out + 4 * lo);
    base += k;
  }
}

/* rows are counted across levels: level l owns rows [sum of earlier levels' H, + H) */
static void run_layout(const job_t* j) {
  int base = 0;
  for (int l = 0; l < NLVL; l++) {
    const int32_t* m = g_meta + 8 * l;
    int H = m[2], W = m[3], HW = H * W;
    int lo = j->b0 > base ? j->b0 : base, hi = j->b1 < base + H ? j->b1 : base + H;
    const uint8_t* src = j->nchw + m[6];
    float* dst = j->out + m[5];
    float s = g_params[l].bb_s, z = (float)g_params[l].bb_z;
    for (int y = lo - base; y < hi - base; y++)
      for (int x = 0; x < W; x++) {
        const uint8_t* sp = src + y * W + x;
        float* dp = dst + 12 * (y * W + x);
        for (int c = 0; c < 12; c++) dp[c] = ((float)sp[c * HW] - z) * s;
      }
    base += H;
  }
}

static void thread_main(void* arg) {
  job_t* j = arg;
  (j->kind ? run_layout : run_decode)(j);
  qurt_thread_exit(0);
}

static int run_jobs(job_t proto, int total, int nthreads) {
  if (nthreads < 1) nthreads = 1;
  if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
  if (nthreads == 1) {
    proto.b0 = 0; proto.b1 = total;
    (proto.kind ? run_layout : run_decode)(&proto);
    return 0;
  }
  job_t jobs[MAX_THREADS];
  qurt_thread_t tids[MAX_THREADS];
  int per = (total + nthreads - 1) / nthreads, started = 0;
  for (int t = 0; t < nthreads; t++) {
    jobs[t] = proto;
    jobs[t].b0 = t * per;
    jobs[t].b1 = (t + 1) * per < total ? (t + 1) * per : total;
    if (jobs[t].b0 >= jobs[t].b1) break;
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
    qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
    qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
    if (qurt_thread_create(&tids[t], &attr, thread_main, &jobs[t]) != QURT_EOK) return -2;
    started++;
  }
  for (int t = 0; t < started; t++) { int st; qurt_thread_join(tids[t], &st); }
  return 0;
}

AEEResult pd_rpc_anchor_grid(remote_handle64 h, int32* mask) {
  *mask = 0;
  for (int l = 0; l < NLVL; l++) *mask |= g_prep[l].grid << l;
  return 0;
}

AEEResult pd_rpc_run(remote_handle64 h, const int32* idx, int idxLen, const float* deltas, int deltasLen,
                     const unsigned char* nchw, int nchwLen, int32 src, int32 fast, int32 nthreads,
                     float* boxes, int boxesLen, uint64* dsp_us) {
  if (!g_ready) return -1;
  int total = 0;
  for (int l = 0; l < NLVL; l++) total += g_meta[8 * l + 1];
  if (idxLen < total || boxesLen < 4 * total) return -1;
  if (src ? nchwLen == 0 : deltasLen == 0) return -1;
  unsigned long long t0 = HAP_perf_get_time_us();
  job_t proto = {0, 0, 0, idx, deltas, nchw, src, fast, boxes};
  int rc = run_jobs(proto, total, nthreads);
  *dsp_us = HAP_perf_get_time_us() - t0;
  return rc;
}

AEEResult pd_rpc_bb_layout(remote_handle64 h, const unsigned char* nchw, int nchwLen, int32 nthreads,
                           float* deltas, int deltasLen, uint64* dsp_us) {
  if (!g_ready) return -1;
  int rows = 0;
  for (int l = 0; l < NLVL; l++) rows += g_meta[8 * l + 2];
  unsigned long long t0 = HAP_perf_get_time_us();
  job_t proto = {1, 0, 0, 0, 0, nchw, 0, 0, deltas};
  int rc = run_jobs(proto, rows, nthreads);
  *dsp_us = HAP_perf_get_time_us() - t0;
  return rc;
}

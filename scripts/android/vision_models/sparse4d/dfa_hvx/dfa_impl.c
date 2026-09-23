/* DSP side of dfa_rpc.idl: one Sparse4D decoder layer's DFA per call (dfa_core.h over the generic
 * msda kernel), anchor blocks handed to up to 8 QuRT threads from a shared counter (each holding an
 * HVX context). The threading follows ../../../msda_hvx/msda_impl.c. */
#include <stdlib.h>

#include "HAP_power.h"
#include "dfa_core.h"
#include "dfa_rpc.h"
#include "qurt.h"

extern unsigned long long HAP_perf_get_time_us(void);

AEEResult dfa_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
  void* ctx = (void*)(uintptr_t)h;
  HAP_power_request_t req = {0};
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = 1;
  int r1 = HAP_power_set(ctx, &req);
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
  int r2 = HAP_power_set(ctx, &d);
  *rc = r1 ? r1 : r2;
  return 0;
}

#define MAX_Q 1024
static float zeros[MAX_Q * DFA_M * DFA_P * 2] __attribute__((aligned(128)));
static float tmp[MAX_Q * DFA_C] __attribute__((aligned(128)));
static uint8_t vis[DFA_CAMS * MAX_Q] __attribute__((aligned(128)));

int dfa_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int dfa_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

#define MAX_THREADS 8
#define STACK_SIZE 65536 /* the msda HVX body keeps ~24 KB on the stack */
#define MAX_QB 256       /* anchors per job with fp16 weights (the per-thread conversion buffer) */
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(128)));
static float wbufs[MAX_THREADS][MAX_QB * DFA_M * DFA_P] __attribute__((aligned(128)));
#define MAX_QB_W2 64 /* anchors per job with v2 weights: the block's transposed weights, 384 KB */
static uint16_t stages[MAX_THREADS][MAX_QB_W2 * DFA_CAMS * DFA_LEVELS * DFA_M * DFA_P] __attribute__((aligned(128)));

typedef struct {
  const dfa_args_t* a;
  volatile int* next;
  int qb, njobs;
  float* wbuf;
  uint16_t* stage;
} job_t;

static void thread_main(void* arg) {
  job_t* j = (job_t*)arg;
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  for (;;) {
    int k = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
    if (k >= j->njobs) break;
    int q0 = k * j->qb, q1 = q0 + j->qb < j->a->Q ? q0 + j->qb : j->a->Q;
    dfa_run(j->a, q0, q1, j->wbuf, j->stage);
  }
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

AEEResult dfa_rpc_run(remote_handle64 h, const uint8* v0, int v0Len, const uint8* v1, int v1Len, const uint8* v2,
                      int v2Len, const uint8* v3, int v3Len, const float* vscale, int vscaleLen, const int32* vzp,
                      int vzpLen, const int32* hw, int hwLen, const float* pts, int ptsLen, const float* w, int wLen,
                      int32 Q, int32 flags, float* out, int outLen, uint64* dsp_us) {
  /* flags bit 16: w holds fp16 (wLen still counts 4-byte words, so half the weights); bit 17: w is
   * fp16 in the v2 layout (Q, 8, 384) (dfa_core.h) */
  const int w2 = (flags >> 17) & 1, w16 = w2 || ((flags >> 16) & 1);
  const long nw = (long)DFA_CAMS * DFA_LEVELS * Q * DFA_M * DFA_P;
  if (Q < 1 || Q > MAX_Q || vscaleLen < DFA_LEVELS || vzpLen < DFA_LEVELS || hwLen < 2 * DFA_LEVELS ||
      ptsLen < DFA_CAMS * Q * DFA_P * 2 || (long)wLen < (w16 ? nw / 2 : nw) || outLen < Q * DFA_C)
    return -1;
  dfa_args_t a;
  memset(&a, 0, sizeof a);
  a.Q = Q;
  const uint8* vs[DFA_LEVELS] = {v0, v1, v2, v3};
  const int vl[DFA_LEVELS] = {v0Len, v1Len, v2Len, v3Len};
  for (int l = 0; l < DFA_LEVELS; l++) {
    a.H[l] = hw[2 * l];
    a.W[l] = hw[2 * l + 1];
    if ((long)vl[l] < (long)DFA_CAMS * a.H[l] * a.W[l] * DFA_C) return -1;
    a.val[l] = vs[l];
    a.scale[l] = vscale[l];
    a.zp[l] = vzp[l];
  }
  a.pts = pts; a.zeros = zeros; a.vis = vis; a.tmp = tmp; a.out = out;
  if (w2) a.w2 = (const uint16_t*)w;
  else if (w16) a.w16 = (const uint16_t*)w;
  else a.w = w;
  if (dfa_check(&a)) return -1;
  unsigned long long t0 = HAP_perf_get_time_us();
  volatile int next = 0;
  int qb = ((flags >> 8) & 0xff) * 16;
  if (qb < 1) qb = 32;
  if (w16 && qb > MAX_QB) qb = MAX_QB;
  if (w2 && qb > MAX_QB_W2) qb = MAX_QB_W2;
  int nthreads = flags & 0xff, rc = 0;
  if (nthreads < 1) nthreads = 1;
  if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
  job_t jobs[MAX_THREADS];
  qurt_thread_t tids[MAX_THREADS];
  int started = 0;
  for (int t = 0; t < nthreads; t++) {
    jobs[t] = (job_t){&a, &next, qb, (Q + qb - 1) / qb, wbufs[t], stages[t]};
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
    qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
    qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
    if (qurt_thread_create(&tids[t], &attr, thread_main, &jobs[t]) != QURT_EOK) { rc = -2; break; }
    started++;
  }
  for (int t = 0; t < started; t++) { int st; qurt_thread_join(tids[t], &st); }
  if (!started) rc = -2;
  *dsp_us = HAP_perf_get_time_us() - t0;
  return rc;
}

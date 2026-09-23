/* DSP side of the NonMaxSuppression FastRPC skel: runs a batch of independent NMS calls (one
 * group of the model's 85: the 5 per-level RPN calls, or the 80 per-class box-head calls) in one
 * RPC, spread across QuRT threads by call (greedy balance on n^2). mode 0 = nms_scalar,
 * mode 1 = nms_hvx (each thread holds the HVX unit via qurt_hvx_lock). Output for call i goes at
 * offset sum(n[0..i)) of `sel`, its count in nsel[i]. DSP time via HAP_perf_get_time_us. */
#include <stdlib.h>
#include "nms_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "nms_kernel.h"

extern unsigned long long HAP_perf_get_time_us(void);

#define MAX_THREADS 8
#define MAX_CALLS 128
#define NMAX 2048
#define STACK_SIZE 16384

AEEResult nms_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
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

int nms_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int nms_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

typedef struct {
  const float *boxes, *scores, *thr; const int *n, *max_out, *boff; int* sel; int* nsel;
  int mode, ncalls, calls[MAX_CALLS];
  int order[NMAX], tmp[NMAX];
} job_t;
static job_t jobs[MAX_THREADS];
static float soa[MAX_THREADS][NMS_SOA * NMAX] __attribute__((aligned(128)));
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8)));

static void run_job(int t) {
  job_t* j = &jobs[t];
  for (int k = 0; k < j->ncalls; k++) {
    int c = j->calls[k], o = j->boff[c], n = j->n[c];
    j->nsel[c] = j->mode ? nms_hvx(j->boxes + 4 * o, j->scores + o, n, j->thr[c], j->max_out[c], j->sel + o, j->order, j->tmp, soa[t])
                         : nms_scalar(j->boxes + 4 * o, j->scores + o, n, j->thr[c], j->max_out[c], j->sel + o, j->order, j->tmp);
  }
}
static void thread_main(void* arg) {
  int t = (int)(uintptr_t)arg;
  int locked = jobs[t].mode && qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  run_job(t);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

AEEResult nms_rpc_run(remote_handle64 h, const float* boxes, int boxesLen, const float* scores,
                      int scoresLen, const int32* n, int nLen, const float* thr, int thrLen,
                      const int32* max_out, int max_outLen, int32 mode, int32 nthreads,
                      int32* sel, int selLen, int32* nsel, int nselLen, uint64* dsp_us) {
  int ncalls = nLen;
  if (ncalls > MAX_CALLS || thrLen < ncalls || max_outLen < ncalls || nselLen < ncalls) return -1;
  static int boff[MAX_CALLS];
  int tot = 0;
  for (int c = 0; c < ncalls; c++) { if (n[c] > NMAX) return -1; boff[c] = tot; tot += n[c]; }
  if (boxesLen < 4 * tot || scoresLen < tot || selLen < tot) return -1;
  if (nthreads < 1) nthreads = 1;
  if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
  unsigned long long t0 = HAP_perf_get_time_us();
  /* greedy load balance: biggest calls first, each to the least-loaded thread (cost ~ n^2) */
  int idx[MAX_CALLS]; long long load[MAX_THREADS] = {0};
  for (int c = 0; c < ncalls; c++) idx[c] = c;
  for (int a = 1; a < ncalls; a++) { int v = idx[a], b = a; while (b > 0 && n[idx[b - 1]] < n[v]) { idx[b] = idx[b - 1]; b--; } idx[b] = v; }
  for (int t = 0; t < nthreads; t++) {
    jobs[t] = (job_t){.boxes = boxes, .scores = scores, .thr = thr, .n = (const int*)n,
                      .max_out = (const int*)max_out, .boff = boff, .sel = (int*)sel, .nsel = (int*)nsel, .mode = mode, .ncalls = 0};
  }
  for (int k = 0; k < ncalls; k++) {
    int c = idx[k], best = 0;
    for (int t = 1; t < nthreads; t++) if (load[t] < load[best]) best = t;
    jobs[best].calls[jobs[best].ncalls++] = c;
    load[best] += (long long)n[c] * n[c] + 1;
  }
  if (nthreads == 1) {
    int locked = mode && qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
    run_job(0);
    if (locked) qurt_hvx_unlock();
  } else {
    qurt_thread_t tids[MAX_THREADS]; int started = 0;
    for (int t = 0; t < nthreads; t++) {
      if (!jobs[t].ncalls) continue;
      qurt_thread_attr_t attr;
      qurt_thread_attr_init(&attr);
      qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
      qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
      qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
      if (qurt_thread_create(&tids[started], &attr, thread_main, (void*)(uintptr_t)t) != QURT_EOK) return -2;
      started++;
    }
    for (int t = 0; t < started; t++) { int st; qurt_thread_join(tids[t], &st); }
  }
  *dsp_us = HAP_perf_get_time_us() - t0;
  return 0;
}

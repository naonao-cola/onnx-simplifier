/* DSP side of msda_rpc.idl: validates the shapes, then runs msda_kernel.h over the queries on up to
 * 8 QuRT threads (each holding an HVX context), handing out 32-query chunks from a shared counter
 * so the SCA's uneven visibility balances. Timed with HAP_perf_get_time_us (client reports the RPC
 * overhead separately). Same power vote as the RoiAlign skel. */
#include <stdlib.h>
#include "msda_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "msda_kernel.h"

extern unsigned long long HAP_perf_get_time_us(void);

AEEResult msda_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
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

int msda_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int msda_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

/* Jobs = blocks of `qb` queries (HVX path: all heads of a query at once, msda_run) or, for shapes
 * only the scalar path handles, (head, query block) pairs handed out head-major so the threads
 * share one head's slice of the value maps in L2. A shared counter hands them out, so the SCA's
 * uneven visibility balances. */
typedef struct {
  const msda_args_t* a;
  volatile int* next;
  int qb, nqb, hvx;
} job_t;

static void work(job_t* j) {
  const int njobs = j->hvx ? j->nqb : MSDA_M * j->nqb;
  for (;;) {
    int k = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
    if (k >= njobs) break;
    int qi = j->hvx ? k : k % j->nqb, q0 = qi * j->qb, q1 = q0 + j->qb < j->a->Q ? q0 + j->qb : j->a->Q;
    if (j->hvx)
      msda_run(j->a, q0, q1);
    else
      msda_run_heads(j->a, q0, q1, k / j->nqb, k / j->nqb + 1);
  }
}

static void thread_main(void* arg) {
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  work((job_t*)arg);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

#define MAX_THREADS 8
#define STACK_SIZE 16384
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(128)));

AEEResult msda_rpc_run(remote_handle64 h, const float* value, int valueLen, const float* ref, int refLen,
                       const float* off, int offLen, const float* attw, int attwLen, const uint8* vis, int visLen,
                       int32 NV, int32 H, int32 W, int32 Q, int32 R, int32 NO, int32 P, int32 flags, float* out,
                       int outLen, uint64* dsp_us) {
  if (NV < 1 || NV > MSDA_MAX_NV || P > MSDA_MAX_P || H < 1 || W < 1 || Q < 1 || R < 1 || P < 1 || (NO != 1 && NO != NV) ||
      (long)valueLen < (long)NV * H * W * MSDA_C || (long)refLen < (long)NV * Q * R * 2 ||
      (long)offLen < (long)Q * MSDA_M * NO * P * 2 || (long)attwLen < (long)Q * MSDA_M * NO * P ||
      (long)visLen < (long)NV * Q || (long)outLen < (long)Q * MSDA_C || ((uintptr_t)value & 127) || ((uintptr_t)out & 127))
    return -1;
  unsigned long long t0 = HAP_perf_get_time_us();
  msda_args_t a = {value, ref, off, attw, vis, out, NV, H, W, Q, R, NO, P};
  volatile int next = 0;
  int qb = ((flags >> 8) & 0xff) * 16;
  if (qb < 1) qb = 64;
  job_t j = {&a, &next, qb, (Q + qb - 1) / qb, msda_hvx_ok(&a)};
  int nthreads = flags & 0xff, rc = 0;
  if (nthreads < 1) nthreads = 1;
  if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
  if (nthreads == 1) {
    work(&j);
  } else {
    qurt_thread_t tids[MAX_THREADS];
    int started = 0;
    for (int t = 0; t < nthreads; t++) {
      qurt_thread_attr_t attr;
      qurt_thread_attr_init(&attr);
      qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
      qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
      qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
      if (qurt_thread_create(&tids[t], &attr, thread_main, &j) != QURT_EOK) { rc = -2; break; }
      started++;
    }
    for (int t = 0; t < started; t++) { int st; qurt_thread_join(tids[t], &st); }
    if (!started) work(&j);
  }
  *dsp_us = HAP_perf_get_time_us() - t0;
  return rc;
}

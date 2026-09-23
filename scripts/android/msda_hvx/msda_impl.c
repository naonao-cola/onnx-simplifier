/* DSP side of msda_rpc.idl: unpacks and checks the shape, then runs msda_kernel.h over blocks of
 * queries on up to 8 QuRT threads (each holding an HVX context), handed out from a shared counter
 * so uneven work (BEVFormer's SCA visibility) balances. 4 threads is the sweet spot on V69.
 * Timed with HAP_perf_get_time_us so clients can report the FastRPC overhead separately. */
#include <stdlib.h>
#include "msda_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "msda_shape.h"

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

typedef struct {
  const msda_args_t* a;
  volatile int* next;
  int qb, njobs;
} job_t;

static void work(job_t* j) {
  for (;;) {
    int k = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
    if (k >= j->njobs) break;
    int q0 = k * j->qb, q1 = q0 + j->qb < j->a->Q ? q0 + j->qb : j->a->Q;
    msda_run(j->a, q0, q1);
  }
}

static void thread_main(void* arg) {
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  work((job_t*)arg);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

#define MAX_THREADS 8
#define STACK_SIZE 65536 /* the HVX body keeps ~24 KB of tap lists and accumulators on the stack */
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(128)));

AEEResult msda_rpc_run(remote_handle64 h, const float* value, int valueLen, const uint8* value_u8, int value_u8Len,
                       const float* vscale, int vscaleLen, const int32* vzp, int vzpLen, const float* loc, int locLen,
                       const float* ref, int refLen, const float* attw, int attwLen, const uint8* vis, int visLen,
                       const int32* shape, int shapeLen, int32 flags, float* out, int outLen, uint64* dsp_us) {
  msda_args_t a;
  memset(&a, 0, sizeof a);
  const int has_vis = msda_shape_unpack(shape, shapeLen, &a);
  a.value = value; a.value_u8 = value_u8; a.vscale = vscale; a.vzp = (const int32_t*)vzp;
  a.loc = loc; a.ref = ref; a.attw = attw; a.vis = has_vis > 0 ? vis : 0; a.out = out;
  const int u8 = a.vdtype == MSDA_U8;
  if (has_vis < 0 || msda_check(&a) || (u8 ? value_u8Len : valueLen) < msda_n_value(&a) ||
      (u8 && (vscaleLen < a.NV || vzpLen < a.NV)) || locLen < msda_n_loc(&a) || refLen < msda_n_ref(&a) ||
      attwLen < msda_n_attw(&a) || (has_vis && visLen < msda_n_vis(&a)) || outLen < msda_n_out(&a))
    return -1;
  unsigned long long t0 = HAP_perf_get_time_us();
  volatile int next = 0;
  int qb = ((flags >> 8) & 0xff) * 16;
  if (qb < 1) qb = 32;
  job_t j = {&a, &next, qb, (a.Q + qb - 1) / qb};
  int nthreads = flags & 0xff, rc = 0;
  if (nthreads < 1) nthreads = 1;
  if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
  /* always on our own threads, even for 1: the HVX body needs ~24 KB of stack, more than the
   * FastRPC thread has (running it there fails the call with AEE_EBADPERMS) */
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
  if (!started) rc = -2;
  *dsp_us = HAP_perf_get_time_us() - t0;
  return rc;
}

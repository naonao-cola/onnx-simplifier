/* DSP side of attn_rpc.idl: packs K / V for all heads, then runs attn_kernel.h's HVX body over
 * (head, row block) jobs on up to 8 QuRT threads, each holding an HVX context. Packed buffers and
 * per-thread scratch are kept across calls (resized when LK grows). */
#include <stdlib.h>
#include "attn_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "attn_kernel.h"

extern unsigned long long HAP_perf_get_time_us(void);

AEEResult attn_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
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

int attn_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int attn_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

#define MAX_THREADS 8
#define STACK_SIZE 16384
#define PACK_KEYS 512 /* keys per pack job */
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(128)));
static int cap_lk;
static uint32_t* g_kp;
static uint8_t* g_vp;
static int32_t* g_kb;
static int32_t* g_srow[MAX_THREADS];
static uint8_t* g_prow[MAX_THREADS];

typedef struct {
  const attn_args_t* a;
  volatile int* next;
  int phase, rows, njobs, tid;
} job_t;

static void work(job_t* j) {
  const attn_args_t* a = j->a;
  for (;;) {
    int k = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
    if (k >= j->njobs) break;
    if (j->phase == 0) {
      const int per = a->LK / PACK_KEYS + (a->LK % PACK_KEYS != 0);
      const int h = k / per, k0 = (k % per) * PACK_KEYS, k1 = k0 + PACK_KEYS < a->LK ? k0 + PACK_KEYS : a->LK;
      attn_pack(a, h, k0, k1);
    } else {
      const int per = (a->LQ + j->rows - 1) / j->rows;
      const int h = k / per, r0 = (k % per) * j->rows, r1 = r0 + j->rows < a->LQ ? r0 + j->rows : a->LQ;
      attn_rows_hvx(a, h, r0, r1, g_srow[j->tid], g_prow[j->tid]);
    }
  }
}

static void thread_main(void* arg) {
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  work((job_t*)arg);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

static int run_phase(job_t* proto, int nthreads) {
  volatile int next = 0;
  job_t jobs[MAX_THREADS];
  qurt_thread_t tids[MAX_THREADS];
  int started = 0;
  for (int t = 0; t < nthreads; t++) {
    jobs[t] = *proto;
    jobs[t].next = &next;
    jobs[t].tid = t;
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
    qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
    qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
    if (qurt_thread_create(&tids[t], &attr, thread_main, &jobs[t]) != QURT_EOK) break;
    started++;
  }
  for (int t = 0; t < started; t++) { int st; qurt_thread_join(tids[t], &st); }
  return started ? 0 : -2;
}

AEEResult attn_rpc_run(remote_handle64 h, const uint8* q, int qLen, const uint8* k, int kLen, const uint8* v, int vLen,
                       const int32* params, int paramsLen, int32 flags, uint8* out, int outLen, uint64* dsp_us,
                       uint64* pack_us) {
  if (paramsLen < 5) return -1;
  attn_args_t a = {params[0], params[1], params[2], params[3], params[4], q, k, v, 0, 0, 0, out};
  if (a.LQ < 1 || a.LK < 128 || a.LK % 128 || qLen < a.LQ * ATTN_C || kLen < a.LK * ATTN_C || vLen < a.LK * ATTN_C ||
      outLen < a.LQ * ATTN_C || a.m16 < 1 || a.m16 > 32767 || a.dcl < 1)
    return -1;
  if (a.LK > cap_lk) {
    free(g_kp); free(g_vp); free(g_kb);
    g_kp = memalign(128, (size_t)ATTN_H * a.LK * 32);
    g_vp = memalign(128, (size_t)ATTN_H * a.LK * ATTN_D);
    g_kb = memalign(128, (size_t)ATTN_H * a.LK * 4);
    for (int t = 0; t < MAX_THREADS; t++) {
      free(g_srow[t]); free(g_prow[t]);
      g_srow[t] = memalign(128, (size_t)ATTN_RB * a.LK * 4);
      g_prow[t] = memalign(128, (size_t)ATTN_RB * a.LK);
      if (!g_srow[t] || !g_prow[t]) { cap_lk = 0; return -3; }
    }
    if (!g_kp || !g_vp || !g_kb) { cap_lk = 0; return -3; }
    cap_lk = a.LK;
  }
  a.kp = g_kp; a.vp = g_vp; a.kb = g_kb;
  int nthreads = flags & 0xff;
  if (nthreads < 1) nthreads = 1;
  if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
  int rows = ((flags >> 8) & 0xff) * 4;
  if (rows < 1) rows = 16;
  unsigned long long t0 = HAP_perf_get_time_us();
  job_t p = {&a, 0, 0, rows, ATTN_H * (a.LK / PACK_KEYS + (a.LK % PACK_KEYS != 0)), 0};
  int rc = run_phase(&p, nthreads);
  unsigned long long t1 = HAP_perf_get_time_us();
  p.phase = 1;
  p.njobs = ATTN_H * ((a.LQ + rows - 1) / rows);
  if (!rc) rc = run_phase(&p, nthreads);
  *pack_us = t1 - t0;
  *dsp_us = HAP_perf_get_time_us() - t0;
  return rc;
}

/* DSP side of fbgather_rpc.idl: fbgather_kernel.h's HVX body split over QuRT threads (each holding
 * an HVX context), timed on the DSP (HAP_perf_get_time_us) so the client can report RPC overhead
 * separately. Power voting as in roialign_u8_impl.c. */
#include <stdlib.h>
#include "fbgather_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "fbgather_kernel.h"

extern unsigned long long HAP_perf_get_time_us(void);
#define NR (200 * 200 * 4)
#define MAX_THREADS 8
#define STACK_SIZE 16384
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(128)));

AEEResult fbgather_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
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

int fbgather_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int fbgather_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

typedef struct {
  const uint8_t* tab[FBG_T];
  const int32_t* lut[FBG_T];
  int32_t rows, r0, r1;
  uint8_t* out;
} slice_t;

static void thread_main(void* arg) {
  slice_t* s = arg;
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  fbg_run(s->tab, s->lut, s->rows, s->r0, s->r1, s->out);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

AEEResult fbgather_rpc_run(remote_handle64 h, const uint8* t0, int t0Len, const uint8* t1, int t1Len,
                           const uint8* t2, int t2Len, const uint8* t3, int t3Len, const int32* l0, int l0Len,
                           const int32* l1, int l1Len, const int32* l2, int l2Len, const int32* l3, int l3Len,
                           int32 rows, int32 flags, uint8* vol, int volLen, uint64* dsp_us) {
  const long tbytes = (rows + 1L) * FBG_C + 64;
  if (rows < 0 || t0Len < tbytes || t1Len < tbytes || t2Len < tbytes || t3Len < tbytes || l0Len < NR ||
      l1Len < NR || l2Len < NR || l3Len < NR || volLen < NR * FBG_T * FBG_C || ((uintptr_t)vol & 127))
    return -1;
  unsigned long long ts = HAP_perf_get_time_us();
  int n = flags & 0xff;
  if (n < 1) n = 1;
  if (n > MAX_THREADS) n = MAX_THREADS;
  slice_t sl[MAX_THREADS];
  qurt_thread_t tid[MAX_THREADS];
  int started = 0, rc = 0, per = (NR + n - 1) / n;
  for (int i = 0; i < n; i++) {
    sl[i] = (slice_t){{t0, t1, t2, t3}, {l0, l1, l2, l3}, rows, i * per, (i + 1) * per > NR ? NR : (i + 1) * per, vol};
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, stacks[i]);
    qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
    qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
    if (qurt_thread_create(&tid[i], &attr, thread_main, &sl[i]) != QURT_EOK) { rc = -2; break; }
    started++;
  }
  for (int i = 0; i < started; i++) { int st; qurt_thread_join(tid[i], &st); }
  *dsp_us = HAP_perf_get_time_us() - ts;
  return rc;
}

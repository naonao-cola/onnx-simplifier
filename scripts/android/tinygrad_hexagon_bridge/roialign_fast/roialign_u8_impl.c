/* DSP side of the merged uint8 RoiAlign FastRPC skel (roialign_u8_rpc.idl): builds the job list from
 * the per-level RoIs + destination rows, optionally sorts it for locality, and runs
 * roialign_u8_kernel.h split across QuRT threads (each holding an HVX context). Timed on the DSP with
 * HAP_perf_get_time_us so the client can report RPC overhead separately. */
#include <stdlib.h>
#include "roialign_u8_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "HAP_compute_res.h"
#include "roialign_u8_kernel.h"

extern unsigned long long HAP_perf_get_time_us(void);

AEEResult roialign_u8_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
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

AEEResult roialign_u8_rpc_vtcm_probe(remote_handle64 h, int32 bytes, int32* query_rc, int32* total, int32* avail,
                                     int32* acquired) {
  unsigned int tot = 0, av = 0;
  compute_res_vtcm_page_t tl, al;
  *query_rc = HAP_compute_res_query_VTCM(0, &tot, &tl, &av, &al);
  *total = (int32)tot; *avail = (int32)av; *acquired = 0;
  compute_res_attr_t attr;
  if (HAP_compute_res_attr_init(&attr) || HAP_compute_res_attr_set_vtcm_param(&attr, (unsigned)bytes, 0)) return 0;
  unsigned int ctx = HAP_compute_res_acquire(&attr, 10000);
  if (ctx) {
    *acquired = HAP_compute_res_attr_get_vtcm_ptr(&attr) != 0;
    HAP_compute_res_release(ctx);
  }
  return 0;
}

int roialign_u8_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int roialign_u8_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

typedef struct {
  const ru8_level_t* lv;
  const ru8_job_t* jobs;
  int njobs, C, OH, OW, sr, z_out, prefetch;
  uint8_t* out;
} slice_t;

static void run_slice(slice_t* s) {
  ru8_run_jobs(s->lv, s->C, s->jobs, s->njobs, s->OH, s->OW, s->sr, s->z_out, s->prefetch, s->out);
}

static void thread_main(void* arg) {
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  run_slice((slice_t*)arg);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

#define MAX_THREADS 8
#define STACK_SIZE 32768
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(128)));

AEEResult roialign_u8_rpc_run(remote_handle64 h, const uint8* map0, int map0Len, const uint8* map1, int map1Len,
                              const uint8* map2, int map2Len, const uint8* map3, int map3Len,
                              const int32* geom, int geomLen, const float* fp, int fpLen,
                              const int32* counts, int countsLen, const float* rois, int roisLen,
                              const int32* rows, int rowsLen, int32 C, int32 OH, int32 OW, int32 sr,
                              float s_out, int32 z_out, int32 flags, uint8* out, int outLen, uint64* dsp_us) {
  const uint8* maps[4] = {map0, map1, map2, map3};
  const int mapLens[4] = {map0Len, map1Len, map2Len, map3Len};
  if (geomLen < 12 || fpLen < 8 || countsLen < 4 || C % 128 || C > 128 * RU8_MAX_HALVES || sr < 1 || sr > RU8_MAX_SR ||
      OH * sr > RU8_MAX_AXIS || OW * sr > RU8_MAX_AXIS)
    return -1;
  unsigned long long t0 = HAP_perf_get_time_us();
  ru8_level_t lv[4];
  for (int k = 0; k < 4; k++) {
    lv[k] = (ru8_level_t){maps[k], geom[3 * k], geom[3 * k + 1], fp[2 * k + 1], geom[3 * k + 2], 0, 0};
    if (counts[k] && mapLens[k] < lv[k].H * lv[k].W * C) return -1;
    ru8_requant_params(fp[2 * k], s_out, sr * sr, &lv[k].mult, &lv[k].shift);
  }
  const long row_bytes = (long)OH * OW * C;
  int n = 0;
  for (int k = 0; k < 4; k++) n += counts[k];
  if (n * 4 > roisLen || n > rowsLen) return -1;
  ru8_job_t* jobs = malloc(sizeof(ru8_job_t) * (n + 1));
  if (!jobs) return -3;
  for (int k = 0, j = 0; k < 4; k++)
    for (int i = 0; i < counts[k]; i++, j++) {
      if (rows[j] < 0 || (rows[j] + 1) * row_bytes > outLen) { free(jobs); return -1; }
      jobs[j].level = k; jobs[j].row = rows[j];
      for (int c = 0; c < 4; c++) jobs[j].box[c] = rois[4 * j + c];
    }
  if (flags & 0x200) ru8_sort_jobs(jobs, n);
  const int prefetch = (flags & 0x400) ? 2 : (flags >> 8) & 1; /* bit 10: + next-RoI footprint prefetch */
  int nthreads = flags & 0xff;
  if (nthreads < 1) nthreads = 1;
  if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
  if (nthreads > n) nthreads = n > 0 ? n : 1;
  int rc = 0;
  if (nthreads == 1) {
    slice_t s = {lv, jobs, n, C, OH, OW, sr, z_out, prefetch, out};
    run_slice(&s);
  } else {
    slice_t sl[MAX_THREADS];
    qurt_thread_t tids[MAX_THREADS];
    int per = (n + nthreads - 1) / nthreads, started = 0;
    for (int t = 0; t < nthreads; t++) {
      int j0 = t * per, j1 = j0 + per > n ? n : j0 + per;
      if (j0 >= j1) break;
      sl[t] = (slice_t){lv, jobs + j0, j1 - j0, C, OH, OW, sr, z_out, prefetch, out};
      qurt_thread_attr_t attr;
      qurt_thread_attr_init(&attr);
      qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
      qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
      qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
      if (qurt_thread_create(&tids[t], &attr, thread_main, &sl[t]) != QURT_EOK) { rc = -2; break; }
      started++;
    }
    for (int t = 0; t < started; t++) { int st; qurt_thread_join(tids[t], &st); }
  }
  free(jobs);
  *dsp_us = HAP_perf_get_time_us() - t0;
  return rc;
}

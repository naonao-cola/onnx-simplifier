/* FastRPC skel for hmx_gemm.h on V69: acquires VTCM + HMX, runs the GEMM on a worker thread that holds
 * the HVX and HMX locks, and times its phases. */
#include <stdlib.h>
#include <string.h>
#include "hmx_gemm_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "HAP_compute_res.h"
#define HMX_NOW() qurt_get_core_pcycles()
#include "hmx_gemm.h"

extern unsigned long long HAP_perf_get_time_us(void);

int hmx_gemm_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return 0;
}
int hmx_gemm_rpc_close(remote_handle64 h) {
  free((void*)(uintptr_t)h);
  return 0;
}

int hmx_gemm_rpc_perf_vote(remote_handle64 h, int flags, int* rc) {
  void* ctx = (void*)hmx_gemm_rpc_perf_vote;
  HAP_power_request_t req = {0};
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = 1;
  int r1 = HAP_power_set(ctx, &req);
  HAP_power_request_t d = {0};
  d.type = HAP_power_set_DCVS_v2;
  d.dcvs_v2.dcvs_enable = 0;
  d.dcvs_v2.set_dcvs_params = 1;
  d.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  if (flags & 1) {
    d.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
  }
  int r2 = HAP_power_set(ctx, &d);
  int r3 = 0;
  if (flags & 2) {
    HAP_power_request_t x = {0};
    x.type = HAP_power_set_HMX;
    x.hmx.power_up = 1;
    r3 = HAP_power_set(ctx, &x);
  }
  *rc = r3 * 1000000 + r1 * 1000 + r2;
  return 0;
}

typedef struct {
  unsigned int ctx;
  uint8_t* vtcm;
  size_t vbytes;
  int mode, M, K, N, iters;
  const uint16_t *a, *wp, *bias;
  uint16_t* c;
  uint64* t;
  int* codes;
} job_t;

#define STACK_SIZE (64 * 1024)
static char g_stack[STACK_SIZE] __attribute__((aligned(128)));

/* mode 1: every weight tile resident in VTCM, MAC + tile store only */
static void mac_only(job_t* j) {
  int mt = (j->M + 31) / 32, kt = j->K / 32, nt = j->N / 32;
  size_t span = (size_t)kt * HMX_TILE_BYTES, off = 0, ao[128], wo[256];
  for (int mb = 0; mb < mt; mb++) ao[mb] = hmx_valloc(&off, span);
  for (int nb = 0; nb < nt; nb++) wo[nb] = hmx_valloc(&off, span);
  uint8_t* ct = j->vtcm + hmx_valloc(&off, (size_t)mt * nt * HMX_TILE_BYTES);
  uint32_t* tbl = (uint32_t*)(j->vtcm + hmx_valloc(&off, 256));
  for (int mb = 0; mb < mt; mb++) hmx_pack_a_f16(j->a, j->M, j->K, mb * 32, (uint16_t*)(j->vtcm + ao[mb]));
  for (int nb = 0; nb < nt; nb++) memcpy(j->vtcm + wo[nb], j->wp + (size_t)nb * kt * 1024, span);
  memset(tbl, 0, 256);
  hmx_set_table(tbl);
  unsigned long long t0 = HAP_perf_get_time_us();
  for (int it = 0; it < j->iters; it++)
    for (int nb = 0; nb < nt; nb++)
      for (int mb = 0; mb < mt; mb++) {
        hmx_mac_f16(j->vtcm + ao[mb], j->vtcm + wo[nb], kt);
        hmx_store_f16(ct + ((size_t)mb * nt + nb) * HMX_TILE_BYTES);
      }
  j->t[2] = HAP_perf_get_time_us() - t0;
  for (int mb = 0; mb < mt; mb++)
    for (int nb = 0; nb < nt; nb++) {
      uint16_t* t = (uint16_t*)(ct + ((size_t)mb * nt + nb) * HMX_TILE_BYTES);
      for (int r = 0; r < 32 && mb * 32 + r < j->M; r++)
        for (int c = 0; c < 32; c++) j->c[(size_t)(mb * 32 + r) * j->N + nb * 32 + c] = t[HMX_IDX(r, c)];
    }
}

static void worker(void* p) {
  job_t* j = (job_t*)p;
  j->codes[1] = qurt_hvx_lock(QURT_HVX_MODE_128B);
  j->codes[2] = HAP_compute_res_hmx_lock(j->ctx);
  unsigned long long prof[4] = {0, 0, 0, 0};
  if (j->codes[2] == 0) {
    unsigned long long t0 = HAP_perf_get_time_us();
    if (j->mode == 1)
      mac_only(j);
    else
      for (int it = 0; it < j->iters; it++)
        j->codes[3] = hmx_gemm_f16_prof(j->a, j->wp, j->bias, j->c, j->M, j->K, j->N, j->vtcm, j->vbytes, prof);
    j->t[0] = HAP_perf_get_time_us() - t0;
    if (j->mode == 0)
    if (j->mode == 0) j->t[1] = prof[0], j->t[2] = prof[1], j->t[3] = prof[2] + (prof[3] << 32);
    HAP_compute_res_hmx_unlock(j->ctx);
  }
  if (j->codes[1] == 0) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

/* mode 2/3: can two threads drive HMX at once? Each thread acquires its own VTCM + HMX context, locks it,
 * and runs the resident MAC loop on its own buffers (mode 3: the same with one thread, for reference). */
typedef struct {
  unsigned int ctx;
  uint8_t* v;
  int M, K, N, iters, lock_rc;
  unsigned long long us;
} dual_t;
static char g_stack2[2][STACK_SIZE] __attribute__((aligned(128)));
static void dual_worker(void* p) {
  dual_t* d = (dual_t*)p;
  qurt_hvx_lock(QURT_HVX_MODE_128B);
  d->lock_rc = HAP_compute_res_hmx_lock(d->ctx);
  if (d->lock_rc == 0) {
    int mt = (d->M + 31) / 32, kt = d->K / 32, nt = d->N / 32;
    size_t span = (size_t)kt * HMX_TILE_BYTES, off = 0, ao[16], wo[64];
    for (int mb = 0; mb < mt; mb++) ao[mb] = hmx_valloc(&off, span);
    for (int nb = 0; nb < nt; nb++) wo[nb] = hmx_valloc(&off, span);
    uint8_t* ct = d->v + hmx_valloc(&off, HMX_TILE_BYTES);
    uint32_t* tbl = (uint32_t*)(d->v + hmx_valloc(&off, 256));
    memset(d->v, 0, off);
    hmx_set_table(tbl);
    unsigned long long t0 = HAP_perf_get_time_us();
    for (int it = 0; it < d->iters; it++)
      for (int nb = 0; nb < nt; nb++)
        for (int mb = 0; mb < mt; mb++) {
          hmx_mac_f16(d->v + ao[mb], d->v + wo[nb], kt);
          hmx_store_f16(ct);
        }
    d->us = HAP_perf_get_time_us() - t0;
    HAP_compute_res_hmx_unlock(d->ctx);
  }
  qurt_hvx_unlock();
  qurt_thread_exit(0);
}
static int dual(int M, int K, int N, int iters, int nthreads, uint64* t, int* codes) {
  if ((M + 31) / 32 > 16 || N / 32 > 64) return AEE_EBADPARM;
  dual_t d[2];
  compute_res_attr_t attr[2];
  memset(d, 0, sizeof d);
  size_t need = (((size_t)(M + 31) / 32 + N / 32) * (K / 32) * HMX_TILE_BYTES * 5 / 4 + HMX_VTCM_WINDOW + 0xFFFF) & ~(size_t)0xFFFF;
  for (int i = 0; i < nthreads; i++) {
    HAP_compute_res_attr_init(&attr[i]);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr[i], need, 0, 0);
    HAP_compute_res_attr_set_hmx_param(&attr[i], 1);
    d[i].ctx = HAP_compute_res_acquire(&attr[i], 100000);
    codes[i] = (int)d[i].ctx;
    if (!d[i].ctx) break;
    void* vp = NULL;
    unsigned int vs = 0;
    HAP_compute_res_attr_get_vtcm_ptr_v2(&attr[i], &vp, &vs);
    d[i].v = (uint8_t*)vp, d[i].M = M, d[i].K = K, d[i].N = N, d[i].iters = iters;
  }
  if (d[0].ctx && (nthreads < 2 || d[1].ctx)) {
    qurt_thread_t tid[2];
    unsigned long long t0 = HAP_perf_get_time_us();
    for (int i = 0; i < nthreads; i++) {
      qurt_thread_attr_t ta;
      qurt_thread_attr_init(&ta);
      qurt_thread_attr_set_stack_addr(&ta, g_stack2[i]);
      qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
      qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
      qurt_thread_create(&tid[i], &ta, dual_worker, &d[i]);
    }
    int st;
    for (int i = 0; i < nthreads; i++) qurt_thread_join(tid[i], &st);
    t[0] = HAP_perf_get_time_us() - t0;
    t[1] = d[0].us, t[2] = d[1].us;
    codes[4] = d[0].lock_rc, codes[5] = d[1].lock_rc;
  }
  for (int i = 0; i < nthreads; i++)
    if (d[i].ctx) HAP_compute_res_release(d[i].ctx);
  return 0;
}

int hmx_gemm_rpc_gemm_f16(remote_handle64 h, int mode, int M, int K, int N, int iters,
                                const uint16* a, int aLen, const uint16* wp, int wpLen, const uint16* bias,
                                int biasLen, uint16* c, int cLen, uint64* t, int tLen, int* codes, int codesLen) {
  if (tLen < 4 || codesLen < 8 || K % 32 || N % 32 || aLen < M * K || wpLen < K * N || cLen < M * N) return AEE_EBADPARM;
  memset(t, 0, tLen * sizeof(uint64));
  memset(codes, 0, codesLen * sizeof(int));
  if (mode == 2 || mode == 3) return dual(M, K, N, iters, mode == 2 ? 2 : 1, t, codes);
  int mt = (M + 31) / 32, kt = K / 32, nt = N / 32;
  size_t need = mode == 1 ? ((size_t)mt * kt + (size_t)nt * kt + (size_t)mt * nt) * HMX_TILE_BYTES * 5 / 4 + HMX_VTCM_WINDOW
                          : hmx_gemm_f16_vtcm(M, K);
  if (mt > 128 || nt > 256) return AEE_EBADPARM;
  need = (need + 0xFFFF) & ~(size_t)0xFFFF;
  compute_res_attr_t attr;
  HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param_v2(&attr, need, 0, 0);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);
  unsigned int ctx = HAP_compute_res_acquire(&attr, 100000);
  codes[0] = (int)ctx;
  if (!ctx) return 0;
  void* vp = NULL;
  unsigned int vs = 0;
  HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vp, &vs);
  codes[4] = (int)vs;
  if (vp && vs >= need) {
    job_t j = {ctx, (uint8_t*)vp, vs, mode, M, K, N, iters, a, wp, biasLen ? bias : NULL, c, t, codes};
    qurt_thread_attr_t ta;
    qurt_thread_attr_init(&ta);
    qurt_thread_attr_set_stack_addr(&ta, g_stack);
    qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
    qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_t tid;
    int st;
    codes[5] = qurt_thread_create(&tid, &ta, worker, &j);
    if (codes[5] == 0) qurt_thread_join(tid, &st);
  } else
    codes[6] = -1;
  HAP_compute_res_release(ctx);
  return 0;
}

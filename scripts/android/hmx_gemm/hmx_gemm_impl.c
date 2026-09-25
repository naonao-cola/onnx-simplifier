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
#include "hmx_gemm_u8.h"

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
  int r4 = 0;
  if (flags & 4) { /* HMX clock at the turbo corner (HAP_power_set_HMX_v2; EBADPARM without a separate HMX clock) */
    HAP_power_request_t x = {0};
    x.type = HAP_power_set_HMX_v2;
    x.hmx_v2.set_power = 1;
    x.hmx_v2.power_up = 1;
    x.hmx_v2.set_clock = 1;
    x.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_TUR;
    x.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_TUR;
    x.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
    x.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
    r4 = HAP_power_set(ctx, &x);
  }
  *rc = r4 * 100000000 + r3 * 1000000 + r1 * 1000 + r2;
  return 0;
}

int hmx_gemm_rpc_clocks(remote_handle64 h, int* v, int vLen) {
  static const HAP_Power_response_type ty[3] = {HAP_power_get_clk_Freq, HAP_power_get_hmx_core_clk_Freq, HAP_power_get_dcvsEnabled};
  for (int i = 0; i < 3 && i < vLen; i++) {
    HAP_power_response_t r;
    memset(&r, 0, sizeof r);
    r.type = ty[i];
    int rc = HAP_power_get(NULL, &r);
    v[i] = rc ? -rc : (i == 2 ? (int)r.dcvsEnabled : (int)r.clkFreqHz);
  }
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

/* ---- int8 cm GEMM (hmx_gemm_u8.h) ---- */
typedef struct {
  unsigned int ctx;
  uint8_t* vtcm;
  size_t vbytes;
  int mode, M, K, N, iters;
  const uint8_t* a;
  const int8_t* wp;
  const uint16_t* scale;
  uint8_t* c;
  uint64* t;
  int* codes;
} job_u8_t;

/* modes 2/3: every A crouton and W block resident in VTCM, only MACs + tile stores are timed */
static void u8_mac_only(job_u8_t* j) {
  int mt = (j->M + 63) / 64, kt = j->K / 32, deep = j->mode == 2;
  size_t a_off, w_off[2], c_off, t_off;
  hmx_gemm_u8_layout(j->M, j->K, j->N, 1, &a_off, w_off, &c_off, &t_off);
  uint8_t *va = j->vtcm + a_off, *vw = j->vtcm + w_off[0], *ct = j->vtcm + c_off;
  uint32_t* tbl = (uint32_t*)(j->vtcm + t_off);
  for (int mb = 0; mb < mt; mb++) hmx_pack_a_u8cm(j->a, j->M, j->K, mb * 64, va + (size_t)mb * kt * 2048);
  if (deep) memcpy(vw, j->wp, (size_t)j->K * j->N);
  else /* 32-column blocks: kt x 1 KB each, from the deep packing (set h of block jb) */
    for (int nb = 0; nb < j->N / 32; nb++)
      for (int kb = 0; kb < kt; kb++)
        memcpy(vw + ((size_t)nb * kt + kb) * 1024, j->wp + ((size_t)(nb / 2) * kt + kb) * 2048 + 1024 * (nb % 2), 1024);
  for (int i = 0; i < 64; i++) tbl[i] = j->scale[i % 32];
  hmx_blk_set_table(tbl);
  unsigned long long p0 = qurt_get_core_pcycles(), t0 = HAP_perf_get_time_us();
  for (int it = 0; it < j->iters; it++)
    if (deep)
      for (int g = 0; g < j->N / 64; g++)
        for (int mb = 0; mb < mt; mb++) {
          hmx_blk_mac_u8cm_deep(va + (size_t)mb * kt * 2048, vw + (size_t)g * kt * 2048, kt);
          hmx_blk_store_u8cm(ct);
          hmx_blk_store_u8cm(ct + 2048);
        }
    else
      for (int nb = 0; nb < j->N / 32; nb++)
        for (int mb = 0; mb < mt; mb++) {
          hmx_blk_mac_u8cm(va + (size_t)mb * kt * 2048, vw + (size_t)nb * kt * 1024, kt);
          hmx_blk_store_u8cm(ct);
        }
  volatile uint8_t sink = ct[0]; /* HVX/scalar read of the last tile: waits for the HMX store */
  (void)sink;
  j->t[0] = HAP_perf_get_time_us() - t0;
  j->t[1] = qurt_get_core_pcycles() - p0;
}

static void worker_u8(void* p) {
  job_u8_t* j = (job_u8_t*)p;
  j->codes[1] = qurt_hvx_lock(QURT_HVX_MODE_128B);
  j->codes[2] = HAP_compute_res_hmx_lock(j->ctx);
  if (j->codes[2] == 0) {
    if (j->mode >= 2)
      u8_mac_only(j);
    else {
      unsigned long long prof[4] = {0, 0, 0, 0};
      int kt = j->K / 32;
      size_t a_off, w_off[2];
      hmx_gemm_u8_layout(j->M, j->K, j->N, j->mode == 1, &a_off, w_off, NULL, NULL);
      const int8_t* w = j->wp;
      if (j->mode == 1) {
        memcpy(j->vtcm + w_off[0], j->wp, (size_t)kt * 32 * j->N);
        w = (const int8_t*)(j->vtcm + w_off[0]);
      }
      unsigned long long t0 = HAP_perf_get_time_us();
      for (int it = 0; it < j->iters; it++)
        j->codes[3] = hmx_gemm_u8_prof(j->a, w, j->scale, j->c, j->M, j->K, j->N, j->vtcm, j->vbytes, j->mode == 1, prof);
      j->t[0] = HAP_perf_get_time_us() - t0;
      j->t[1] = prof[0], j->t[2] = prof[1], j->t[3] = prof[2] + (prof[3] << 32);
    }
    HAP_compute_res_hmx_unlock(j->ctx);
  }
  if (j->codes[1] == 0) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

int hmx_gemm_rpc_gemm_u8(remote_handle64 h, int mode, int M, int K, int N, int iters, const uint8* a, int aLen,
                         const int8* wp, int wpLen, const uint16* scale, int scaleLen, uint8* c, int cLen, uint64* t,
                         int tLen, int* codes, int codesLen) {
  if (tLen < 4 || codesLen < 8 || K % 64 || N % 64 || aLen < M * K || wpLen < K * N || scaleLen < N || cLen < M * N)
    return AEE_EBADPARM;
  memset(t, 0, tLen * sizeof(uint64));
  memset(codes, 0, codesLen * sizeof(int));
  size_t need = hmx_gemm_u8_layout(M, K, N, mode >= 1, NULL, NULL, NULL, NULL);
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
    job_u8_t j = {ctx, (uint8_t*)vp, vs, mode, M, K, N, iters, a, wp, scale, c, t, codes};
    qurt_thread_attr_t ta;
    qurt_thread_attr_init(&ta);
    qurt_thread_attr_set_stack_addr(&ta, g_stack);
    qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
    qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_t tid;
    int st;
    codes[5] = qurt_thread_create(&tid, &ta, worker_u8, &j);
    if (codes[5] == 0) qurt_thread_join(tid, &st);
  } else
    codes[6] = -1;
  HAP_compute_res_release(ctx);
  return 0;
}

/* ---- chained int8 layers, activations resident in VTCM (hmx_layer_u8cm) ---- */
typedef struct {
  unsigned int ctx;
  uint8_t* vtcm;
  int flags, M, C, L, iters;
  const uint8_t* a;
  const int8_t* wp;
  const uint16_t* scale;
  uint8_t* out;
  uint64* t;
  int* codes;
  /* weight prefetch thread */
  /* one waiter per QuRT signal object (two threads waiting on one signal raised an exception): */
  qurt_signal_t sig;  /* copier waits: bit 0 request posted, bit 2 quit */
  qurt_signal_t sigd; /* worker waits: bit 1 copy done */
  volatile int req, done;
  uint8_t* wbuf[2];
} chain_t;
#define SIG_REQ 1u
#define SIG_DONE 2u
#define SIG_QUIT 4u
static char g_stack3[STACK_SIZE] __attribute__((aligned(128)));

static void copier(void* p) {
  chain_t* c = (chain_t*)p;
  qurt_hvx_lock(QURT_HVX_MODE_128B);
  for (;;) {
    unsigned m = qurt_signal_wait_any(&c->sig, SIG_REQ | SIG_QUIT);
    if (m & SIG_QUIT) break;
    qurt_signal_clear(&c->sig, SIG_REQ);
    int l = c->req;
    hmx_copy_hvx(c->wbuf[l & 1], c->wp + (size_t)(l % c->L) * c->C * c->C, (size_t)c->C * c->C);
    c->done = l;
    qurt_signal_set(&c->sigd, SIG_DONE);
  }
  qurt_hvx_unlock();
  qurt_thread_exit(0);
}

static void chain_worker(void* p) {
  chain_t* c = (chain_t*)p;
  c->codes[1] = qurt_hvx_lock(QURT_HVX_MODE_128B);
  c->codes[2] = HAP_compute_res_hmx_lock(c->ctx);
  if (c->codes[2] == 0) {
    int mt = (c->M + 63) / 64, kt = c->C / 32, nt = c->C / 32;
    size_t act = (size_t)mt * kt * 2048, wsz = (size_t)c->C * c->C;
    uint8_t *X = c->vtcm, *Y = X + act;
    c->wbuf[0] = Y + act, c->wbuf[1] = c->wbuf[0] + wsz;
    uint32_t* tbl = (uint32_t*)(c->wbuf[1] + wsz);
    for (int l = 0; l < c->L; l++)
      for (int j = 0; j < nt; j++)
        for (int k = 0; k < 64; k++) tbl[((size_t)l * nt + j) * 64 + k] = k < 32 ? c->scale[(size_t)l * c->C + 32 * j + k] : 0;
    unsigned long long t0 = HAP_perf_get_time_us();
    for (int mb = 0; mb < mt; mb++) hmx_pack_a_u8cm(c->a, c->M, c->C, mb * 64, X + (size_t)mb * kt * 2048);
    c->t[3] = HAP_perf_get_time_us() - t0;
    int prefetch = c->flags & 1, nocopy = c->flags & 2, total = c->L * (1 + c->iters), seq = 0;
    unsigned long long wait = 0;
    const volatile uint8_t* last_tile = NULL; /* last tile the previous layer stored: reading it waits for that layer */
    if (nocopy) hmx_copy_hvx(c->wbuf[0], c->wp, wsz);
    if (prefetch && !nocopy) {
      c->req = 0, c->done = -1;
      qurt_signal_set(&c->sig, SIG_REQ);
    }
    for (int run = 0; run < 1 + c->iters; run++) {
      unsigned long long r0 = HAP_perf_get_time_us();
      uint8_t *src = X, *dst = Y;
      for (int l = 0; l < c->L; l++, seq++) {
        const uint8_t* w = c->wbuf[0];
        if (!nocopy) {
          unsigned long long w0 = HAP_perf_get_time_us();
          if (prefetch) {
            while (c->done < seq) {
              qurt_signal_wait_any(&c->sigd, SIG_DONE);
              qurt_signal_clear(&c->sigd, SIG_DONE);
            }
            /* the buffer about to be refilled held layer seq-1's weights: make sure its HMX work is done */
            if (last_tile) (void)*last_tile;
            if (seq + 1 < total) {
              c->req = seq + 1;
              qurt_signal_set(&c->sig, SIG_REQ);
            }
          } else {
            if (last_tile) (void)*last_tile;
            hmx_copy_hvx(c->wbuf[seq & 1], c->wp + (size_t)l * wsz, wsz);
          }
          wait += HAP_perf_get_time_us() - w0;
          w = c->wbuf[seq & 1];
        }
        hmx_layer_u8cm(src, dst, w, tbl + (size_t)l * nt * 64, mt, kt, c->C);
        last_tile = dst + ((size_t)mt * nt - 1) * 2048;
        uint8_t* tmp = src;
        src = dst, dst = tmp;
      }
      (void)*last_tile; /* waits for the last HMX store */
      if (run == 0) {
        c->t[0] = HAP_perf_get_time_us() - r0;
        hmx_unpack_rows_u8cm(src, c->out, c->M, c->C);
        /* restore the input for the timing runs (they recompute the same chain) */
        for (int mb = 0; mb < mt; mb++) hmx_pack_a_u8cm(c->a, c->M, c->C, mb * 64, X + (size_t)mb * kt * 2048);
      } else
        c->t[1] += HAP_perf_get_time_us() - r0;
    }
    c->t[2] = wait;
    HAP_compute_res_hmx_unlock(c->ctx);
  }
  if (c->codes[1] == 0) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

int hmx_gemm_rpc_layers_u8(remote_handle64 h, int flags, int M, int C, int L, int iters, const uint8* a, int aLen,
                           const int8* wp, int wpLen, const uint16* scale, int scaleLen, uint8* o, int oLen,
                           uint64* t, int tLen, int* codes, int codesLen) {
  if (tLen < 4 || codesLen < 8 || C % 64 || L < 1 || aLen < M * C || wpLen < L * C * C || scaleLen < L * C ||
      oLen < M * C)
    return AEE_EBADPARM;
  memset(t, 0, tLen * sizeof(uint64));
  memset(codes, 0, codesLen * sizeof(int));
  size_t act = (size_t)((M + 63) / 64) * (C / 32) * 2048;
  size_t need = 2 * act + 2 * (size_t)C * C + (size_t)L * (C / 32) * 256;
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
    static chain_t c;
    memset(&c, 0, sizeof c);
    c.ctx = ctx, c.vtcm = (uint8_t*)vp, c.flags = flags, c.M = M, c.C = C, c.L = L, c.iters = iters;
    c.a = a, c.wp = wp, c.scale = scale, c.out = o, c.t = t, c.codes = codes;
    qurt_signal_init(&c.sig);
    qurt_signal_init(&c.sigd);
    qurt_thread_attr_t ta;
    qurt_thread_t tid, cid = 0;
    int st;
    if (flags & 1) {
      qurt_thread_attr_init(&ta);
      qurt_thread_attr_set_stack_addr(&ta, g_stack3);
      qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
      qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
      codes[6] = qurt_thread_create(&cid, &ta, copier, &c);
    }
    qurt_thread_attr_init(&ta);
    qurt_thread_attr_set_stack_addr(&ta, g_stack);
    qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
    qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
    codes[5] = qurt_thread_create(&tid, &ta, chain_worker, &c);
    if (codes[5] == 0) qurt_thread_join(tid, &st);
    if (flags & 1) {
      qurt_signal_set(&c.sig, SIG_QUIT);
      if (codes[6] == 0) qurt_thread_join(cid, &st);
    }
    qurt_signal_destroy(&c.sig);
    qurt_signal_destroy(&c.sigd);
  } else
    codes[7] = -1;
  HAP_compute_res_release(ctx);
  return 0;
}

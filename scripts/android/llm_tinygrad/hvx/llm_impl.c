/* DSP side of the LLM GEMV skel: runs one decode-step GEMV `reps` times on real buffers, timed on the DSP
 * (HAP_perf_get_time_us), single thread holding the HVX unit. Kernels:
 *   0/1: tinygrad-generated "up" GEMV (576x1536): tc / int32 variants (kernels.h from llm_kernels.py)
 *   2/3: tinygrad-generated "down" GEMV (1536x576)
 *   4  : hand vrmpybusv GEMV on the packed layout Wp[N/32][K/4][32][4], any K%4==0, N%32==0 (the roofline
 *        reference: one 128-byte weight load + one vrmpy per 32x4 MACs, 4 independent accumulators) */
#include <stdlib.h>
#include <string.h>
#include "llm_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "kernels.h"

extern unsigned long long HAP_perf_get_time_us(void);

AEEResult llm_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
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

int llm_rpc_open(const char* uri, remote_handle64* h) { *h = (remote_handle64)(uintptr_t)malloc(1); return *h ? 0 : -1; }
int llm_rpc_close(remote_handle64 h) { if (h) free((void*)(uintptr_t)h); return 0; }

static void hand_gemv(int* out, const unsigned char* x, const signed char* wp, int K, int N) {
  const HVX_Vector* w = (const HVX_Vector*)wp;
  const int K4 = K / 4;
  for (int nb = 0; nb < N / 32; nb++) {
    HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero(), a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
    const HVX_Vector* wb = w + (size_t)nb * K4;
    int k = 0;
    for (; k + 4 <= K4; k += 4) {
      Q6_dcfetch_A((void*)(wb + k + 16));
      int x0, x1, x2, x3;
      memcpy(&x0, x + 4 * k, 4); memcpy(&x1, x + 4 * k + 4, 4); memcpy(&x2, x + 4 * k + 8, 4); memcpy(&x3, x + 4 * k + 12, 4);
      a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R(x0), wb[k]);
      a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R(x1), wb[k + 1]);
      a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R(x2), wb[k + 2]);
      a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R(x3), wb[k + 3]);
    }
    for (; k < K4; k++) {
      int xk;
      memcpy(&xk, x + 4 * k, 4);
      a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R(xk), wb[k]);
    }
    HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1), Q6_Vw_vadd_VwVw(a2, a3));
    ((HVX_Vector*)out)[nb] = s;
  }
}

static void call(int kern, void* o, const void* x, const void* w, int K, int N) {
  switch (kern) {
    case 0: llm_up_tc(o, (void*)x, (void*)w); break;
    case 1: llm_up_int32(o, (void*)x, (void*)w); break;
    case 2: llm_down_tc(o, (void*)x, (void*)w); break;
    case 3: llm_down_int32(o, (void*)x, (void*)w); break;
    case 4: hand_gemv(o, x, w, K, N); break;
    case 5: llm_up_tcp(o, (void*)x, (void*)w); break;
    case 6: llm_down_tcp(o, (void*)x, (void*)w); break;
    case 7: llm_qo_tcp(o, (void*)x, (void*)w); break;
    case 8: llm_kv_tcp(o, (void*)x, (void*)w); break;
  }
}

#define BENCH_STACK (256 * 1024)
typedef struct {
  int kern, reps, K, N;
  const unsigned char *x, *w;
  unsigned char* result;
  unsigned long long t[64];
} bench_job_t;

static void bench_thread(void* arg) {
  bench_job_t* j = (bench_job_t*)arg;
  qurt_hvx_lock(QURT_HVX_MODE_128B);
  for (int r = 0; r < j->reps; r++) {
    unsigned long long t0 = HAP_perf_get_time_us();
    call(j->kern, j->result, j->x, j->w, j->K, j->N);
    j->t[r] = HAP_perf_get_time_us() - t0;
  }
  qurt_hvx_unlock();
  qurt_thread_exit(0);
}

static int cmp_u64(const void* a, const void* b) {
  unsigned long long x = *(const unsigned long long*)a, y = *(const unsigned long long*)b; return x < y ? -1 : x > y;
}

AEEResult llm_rpc_run(remote_handle64 h, int32 kern, int32 reps, int32 K, int32 N, const unsigned char* x, int xLen,
                      const unsigned char* w, int wLen, unsigned char* result, int resultLen, uint64* min_us,
                      uint64* med_us) {
  if (kern < 0 || kern > 8 || reps < 1 || reps > 64 || K % 16 || N % 32 || wLen < K * N || resultLen < 4 * N)
    return AEE_EBADPARM;
  bench_job_t j = {kern, reps, K, N, x, w, result, {0}};
  /* on a QuRT thread of our own: the FastRPC thread's stack is too small for tinygrad's vrmpy kernels, whose
     128-512-byte-aligned vector temporaries (and spills) crash the process there */
  qurt_thread_attr_t attr;
  qurt_thread_t tid;
  int status;
  void* stack = malloc(BENCH_STACK + 128);
  if (!stack) return AEE_ENOMEMORY;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "llm_bench");
  qurt_thread_attr_set_stack_addr(&attr, (void*)(((uintptr_t)stack + 127) & ~(uintptr_t)127));
  qurt_thread_attr_set_stack_size(&attr, BENCH_STACK);
  qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()) - 1);
  if (qurt_thread_create(&tid, &attr, bench_thread, &j) != QURT_EOK) { free(stack); return AEE_EFAILED; }
  qurt_thread_join(tid, &status);
  free(stack);
  unsigned long long* t = j.t;
  qsort(t, reps, sizeof t[0], cmp_u64);
  *min_us = t[0];
  *med_us = t[reps / 2];
  return 0;
}

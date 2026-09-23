/* HMX probe for Hexagon V69 (SM8475): can an unsigned PD reserve VTCM, lock HMX and run tile MACs?
 * HMX instructions are inline asm (open-access Hexagon tools 19.x accept them with -mhmx; the v73+
 * `cvt.* = acc(..)` read-out does not exist on v69, which stores the accumulator directly with
 * `mxmem(Rs,Rt):after... = acc`). VTCM/HMX come from HAP_compute_res (SDK header, weak symbols
 * resolved by the DSP runtime). */
#include <stdlib.h>
#include <string.h>
#include "hmx_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "HAP_compute_res.h"

extern unsigned long long HAP_perf_get_time_us(void);

#define VTCM_SIZE (256 * 1024)
#define OFF_ACT 0
#define OFF_WEI (64 * 1024)
#define OFF_OUT (128 * 1024)
#define OFF_BIAS (192 * 1024)
#define OUT_SPAN (32 * 1024)

AEEResult hmx_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
  void* ctx = (void*)hmx_rpc_perf_vote;
  HAP_power_request_t req = {0};
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = 1;
  int r1 = HAP_power_set(ctx, &req);
  HAP_power_request_t d = {0};
  d.type = HAP_power_set_DCVS_v2;
  d.dcvs_v2.dcvs_enable = 0;
  d.dcvs_v2.set_dcvs_params = 1;
  d.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  if (turbo) {
    d.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
  }
  int r2 = HAP_power_set(ctx, &d);
  *rc = r1 * 1000 + r2;
  return 0;
}

int hmx_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return 0;
}
int hmx_rpc_close(remote_handle64 h) {
  free((void*)(uintptr_t)h);
  return 0;
}

/* variant: bits 0-3 load form (0: ub x b, 1: hf x hf, 2: ub x b no :deep), bits 4-7 store form
 * (0: :after:sat.ub, 1: :after.hf, 2: :after:sat.uh 2x1, 3: :before:sat.ub), bit 8: load bias first. */
static void hmx_seq(unsigned char* v, int variant, int limit, int iters) {
  unsigned char* a = v + OFF_ACT;
  unsigned char* w = v + OFF_WEI;
  unsigned char* o = v + OFF_OUT;
  unsigned char* b = v + OFF_BIAS;
  int ld = variant & 15, st = (variant >> 4) & 15;
  if (variant & 256) asm volatile("bias = mxmem(%0)" ::"r"(b) : "memory");
  for (int i = 0; i < iters; i++) {
    if (ld == 0)
      asm volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }" ::"r"(a), "r"(limit), "r"(w),
                   "r"(limit)
                   : "memory");
    else if (ld == 1)
      asm volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" ::"r"(a), "r"(limit), "r"(w),
                   "r"(limit)
                   : "memory");
    else
      asm volatile("{ activation.ub = mxmem(%0,%1)\n weight.b = mxmem(%2,%3) }" ::"r"(a), "r"(limit), "r"(w),
                   "r"(limit)
                   : "memory");
  }
  if (st == 0)
    asm volatile("mxmem(%0,%1):after:sat.ub = acc" ::"r"(o), "r"(0) : "memory");
  else if (st == 1)
    asm volatile("mxmem(%0,%1):after.hf = acc" ::"r"(o), "r"(0) : "memory");
  else if (st == 2)
    asm volatile("mxmem(%0,%1):after:sat.uh = acc:2x1" ::"r"(o), "r"(0) : "memory");
  else
    asm volatile("mxmem(%0,%1):before:sat.ub = acc" ::"r"(o), "r"(0) : "memory");
}

typedef struct {
  unsigned int ctx;
  unsigned char* vtcm;
  int mode, variant, limit, iters;
  int32* codes;
  unsigned long long us;
} job_t;

#define STACK_SIZE (64 * 1024)
static char g_stack[STACK_SIZE] __attribute__((aligned(128)));

static void worker(void* p) {
  job_t* j = (job_t*)p;
  j->codes[4] = qurt_hvx_lock(QURT_HVX_MODE_128B);
  j->codes[5] = HAP_compute_res_hmx_lock(j->ctx);
  if (j->codes[5] == 0) {
    if (j->mode >= 1) {
      unsigned long long t0 = HAP_perf_get_time_us();
      hmx_seq(j->vtcm, j->variant, j->limit, j->mode == 2 ? j->iters : 1);
      j->us = HAP_perf_get_time_us() - t0;
      j->codes[6] = 1;
    }
    j->codes[7] = HAP_compute_res_hmx_unlock(j->ctx);
  }
  if (j->codes[4] == 0) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

AEEResult hmx_rpc_probe(remote_handle64 h, int32 mode, int32 variant, int32 limit, int32 iters, const uint8* act,
                        int actLen, const uint8* wei, int weiLen, const uint8* bias, int biasLen, uint8* out, int resultLen,
                        int32* codes, int codesLen, uint64* dsp_us) {
  if (codesLen < 12) return AEE_EBADPARM;
  memset(codes, 0, codesLen * sizeof(int32));
  *dsp_us = 0;
  compute_res_attr_t attr;
  codes[0] = HAP_compute_res_attr_init(&attr);
  codes[1] = HAP_compute_res_attr_set_vtcm_param_v2(&attr, VTCM_SIZE, 0, 0);
  codes[2] = HAP_compute_res_attr_set_hmx_param(&attr, 1);
  unsigned int ctx = HAP_compute_res_acquire(&attr, 100000);
  codes[3] = (int32)ctx;
  if (!ctx) return 0;
  void* vp = NULL;
  unsigned int vs = 0;
  codes[8] = HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vp, &vs);
  codes[9] = (int32)vs;
  codes[10] = (int32)(uintptr_t)vp;
  unsigned char* v = (unsigned char*)vp;
  if (v && vs >= VTCM_SIZE) {
    memset(v, 0, VTCM_SIZE);
    memset(v + OFF_OUT, 0xCD, OUT_SPAN);
    if (actLen) memcpy(v + OFF_ACT, act, actLen > 65536 ? 65536 : actLen);
    if (weiLen) memcpy(v + OFF_WEI, wei, weiLen > 65536 ? 65536 : weiLen);
    if (biasLen) memcpy(v + OFF_BIAS, bias, biasLen > 65536 ? 65536 : biasLen);
    job_t j = {ctx, v, mode, variant, limit, iters, codes, 0};
    qurt_thread_attr_t ta;
    qurt_thread_attr_init(&ta);
    qurt_thread_attr_set_stack_addr(&ta, g_stack);
    qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
    qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()) - 1);
    qurt_thread_t tid;
    int st;
    codes[11] = qurt_thread_create(&tid, &ta, worker, &j);
    if (codes[11] == 0) qurt_thread_join(tid, &st);
    *dsp_us = j.us;
    if (resultLen) memcpy(out, v + OFF_OUT, resultLen > OUT_SPAN ? OUT_SPAN : resultLen);
  }
  HAP_compute_res_release(ctx);
  return 0;
}

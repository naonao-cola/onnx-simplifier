/* DSP side of the tinygrad-codegen comparison skel: runs one of kernels.h's functions (plain tinygrad Tensor code vs the
 * hand custom_kernel for the same op) `reps` times on real buffers, timed on the DSP with HAP_perf_get_time_us so the
 * FastRPC round trip is reported separately by the client. Single thread, holding the HVX unit (qurt_hvx_lock). */
#include <stdlib.h>
#include "tgk_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "kernels.h"

extern unsigned long long HAP_perf_get_time_us(void);

AEEResult tgk_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
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

int tgk_rpc_open(const char* uri, remote_handle64* h) { *h = (remote_handle64)(uintptr_t)malloc(1); return *h ? 0 : -1; }
int tgk_rpc_close(remote_handle64 h) { if (h) free((void*)(uintptr_t)h); return 0; }

static void call(int kern, void* o, void* a, void* b) {
  switch (kern) {
    case 0: gen_add(o, a, b); break;
    case 1: hand_add(o, a, b); break;
    case 2: gen_maxpool(o, a); break;
    case 3: hand_maxpool(o, a); break;
    case 4: gen_requant(o, a); break;
    case 5: hand_requant(o, a); break;
  }
}

static int cmp_u64(const void* x, const void* y) {
  unsigned long long a = *(const unsigned long long*)x, b = *(const unsigned long long*)y; return a < b ? -1 : a > b;
}

AEEResult tgk_rpc_run(remote_handle64 h, int32 kern, int32 reps, const unsigned char* in0, int in0Len,
                      const unsigned char* in1, int in1Len, unsigned char* result, int resultLen,
                      uint64* min_us, uint64* med_us) {
  if (kern < 0 || kern > 5 || reps < 1 || reps > 64) return AEE_EBADPARM;
  unsigned long long t[64];
  qurt_hvx_lock(QURT_HVX_MODE_128B);
  for (int r = 0; r < reps; r++) {
    unsigned long long t0 = HAP_perf_get_time_us();
    call(kern, result, (void*)in0, (void*)in1);
    t[r] = HAP_perf_get_time_us() - t0;
  }
  qurt_hvx_unlock();
  qsort(t, reps, sizeof t[0], cmp_u64);
  *min_us = t[0]; *med_us = t[reps / 2];
  return 0;
}

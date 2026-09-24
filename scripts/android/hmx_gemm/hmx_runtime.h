/* HMX runtime setup for an unsigned FastRPC skel on V69 -- kept apart from the kernels (hmx_block.h).
 *
 *   hmx_rt_power();                                   // once per session: HVX + HMX power, DCVS turbo
 *   hmx_rt_t rt; hmx_rt_acquire(&rt, bytes);          // VTCM + HMX context (one per PD: a second HMX
 *                                                     //   acquire in the same PD returns ctx 0)
 *   // on the thread that issues HMX ops (a QuRT worker with >= 64 KB stack, not the FastRPC thread):
 *   hmx_rt_lock(&rt);  ... hmx_blk_* on rt.vtcm ...  hmx_rt_unlock(&rt);
 *   hmx_rt_release(&rt);
 *
 * HAP_power_set_HMX is mandatory: without it the first HMX tile op hung the cDSP until the phone was
 * rebooted (scripts/android/hmx_probe, PRs #1890 / #1905). */
#ifndef HMX_RUNTIME_H
#define HMX_RUNTIME_H
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "qurt.h"

typedef struct {
  unsigned int ctx;
  unsigned char* vtcm;
  unsigned int vtcm_bytes;
} hmx_rt_t;

static inline int hmx_rt_power(void* client_ctx, int turbo) {
  HAP_power_request_t r = {0};
  r.type = HAP_power_set_HVX;
  r.hvx.power_up = 1;
  int rc = HAP_power_set(client_ctx, &r);
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
  rc |= HAP_power_set(client_ctx, &d);
  HAP_power_request_t x = {0};
  x.type = HAP_power_set_HMX;
  x.hmx.power_up = 1;
  return rc | HAP_power_set(client_ctx, &x);
}

/* bytes: VTCM size (rounded up to 64 KB); returns 0 on success */
static inline int hmx_rt_acquire(hmx_rt_t* rt, unsigned int bytes) {
  compute_res_attr_t attr;
  HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param_v2(&attr, (bytes + 0xFFFF) & ~0xFFFFu, 0, 0);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);
  rt->ctx = HAP_compute_res_acquire(&attr, 100000);
  if (!rt->ctx) return -1;
  void* p = 0;
  HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &p, &rt->vtcm_bytes);
  rt->vtcm = (unsigned char*)p;
  return p ? 0 : -1;
}
static inline int hmx_rt_lock(hmx_rt_t* rt) {
  int rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
  return rc ? rc : HAP_compute_res_hmx_lock(rt->ctx);
}
static inline void hmx_rt_unlock(hmx_rt_t* rt) {
  HAP_compute_res_hmx_unlock(rt->ctx);
  qurt_hvx_unlock();
}
static inline void hmx_rt_release(hmx_rt_t* rt) { HAP_compute_res_release(rt->ctx); }
#endif

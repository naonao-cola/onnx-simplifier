/* StreamPETR cross-attention on the HVX: softmax(Q K^T) V for 8 heads of 32 dims, uint8 in and out.
 * The integer contract is attn_hvx/emulate.py's attn_int(exp="poly"), bit-exact:
 *   s_j  = q . k_j - zq * sum(k_j)   (= (q - zq).(k - zk) + a per-row constant, which softmax ignores)
 *   p_j  = exp_u8(max s - s_j)       round(255 * e^-(d * step)), step = sq * sk, via 2^-t in Q11
 *   out  = round(sum p_j v_j / sum p_j)   uint8 with V's own (scale, zero point)
 * Layout: q (LQ, 256), k and v (LK, 256), out (LQ, 256), row-major uint8 as the HTP emits them.
 * Packed per call (attn_pack): kp[h][LK/32][8][32] words = the 4 dims 4g..4g+3 of key 32b+j (vrmpy's
 * 4-byte groups with keys across lanes), vp[h][LK/4][32][4] bytes = dim d of keys 4g..4g+3 (dims
 * across lanes), kb[h][LK] = zq * sum_d k. Then per (head, rows): QK into a row of LK int32, max,
 * exp to uint8, AV into 32 lanes, one reciprocal per row. LK must be a multiple of 128. */
#pragma once
#include <stdint.h>
#include <string.h>

#define ATTN_H 8
#define ATTN_D 32
#define ATTN_C 256
#define ATTN_C1 22663
#define ATTN_C2 7582
#define ATTN_C3 1307
#define ATTN_TMAX (9 * 2048 + 2047)

typedef struct {
  int LQ, LK, zq, m16, dcl;
  const uint8_t *q, *k, *v;
  uint32_t* kp;  // ATTN_H * LK words
  uint8_t* vp;   // ATTN_H * LK * 32 bytes
  int32_t* kb;   // ATTN_H * LK
  uint8_t* out;
} attn_args_t;

static inline int attn_mulq15(int a, int b) { return (a * b + (1 << 14)) >> 15; }

static inline int attn_exp_u8(int32_t d, int m16, int dcl) {
  int t = (int)(((int64_t)(d < dcl ? d : dcl) * m16) >> 4);
  if (t > ATTN_TMAX) t = ATTN_TMAX;
  int n = t >> 11, x = (t & 2047) << 4;
  int x2 = attn_mulq15(x, x), x3 = attn_mulq15(x2, x);
  int y = (32767 - attn_mulq15(ATTN_C1, x) + attn_mulq15(ATTN_C2, x2) - attn_mulq15(ATTN_C3, x3)) >> n;
  return (y * 255 + (1 << 14)) >> 15;
}

/* round(acc / sp) as (acc + sp / 2) * R >> 32 with R = 2^32 / sp + 1: one division per row */
static inline uint64_t attn_recip(uint32_t sp) { return (((uint64_t)1) << 32) / sp + 1; }
static inline int attn_div_round(uint32_t acc, uint32_t sp, uint64_t r) {
  return (int)(((uint64_t)(acc + (sp >> 1)) * r) >> 32);
}

/* packs head h's keys [k0, k1) (multiples of 32) */
static void attn_pack(const attn_args_t* a, int h, int k0, int k1) {
  const int LK = a->LK;
  uint32_t* kp = a->kp + (size_t)h * LK * 8;
  uint8_t* vp = a->vp + (size_t)h * LK * ATTN_D;
  int32_t* kb = a->kb + (size_t)h * LK;
  for (int j = k0; j < k1; j++) {
    const uint8_t* kr = a->k + (size_t)j * ATTN_C + h * ATTN_D;
    uint32_t w[8];
    memcpy(w, kr, 32);
    int s = 0;
    for (int d = 0; d < ATTN_D; d++) s += kr[d];
    kb[j] = a->zq * s;
    for (int g = 0; g < 8; g++) kp[((size_t)(j >> 5) * 8 + g) * 32 + (j & 31)] = w[g];
    const uint8_t* vr = a->v + (size_t)j * ATTN_C + h * ATTN_D;
    uint8_t* vo = vp + (size_t)(j >> 2) * 128 + (j & 3);
    for (int d = 0; d < ATTN_D; d++) vo[4 * d] = vr[d];
  }
}

/* scalar body over the packed layout (the reference the HVX body must match) */
static void attn_rows_scalar(const attn_args_t* a, int h, int r0, int r1, int32_t* srow, uint8_t* prow) {
  const int LK = a->LK;
  const uint32_t* kp = a->kp + (size_t)h * LK * 8;
  const uint8_t* vp = a->vp + (size_t)h * LK * ATTN_D;
  const int32_t* kb = a->kb + (size_t)h * LK;
  for (int i = r0; i < r1; i++) {
    const uint8_t* qr = a->q + (size_t)i * ATTN_C + h * ATTN_D;
    int32_t mx = INT32_MIN;
    for (int j = 0; j < LK; j++) {
      const uint8_t* kw = (const uint8_t*)(kp + ((size_t)(j >> 5) * 8) * 32 + (j & 31));
      uint32_t s = 0;
      for (int g = 0; g < 8; g++)
        for (int b = 0; b < 4; b++) s += (uint32_t)qr[4 * g + b] * kw[g * 128 + b];
      srow[j] = (int32_t)s - kb[j];
      if (srow[j] > mx) mx = srow[j];
    }
    uint32_t sp = 0;
    for (int j = 0; j < LK; j++) sp += prow[j] = (uint8_t)attn_exp_u8(mx - srow[j], a->m16, a->dcl);
    uint8_t* o = a->out + (size_t)i * ATTN_C + h * ATTN_D;
    const uint64_t rcp = attn_recip(sp);
    for (int d = 0; d < ATTN_D; d++) {
      uint32_t acc = 0;
      for (int j = 0; j < LK; j++) acc += (uint32_t)prow[j] * vp[(size_t)(j >> 2) * 128 + 4 * d + (j & 3)];
      o[d] = (uint8_t)attn_div_round(acc, sp, rcp);
    }
  }
}

#ifdef __HVX__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define ATTN_RB 4 /* query rows sharing each K / V vector load */
#ifdef ATTN_PROF
static unsigned long long attn_prof[4];
#define ATTN_T(i) unsigned long long _t##i = hexagon_sim_read_pcycles()
#define ATTN_ACC(k, a, b) attn_prof[k] += _t##b - _t##a
#else
#define ATTN_T(i)
#define ATTN_ACC(k, a, b)
#endif

/* HVX body: rows [r0, r1) of head h. srow: ATTN_RB * LK int32, prow: ATTN_RB * LK bytes, both
 * 128-byte aligned scratch owned by the calling thread. */
static void attn_rows_hvx(const attn_args_t* a, int h, int r0, int r1, int32_t* srow, uint8_t* prow) {
  const int LK = a->LK, NKB = LK / 32;
  const HVX_Vector* kp = (const HVX_Vector*)(a->kp + (size_t)h * LK * 8);
  const HVX_Vector* vp = (const HVX_Vector*)(a->vp + (size_t)h * LK * ATTN_D);
  const HVX_Vector* kb = (const HVX_Vector*)(a->kb + (size_t)h * LK);
  const HVX_Vector vc1 = Q6_Vh_vsplat_R(ATTN_C1), vc2 = Q6_Vh_vsplat_R(ATTN_C2), vc3 = Q6_Vh_vsplat_R(ATTN_C3);
  const HVX_Vector v32767 = Q6_Vh_vsplat_R(32767), vtmax = Q6_V_vsplat_R(ATTN_TMAX), v2047 = Q6_Vh_vsplat_R(2047);
  const HVX_Vector vdcl = Q6_V_vsplat_R(a->dcl);
  for (int i0 = r0; i0 < r1; i0 += ATTN_RB) {
    const int nr = r1 - i0 < ATTN_RB ? r1 - i0 : ATTN_RB;
    uint32_t qw[ATTN_RB][8];
    for (int r = 0; r < ATTN_RB; r++)
      memcpy(qw[r], a->q + (size_t)(i0 + (r < nr ? r : 0)) * ATTN_C + h * ATTN_D, 32);
    ATTN_T(0);
    /* QK: 32 keys per vector, 8 vrmpy over the 32 dims, 4 rows per K load */
    HVX_Vector vmx[ATTN_RB];
    for (int r = 0; r < ATTN_RB; r++) vmx[r] = Q6_V_vsplat_R(INT32_MIN);
    for (int b = 0; b < NKB; b++) {
      HVX_Vector acc[ATTN_RB];
      for (int r = 0; r < ATTN_RB; r++) acc[r] = Q6_V_vzero();
      for (int g = 0; g < 8; g++) {
        HVX_Vector kv = kp[b * 8 + g];
        for (int r = 0; r < ATTN_RB; r++) acc[r] = Q6_Vuw_vrmpyacc_VuwVubRub(acc[r], kv, qw[r][g]);
      }
      for (int r = 0; r < ATTN_RB; r++) {
        HVX_Vector s = Q6_Vw_vsub_VwVw(acc[r], kb[b]);
        ((HVX_Vector*)(srow + (size_t)r * LK))[b] = s;
        vmx[r] = Q6_Vw_vmax_VwVw(vmx[r], s);
      }
    }
    ATTN_T(1);
    ATTN_ACC(0, 0, 1);
    /* per row: horizontal max, exp to uint8 (64 keys per halfword vector), sum */
    uint32_t sp[ATTN_RB];
    for (int r = 0; r < ATTN_RB; r++) {
      int32_t m[32] __attribute__((aligned(128)));
      *(HVX_Vector*)m = vmx[r];
      int32_t mx = m[0];
      for (int l = 1; l < 32; l++) mx = m[l] > mx ? m[l] : mx;
      const HVX_Vector vm = Q6_V_vsplat_R(mx);
      const HVX_Vector* s = (const HVX_Vector*)(srow + (size_t)r * LK);
      HVX_Vector* p = (HVX_Vector*)(prow + (size_t)r * LK);
      HVX_Vector vsum = Q6_V_vzero();
      for (int b = 0; b < NKB; b += 4) {
        HVX_Vector th[2];
        for (int u = 0; u < 2; u++) {
          HVX_Vector tw[2];
          for (int e = 0; e < 2; e++) {
            HVX_Vector d = Q6_Vw_vmin_VwVw(Q6_Vw_vsub_VwVw(vm, s[b + 2 * u + e]), vdcl);
            tw[e] = Q6_Vw_vmin_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vmpyi_VwRh(d, (a->m16 << 16) | a->m16), 4), vtmax);
          }
          HVX_Vector t = Q6_Vh_vpacke_VwVw(tw[1], tw[0]); /* keys in order */
          HVX_Vector n = Q6_Vh_vasr_VhR(t, 11);
          HVX_Vector x = Q6_Vh_vasl_VhR(Q6_V_vand_VV(t, v2047), 4);
          HVX_Vector x2 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(x, x);
          HVX_Vector x3 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(x2, x);
          HVX_Vector y = Q6_Vh_vsub_VhVh(v32767, Q6_Vh_vmpy_VhVh_s1_rnd_sat(vc1, x));
          y = Q6_Vh_vadd_VhVh(y, Q6_Vh_vmpy_VhVh_s1_rnd_sat(vc2, x2));
          y = Q6_Vh_vsub_VhVh(y, Q6_Vh_vmpy_VhVh_s1_rnd_sat(vc3, x3));
          y = Q6_Vh_vasr_VhVh(y, n);
          th[u] = Q6_Vh_vmpy_VhRh_s1_rnd_sat(y, (255 << 16) | 255);
        }
        HVX_Vector pv = Q6_Vub_vpack_VhVh_sat(th[1], th[0]);
        p[b >> 2] = pv;
        vsum = Q6_Vuw_vrmpyacc_VuwVubRub(vsum, pv, 0x01010101);
      }
      uint32_t w[32] __attribute__((aligned(128)));
      *(HVX_Vector*)w = vsum;
      uint32_t t = 0;
      for (int l = 0; l < 32; l++) t += w[l];
      sp[r] = t;
    }
    ATTN_T(2);
    ATTN_ACC(1, 1, 2);
    /* AV: dims across lanes, 4 keys per vrmpy, 4 rows per V load */
    HVX_Vector acc[ATTN_RB];
    for (int r = 0; r < ATTN_RB; r++) acc[r] = Q6_V_vzero();
    const uint32_t* pw[ATTN_RB];
    for (int r = 0; r < ATTN_RB; r++) pw[r] = (const uint32_t*)(prow + (size_t)r * LK);
    for (int g = 0; g < LK / 4; g++) {
      HVX_Vector vv = vp[g];
      for (int r = 0; r < ATTN_RB; r++) acc[r] = Q6_Vuw_vrmpyacc_VuwVubRub(acc[r], vv, pw[r][g]);
    }
    ATTN_T(3);
    ATTN_ACC(2, 2, 3);
    for (int r = 0; r < nr; r++) {
      uint32_t w[32] __attribute__((aligned(128)));
      *(HVX_Vector*)w = acc[r];
      uint8_t* o = a->out + (size_t)(i0 + r) * ATTN_C + h * ATTN_D;
      const uint64_t rcp = attn_recip(sp[r]);
      for (int d = 0; d < ATTN_D; d++) o[d] = (uint8_t)attn_div_round(w[d], sp[r], rcp);
    }
  }
}
#endif

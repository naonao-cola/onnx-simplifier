// NSS v1 "high" pre/post-processing as OpenCL kernels (Adreno GPU; also runs on any host OpenCL device).
//
// A fresh translation of the Apache-2.0 torch reference in arm/neural-graphics-model-gym
// (usecases/nss/model/torch_preprocess, torch_postprocess, pinned in nss_gym.py), restricted to the
// "high" quality path: full-resolution preprocess, 2x2 depth scatter, dense 6x6 KPN with 9 filter taps,
// Catmull-Rom history, YCoCg luma derivative, sharp theta. Arm's own GLSL shaders are not used.
//
// Layout: planar float NCHW (N = 1), coordinates are (row, column) = (y, x) like the reference. The
// CNN boundary is uint8 NHWC: `preprocess` writes the 12-channel CNN input (round(x * 255), the HTP
// model's 1/255 input scale) and `postprocess` reads the HTP's uint8 KPN / temporal outputs (x / 255).
// Floating-point contraction is off so each op rounds like the eager torch reference; the reference's
// explicit fused multiply-adds (float64 multiply-add, one rounding) use fma().
#pragma OPENCL FP_CONTRACT OFF

#define EPS 1e-07f
#define MAX_HALF 65504.0f
#define INT32_MAXF 2147483648.0f  // float(INT32_MAX) as a float32 tensor holds it

inline int clampi(int v, int lo, int hi) { return min(max(v, lo), hi); }
inline float satf(float v) { return clamp(v, 0.0f, 1.0f); }
// torch.lerp: start + w * (end - start) for |w| < 0.5, else end - (end - start) * (1 - w)
inline float lerpt(float a, float b, float w) {
  return fabs(w) < 0.5f ? a + w * (b - a) : b - (b - a) * (1.0f - w);
}
inline int reflect1(int v, int size) {
  v = v < 0 ? -v - 1 : v;
  return v >= size ? 2 * size - v - 1 : v;
}
inline float load(__global const float* t, int H, int W, int ch, int y, int x) {
  return t[(ch * H + clampi(y, 0, H - 1)) * W + clampi(x, 0, W - 1)];
}

// bilinear_sample / sample_bilinear (identical in the pre and post references): uv in [0, 1] (y, x),
// taps clamped to the edge; clamp_to_edge = 0 zeroes the weight of off-screen taps instead.
inline void bilinear(__global const float* t, int C, int H, int W, float uvy, float uvx, int clamp_to_edge,
                     float* out) {
  float py = uvy * (float)H, px = uvx * (float)W;
  float gy0 = floor(py - 0.5f), gx0 = floor(px - 0.5f);
  float gy1 = gy0 + 1.0f, gx1 = gx0 + 1.0f;
  float wy0 = fmax(1.0f - fabs(gy0 + 0.5f - py), 0.0f), wx0 = fmax(1.0f - fabs(gx0 + 0.5f - px), 0.0f);
  float wy1 = fmax(1.0f - fabs(gy1 + 0.5f - py), 0.0f), wx1 = fmax(1.0f - fabs(gx1 + 0.5f - px), 0.0f);
  if (!clamp_to_edge) {
    wy0 *= (gy0 >= 0.0f && gy0 < (float)H) ? 1.0f : 0.0f;
    wx0 *= (gx0 >= 0.0f && gx0 < (float)W) ? 1.0f : 0.0f;
    wy1 *= (gy1 >= 0.0f && gy1 < (float)H) ? 1.0f : 0.0f;
    wx1 *= (gx1 >= 0.0f && gx1 < (float)W) ? 1.0f : 0.0f;
  }
  int y0 = (int)gy0, x0 = (int)gx0, y1 = (int)gy1, x1 = (int)gx1;
  for (int c = 0; c < C; c++) {
    float tl = load(t, H, W, c, y0, x0) * wy0 * wx0;
    float tr = load(t, H, W, c, y0, x1) * wy0 * wx1;
    float bl = load(t, H, W, c, y1, x0) * wy1 * wx0;
    float br = load(t, H, W, c, y1, x1) * wy1 * wx1;
    out[c] = tl + tr + bl + br;
  }
}

inline void karis3(float* c) {  // tonemap_forward(..., Karis): x / (1 + max(x)), on non-negative x
  float r = fmax(c[0], 0.0f), g = fmax(c[1], 0.0f), b = fmax(c[2], 0.0f);
  float m = fmax(fmax(r, g), b);
  float s = 1.0f / (1.0f + m);
  c[0] = satf(r * s);
  c[1] = satf(g * s);
  c[2] = satf(b * s);
}

// ------------------------------------------------------------------------------------------------
// depth_scatter (non-quarter): for every depth-res pixel, the nearest of the 2x2 source pixels, its
// motion reprojects it, and its depth is atomic-min'ed into the 4 bilinear neighbours (weight > 0.1)
// of the reprojected position, as int32 depth * INT32_MAX.
__kernel void depth_scatter_init(__global int* out) { out[get_global_id(0)] = 2147483647; }

__kernel void depth_scatter(__global const float* motion, __global const float* depth, int H, int W,
                            __global int* out, int Ho, int Wo) {
  int oy = get_global_id(1), ox = get_global_id(0);
  if (oy >= Ho || ox >= Wo) return;
  float inv_oy = 1.0f / (float)Ho, inv_ox = 1.0f / (float)Wo;
  float guy = ((float)oy + 0.5f) * inv_oy, gux = ((float)ox + 0.5f) * inv_ox;
  int by = (int)floor(guy * (float)H - 0.5f), bx = (int)floor(gux * (float)W - 0.5f);
  float nd = load(depth, H, W, 0, by, bx);
  float my = load(motion, H, W, 0, by, bx), mx = load(motion, H, W, 1, by, bx);
  const int oyo[3] = {0, 1, 1}, oxo[3] = {1, 0, 1};  // (0,1), (1,0), (1,1) after the initial (0,0)
  for (int i = 0; i < 3; i++) {
    int y = by + oyo[i], x = bx + oxo[i];
    float cd = load(depth, H, W, 0, y, x);
    float take = cd <= nd ? 1.0f : 0.0f;
    nd = nd + take * (cd - nd);
    my = my + take * (load(motion, H, W, 0, y, x) - my);
    mx = mx + take * (load(motion, H, W, 1, y, x) - mx);
  }
  float isy = 1.0f / ((float)H / (float)Ho), isx = 1.0f / ((float)W / (float)Wo);
  float sy = my * isy, sx = mx * isx;
  float th = sqrt(sy * sy + sx * sx) > 0.1f ? 1.0f : 0.0f;
  sy *= th;
  sx *= th;
  float ry = guy - sy * inv_oy, rx = gux - sx * inv_ox;
  // _reconstruct_previous_depth
  float psy = ry * (float)Ho - 0.5f, psx = rx * (float)Wo - 0.5f;
  float fy0 = floor(psy), fx0 = floor(psx);
  float fy = psy - fy0, fx = psx - fx0;
  int b0y = (int)fy0, b0x = (int)fx0;
  float v = nd * INT32_MAXF;
  int d = v >= INT32_MAXF ? 2147483647 : (v <= -INT32_MAXF ? (int)0x80000000 : (int)v);
  float w[4] = {(1.0f - fy) * (1.0f - fx), fy * (1.0f - fx), (1.0f - fy) * fx, fy * fx};
  const int ry4[4] = {0, 1, 0, 1}, rx4[4] = {0, 0, 1, 1};
  for (int i = 0; i < 4; i++) {
    int y = b0y + ry4[i], x = b0x + rx4[i];
    if (y >= 0 && y < Ho && x >= 0 && x < Wo && w[i] > 0.1f) atomic_min(&out[y * Wo + x], d);
  }
}

// ------------------------------------------------------------------------------------------------
// preprocess helpers
inline float view_depth(float d, float dtv0, float dtv1) { return dtv1 / (d - dtv0); }

inline float depth_clip(__global const int* dtm1, int Hd, int Wd, float uvy, float uvx, float rs0, float rs1,
                        float cur, float4 dtv) {
  float cur_view = view_depth(cur, dtv.x, dtv.y);
  float psy = uvy * (float)Hd - 0.5f, psx = uvx * (float)Wd - 0.5f;
  float fy0 = floor(psy), fx0 = floor(psx);
  float fy = psy - fy0, fx = psx - fx0;
  int by = (int)fy0, bx = (int)fx0;
  float w[4] = {(1.0f - fy) * (1.0f - fx), fy * (1.0f - fx), (1.0f - fy) * fx, fy * fx};
  const int oy4[4] = {0, 1, 0, 1}, ox4[4] = {0, 0, 1, 1};
  float acc = 0.0f, wsum = 0.0f;
  float vs0 = (float)(long)rs0, vs1 = (float)(long)rs1;
  float cp0 = (float)(long)(rs0 * 0.5f), cp1 = (float)(long)(rs1 * 0.5f);
  float half_vw = sqrt(rs0 * rs0 + rs1 * rs1);
  float refl = sqrt(1080.0f * 1080.0f + 1920.0f * 1920.0f);
  float power = 1.0f + (3.0f - 1.0f) * satf(half_vw / refl);
  for (int i = 0; i < 4; i++) {
    int y = by + oy4[i], x = bx + ox4[i];
    int on = y >= 0 && y < Hd && x >= 0 && x < Wd;
    wsum = wsum + (on ? 0.0f : w[i]);
    int aw = on && w[i] > 0.1f;
    float prev = (float)dtm1[clampi(y, 0, Hd - 1) * Wd + clampi(x, 0, Wd - 1)] * (1.0f / INT32_MAXF);
    float prev_view = view_depth(prev, dtv.x, dtv.y);
    float diff = cur_view - prev_view;
    int active = aw && diff > 0.0f;
    float plane = fmax(prev, cur);
    float vd = view_depth(plane, dtv.x, dtv.y);
    // center = view position of the viewport center, corner = of (0, 0)
    float s0 = cp0 / vs0, s1 = cp1 / vs1;
    float c0 = dtv.z * (s0 * 2.0f - 1.0f) * vd, c1 = dtv.w * (s1 * -2.0f + 1.0f) * vd;
    float k0 = dtv.z * (0.0f * 2.0f - 1.0f) * vd, k1 = dtv.w * (0.0f * -2.0f + 1.0f) * vd;
    float len_center = sqrt(c0 * c0 + c1 * c1 + vd * vd);
    float len_corner = sqrt(k0 * k0 + k1 * k1 + vd * vd);
    float thr = fmax(cur_view, prev_view);
    float req = 1.37e-05f * (len_corner / len_center) * half_vw * thr + 0.0f;
    float ratio = satf(req / diff);
    float contrib = pow(ratio, power) * w[i];
    acc = acc + (active ? contrib : 0.0f);
    wsum = wsum + (active ? w[i] : 0.0f);
  }
  return wsum > 0.0f ? satf(1.0f - acc / wsum) : 0.0f;
}

inline void ycocg_load(__global const float* col, int H, int W, int y, int x, float e, float* o) {
  y = reflect1(y, H);
  x = reflect1(x, W);
  float rgb[3];
  for (int c = 0; c < 3; c++) rgb[c] = sqrt(fmax(load(col, H, W, c, y, x) * e, 0.0f));
  float co = rgb[0] - rgb[2];
  float tmp = rgb[2] + co * 0.5f;
  float cg = rgb[1] - tmp;
  o[0] = tmp + cg * 0.5f;
  o[1] = co;
  o[2] = cg;
}
inline float ydelta(const float* a, const float* b) {
  float dl = a[0] - b[0], dc = (a[1] - b[1]) * 1.25f, dg = (a[2] - b[2]) * 1.25f;
  return sqrt(dl * dl + dc * dc + dg * dg);
}

// One thread per padded CNN-grid pixel (Hp x Wp, >= H x W): the 12-channel CNN input (+ its uint8 NHWC
// copy), and for the pixels inside the H x W input the recurrent derivative state, the disocclusion
// mask and the nearest-depth offset code the postprocess needs.
__kernel void preprocess(
    __global const float* color, __global const float* history, __global const float* motion,
    __global const float* depth, __global const float* feedback_tm1, __global const float* derivative_tm1,
    __global const int* recon_depth, int H, int W, int Hp, int Wp, int Hh, int Wh, int Hd, int Wd,
    float jy, float jx, float exposure, float rs0, float rs1, float4 dtv,
    __global float* cnn_in, __global uchar* cnn_in_u8, __global float* derivative_out,
    __global float* disocc_out, __global uchar* nearest_code) {
  int py = get_global_id(1), px = get_global_id(0);
  if (py >= Hp || px >= Wp) return;
  int ry = reflect1(py, H), rx = reflect1(px, W);
  float iiy = 1.0f / (float)H, iix = 1.0f / (float)W;
  float uvy = ((float)ry + 0.5f) * iiy, uvx = ((float)rx + 0.5f) * iix;
  float upy = ((float)py + 0.5f) * (1.0f / (float)Hp), upx = ((float)px + 0.5f) * (1.0f / (float)Wp);
  // find_nearest_depth_4x4 (high offsets, strictly closer)
  int cy = (int)(uvy * (float)H), cx = (int)(uvx * (float)W);
  const int ofy[16] = {0, 1, 0, 0, -1, -1, 1, -1, 1, -1, 0, 1, 2, 2, 2, 2};
  const int ofx[16] = {0, 0, 1, -1, 0, 1, 1, -1, -1, 2, 2, 2, 2, 1, 0, -1};
  float nd = load(depth, H, W, 0, cy, cx);
  int ny = cy, nx = cx, oy = 0, ox = 0;
  for (int i = 1; i < 16; i++) {
    int sy = cy + ofy[i], sx = cx + ofx[i];
    int on = sy >= 0 && sy < H && sx >= 0 && sx < W;
    float sd = load(depth, H, W, 0, sy, sx);
    if (on && sd < nd) {
      nd = sd;
      ny = sy;
      nx = sx;
      oy = ofy[i];
      ox = ofx[i];
    }
  }
  float m[2];
  bilinear(motion, 2, H, W, ((float)ny + 0.5f) / (float)H, ((float)nx + 0.5f) / (float)W, 1, m);
  float mth = sqrt(m[0] * m[0] + m[1] * m[1]) > 0.1f ? 1.0f : 0.0f;
  m[0] *= mth;
  m[1] *= mth;
  float rpy = uvy - m[0] * iiy, rpx = uvx - m[1] * iix;
  float ujy = uvy - jy * iiy, ujx = uvx - jx * iix;
  int dcy = ry >> 1, dcx = rx >> 1;
  float rdy = ((float)dcy + 0.5f) * (1.0f / (float)Hd) - m[0] * iiy;
  float rdx = ((float)dcx + 0.5f) * (1.0f / (float)Wd) - m[1] * iix;
  float rppy = upy - m[0] * (1.0f / (float)Hp), rppx = upx - m[1] * (1.0f / (float)Wp);
  float dis = depth_clip(recon_depth, Hd, Wd, rdy, rdx, rs0, rs1, nd, dtv);
  float uc[3];
  bilinear(color, 3, H, W, ujy, ujx, 1, uc);
  for (int c = 0; c < 3; c++) uc[c] *= exposure;
  karis3(uc);
  float wh[3];
  bilinear(history, 3, Hh, Wh, rpy, rpx, 0, wh);
  for (int c = 0; c < 3; c++) wh[c] *= exposure;
  karis3(wh);
  // calculate_ycocg_derivative (not low/mid)
  float dt[4];
  bilinear(derivative_tm1, 4, H, W, rpy, rpx, 0, dt);
  float yc[3], yn[3], ys[3], ye[3], yw[3];
  ycocg_load(color, H, W, ry, rx, exposure, yc);
  ycocg_load(color, H, W, ry, rx - 1, exposure, yn);
  ycocg_load(color, H, W, ry, rx + 1, exposure, ys);
  ycocg_load(color, H, W, ry + 1, rx, exposure, ye);
  ycocg_load(color, H, W, ry - 1, rx, exposure, yw);
  float d_c = ydelta(yc, dt), d_n = ydelta(yc, yn), d_s = ydelta(yc, ys), d_e = ydelta(yc, ye),
        d_w = ydelta(yc, yw);
  float s_sum = d_n + d_s + d_e + d_w;
  float s_max = fmax(fmax(d_n, d_s), fmax(d_e, d_w));
  float prev = dt[3];
  float support = fmax(s_sum - s_max, 0.0f) * 0.3333333432674408f;
  float sup = lerpt(d_c, support, 0.3f) * 0.75f;
  float excursion = fmax(sup - prev, 0.0f);
  float recall = satf((sup - 0.065f) * 2.816901445388794f);
  float exc = satf((excursion - 0.025f) * 7.407407283782959f);
  float mean_gate = satf((sup - 0.07f) * 6.25f);
  float raw_entry = sqrt(recall) * sqrt(exc) * mean_gate;
  float heat = satf((prev - 0.11f) * 10.0f);
  float s_support = lerpt(prev, sup, 0.3f);
  float s_floor = lerpt(0.177f, 0.157f, heat), s_ceil = lerpt(0.33f, 0.305f, heat);
  float s_gate = satf((s_support - s_floor) * (1.0f / (s_ceil - s_floor)));
  s_gate = s_gate * s_gate;
  float hot = satf((prev - 0.18f) * 10.0f);
  hot = hot * hot;
  float carry = fmax(s_gate, hot * 0.12f);
  float raw_sustain = prev * carry * 0.8f;
  float raw_inst = fmax(raw_entry, raw_sustain);
  float decay = fmax(s_gate, heat * heat * 0.5f);
  float fall = lerpt(0.24f, 0.05f, decay);
  float rise = lerpt(0.08f, 0.22f, sqrt(recall * mean_gate));
  float filt = lerpt(prev, raw_inst, raw_inst > prev ? rise : fall);
  float vis = lerpt(prev, filt, filt > prev ? 0.75f : 0.8f);
  float disb = dis > 0.01f ? 1.0f : 0.0f;
  float uninit = (fabs(dt[0]) + fabs(dt[1]) + fabs(dt[2]) + fabs(dt[3])) < 0.0001f ? 1.0f : 0.0f;
  vis = vis * (1.0f - disb);
  float state[4] = {yc[0], yc[1], yc[2], filt};
  float reset_state[4] = {yc[0], yc[1], yc[2], 0.0f};
  for (int c = 0; c < 4; c++) state[c] = lerpt(lerpt(state[c], reset_state[c], disb), reset_state[c], uninit);
  vis = lerpt(vis, 0.0f, uninit);
  // feedback, motion detector
  float fb[4];
  bilinear(feedback_tm1, 4, Hp, Wp, rppy, rppx, 0, fb);
  for (int c = 0; c < 4; c++) fb[c] = lerpt(fb[c], 0.0f, disb);
  float pmin = sqrt((1.0f / rs0) * (1.0f / rs0) + (1.0f / rs1) * (1.0f / rs1));
  float pmax = sqrt((200.0f / rs0) * (200.0f / rs0) + (200.0f / rs1) * (200.0f / rs1));
  float nm0 = m[0] / rs0, nm1 = m[1] / rs1;
  float mlen = clamp(sqrt(nm0 * nm0 + nm1 * nm1), pmin, pmax);
  float md = sqrt((mlen - pmin) * (1.0f / (pmax - pmin)));
  float in12[12] = {wh[0], wh[1], wh[2], uc[0], uc[1], uc[2], md, fb[0], fb[1], fb[2], fb[3], vis};
  int n = Hp * Wp, p = py * Wp + px;
  for (int c = 0; c < 12; c++) {
    if (cnn_in) cnn_in[c * n + p] = in12[c];
    cnn_in_u8[p * 12 + c] = (uchar)clamp(rint(in12[c] * 255.0f), 0.0f, 255.0f);
  }
  if (py < H && px < W) {
    int q = py * W + px, nq = H * W;
    for (int c = 0; c < 4; c++) derivative_out[c * nq + q] = state[c];
    if (disocc_out) disocc_out[q] = dis;
    nearest_code[q] = (uchar)(((clampi(ox, -2, 2) + 2) << 3) | (clampi(oy, -2, 2) + 2));
  }
}

// ------------------------------------------------------------------------------------------------
// postprocess helpers
inline void catmull_rom(__global const float* t, int H, int W, float uvy, float uvx, float* out) {
  float sy = (float)H, sx = (float)W;
  float isy = 1.0f / sy, isx = 1.0f / sx;
  float suy = uvy * sy, sux = uvx * sx;
  float tcy = floor(suy - 0.5f) + 0.5f, tcx = floor(sux - 0.5f) + 0.5f;
  float f[2] = {suy - tcy, sux - tcx}, tc[2] = {tcy, tcx}, is[2] = {isy, isx};
  float cw[3][2], pos[3][2];
  for (int a = 0; a < 2; a++) {
    float f1 = f[a], f2 = f1 * f1, f3 = f2 * f1;
    float w0 = f2 - 0.5f * (f3 + f1);
    float w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
    float w3 = 0.5f * (f3 - f2);
    float w2 = 1.0f - w0 - w1 - w3;
    cw[0][a] = w0;
    cw[1][a] = w1 + w2;
    cw[2][a] = w3;
    pos[0][a] = (tc[a] - 1.0f) * is[a];
    pos[1][a] = (tc[a] + w2 / cw[1][a]) * is[a];
    pos[2][a] = (tc[a] + 2.0f) * is[a];
  }
  // cross taps (y index, x index): (m, l), (l, m), (m, m), (h, m), (m, h)
  const int ty[5] = {1, 0, 1, 2, 1}, tx[5] = {0, 1, 1, 1, 2};
  float ws = 0.0f, acc[3] = {0.0f, 0.0f, 0.0f}, mn[3], mx[3];
  for (int c = 0; c < 3; c++) {
    mn[c] = MAX_HALF;
    mx[c] = -MAX_HALF;
  }
  for (int k = 0; k < 5; k++) {
    float s[3];
    bilinear(t, 3, H, W, pos[ty[k]][0], pos[tx[k]][1], 1, s);
    float w = cw[ty[k]][0] * cw[tx[k]][1];
    for (int c = 0; c < 3; c++) {
      float v = s[c] * w;
      acc[c] = k == 0 ? v : acc[c] + v;
      mn[c] = fmin(mn[c], s[c]);
      mx[c] = fmax(mx[c], s[c]);
    }
    ws = k == 0 ? w : ws + w;
  }
  float fm = 1.0f / ws;
  int neg = 0;
  for (int c = 0; c < 3; c++) {
    out[c] = acc[c] * fm;
    neg |= out[c] < 0.0f;
  }
  if (neg)
    for (int c = 0; c < 3; c++) out[c] = fmax(fmin(out[c], mx[c]), mn[c]);
}

// One thread per output pixel (Ho x Wo): the filtered + temporally accumulated output (linear, which is
// also next frame's history) and its reinhard-tonemapped RGBA8 display copy.
__kernel void postprocess(
    __global const float* color, __global const float* history, __global const float* motion,
    __global const uchar* nearest_code, __global const uchar* kpn_u8, __global const uchar* temporal_u8,
    __global const float* offset_lut, int H, int W, int Ho, int Wo, int Hk, int Wk, int Kc, int Ht, int Wt,
    int mod_h, int mod_w, int taps, float exposure, float reset, __global float* out_linear,
    __global uchar* out_rgba) {
  int oy = get_global_id(1), ox = get_global_id(0);
  if (oy >= Ho || ox >= Wo) return;
  float e = exposure, ie = 1.0f / e;
  float scy = (float)Ho / (float)H, scx = (float)Wo / (float)W;
  float isy = 1.0f / scy, isx = 1.0f / scx;
  // filter_color (dense 6x6 KPN, full-res preprocess)
  int li = (oy % mod_h) * mod_w + (ox % mod_w);
  int nl = mod_h * mod_w * taps;
  float kpsy = (float)Hk / (float)Ht, kpsx = (float)Wk / (float)Wt;
  float m1[3] = {0, 0, 0}, m2[3] = {0, 0, 0}, wsum = 0.0f, cc[3] = {0, 0, 0}, cv = 0.0f;
  for (int k = 0; k < taps; k++) {
    int j = li * taps + k;
    float t0 = offset_lut[j], t1 = offset_lut[nl + j], t2 = offset_lut[2 * nl + j];
    float t3 = offset_lut[3 * nl + j], t4 = offset_lut[4 * nl + j], t5 = offset_lut[5 * nl + j];
    int ly = (int)floor(((float)(oy + (int)t3) + 0.5f) * isy + 0.001f) + (int)t0;
    int lx = (int)floor(((float)(ox + (int)t4) + 0.5f) * isx + 0.001f) + (int)t1;
    ly = clampi(ly, 0, H - 1);
    lx = clampi(lx, 0, W - 1);
    float ct[3];
    for (int c = 0; c < 3; c++) ct[c] = fmin(color[(c * H + ly) * W + lx] * e, MAX_HALF);
    if (t3 == 0.0f && t4 == 0.0f) {
      cc[0] = ct[0];
      cc[1] = ct[1];
      cc[2] = ct[2];
      cv = 1.0f;
    }
    int ky = clampi((int)floor(((float)ly + 0.5f + 0.001f) * kpsy), 0, Hk - 1);
    int kx = clampi((int)floor(((float)lx + 0.5f + 0.001f) * kpsx), 0, Wk - 1);
    int ch = clampi((int)t5, 0, Kc - 1);
    float raw = (float)kpn_u8[(ky * Wk + kx) * Kc + ch] * (1.0f / 255.0f);
    float w = fmax(raw, EPS) * t2;
    for (int c = 0; c < 3; c++) {
      m1[c] = m1[c] + ct[c] * w;
      m2[c] = m2[c] + (ct[c] * ct[c]) * w;
    }
    wsum = wsum + w;
  }
  float den = fmax(wsum, EPS);
  for (int c = 0; c < 3; c++) {
    m1[c] = m1[c] / den;
    m2[c] = m2[c] / den;
  }
  // sample_temporal_params (sharp theta); the temporal map is uint8 NHWC 4 channels
  float iy = 1.0f / (float)Ho, ix = 1.0f / (float)Wo;
  float uvy = ((float)oy + 0.5f) * iy, uvx = ((float)ox + 0.5f) * ix;
  float suy = uvy * ((float)H * (1.0f / (float)Ht)), sux = uvx * ((float)W * (1.0f / (float)Wt));
  float par[3];
  {
    float py = suy * (float)Ht, px = sux * (float)Wt;
    float gy0 = floor(py - 0.5f), gx0 = floor(px - 0.5f), gy1 = gy0 + 1.0f, gx1 = gx0 + 1.0f;
    float wy0 = fmax(1.0f - fabs(gy0 + 0.5f - py), 0.0f), wx0 = fmax(1.0f - fabs(gx0 + 0.5f - px), 0.0f);
    float wy1 = fmax(1.0f - fabs(gy1 + 0.5f - py), 0.0f), wx1 = fmax(1.0f - fabs(gx1 + 0.5f - px), 0.0f);
    int y0 = clampi((int)gy0, 0, Ht - 1), x0 = clampi((int)gx0, 0, Wt - 1);
    int y1 = clampi((int)gy1, 0, Ht - 1), x1 = clampi((int)gx1, 0, Wt - 1);
    for (int c = 0; c < 3; c++) {
      float tl = (float)temporal_u8[(y0 * Wt + x0) * 4 + c] * (1.0f / 255.0f) * wy0 * wx0;
      float tr = (float)temporal_u8[(y0 * Wt + x1) * 4 + c] * (1.0f / 255.0f) * wy0 * wx1;
      float bl = (float)temporal_u8[(y1 * Wt + x0) * 4 + c] * (1.0f / 255.0f) * wy1 * wx0;
      float br = (float)temporal_u8[(y1 * Wt + x1) * 4 + c] * (1.0f / 255.0f) * wy1 * wx1;
      par[c] = tl + tr + bl + br;
    }
  }
  float theta = satf(par[0]);
  float th2 = theta * theta, it = 1.0f - theta, it2 = it * it;
  theta = th2 / fmax(th2 + it2, 1e-06f);
  float alpha = par[1] * 0.35f + 0.05f;
  float gamma = par[2] * 2.0f;
  // load_motion via the nearest-depth offset code of the low-res pixel
  int icy = (int)floor((float)oy * isy), icx = (int)floor((float)ox * isx);
  int code = nearest_code[clampi(icy, 0, H - 1) * W + clampi(icx, 0, W - 1)];
  int sy = clampi(icy + (code & 7) - 2, 0, H - 1), sx = clampi(icx + ((code >> 3) & 7) - 2, 0, W - 1);
  float mvy = motion[sy * W + sx] * scy, mvx = motion[(H + sy) * W + sx] * scx;
  float len = sqrt(fma(mvy, mvy, mvx * mvx));
  float mth = len > 0.1f ? 1.0f : 0.0f;
  mvy *= mth;
  mvx *= mth;
  float rpy = fma(-mvy, iy, uvy), rpx = fma(-mvx, ix, uvx);
  float onscreen = (rpy >= 0.0f && rpx >= 0.0f && rpy <= 1.0f && rpx <= 1.0f) ? 1.0f : 0.0f;
  float wh[3];
  catmull_rom(history, Ho, Wo, rpy, rpx, wh);
  // rectify_history (variance via fma), accumulate, karis inverse
  float rect[3], cen[3];
  for (int c = 0; c < 3; c++) {
    float w = fmin(wh[c] * e, MAX_HALF);
    float var = fmax(fabs(fma(-m1[c], m1[c], m2[c])), EPS);
    float sigma = sqrt(var) * gamma;
    float hc = lerpt(m1[c], fmax(fmin(w, m1[c] + sigma), m1[c] - sigma), reset);
    rect[c] = lerpt(hc, w, theta * onscreen * reset);
    cen[c] = cc[c];
  }
  karis3(rect);
  karis3(cen);
  float a = alpha * cv * reset;
  float acc[3];
  for (int c = 0; c < 3; c++) acc[c] = clamp(lerpt(rect[c], cen[c], a), 0.0f, 1.0f - EPS);
  float lim = 65504.0f * (1.0f / (1.0f + 65504.0f));  // karis_forward of the MAX_HALF limit
  float mx = 0.0f, cl[3];
  for (int c = 0; c < 3; c++) {
    cl[c] = fmin(fmax(acc[c], 0.0f), lim);
    mx = fmax(mx, cl[c]);
  }
  float inv = 1.0f / (1.0f - mx);
  int p = oy * Wo + ox, n = Ho * Wo;
  for (int c = 0; c < 3; c++) {
    float lin = cl[c] * inv * ie;
    out_linear[c * n + p] = lin;
    float x = fmax(lin * e, 0.0f);
    x = satf(x * (1.0f / (1.0f + x)));  // reinhard tonemap for display
    out_rgba[p * 4 + c] = (uchar)rint(x * 255.0f);
  }
  out_rgba[p * 4 + 3] = 255;
}

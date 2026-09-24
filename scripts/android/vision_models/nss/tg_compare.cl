// Hand-written twins of the two NSS stages tg_nss.py expresses in tinygrad, with the same math and the
// same planar float layout (C, H, W) as tinygrad's graphs, one thread per pixel. Used only by tg_nss.py.
#pragma OPENCL FP_CONTRACT OFF
#define EPS 1e-07f

inline float lerpt(float a, float b, float w) {
  return fabs(w) < 0.5f ? a + w * (b - a) : b - (b - a) * (1.0f - w);
}
inline float satf(float v) { return clamp(v, 0.0f, 1.0f); }

// postprocess tail: rectify the warped history against the filtered moments, Karis blend with the centre
// sample, inverse tonemap -> linear output, plus a reinhard display copy. n = H * W.
__kernel void hw_accum(__global const float* m1, __global const float* m2, __global const float* wh,
                       __global const float* cc, __global const float* theta, __global const float* alpha,
                       __global const float* gamma, __global const float* onscreen, __global const float* cv,
                       float e, float reset, int n, __global float* lin, __global float* disp) {
  int i = get_global_id(0);
  if (i >= n) return;
  float rect[3], cen[3];
  float tw = theta[i] * onscreen[i] * reset;
  for (int c = 0; c < 3; c++) {
    float a1 = m1[c * n + i], w = wh[c * n + i];
    float var = fmax(fabs(m2[c * n + i] - a1 * a1), EPS);
    float sigma = sqrt(var) * gamma[i];
    float hcl = fmax(fmin(w, a1 + sigma), a1 - sigma);
    rect[c] = lerpt(lerpt(a1, hcl, reset), w, tw);
    cen[c] = cc[c * n + i];
  }
  float rn[3], cn[3], mr = 0.0f, mc = 0.0f;
  for (int c = 0; c < 3; c++) {
    rn[c] = fmax(rect[c], 0.0f);
    cn[c] = fmax(cen[c], 0.0f);
    mr = fmax(mr, rn[c]);
    mc = fmax(mc, cn[c]);
  }
  float sr = 1.0f / (1.0f + mr), sc = 1.0f / (1.0f + mc);
  float a = alpha[i] * cv[i] * reset;
  float lim = 65504.0f * (1.0f / (1.0f + 65504.0f));
  float cl[3], mx = 0.0f;
  for (int c = 0; c < 3; c++) {
    float acc = clamp(lerpt(satf(rn[c] * sr), satf(cn[c] * sc), a), 0.0f, 1.0f - EPS);
    cl[c] = fmin(fmax(acc, 0.0f), lim);
    mx = fmax(mx, cl[c]);
  }
  float inv = 1.0f / (1.0f - mx), ie = 1.0f / e;
  for (int c = 0; c < 3; c++) {
    float l = cl[c] * inv * ie;
    lin[c * n + i] = l;
    float x = fmax(l * e, 0.0f);
    disp[c * n + i] = satf(x * (1.0f / (1.0f + x)));
  }
}

inline float3 ycocg(__global const float* col, int H, int W, int y, int x, float e) {
  y = clamp(y, 0, H - 1);  // a 1-pixel symmetric reflect == replicate
  x = clamp(x, 0, W - 1);
  int n = H * W, p = y * W + x;
  float r = sqrt(fmax(col[p] * e, 0.0f)), g = sqrt(fmax(col[n + p] * e, 0.0f)), b = sqrt(fmax(col[2 * n + p] * e, 0.0f));
  float co = r - b, tmp = b + co * 0.5f, cg = g - tmp;
  return (float3)(tmp + cg * 0.5f, co, cg);
}
inline float ydelta(float3 a, float3 b) {
  float dl = a.x - b.x, dc = (a.y - b.y) * 1.25f, dg = (a.z - b.z) * 1.25f;
  return sqrt(dl * dl + dc * dc + dg * dg);
}

// preprocess YCoCg luma derivative / instability state (the stencil + state machine; dt = the already
// warped previous state, dis = the disocclusion mask)
__kernel void hw_deriv(__global const float* col, __global const float* dt, __global const float* dis, float e, int H,
                       int W, __global float* state, __global float* vis_out) {
  int i = get_global_id(0), n = H * W;
  if (i >= n) return;
  int y = i / W, x = i - y * W;
  float3 yc = ycocg(col, H, W, y, x, e);
  float3 d3 = (float3)(dt[i], dt[n + i], dt[2 * n + i]);
  float prev = dt[3 * n + i];
  float d_c = ydelta(yc, d3);
  float d_n = ydelta(yc, ycocg(col, H, W, y, x - 1, e)), d_s = ydelta(yc, ycocg(col, H, W, y, x + 1, e));
  float d_e = ydelta(yc, ycocg(col, H, W, y + 1, x, e)), d_w = ydelta(yc, ycocg(col, H, W, y - 1, x, e));
  float s_sum = d_n + d_s + d_e + d_w, s_max = fmax(fmax(d_n, d_s), fmax(d_e, d_w));
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
  float raw_sustain = prev * fmax(s_gate, hot * 0.12f) * 0.8f;
  float raw_inst = fmax(raw_entry, raw_sustain);
  float fall = lerpt(0.24f, 0.05f, fmax(s_gate, heat * heat * 0.5f));
  float rise = lerpt(0.08f, 0.22f, sqrt(recall * mean_gate));
  float filt = lerpt(prev, raw_inst, raw_inst > prev ? rise : fall);
  float vis = lerpt(prev, filt, filt > prev ? 0.75f : 0.8f);
  float disb = dis[i] > 0.01f ? 1.0f : 0.0f;
  float uninit = (fabs(dt[i]) + fabs(dt[n + i]) + fabs(dt[2 * n + i]) + fabs(prev)) < 0.0001f ? 1.0f : 0.0f;
  vis = vis * (1.0f - disb);
  float st[4] = {yc.x, yc.y, yc.z, filt}, rs[4] = {yc.x, yc.y, yc.z, 0.0f};
  for (int c = 0; c < 4; c++) state[c * n + i] = lerpt(lerpt(st[c], rs[c], disb), rs[c], uninit);
  vis_out[i] = lerpt(vis, 0.0f, uninit);
}

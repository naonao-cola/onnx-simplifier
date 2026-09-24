// NSS v1 per-frame filter offset LUT from the camera jitter, on the device (no host script).
//
// A translation of the gym's Apache-2.0 torch reference: graphics_utils.calculate_lr_to_hr_modulo /
// generate_lr_to_hr_tile and NSSV1Model._compute_lut (6x6 window, Gaussian sigma 2 with a tiny
// order bias, top-k valid taps). All arithmetic in float32 like the reference.
//
// Output: lut[6][tiles * taps] (dy, dx, valid, tap_dy, tap_dx, tap_channel), tiles = mod_h * mod_w.
#pragma once
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <vector>

struct NssLut {
  int mod_h, mod_w, taps;
  std::vector<float> lut;  // 6 x (mod_h * mod_w * taps)
};

inline int nss_gcd(int a, int b) { return b ? nss_gcd(b, a % b) : a; }
inline int nss_pymod(int a, int m) { return ((a % m) + m) % m; }

inline NssLut nss_offset_lut(int h_in, int w_in, int h_out, int w_out, float jy, float jx, int taps) {
  // _reduce_fraction(lr, hr) -> the smallest (lr, hr) tile with the same ratio
  int gh = nss_gcd(h_in, h_out), gw = nss_gcd(w_in, w_out);
  int h_lr = h_in / gh, h_hr = h_out / gh, w_lr = w_in / gw, w_hr = w_out / gw;
  float sy = (float)h_out / (float)h_in, sx = (float)w_out / (float)w_in;
  // generate_lr_to_hr_tile: base[4][h_hr][w_hr] = (dy, dx, valid, 0)
  std::vector<float> base(4 * h_hr * w_hr, 0.0f);
  for (int y = 0; y < h_lr; y++)
    for (int x = 0; x < w_lr; x++) {
      int hy = (int)std::floor(((float)y + jy + 0.5f) * sy);
      int hx = (int)std::floor(((float)x + jx + 0.5f) * sx);
      int ly = (int)std::floor(((float)hy + 0.5f) / sy), lx = (int)std::floor(((float)hx + 0.5f) / sx);
      if (hy < 0 || hy >= h_hr || hx < 0 || hx >= w_hr) continue;
      int i = hy * w_hr + hx;
      base[i] = (float)(y - ly);
      base[h_hr * w_hr + i] = (float)(x - lx);
      base[2 * h_hr * w_hr + i] = 1.0f;
    }
  // _compute_lut
  const int kw = 6, no = kw * kw, start = -kw / 2 + 1;
  const float sigma = kw / 3.0f, eps = FLT_EPSILON * 16.0f;
  float wts[no], tdy[no], tdx[no];
  for (int t = 0; t < no; t++) {
    int dx = start + t / kw, dy = start + t % kw;  // dx: repeat_interleave, dy: repeat
    tdy[t] = (float)dy;
    tdx[t] = (float)dx;
    float d2 = tdy[t] * tdy[t] + tdx[t] * tdx[t];
    wts[t] = std::exp(-0.5f * d2 / (sigma * sigma)) + ((float)no - (float)t) * eps;
  }
  NssLut r{h_hr, w_hr, taps, std::vector<float>(6 * h_hr * w_hr * taps, 0.0f)};
  const int tiles = h_hr * w_hr, nl = tiles * taps;
  for (int by = 0; by < h_hr; by++)
    for (int bx = 0; bx < w_hr; bx++) {
      int tile = by * w_hr + bx;
      float score[no], dyv[no], dxv[no], val[no];
      for (int t = 0; t < no; t++) {
        int ty = nss_pymod(by + (int)tdy[t], h_hr), tx = nss_pymod(bx + (int)tdx[t], w_hr);
        int li = ty * w_hr + tx;
        dyv[t] = base[li];
        dxv[t] = base[tiles + li];
        val[t] = base[2 * tiles + li];
        score[t] = wts[t] * val[t];
      }
      int ord[no];
      std::iota(ord, ord + no, 0);
      // torch.topk: descending scores; valid scores are unique (order bias), and the masked-out (zero)
      // entries become all-zero rows whichever are picked
      std::stable_sort(ord, ord + no, [&](int a, int b) { return score[a] > score[b]; });
      for (int k = 0; k < taps; k++) {
        int t = ord[k];
        float m = val[t] > 0.0f ? 1.0f : 0.0f;
        int j = tile * taps + k;
        r.lut[j] = dyv[t] * m;
        r.lut[nl + j] = dxv[t] * m;
        r.lut[2 * nl + j] = m;
        r.lut[3 * nl + j] = tdy[t] * m;
        r.lut[4 * nl + j] = tdx[t] * m;
        r.lut[5 * nl + j] = (float)t * m;
      }
    }
  return r;
}

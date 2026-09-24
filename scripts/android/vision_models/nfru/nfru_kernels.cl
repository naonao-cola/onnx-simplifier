// NFRU v1 (frame generation) pre/post-processing as OpenCL kernels (Adreno GPU; also any host OpenCL device).
//
// A fresh translation of the gym's Apache-2.0 torch reference (arm/neural-graphics-model-gym @ fc5fdaf6:
// usecases/nfru/model/{optical_flow/blockmatch_v321.py, torch_processing/*.py}, utils/color_pipeline.py);
// Arm's GLSL shaders are not used. Planar float NCHW (N = 1) unless noted; coordinates (row, col).
// fp16 steps of the reference (the block matcher's vectors and sub-pixel fit) round through half
// (vstore_half_rte). Lessons from ../nss: no dynamically indexed private arrays in hot loops, no fma().
#pragma OPENCL FP_CONTRACT OFF

#define HALF_MAX 65504.0f

// k / 255 for uint8 k, correctly rounded (the reference divides; Adreno division is not correctly rounded)
__constant float U8F[256] = {0.0f, 0.003921568859368563f, 0.007843137718737125f, 0.0117647061124444f, 0.01568627543747425f, 0.019607843831181526f, 0.0235294122248888f, 0.027450980618596077f, 0.0313725508749485f, 0.03529411926865578f, 0.03921568766236305f, 0.04313725605607033f, 0.0470588244497776f, 0.05098039284348488f, 0.054901961237192154f, 0.05882352963089943f, 0.062745101749897f, 0.06666667014360428f, 0.07058823853731155f, 0.07450980693101883f, 0.0784313753247261f, 0.08235294371843338f, 0.08627451211214066f, 0.09019608050584793f, 0.0941176488995552f, 0.09803921729326248f, 0.10196078568696976f, 0.10588235408067703f, 0.10980392247438431f, 0.11372549086809158f, 0.11764705926179886f, 0.12156862765550613f, 0.125490203499794f, 0.12941177189350128f, 0.13333334028720856f, 0.13725490868091583f, 0.1411764770746231f, 0.14509804546833038f, 0.14901961386203766f, 0.15294118225574493f, 0.1568627506494522f, 0.16078431904315948f, 0.16470588743686676f, 0.16862745583057404f, 0.1725490242242813f, 0.1764705926179886f, 0.18039216101169586f, 0.18431372940540314f, 0.1882352977991104f, 0.1921568661928177f, 0.19607843458652496f, 0.20000000298023224f, 0.20392157137393951f, 0.2078431397676468f, 0.21176470816135406f, 0.21568627655506134f, 0.21960784494876862f, 0.2235294133424759f, 0.22745098173618317f, 0.23137255012989044f, 0.23529411852359772f, 0.239215686917305f, 0.24313725531101227f, 0.24705882370471954f, 0.250980406999588f, 0.2549019753932953f, 0.25882354378700256f, 0.26274511218070984f, 0.2666666805744171f, 0.2705882489681244f, 0.27450981736183167f, 0.27843138575553894f, 0.2823529541492462f, 0.2862745225429535f, 0.29019609093666077f, 0.29411765933036804f, 0.2980392277240753f, 0.3019607961177826f, 0.30588236451148987f, 0.30980393290519714f, 0.3137255012989044f, 0.3176470696926117f, 0.32156863808631897f, 0.32549020648002625f, 0.3294117748737335f, 0.3333333432674408f, 0.33725491166114807f, 0.34117648005485535f, 0.3450980484485626f, 0.3490196168422699f, 0.3529411852359772f, 0.35686275362968445f, 0.3607843220233917f, 0.364705890417099f, 0.3686274588108063f, 0.37254902720451355f, 0.3764705955982208f, 0.3803921639919281f, 0.3843137323856354f, 0.38823530077934265f, 0.3921568691730499f, 0.3960784375667572f, 0.4000000059604645f, 0.40392157435417175f, 0.40784314274787903f, 0.4117647111415863f, 0.4156862795352936f, 0.41960784792900085f, 0.42352941632270813f, 0.4274509847164154f, 0.4313725531101227f, 0.43529412150382996f, 0.43921568989753723f, 0.4431372582912445f, 0.4470588266849518f, 0.45098039507865906f, 0.45490196347236633f, 0.4588235318660736f, 0.4627451002597809f, 0.46666666865348816f, 0.47058823704719543f, 0.4745098054409027f, 0.47843137383461f, 0.48235294222831726f, 0.48627451062202454f, 0.4901960790157318f, 0.4941176474094391f, 0.49803921580314636f, 0.501960813999176f, 0.5058823823928833f, 0.5098039507865906f, 0.5137255191802979f, 0.5176470875740051f, 0.5215686559677124f, 0.5254902243614197f, 0.529411792755127f, 0.5333333611488342f, 0.5372549295425415f, 0.5411764979362488f, 0.545098066329956f, 0.5490196347236633f, 0.5529412031173706f, 0.5568627715110779f, 0.5607843399047852f, 0.5647059082984924f, 0.5686274766921997f, 0.572549045085907f, 0.5764706134796143f, 0.5803921818733215f, 0.5843137502670288f, 0.5882353186607361f, 0.5921568870544434f, 0.5960784554481506f, 0.6000000238418579f, 0.6039215922355652f, 0.6078431606292725f, 0.6117647290229797f, 0.615686297416687f, 0.6196078658103943f, 0.6235294342041016f, 0.6274510025978088f, 0.6313725709915161f, 0.6352941393852234f, 0.6392157077789307f, 0.6431372761726379f, 0.6470588445663452f, 0.6509804129600525f, 0.6549019813537598f, 0.658823549747467f, 0.6627451181411743f, 0.6666666865348816f, 0.6705882549285889f, 0.6745098233222961f, 0.6784313917160034f, 0.6823529601097107f, 0.686274528503418f, 0.6901960968971252f, 0.6941176652908325f, 0.6980392336845398f, 0.7019608020782471f, 0.7058823704719543f, 0.7098039388656616f, 0.7137255072593689f, 0.7176470756530762f, 0.7215686440467834f, 0.7254902124404907f, 0.729411780834198f, 0.7333333492279053f, 0.7372549176216125f, 0.7411764860153198f, 0.7450980544090271f, 0.7490196228027344f, 0.7529411911964417f, 0.7568627595901489f, 0.7607843279838562f, 0.7647058963775635f, 0.7686274647712708f, 0.772549033164978f, 0.7764706015586853f, 0.7803921699523926f, 0.7843137383460999f, 0.7882353067398071f, 0.7921568751335144f, 0.7960784435272217f, 0.800000011920929f, 0.8039215803146362f, 0.8078431487083435f, 0.8117647171020508f, 0.8156862854957581f, 0.8196078538894653f, 0.8235294222831726f, 0.8274509906768799f, 0.8313725590705872f, 0.8352941274642944f, 0.8392156958580017f, 0.843137264251709f, 0.8470588326454163f, 0.8509804010391235f, 0.8549019694328308f, 0.8588235378265381f, 0.8627451062202454f, 0.8666666746139526f, 0.8705882430076599f, 0.8745098114013672f, 0.8784313797950745f, 0.8823529481887817f, 0.886274516582489f, 0.8901960849761963f, 0.8941176533699036f, 0.8980392217636108f, 0.9019607901573181f, 0.9058823585510254f, 0.9098039269447327f, 0.9137254953384399f, 0.9176470637321472f, 0.9215686321258545f, 0.9254902005195618f, 0.929411768913269f, 0.9333333373069763f, 0.9372549057006836f, 0.9411764740943909f, 0.9450980424880981f, 0.9490196108818054f, 0.9529411792755127f, 0.95686274766922f, 0.9607843160629272f, 0.9647058844566345f, 0.9686274528503418f, 0.9725490212440491f, 0.9764705896377563f, 0.9803921580314636f, 0.9843137264251709f, 0.9882352948188782f, 0.9921568632125854f, 0.9960784316062927f, 1.0f};
inline int clampi(int v, int lo, int hi) { return min(max(v, lo), hi); }
inline float rh(float x) {  // round to fp16 (round to nearest even) and back
  ushort u;
  vstore_half_rte(x, 0, (__private half*)&u);
  return vload_half(0, (__private const half*)&u);
}
inline float to_u8f(float x) { return clamp(rint(x * 255.0f), 0.0f, 255.0f); }  // cast(float -> uint8)

// ---------------------------------------------------------------------------------------------------------
// colour pipeline (test split: rgb * exp(2), clamp to half max, reinhard) + block-matching luma (uint8)
__kernel void colour_luma(__global const float* lin, float expo, int n, __global float* rgb, __global uchar* y8) {
  int i = get_global_id(0);
  if (i >= n) return;
  float c[3];
  for (int k = 0; k < 3; k++) {
    float x = clamp(lin[k * n + i] * expo, 0.0f, HALF_MAX);
    x = fmax(x, 0.0f);
    x = clamp(x * (1.0f / (1.0f + x)), 0.0f, 1.0f);
    c[k] = x;
    rgb[k * n + i] = x;
  }
  float yv = 0.25f * ((c[0] + 2.0f * c[1]) + c[2]);
  y8[i] = (uchar)to_u8f(yv);
}

// the luma alone, from an already colour-processed image
__kernel void luma8(__global const float* rgb, int n, __global uchar* y8) {
  int i = get_global_id(0);
  if (i >= n) return;
  y8[i] = (uchar)to_u8f(0.25f * ((rgb[i] + 2.0f * rgb[n + i]) + rgb[2 * n + i]));
}

// ---------------------------------------------------------------------------------------------------------
// pyramid: one level down (optional binomial blur, bilinear x0.5), cast to uint8, replicate-pad to even
inline float px8(__global const uchar* s, int H, int W, int y, int x) {
  return U8F[s[clampi(y, 0, H - 1) * W + clampi(x, 0, W - 1)]];
}
inline float blur_at(__global const uchar* s, int H, int W, int y, int x) {  // binomial_filter, replicate
  float v[3];
  for (int d = -1; d <= 1; d++)
    v[d + 1] = ((px8(s, H, W, y - 1, x + d) + 2.0f * px8(s, H, W, y, x + d)) + px8(s, H, W, y + 1, x + d)) / 4.0f;
  return ((v[0] + 2.0f * v[1]) + v[2]) / 4.0f;
}
__kernel void pyr_down(__global const uchar* s, int H, int W, int blur, int quad, __global uchar* o, int Ho, int Wo,
                       int Hd, int Wd) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= Ho || x >= Wo) return;
  int sy = min(y, Hd - 1), sx = min(x, Wd - 1);  // (Ho, Wo) = (Hd, Wd) padded to even by replication
  float a, b, c, d;
  if (blur) {
    a = blur_at(s, H, W, 2 * sy, 2 * sx), b = blur_at(s, H, W, 2 * sy, 2 * sx + 1);
    c = blur_at(s, H, W, 2 * sy + 1, 2 * sx), d = blur_at(s, H, W, 2 * sy + 1, 2 * sx + 1);
  } else {
    a = px8(s, H, W, 2 * sy, 2 * sx), b = px8(s, H, W, 2 * sy, 2 * sx + 1);
    c = px8(s, H, W, 2 * sy + 1, 2 * sx), d = px8(s, H, W, 2 * sy + 1, 2 * sx + 1);
  }
  // F.interpolate(x0.5, bilinear) on the CPU rounds two ways depending on its code path (a size heuristic):
  // row-separable, or the four weighted taps summed -- the reference takes the latter for 68x120 -> 34x60 only
  float v = quad ? ((0.25f * a + 0.25f * b) + 0.25f * c) + 0.25f * d
                 : 0.5f * (0.5f * a + 0.5f * b) + 0.5f * (0.5f * c + 0.5f * d);
  o[y * Wo + x] = (uchar)to_u8f(v);
}

// ---------------------------------------------------------------------------------------------------------
// block matching, one pyramid level
// vector_prev: bilinear x2 upsample (align_corners=False) of the previous level's (cropped) vectors, * 2
__kernel void bm_upsample(__global const float* vin, int Hi, int Wi, __global float* vp, int H, int W) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  float sy = fmax(0.5f * ((float)y + 0.5f) - 0.5f, 0.0f), sx = fmax(0.5f * ((float)x + 0.5f) - 0.5f, 0.0f);
  int y0 = (int)sy, x0 = (int)sx, y1 = y0 + (y0 < Hi - 1), x1 = x0 + (x0 < Wi - 1);
  float ly1 = sy - y0, lx1 = sx - x0, ly0 = 1.0f - ly1, lx0 = 1.0f - lx1;
  for (int c = 0; c < 2; c++) {
    __global const float* p = vin + c * Hi * Wi;
    float r0 = lx0 * p[y0 * Wi + x0] + lx1 * p[y0 * Wi + x1];
    float r1 = lx0 * p[y1 * Wi + x0] + lx1 * p[y1 * Wi + x1];
    vp[c * H * W + y * W + x] = (ly0 * r0 + ly1 * r1) * 2.0f;
  }
}

// dense_image_warp (grid_sample bilinear, border): out(p) = img(p - flow(p)); uint8 in/out. The flow is
// (row, col) planar at stride fW (fH x fW >= H x W: the hint flow is a crop of a larger field).
__kernel void bm_warp(__global const uchar* img, int H, int W, __global const float* flow, int fH, int fW,
                      __global uchar* out) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  float fy = flow[y * fW + x], fx = flow[fH * fW + y * fW + x];
  float qx = (float)x - fx, qy = (float)y - fy;
  float gu = (qx + 0.5f) / (float)W * 2.0f - 1.0f, gv = (qy + 0.5f) / (float)H * 2.0f - 1.0f;
  float ix = (gu + 1.0f) * ((float)W / 2.0f) - 0.5f, iy = (gv + 1.0f) * ((float)H / 2.0f) - 0.5f;
  ix = clamp(ix, 0.0f, (float)(W - 1));
  iy = clamp(iy, 0.0f, (float)(H - 1));
  float x0f = floor(ix), y0f = floor(iy);
  int x0 = (int)x0f, y0 = (int)y0f, x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
  float wx = ix - x0f, wy = iy - y0f;
  float nw = (1.0f - wx) * (1.0f - wy), ne = wx * (1.0f - wy), sw = (1.0f - wx) * wy, se = wx * wy;
  float v = px8(img, H, W, y0, x0) * nw + px8(img, H, W, y0, x1) * ne + px8(img, H, W, y1, x0) * sw +
            px8(img, H, W, y1, x1) * se;
  out[y * W + x] = (uchar)to_u8f(v);
}

inline int u8z(__global const uchar* s, int H, int W, int y, int x) {  // zero padding
  return (y >= 0 && y < H && x >= 0 && x < W) ? (int)s[y * W + x] : 0;
}
// spiral order of the 7x7 search window (ArgMinCentered 'spiral': ties go to the lowest spiral number)
__constant uchar SPIRAL[49] = {44, 43, 42, 41, 40, 39, 38, 45, 22, 21, 20, 19, 18, 37, 46, 23, 8,
                               7,  6,  17, 36, 47, 24, 9,  1,  5,  16, 35, 48, 25, 2,  3,  4,  15,
                               34, 49, 10, 11, 12, 13, 14, 33, 26, 27, 28, 29, 30, 31, 32};

// SAD over 5x5 templates for the 7x7 displacements (+ the warped motion-vector hint at the target level),
// spiral argmin, least-squares sub-pixel refinement (fp16 like the reference), + vector_prev.
// out: the vector before the median (fp16 values) and the hint mask (1 = the hint won).
__kernel void bm_match(__global const uchar* srch, __global const uchar* tmpl, __global const uchar* hint, int H,
                       int W, __global const float* vp, __global float* vec, __global uchar* hint_won) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  int best = 0x7fffffff, bi = 0, bs = 99;
  for (int sy = 0; sy < 7; sy++)
    for (int sx = 0; sx < 7; sx++) {
      int cost = 0;
      for (int ky = -2; ky <= 2; ky++)
        for (int kx = -2; kx <= 2; kx++)
          cost += abs(u8z(srch, H, W, y + sy - 3 + ky, x + sx - 3 + kx) - u8z(tmpl, H, W, y + ky, x + kx));
      int c = sy * 7 + sx, sp = SPIRAL[c];
      if (cost < best || (cost == best && sp < bs)) {
        best = cost;
        bi = c;
        bs = sp;
      }
    }
  int use_hint = 0;
  if (hint) {
    int hc = 0;
    for (int ky = -2; ky <= 2; ky++)
      for (int kx = -2; kx <= 2; kx++) hc += abs(u8z(hint, H, W, y + ky, x + kx) - u8z(tmpl, H, W, y + ky, x + kx));
    use_hint = hc < best;
  }
  int dy = bi / 7 - 3, dx = bi % 7 - 3;  // the chosen candidate's source (the hint, if it won)
  // calculate_subpixel: f = template, g = the chosen candidate, gradients of the template image; border
  // pixels of the 5x5 only (pred mask), fp16 products, fp32 sums rounded to fp16
  float a = 0.0f, b = 0.0f, d = 0.0f, p = 0.0f, q = 0.0f;
  for (int ky = -2; ky <= 2; ky++)
    for (int kx = -2; kx <= 2; kx++) {
      if (ky > -2 && ky < 2 && kx > -2 && kx < 2) continue;
      int ty = y + ky, tx = x + kx;
      float gy = 0.0f, gx = 0.0f;
      if (ty >= 0 && ty < H && tx >= 0 && tx < W) {  // gradient (replicate pad) of the fp16 image
        float up = rh(U8F[tmpl[clampi(ty - 1, 0, H - 1) * W + tx]]);
        float dn = rh(U8F[tmpl[clampi(ty + 1, 0, H - 1) * W + tx]]);
        float lf = rh(U8F[tmpl[ty * W + clampi(tx - 1, 0, W - 1)]]);
        float rt = rh(U8F[tmpl[ty * W + clampi(tx + 1, 0, W - 1)]]);
        gy = rh(rh(dn - up) / 2.0f);
        gx = rh(rh(rt - lf) / 2.0f);
      }
      float fv = rh(U8F[u8z(tmpl, H, W, ty, tx)]);
      int gvi = use_hint ? u8z(hint, H, W, ty, tx) : u8z(srch, H, W, y + dy + ky, x + dx + kx);
      float z = rh(rh(U8F[gvi]) - fv);
      a += rh(gx * gx);
      b += rh(gx * gy);
      d += rh(gy * gy);
      p += rh(z * gx);
      q += rh(z * gy);
    }
  a = rh(a), b = rh(b), d = rh(d), p = rh(p), q = rh(q);
  float det = a * d - b * b;
  float su = d * p - b * q, sv = a * q - b * p;
  float s0 = sv / (det + 0.0f), s1 = su / (det + 0.0f);
  s0 = (det <= 1e-7f || fabs(s0) >= 1.0f) ? 0.0f : s0;
  s1 = (det <= 1e-7f || fabs(s1) >= 1.0f) ? 0.0f : s1;
  float v0 = rh((float)(-dy) + rh(s0)), v1 = rh((float)(-dx) + rh(s1));
  if (vp) {
    v0 = rh(v0 + rh(vp[y * W + x]));
    v1 = rh(v1 + rh(vp[H * W + y * W + x]));
  }
  vec[y * W + x] = v0;
  vec[H * W + y * W + x] = v1;
  hint_won[y * W + x] = (uchar)use_hint;
}

// 3x3 median (replicate), per channel
__kernel void bm_median(__global const float* vin, int H, int W, __global float* vout) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  for (int c = 0; c < 2; c++) {
    __global const float* p = vin + c * H * W;
    float v0 = p[clampi(y - 1, 0, H - 1) * W + clampi(x - 1, 0, W - 1)], v1 = p[clampi(y - 1, 0, H - 1) * W + x],
          v2 = p[clampi(y - 1, 0, H - 1) * W + clampi(x + 1, 0, W - 1)], v3 = p[y * W + clampi(x - 1, 0, W - 1)],
          v4 = p[y * W + x], v5 = p[y * W + clampi(x + 1, 0, W - 1)],
          v6 = p[clampi(y + 1, 0, H - 1) * W + clampi(x - 1, 0, W - 1)], v7 = p[clampi(y + 1, 0, H - 1) * W + x],
          v8 = p[clampi(y + 1, 0, H - 1) * W + clampi(x + 1, 0, W - 1)];
#define SRT(a, b)       \
  {                     \
    float t = fmin(a, b); \
    b = fmax(a, b);     \
    a = t;              \
  }
    // Paeth's 9-element median network
    SRT(v1, v2) SRT(v4, v5) SRT(v7, v8) SRT(v0, v1) SRT(v3, v4) SRT(v6, v7) SRT(v1, v2) SRT(v4, v5) SRT(v7, v8)
    SRT(v0, v3) SRT(v5, v8) SRT(v4, v7) SRT(v3, v6) SRT(v1, v4) SRT(v2, v5) SRT(v4, v7) SRT(v4, v2) SRT(v6, v4)
    SRT(v4, v2)
    vout[c * H * W + y * W + x] = v4;
  }
}

// joint bilateral 5x5 (zero-padded blocks, like F.unfold), then at the target level the hint replace, and
// the crop to the level's true size (Ht x Wt <= H x W): out is Ht x Wt
__kernel void bm_jbf(__global const float* vin, __global const uchar* tmpl, int H, int W, __global const uchar* hint_won,
                     __global const float* hint_mv, int hH, int hW, __global float* vout, int Ht, int Wt) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= Ht || x >= Wt) return;
  const float kern = -0.009130752645432949f, sid = 25.510204315185547f;
  float cp = U8F[tmpl[y * W + x]];
  float n0 = 0.0f, n1 = 0.0f, den = 0.0f;
  for (int ky = -2; ky <= 2; ky++)
    for (int kx = -2; kx <= 2; kx++) {
      int yy = y + ky, xx = x + kx;
      int in = yy >= 0 && yy < H && xx >= 0 && xx < W;
      float ck = in ? U8F[tmpl[yy * W + xx]] : 0.0f;
      float diff = (cp - ck) * (cp - ck);
      float co = clamp(1.0f - fabs(kern - diff * sid), 0.0f, 1.0f);
      float a = in ? vin[yy * W + xx] : 0.0f, b = in ? vin[H * W + yy * W + xx] : 0.0f;
      n0 += a * co;
      n1 += b * co;
      den += co;
    }
  float o0 = rh(n0 / den), o1 = rh(n1 / den);
  if (hint_won && hint_won[y * W + x]) {
    o0 = hint_mv[y * hW + x];
    o1 = hint_mv[hH * hW + y * hW + x];
  }
  vout[y * Wt + x] = o0;
  vout[Ht * Wt + y * Wt + x] = o1;
}

// the motion-vector hint: upscale_and_dilate_flow(sy_m1_f30_p1, depth_m1) (3x3 reflect window, the flow at
// the nearest depth), / 4 (granularity), * -1 (polarity), fp16 -- only the top-left H x W crop is used
inline int refl(int v, int n) { return v < 0 ? -v : (v >= n ? 2 * n - 2 - v : v); }
__kernel void hint_mv(__global const float* sy, __global const float* depth, int H, int W, __global float* out,
                      int Ho, int Wo) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= Ho || x >= Wo) return;
  float best = INFINITY;
  int by = 0, bx = 0;
  for (int ky = -1; ky <= 1; ky++)
    for (int kx = -1; kx <= 1; kx++) {
      int yy = refl(y + ky, H), xx = refl(x + kx, W);
      float d = depth[yy * W + xx];
      if (d < best) {
        best = d;
        by = yy;
        bx = xx;
      }
    }
  out[y * Wo + x] = rh(sy[by * W + bx] / 4.0f * -1.0f);
  out[Ho * Wo + y * Wo + x] = rh(sy[H * W + by * W + bx] / 4.0f * -1.0f);
}

// ---------------------------------------------------------------------------------------------------------
// motion (540p / 270p)
typedef struct {
  float4 r0, r1, r2, r3;
} Mat4;
inline float4 mulm(Mat4 m, float4 v) {
  return (float4)(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v), dot(m.r3, v));
}
inline float4 mv4(__constant float* m, float4 v) {  // row-major 4x4 times column vector
  return (float4)(((m[0] * v.x + m[1] * v.y) + m[2] * v.z) + m[3] * v.w,
                  ((m[4] * v.x + m[5] * v.y) + m[6] * v.z) + m[7] * v.w,
                  ((m[8] * v.x + m[9] * v.y) + m[10] * v.z) + m[11] * v.w,
                  ((m[12] * v.x + m[13] * v.y) + m[14] * v.z) + m[15] * v.w);
}
// _calculate_camera_motion: (motion (row, col) in uv units, invalid)
inline float2 cam_motion(__constant float* m, float uy, float ux, float depth, float* invalid) {
  float tx = ux, ty = 1.0f - uy;
  float4 r = mv4(m, (float4)(2.0f * tx - 1.0f, 2.0f * ty - 1.0f, depth, 1.0f));
  *invalid = r.w < 0.0f ? 1.0f : 0.0f;
  float px = (r.x / r.w + 1.0f) * 0.5f, py = (r.y / r.w + 1.0f) * 0.5f;
  float vx = tx - px, vy = ty - py;
  int zero = fabs(vx) < 1e-05f && fabs(vy) < 1e-05f;
  int bad = isnan(vx) || isnan(vy) || isinf(vx) || isinf(vy);
  if (zero || bad) vx = vy = 0.0f;
  return (float2)(vy, -vx);
}
inline float len2(float2 v) { return sqrt(v.x * v.x + v.y * v.y); }

// previous_dynamic_mask (not runtime-accurate): 1 where the rendered motion disagrees with camera motion
__kernel void dyn_mask(__global const float* depth, __global const float* mv, __constant float* m, int H, int W,
                       __global float* out) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  float uy = ((float)y + 0.5f) * (1.0f / (float)H), ux = ((float)x + 0.5f) * (1.0f / (float)W);
  float inv;
  float2 cm = cam_motion(m, uy, ux, 1.0f - depth[y * W + x], &inv);
  float2 rm = (float2)(mv[y * W + x], mv[H * W + y * W + x]);
  float diff = len2(cm - rm);
  float den = fmax(fmax(len2(cm), len2(rm)), 0.001f);
  out[y * W + x] = diff / den >= 0.01f ? 1.0f : 0.0f;
}

// pack_depth_motion / decode_motion (14-bit motion, 4-bit exponent, 11-bit depth; larger = nearer)
inline float symceil(float v) { return v > 0.0f ? ceil(v) : (v < 0.0f ? -ceil(-v) : 0.0f); }
inline int pack_dm(float m0, float m1, float depth) {
  const float sc = 1.601466370e+01f;  // float32(16383 / 1023)
  int ix = (int)clamp(symceil(m0 * sc), -16383.0f, 16383.0f), iy = (int)clamp(symceil(m1 * sc), -16383.0f, 16383.0f);
  int dcode = (int)floor(depth * 1.860909058e+03f + 0.5f);
  int ax = (int)abs(ix), ay = (int)abs(iy), am = max(ax, ay);
  int e = am == 0 ? 0 : min(31 - (int)clz(am), 15);
  int sh = max(e - 6, 0);
  int mx = min(ax >> sh, 127), my = min(ay >> sh, 127);
  int code = (dcode & 2047) << 20;
  code |= (e & 15) << 16;
  code |= (ix < 0) << 15;
  code |= (iy < 0) << 14;
  code |= (mx & 127) << 7;
  code |= my & 127;
  return code & 0x7fffffff;
}
inline float2 decode_dm(int code) {
  int v = code & 0x7fffffff;
  int e = (v >> 16) & 15, sh = max(e - 6, 0);
  int ix = (((v >> 7) & 127) << sh) * (1 - 2 * ((v >> 15) & 1)), iy = ((v & 127) << sh) * (1 - 2 * ((v >> 14) & 1));
  const float inv = 6.244277582e-02f;  // float32(1023 / 16383)
  return (float2)((float)ix * inv, (float)iy * inv);
}
inline float norm_inv_depth(float d, int y, int x, int H, int W) {  // _normalize_and_invert_depth
  float nd = 1.0f - log(-(1200.0f * (d - 1.0f)) + 1.0f) / 7.090909958e+00f /* log(1201) */;
  if (nd == 1.0f) nd = nd + (float)(y + x * H) / (float)(H * W) * 0.1f;
  return 1.1f - nd;
}
inline void scatter(__global int* packed, int H, int W, int y, int x, float2 vec, float t, int code) {
  int dy = y + (int)floor(vec.x * t), dx = x + (int)floor(vec.y * t);
  if (dy >= 0 && dy < H && dx >= 0 && dx < W) atomic_max(&packed[dy * W + dx], code);
}
__kernel void zero_i32(__global int* p) { p[get_global_id(0)] = 0; }

// warp_mv: splat the rendered motion (timestep 1 - t) and the camera motion of static pixels (t) into
// packed depth/motion codes; mark the hole targets of the nearest-depth motion at 1 - t and 1
// (NFRUv1Core passes current = frame p1's depth, previous = frame m1's, mv = mv_p1_f30_m1)
__kernel void warp_mv(__global const float* depth_cur, __global const float* depth_prev, __global const float* mv,
                      __global const float* dyn, __constant float* m_prev_cur, int H, int W, float t,
                      __global int* packed, __global int* holes_t, __global int* holes_m1) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  float uy = ((float)y + 0.5f) * (1.0f / (float)H), ux = ((float)x + 0.5f) * (1.0f / (float)W);
  float dc = depth_cur[y * W + x], dp = depth_prev[y * W + x];
  float2 mo = (float2)(mv[y * W + x], mv[H * W + y * W + x]);
  float2 vec = (float2)(mo.x * (float)H, mo.y * (float)W);
  scatter(packed, H, W, y, x, vec, 1.0f - t, pack_dm(vec.x, vec.y, norm_inv_depth(dc, y, x, H, W)));
  // ordered_nearest_depth (9 offsets, clamped, strictly nearer)
  float nd = depth_cur[y * W + x];
  int oy = 0, ox = 0;
#define NEAR(a, b)                                                             \
  {                                                                            \
    float cd = depth_cur[clampi(y + (a), 0, H - 1) * W + clampi(x + (b), 0, W - 1)]; \
    if (cd < nd) {                                                             \
      nd = cd;                                                                 \
      oy = (a);                                                                \
      ox = (b);                                                                \
    }                                                                          \
  }
  NEAR(1, 0) NEAR(0, 1) NEAR(0, -1) NEAR(-1, 0) NEAR(-1, 1) NEAR(1, 1) NEAR(-1, -1) NEAR(1, -1)
  int sy = y + oy, sx = x + ox;
  int ins = sy >= 0 && sy < H && sx >= 0 && sx < W;
  float2 nm = ins ? (float2)(mv[sy * W + sx], mv[H * W + sy * W + sx]) : (float2)(0.0f);
  int hy = y + (int)floor(nm.x * (float)H * (1.0f - t)), hx = x + (int)floor(nm.y * (float)W * (1.0f - t));
  if (hy >= 0 && hy < H && hx >= 0 && hx < W) atomic_max(&holes_t[hy * W + hx], 1);
  hy = y + (int)floor(nm.x * (float)H), hx = x + (int)floor(nm.y * (float)W);
  if (hy >= 0 && hy < H && hx >= 0 && hx < W) atomic_max(&holes_m1[hy * W + hx], 1);
  // static pixels of the previous frame, by camera motion
  float inv_prev;
  float2 cp = cam_motion(m_prev_cur, uy, ux, 1.0f - dp, &inv_prev);
  if (dyn[y * W + x] + inv_prev <= 0.0f) {
    float2 v2 = (float2)(cp.x * (float)H, cp.y * (float)W);
    scatter(packed, H, W, y, x, v2, t, pack_dm(-v2.x, -v2.y, norm_inv_depth(dp, y, x, H, W)));
  }
}

// warp_flow: splat the optical flow (timestep t) with the depth of the previous frame (depth at dH x dW)
__kernel void warp_flow(__global const float* depth, int dH, int dW, __global const float* flow, int H, int W, float t,
                        __global int* packed) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  float uy = ((float)y + 0.5f) * (1.0f / (float)H), ux = ((float)x + 0.5f) * (1.0f / (float)W);
  int dy = (int)floor(uy * (float)dH), dx = (int)floor(ux * (float)dW);
  float d = depth[dy * dW + dx];
  float2 vec = (float2)(flow[y * W + x] * (float)H, flow[H * W + y * W + x] * (float)W);
  scatter(packed, H, W, y, x, vec, t, pack_dm(vec.x, vec.y, norm_inv_depth(d, y, x, H, W)));
}

// normalize_mvs: (row, col) motion (times mul) / (H, W)
__kernel void norm_mv(__global const float* in, int n, float mul, float dh, float dw, __global float* out) {
  int i = get_global_id(0);
  if (i >= n) return;
  out[i] = in[i] * mul / dh;
  out[n + i] = in[n + i] * mul / dw;
}

// fill: the largest (nearest) code in the 3x3 neighbourhood (ordered), decoded to uv units
__kernel void fill_mv(__global const int* packed, int H, int W, __global float* out) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  int best = packed[y * W + x];
#define PK(a, b) best = max(best, packed[clampi(y + (a), 0, H - 1) * W + clampi(x + (b), 0, W - 1)]);
  PK(1, 0) PK(0, 1) PK(0, -1) PK(-1, 0) PK(-1, 1) PK(1, 1) PK(-1, -1) PK(1, -1)
  float2 m = decode_dm(best);
  out[y * W + x] = m.x * (1.0f / (float)H);
  out[H * W + y * W + x] = m.y * (1.0f / (float)W);
}

// ---------------------------------------------------------------------------------------------------------
// preprocess (270p): 16 network inputs, uint8 NHWC (x 255, rounded) for the HTP; netf (planar float) optional
inline float hash01(int seed_base, int ch, uint seed) {
  uint v = (uint)seed_base + 2654435769u * (uint)ch + seed;
  v = (v ^ 61u) ^ (v >> 16);
  v = v * 9u;
  v = v ^ (v >> 4);
  v = v * 668265261u;
  v = v ^ (v >> 15);
  return as_float((v >> 9) | 1065353216u) - 1.0f;
}
inline int oob(float uy, float ux) { return uy <= 0.0f || uy >= 1.0f || ux <= 0.0f || ux >= 1.0f; }
inline float bil1(__global const float* s, int H, int W, float uy, float ux) {  // sampling.bilinear_sample
  float py = uy * (float)H, px = ux * (float)W;
  float gy0 = floor(py - 0.5f), gx0 = floor(px - 0.5f), gy1 = gy0 + 1.0f, gx1 = gx0 + 1.0f;
  float wy0 = fmax(1.0f - fabs(gy0 + 0.5f - py), 0.0f), wx0 = fmax(1.0f - fabs(gx0 + 0.5f - px), 0.0f);
  float wy1 = fmax(1.0f - fabs(gy1 + 0.5f - py), 0.0f), wx1 = fmax(1.0f - fabs(gx1 + 0.5f - px), 0.0f);
  int y0 = clampi((int)gy0, 0, H - 1), x0 = clampi((int)gx0, 0, W - 1), y1 = clampi((int)gy1, 0, H - 1),
      x1 = clampi((int)gx1, 0, W - 1);
  return s[y0 * W + x0] * wy0 * wx0 + s[y0 * W + x1] * wy0 * wx1 + s[y1 * W + x0] * wy1 * wx0 +
         s[y1 * W + x1] * wy1 * wx1;
}
inline float gz(__global const float* s, int H, int W, int y, int x) {
  return (y >= 0 && y < H && x >= 0 && x < W) ? s[y * W + x] : 0.0f;
}
inline float view_depth(float d, float4 p) { return p.y / (d - p.x); }
inline float depth_clip1(float cur, float prev, float4 p, float H, float W) {  // _single_tap_depth_clip
  float cv = view_depth(cur, p), pv = view_depth(prev, p);
  float diff = cv - pv;
  float plane = fmax(prev, cur);
  float z = view_depth(plane, p);
  float cy = floor(H * 0.5f), cx = floor(W * 0.5f);
  float c0 = p.z * (cy / H * 2.0f + -1.0f) * z, c1 = p.w * (cx / W * -2.0f + 1.0f) * z;
  float k0 = p.z * (0.0f / H * 2.0f + -1.0f) * z, k1 = p.w * (0.0f / W * -2.0f + 1.0f) * z;
  float vl = sqrt(H * H + W * W);
  float thr = fmax(cv, pv);
  float fov = sqrt((k0 * k0 + k1 * k1) + z * z) / sqrt((c0 * c0 + c1 * c1) + z * z);
  float req = 1.37e-05f * fov * vl * thr;
  float rf = clamp(vl / sqrt(1080.0f * 1080.0f + 1920.0f * 1920.0f), 0.0f, 1.0f);
  float pw = 1.0f + (3.0f - 1.0f) * rf;
  float r = 1.0f - pow(clamp(req / diff, 0.0f, 1.0f), pw);
  return diff <= 0.0f ? 0.0f : r;
}
inline float reproj_depth(__constant float* m, float uy, float ux, float d) {
  float4 r = mv4(m, (float4)(2.0f * ux - 1.0f, 2.0f * (1.0f - uy) - 1.0f, d, 1.0f));
  return 1.0f - r.z / r.w;
}
inline float norm_depth(float d) { return 1.0f - log(-(1200.0f * (d - 1.0f)) + 1.0f) / 7.090909958e+00f /* log(1201) */; }

__kernel void preprocess(__global const float* flow_t, __global const float* mv_t, int mH, int mW,
                         __global const float* rgb_m1, __global const float* rgb_p1, int cH, int cW,
                         __global const float* depth_m1, __global const float* depth_p1,
                         __global const int* holes_t, __global const int* holes_m1, __constant float* m_m1p1,
                         __constant float* m_p1m1, float4 dparams, float t, uint seed, int H, int W,
                         __global float* netf, __global uchar* net_u8) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  float uy = ((float)y + 0.5f) * (1.0f / (float)H), ux = ((float)x + 0.5f) * (1.0f / (float)W);
  int my = (int)floor(uy * (float)mH), mx = (int)floor(ux * (float)mW);
  float fy = flow_t[y * W + x], fx = flow_t[H * W + y * W + x];  // flow is at H x W
  float vy = gz(mv_t, mH, mW, my, mx), vx = gz(mv_t + mH * mW, mH, mW, my, mx);
  float a_y = uy + vy * t, a_x = ux + vx * t;                      // m1, mv
  float b_y = uy - vy * (1.0f - t), b_x = ux - vx * (1.0f - t);    // p1, mv
  float c_y = uy - fy * t, c_x = ux - fx * t;                      // m1, flow
  float d_y = uy + fy * (1.0f - t), d_x = ux + fx * (1.0f - t);    // p1, flow
  int o_a = oob(a_y, a_x), o_b = oob(b_y, b_x), o_c = oob(c_y, c_x), o_d = oob(d_y, d_x);
  // holes: bilinear of the m1 hole markers at the m1 position, the t markers at this pixel
  float hm = 0.0f;
  {
    float py = a_y * (float)mH, px = a_x * (float)mW;
    float gy0 = floor(py - 0.5f), gx0 = floor(px - 0.5f), gy1 = gy0 + 1.0f, gx1 = gx0 + 1.0f;
    float wy0 = fmax(1.0f - fabs(gy0 + 0.5f - py), 0.0f), wx0 = fmax(1.0f - fabs(gx0 + 0.5f - px), 0.0f);
    float wy1 = fmax(1.0f - fabs(gy1 + 0.5f - py), 0.0f), wx1 = fmax(1.0f - fabs(gx1 + 0.5f - px), 0.0f);
    int y0 = clampi((int)gy0, 0, mH - 1), x0 = clampi((int)gx0, 0, mW - 1), y1 = clampi((int)gy1, 0, mH - 1),
        x1 = clampi((int)gx1, 0, mW - 1);
    hm = (float)holes_m1[y0 * mW + x0] * wy0 * wx0 + (float)holes_m1[y0 * mW + x1] * wy0 * wx1 +
         (float)holes_m1[y1 * mW + x0] * wy1 * wx0 + (float)holes_m1[y1 * mW + x1] * wy1 * wx1;
  }
  float hole_m1 = (hm != 0.0f ? 1.0f : 0.0f) * (o_b ? 0.0f : 1.0f);
  float hole_t = ((my >= 0 && my < mH && mx >= 0 && mx < mW) ? (float)holes_t[my * mW + mx] : 0.0f) * (o_b ? 0.0f : 1.0f);
  float dd = (hole_t - hole_m1 < 0.0f) ? 1.0f : 0.0f;
  float o[16];
  int sb = y * 10000 + x;
#define COL(g, src, qy, qx, ob)                                                        \
  {                                                                                    \
    int cy = (int)floor((qy) * (float)cH), cx = (int)floor((qx) * (float)cW);          \
    for (int k = 0; k < 3; k++)                                                        \
      o[(g) * 3 + k] = (ob) ? hash01(sb, (g) * 3 + k, seed) : gz(src + k * cH * cW, cH, cW, cy, cx); \
  }
  COL(0, rgb_m1, a_y, a_x, o_a)
  COL(1, rgb_p1, b_y, b_x, o_b)
  COL(2, rgb_m1, c_y, c_x, o_c)
  COL(3, rgb_p1, d_y, d_x, o_d)
  float dm = gz(depth_m1, mH, mW, (int)floor(a_y * (float)mH), (int)floor(a_x * (float)mW));
  float dp = gz(depth_p1, mH, mW, (int)floor(b_y * (float)mH), (int)floor(b_x * (float)mW));
  o[12] = norm_depth(dm);
  o[13] = norm_depth(dp);
  float dm_t = reproj_depth(m_m1p1, b_y, b_x, 1.0f - dm);
  float dp_t = reproj_depth(m_p1m1, a_y, a_x, 1.0f - dp);
  o[14] = clamp(depth_clip1(dp, dm_t, dparams, (float)mH, (float)mW) + dd, 0.0f, 1.0f);
  o[15] = clamp(depth_clip1(dm, dp_t, dparams, (float)mH, (float)mW) + dd, 0.0f, 1.0f);
  int n = H * W, pix = y * W + x;
  for (int k = 0; k < 16; k++) {
    if (netf) netf[k * n + pix] = o[k];
    net_u8[pix * 16 + k] = (uchar)to_u8f(o[k]);
  }
}

// ---------------------------------------------------------------------------------------------------------
// postprocess (full res): softmax(logits) blend of the four warped candidates. params: the HTP's uint8 NHWC
// logits (4 x pH x pW, dequantized with scale/zero point)
inline float4 bil_rgb(__global const float* s, int H, int W, float uy, float ux) {
  float py = uy * (float)H, px = ux * (float)W;
  float gy0 = floor(py - 0.5f), gx0 = floor(px - 0.5f), gy1 = gy0 + 1.0f, gx1 = gx0 + 1.0f;
  float wy0 = fmax(1.0f - fabs(gy0 + 0.5f - py), 0.0f), wx0 = fmax(1.0f - fabs(gx0 + 0.5f - px), 0.0f);
  float wy1 = fmax(1.0f - fabs(gy1 + 0.5f - py), 0.0f), wx1 = fmax(1.0f - fabs(gx1 + 0.5f - px), 0.0f);
  int y0 = clampi((int)gy0, 0, H - 1), x0 = clampi((int)gx0, 0, W - 1), y1 = clampi((int)gy1, 0, H - 1),
      x1 = clampi((int)gx1, 0, W - 1);
  int n = H * W;
  float4 r;
  r.x = s[y0 * W + x0] * wy0 * wx0 + s[y0 * W + x1] * wy0 * wx1 + s[y1 * W + x0] * wy1 * wx0 + s[y1 * W + x1] * wy1 * wx1;
  r.y = s[n + y0 * W + x0] * wy0 * wx0 + s[n + y0 * W + x1] * wy0 * wx1 + s[n + y1 * W + x0] * wy1 * wx0 +
        s[n + y1 * W + x1] * wy1 * wx1;
  r.z = s[2 * n + y0 * W + x0] * wy0 * wx0 + s[2 * n + y0 * W + x1] * wy0 * wx1 + s[2 * n + y1 * W + x0] * wy1 * wx0 +
        s[2 * n + y1 * W + x1] * wy1 * wx1;
  r.w = 0.0f;
  return r;
}
__kernel void postprocess(__global const float* flow_t, int fH, int fW, __global const float* mv_t, int mH, int mW,
                          __global const uchar* params, int pH, int pW, float psc, float pzp,
                          __global const float* rgb_m1, __global const float* rgb_p1, int H, int W, float t,
                          __global float* out, __global uchar* out_rgba) {
  int y = get_global_id(1), x = get_global_id(0);
  if (y >= H || x >= W) return;
  float uy = ((float)y + 0.5f) * (1.0f / (float)H), ux = ((float)x + 0.5f) * (1.0f / (float)W);
  int fy = (int)floor(uy * (float)fH), fx = (int)floor(ux * (float)fW);
  int my = (int)floor(uy * (float)mH), mx = (int)floor(ux * (float)mW);
  float f0 = gz(flow_t, fH, fW, fy, fx), f1 = gz(flow_t + fH * fW, fH, fW, fy, fx);
  float m0 = gz(mv_t, mH, mW, my, mx), m1 = gz(mv_t + mH * mW, mH, mW, my, mx);
  // bilinear of the dequantized logits
  float lg[4];
  {
    float py = uy * (float)pH, px = ux * (float)pW;
    float gy0 = floor(py - 0.5f), gx0 = floor(px - 0.5f), gy1 = gy0 + 1.0f, gx1 = gx0 + 1.0f;
    float wy0 = fmax(1.0f - fabs(gy0 + 0.5f - py), 0.0f), wx0 = fmax(1.0f - fabs(gx0 + 0.5f - px), 0.0f);
    float wy1 = fmax(1.0f - fabs(gy1 + 0.5f - py), 0.0f), wx1 = fmax(1.0f - fabs(gx1 + 0.5f - px), 0.0f);
    int y0 = clampi((int)gy0, 0, pH - 1), x0 = clampi((int)gx0, 0, pW - 1), y1 = clampi((int)gy1, 0, pH - 1),
        x1 = clampi((int)gx1, 0, pW - 1);
    for (int k = 0; k < 4; k++) {
#define LQ(yy, xx) (((float)params[((yy) * pW + (xx)) * 4 + k] - pzp) * psc)
      lg[k] = LQ(y0, x0) * wy0 * wx0 + LQ(y0, x1) * wy0 * wx1 + LQ(y1, x0) * wy1 * wx0 + LQ(y1, x1) * wy1 * wx1;
    }
  }
  float4 c0 = bil_rgb(rgb_m1, H, W, uy + m0 * t, ux + m1 * t);
  float4 c1 = bil_rgb(rgb_p1, H, W, uy - m0 * (1.0f - t), ux - m1 * (1.0f - t));
  float4 c2 = bil_rgb(rgb_m1, H, W, uy - f0 * t, ux - f1 * t);
  float4 c3 = bil_rgb(rgb_p1, H, W, uy + f0 * (1.0f - t), ux + f1 * (1.0f - t));
  float mxl = fmax(fmax(fmax(lg[0], lg[1]), lg[2]), lg[3]);
  float e0 = exp(lg[0] - mxl), e1 = exp(lg[1] - mxl), e2 = exp(lg[2] - mxl), e3 = exp(lg[3] - mxl);
  float den = ((e0 + e1) + e2) + e3 + 1.175494351e-38f;
  float4 r = c0 * (e0 / den);
  r = r + c1 * (e1 / den);
  r = r + c2 * (e2 / den);
  r = r + c3 * (e3 / den);
  int n = H * W, p = y * W + x;
  r = select(r, (float4)(0.0f), isnan(r));
  out[p] = r.x;
  out[n + p] = r.y;
  out[2 * n + p] = r.z;
  out_rgba[p * 4] = (uchar)to_u8f(r.x);
  out_rgba[p * 4 + 1] = (uchar)to_u8f(r.y);
  out_rgba[p * 4 + 2] = (uchar)to_u8f(r.z);
  out_rgba[p * 4 + 3] = 255;
}

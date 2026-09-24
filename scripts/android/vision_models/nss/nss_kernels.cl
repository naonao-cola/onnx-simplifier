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
// explicit fused multiply-adds (float64 multiply-add, one rounding) use fma1() (see there).
#pragma OPENCL FP_CONTRACT OFF

#define EPS 1e-07f
#define MAX_HALF 65504.0f
#define INT32_MAXF 2147483648.0f  // float(INT32_MAX) as a float32 tensor holds it

// k / 255 for k in 0..255, exactly as the reference's float32 division (no per-tap divide)
__constant float U8F[256] = {0.0f, 0.003921568859368563f, 0.007843137718737125f, 0.0117647061124444f, 0.01568627543747425f, 0.019607843831181526f, 0.0235294122248888f, 0.027450980618596077f, 0.0313725508749485f, 0.03529411926865578f, 0.03921568766236305f, 0.04313725605607033f, 0.0470588244497776f, 0.05098039284348488f, 0.054901961237192154f, 0.05882352963089943f, 0.062745101749897f, 0.06666667014360428f, 0.07058823853731155f, 0.07450980693101883f, 0.0784313753247261f, 0.08235294371843338f, 0.08627451211214066f, 0.09019608050584793f, 0.0941176488995552f, 0.09803921729326248f, 0.10196078568696976f, 0.10588235408067703f, 0.10980392247438431f, 0.11372549086809158f, 0.11764705926179886f, 0.12156862765550613f, 0.125490203499794f, 0.12941177189350128f, 0.13333334028720856f, 0.13725490868091583f, 0.1411764770746231f, 0.14509804546833038f, 0.14901961386203766f, 0.15294118225574493f, 0.1568627506494522f, 0.16078431904315948f, 0.16470588743686676f, 0.16862745583057404f, 0.1725490242242813f, 0.1764705926179886f, 0.18039216101169586f, 0.18431372940540314f, 0.1882352977991104f, 0.1921568661928177f, 0.19607843458652496f, 0.20000000298023224f, 0.20392157137393951f, 0.2078431397676468f, 0.21176470816135406f, 0.21568627655506134f, 0.21960784494876862f, 0.2235294133424759f, 0.22745098173618317f, 0.23137255012989044f, 0.23529411852359772f, 0.239215686917305f, 0.24313725531101227f, 0.24705882370471954f, 0.250980406999588f, 0.2549019753932953f, 0.25882354378700256f, 0.26274511218070984f, 0.2666666805744171f, 0.2705882489681244f, 0.27450981736183167f, 0.27843138575553894f, 0.2823529541492462f, 0.2862745225429535f, 0.29019609093666077f, 0.29411765933036804f, 0.2980392277240753f, 0.3019607961177826f, 0.30588236451148987f, 0.30980393290519714f, 0.3137255012989044f, 0.3176470696926117f, 0.32156863808631897f, 0.32549020648002625f, 0.3294117748737335f, 0.3333333432674408f, 0.33725491166114807f, 0.34117648005485535f, 0.3450980484485626f, 0.3490196168422699f, 0.3529411852359772f, 0.35686275362968445f, 0.3607843220233917f, 0.364705890417099f, 0.3686274588108063f, 0.37254902720451355f, 0.3764705955982208f, 0.3803921639919281f, 0.3843137323856354f, 0.38823530077934265f, 0.3921568691730499f, 0.3960784375667572f, 0.4000000059604645f, 0.40392157435417175f, 0.40784314274787903f, 0.4117647111415863f, 0.4156862795352936f, 0.41960784792900085f, 0.42352941632270813f, 0.4274509847164154f, 0.4313725531101227f, 0.43529412150382996f, 0.43921568989753723f, 0.4431372582912445f, 0.4470588266849518f, 0.45098039507865906f, 0.45490196347236633f, 0.4588235318660736f, 0.4627451002597809f, 0.46666666865348816f, 0.47058823704719543f, 0.4745098054409027f, 0.47843137383461f, 0.48235294222831726f, 0.48627451062202454f, 0.4901960790157318f, 0.4941176474094391f, 0.49803921580314636f, 0.501960813999176f, 0.5058823823928833f, 0.5098039507865906f, 0.5137255191802979f, 0.5176470875740051f, 0.5215686559677124f, 0.5254902243614197f, 0.529411792755127f, 0.5333333611488342f, 0.5372549295425415f, 0.5411764979362488f, 0.545098066329956f, 0.5490196347236633f, 0.5529412031173706f, 0.5568627715110779f, 0.5607843399047852f, 0.5647059082984924f, 0.5686274766921997f, 0.572549045085907f, 0.5764706134796143f, 0.5803921818733215f, 0.5843137502670288f, 0.5882353186607361f, 0.5921568870544434f, 0.5960784554481506f, 0.6000000238418579f, 0.6039215922355652f, 0.6078431606292725f, 0.6117647290229797f, 0.615686297416687f, 0.6196078658103943f, 0.6235294342041016f, 0.6274510025978088f, 0.6313725709915161f, 0.6352941393852234f, 0.6392157077789307f, 0.6431372761726379f, 0.6470588445663452f, 0.6509804129600525f, 0.6549019813537598f, 0.658823549747467f, 0.6627451181411743f, 0.6666666865348816f, 0.6705882549285889f, 0.6745098233222961f, 0.6784313917160034f, 0.6823529601097107f, 0.686274528503418f, 0.6901960968971252f, 0.6941176652908325f, 0.6980392336845398f, 0.7019608020782471f, 0.7058823704719543f, 0.7098039388656616f, 0.7137255072593689f, 0.7176470756530762f, 0.7215686440467834f, 0.7254902124404907f, 0.729411780834198f, 0.7333333492279053f, 0.7372549176216125f, 0.7411764860153198f, 0.7450980544090271f, 0.7490196228027344f, 0.7529411911964417f, 0.7568627595901489f, 0.7607843279838562f, 0.7647058963775635f, 0.7686274647712708f, 0.772549033164978f, 0.7764706015586853f, 0.7803921699523926f, 0.7843137383460999f, 0.7882353067398071f, 0.7921568751335144f, 0.7960784435272217f, 0.800000011920929f, 0.8039215803146362f, 0.8078431487083435f, 0.8117647171020508f, 0.8156862854957581f, 0.8196078538894653f, 0.8235294222831726f, 0.8274509906768799f, 0.8313725590705872f, 0.8352941274642944f, 0.8392156958580017f, 0.843137264251709f, 0.8470588326454163f, 0.8509804010391235f, 0.8549019694328308f, 0.8588235378265381f, 0.8627451062202454f, 0.8666666746139526f, 0.8705882430076599f, 0.8745098114013672f, 0.8784313797950745f, 0.8823529481887817f, 0.886274516582489f, 0.8901960849761963f, 0.8941176533699036f, 0.8980392217636108f, 0.9019607901573181f, 0.9058823585510254f, 0.9098039269447327f, 0.9137254953384399f, 0.9176470637321472f, 0.9215686321258545f, 0.9254902005195618f, 0.929411768913269f, 0.9333333373069763f, 0.9372549057006836f, 0.9411764740943909f, 0.9450980424880981f, 0.9490196108818054f, 0.9529411792755127f, 0.95686274766922f, 0.9607843160629272f, 0.9647058844566345f, 0.9686274528503418f, 0.9725490212440491f, 0.9764705896377563f, 0.9803921580314636f, 0.9843137264251709f, 0.9882352948188782f, 0.9921568632125854f, 0.9960784316062927f, 1.0f};

inline int clampi(int v, int lo, int hi) { return min(max(v, lo), hi); }
inline float satf(float v) { return clamp(v, 0.0f, 1.0f); }
// torch.lerp: start + w * (end - start) for |w| < 0.5, else end - (end - start) * (1 - w)
inline float lerpt(float a, float b, float w) {
  return fabs(w) < 0.5f ? a + w * (b - a) : b - (b - a) * (1.0f - w);
}
// a * b + c with one rounding, like the reference's float64 multiply-add, without fma(): Adreno has no
// native fused fp32 FMA, so OpenCL's correctly rounded fma() is emulated in software (measured: ~50 ms for
// the rectify step at 1080p). Error-free transforms instead: Dekker's exact product p + e = a * b, then
// TwoSum(p, c); agrees with the single rounding except in rare double-rounding ties.
inline float fma1(float a, float b, float c) {
  float p = a * b;
  float t = 4097.0f * a, ah = t - (t - a), al = a - ah;
  t = 4097.0f * b;
  float bh = t - (t - b), bl = b - bh;
  float e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  float s = p + c, bb = s - p;
  float err = (p - (s - bb)) + (c - bb);
  return s + (err + e);
}
inline float4 fma4(float4 a, float4 b, float4 c) {
  return (float4)(fma1(a.x, b.x, c.x), fma1(a.y, b.y, c.y), fma1(a.z, b.z, c.z), fma1(a.w, b.w, c.w));
}

inline int reflect1(int v, int size) {
  v = v < 0 ? -v - 1 : v;
  return v >= size ? 2 * size - v - 1 : v;
}
inline int idx2(int H, int W, int y, int x) { return clampi(y, 0, H - 1) * W + clampi(x, 0, W - 1); }
inline float load(__global const float* t, int H, int W, int ch, int y, int x) {
  return t[(ch * H + clampi(y, 0, H - 1)) * W + clampi(x, 0, W - 1)];
}

// bilinear_sample / sample_bilinear (identical in the pre and post references): uv in [0, 1] (y, x), taps
// clamped to the edge; clamp_to_edge = 0 zeroes the weight of off-screen taps instead. The weights and the
// corner coordinates, then one typed gather per corner (float4 RGBA / float2 / uchar4 / scalar).
typedef struct {
  float w00, w01, w10, w11;
  int y0, x0, y1, x1;
} Bil;
inline Bil bil(int H, int W, float uvy, float uvx, int clamp_to_edge) {
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
  Bil b;
  b.w00 = wy0;  // row weights; the per-corner products are formed in the reference's order below
  b.w01 = wx0;
  b.w10 = wy1;
  b.w11 = wx1;
  b.y0 = clampi((int)gy0, 0, H - 1);
  b.x0 = clampi((int)gx0, 0, W - 1);
  b.y1 = clampi((int)gy1, 0, H - 1);
  b.x1 = clampi((int)gx1, 0, W - 1);
  return b;
}
// tl + tr + bl + br with each term (value * row weight) * column weight, as the reference
#define BIL_SUM(T, t, W, b)                                                                      \
  ((t)[(b).y0 * (W) + (b).x0] * (b).w00 * (b).w01 + (t)[(b).y0 * (W) + (b).x1] * (b).w00 * (b).w11 + \
   (t)[(b).y1 * (W) + (b).x0] * (b).w10 * (b).w01 + (t)[(b).y1 * (W) + (b).x1] * (b).w10 * (b).w11)
inline float4 bil4(__global const float4* t, int H, int W, float uvy, float uvx, int ce) {
  Bil b = bil(H, W, uvy, uvx, ce);
  return BIL_SUM(float4, t, W, b);
}
// the same over an RGBA32F image read with a nearest, clamp-to-edge sampler (exact texel values; the
// texture path caches 2D gathers far better than buffer loads on Adreno)
__constant sampler_t NEAREST = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
inline float4 bil4i(__read_only image2d_t t, int H, int W, float uvy, float uvx, int ce) {
  Bil b = bil(H, W, uvy, uvx, ce);
  return read_imagef(t, NEAREST, (int2)(b.x0, b.y0)) * b.w00 * b.w01 +
         read_imagef(t, NEAREST, (int2)(b.x1, b.y0)) * b.w00 * b.w11 +
         read_imagef(t, NEAREST, (int2)(b.x0, b.y1)) * b.w10 * b.w01 +
         read_imagef(t, NEAREST, (int2)(b.x1, b.y1)) * b.w10 * b.w11;
}
// uint8 RGBA image (the CNN's temporal output viewed as an image) -> k / 255 via the local table
inline float4 u8i(__read_only image2d_t t, int x, int y, __local const float* u8f) {
  uint4 v = read_imageui(t, NEAREST, (int2)(x, y));
  return (float4)(u8f[v.x], u8f[v.y], u8f[v.z], u8f[v.w]);
}
inline float4 bilu8i(__read_only image2d_t t, int H, int W, float uvy, float uvx, int ce, __local const float* u8f) {
  Bil b = bil(H, W, uvy, uvx, ce);
  return u8i(t, b.x0, b.y0, u8f) * b.w00 * b.w01 + u8i(t, b.x1, b.y0, u8f) * b.w00 * b.w11 +
         u8i(t, b.x0, b.y1, u8f) * b.w10 * b.w01 + u8i(t, b.x1, b.y1, u8f) * b.w10 * b.w11;
}
inline float2 bil2(__global const float2* t, int H, int W, float uvy, float uvx, int ce) {
  Bil b = bil(H, W, uvy, uvx, ce);
  return BIL_SUM(float2, t, W, b);
}
// uint8 -> k / 255 through a work-group-local copy of U8F: divergent __constant lookups serialize on Adreno
inline float4 u8x4(__global const uchar4* t, int i, __local const float* u8f) {
  uchar4 v = t[i];
  return (float4)(u8f[v.x], u8f[v.y], u8f[v.z], u8f[v.w]);
}
inline float4 bilu8(__global const uchar4* t, int H, int W, float uvy, float uvx, int ce, __local const float* u8f) {
  Bil b = bil(H, W, uvy, uvx, ce);
  return u8x4(t, b.y0 * W + b.x0, u8f) * b.w00 * b.w01 + u8x4(t, b.y0 * W + b.x1, u8f) * b.w00 * b.w11 +
         u8x4(t, b.y1 * W + b.x0, u8f) * b.w10 * b.w01 + u8x4(t, b.y1 * W + b.x1, u8f) * b.w10 * b.w11;
}

inline float4 karis4(float4 c) {  // tonemap_forward(..., Karis) on rgb: x / (1 + max(x)), non-negative x
  float4 n = fmax(c, 0.0f);
  float s = 1.0f / (1.0f + fmax(fmax(n.x, n.y), n.z));
  return clamp(n * s, 0.0f, 1.0f);
}

// ------------------------------------------------------------------------------------------------
// depth_scatter (non-quarter): for every depth-res pixel, the nearest of the 2x2 source pixels, its
// motion reprojects it, and its depth is atomic-min'ed into the 4 bilinear neighbours (weight > 0.1)
// of the reprojected position, as int32 depth * INT32_MAX.
__kernel void depth_scatter_init(__global int* out) { out[get_global_id(0)] = 2147483647; }

__kernel void depth_scatter(__global const float2* motion, __global const float* depth, int H, int W,
                            __global int* out, int Ho, int Wo) {
  int oy = get_global_id(1), ox = get_global_id(0);
  if (oy >= Ho || ox >= Wo) return;
  float inv_oy = 1.0f / (float)Ho, inv_ox = 1.0f / (float)Wo;
  float guy = ((float)oy + 0.5f) * inv_oy, gux = ((float)ox + 0.5f) * inv_ox;
  int by = (int)floor(guy * (float)H - 0.5f), bx = (int)floor(gux * (float)W - 0.5f);
  int i0 = idx2(H, W, by, bx);
  float nd = depth[i0];
  float2 mv = motion[i0];
  // candidates (0,1), (1,0), (1,1) after the initial (0,0), unrolled (no indexed private arrays on Adreno)
#define DS_TAKE(dy, dx)                         \
  {                                             \
    int j = idx2(H, W, by + (dy), bx + (dx));  \
    float cd = depth[j];                        \
    float take = cd <= nd ? 1.0f : 0.0f;        \
    nd = nd + take * (cd - nd);                 \
    mv = mv + take * (motion[j] - mv);          \
  }
  DS_TAKE(0, 1) DS_TAKE(1, 0) DS_TAKE(1, 1)
  float isy = 1.0f / ((float)H / (float)Ho), isx = 1.0f / ((float)W / (float)Wo);
  float sy = mv.x * isy, sx = mv.y * isx;
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
#define DS_SCATTER(dy, dx, w)                                                                  \
  {                                                                                            \
    int y = b0y + (dy), x = b0x + (dx);                                                        \
    if (y >= 0 && y < Ho && x >= 0 && x < Wo && (w) > 0.1f) atomic_min(&out[y * Wo + x], d);  \
  }
  DS_SCATTER(0, 0, (1.0f - fy) * (1.0f - fx))
  DS_SCATTER(1, 0, fy * (1.0f - fx))
  DS_SCATTER(0, 1, (1.0f - fy) * fx)
  DS_SCATTER(1, 1, fy * fx)
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
  float acc = 0.0f, wsum = 0.0f;
  float vs0 = (float)(long)rs0, vs1 = (float)(long)rs1;
  float cp0 = (float)(long)(rs0 * 0.5f), cp1 = (float)(long)(rs1 * 0.5f);
  float half_vw = sqrt(rs0 * rs0 + rs1 * rs1);
  float refl = sqrt(1080.0f * 1080.0f + 1920.0f * 1920.0f);
  float power = 1.0f + (3.0f - 1.0f) * satf(half_vw / refl);
  float s0 = cp0 / vs0, s1 = cp1 / vs1;
#define DC_TAP(dy, dx, wexpr)                                                                 \
  {                                                                                           \
    float w = (wexpr);                                                                        \
    int y = by + (dy), x = bx + (dx);                                                         \
    int on = y >= 0 && y < Hd && x >= 0 && x < Wd;                                            \
    wsum = wsum + (on ? 0.0f : w);                                                            \
    int aw = on && w > 0.1f;                                                                  \
    float prev = (float)dtm1[clampi(y, 0, Hd - 1) * Wd + clampi(x, 0, Wd - 1)] * (1.0f / INT32_MAXF); \
    float prev_view = view_depth(prev, dtv.x, dtv.y);                                         \
    float diff = cur_view - prev_view;                                                        \
    int active = aw && diff > 0.0f;                                                           \
    float vd = view_depth(fmax(prev, cur), dtv.x, dtv.y);                                     \
    /* view positions of the viewport center and of the (0, 0) corner */                     \
    float c0 = dtv.z * (s0 * 2.0f - 1.0f) * vd, c1 = dtv.w * (s1 * -2.0f + 1.0f) * vd;        \
    float k0 = dtv.z * (0.0f * 2.0f - 1.0f) * vd, k1 = dtv.w * (0.0f * -2.0f + 1.0f) * vd;    \
    float len_center = sqrt(c0 * c0 + c1 * c1 + vd * vd);                                     \
    float len_corner = sqrt(k0 * k0 + k1 * k1 + vd * vd);                                     \
    float thr = fmax(cur_view, prev_view);                                                    \
    float req = 1.37e-05f * (len_corner / len_center) * half_vw * thr + 0.0f;                 \
    float contrib = pow(satf(req / diff), power) * w;                                         \
    acc = acc + (active ? contrib : 0.0f);                                                    \
    wsum = wsum + (active ? w : 0.0f);                                                        \
  }
  DC_TAP(0, 0, (1.0f - fy) * (1.0f - fx))
  DC_TAP(1, 0, fy * (1.0f - fx))
  DC_TAP(0, 1, (1.0f - fy) * fx)
  DC_TAP(1, 1, fy * fx)
  return wsum > 0.0f ? satf(1.0f - acc / wsum) : 0.0f;
}

inline float4 ycocg_load(__read_only image2d_t col, int H, int W, int y, int x, float e) {
  float4 c = sqrt(fmax(read_imagef(col, NEAREST, (int2)(reflect1(x, W), reflect1(y, H))) * e, 0.0f));
  float co = c.x - c.z;
  float tmp = c.z + co * 0.5f;
  float cg = c.y - tmp;
  return (float4)(tmp + cg * 0.5f, co, cg, 0.0f);
}
inline float ydelta(float4 a, float4 b) {
  float dl = a.x - b.x, dc = (a.y - b.y) * 1.25f, dg = (a.z - b.z) * 1.25f;
  return sqrt(dl * dl + dc * dc + dg * dg);
}

// One thread per padded CNN-grid pixel (Hp x Wp, >= H x W): the 12-channel CNN input as uint8 NHWC (plus,
// if cnn_in != 0, float planar for validation), and inside the H x W input the recurrent derivative state,
// the disocclusion mask (if disocc_out != 0) and the nearest-depth offset code the postprocess needs.
// feedback is the previous frame's uint8 NHWC temporal CNN output (Hp x Wp x 4).
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void preprocess(
    __read_only image2d_t color, __read_only image2d_t history, __global const float2* motion,
    __global const float* depth, __read_only image2d_t feedback_tm1, __read_only image2d_t derivative_tm1,
    __global const int* recon_depth, int H, int W, int Hp, int Wp, int Hh, int Wh, int Hd, int Wd,
    float jy, float jx, float exposure, float rs0, float rs1, float4 dtv,
    __global float* cnn_in, __global uchar* cnn_in_u8, __write_only image2d_t derivative_out,
    __global float* disocc_out, __global uchar* nearest_code) {
  __local float u8f[256];
  u8f[get_local_id(1) * 32 + get_local_id(0)] = U8F[get_local_id(1) * 32 + get_local_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  int py = get_global_id(1), px = get_global_id(0);
  if (py >= Hp || px >= Wp) return;
  int ry = reflect1(py, H), rx = reflect1(px, W);
  float iiy = 1.0f / (float)H, iix = 1.0f / (float)W;
  float uvy = ((float)ry + 0.5f) * iiy, uvx = ((float)rx + 0.5f) * iix;
  float upy = ((float)py + 0.5f) * (1.0f / (float)Hp), upx = ((float)px + 0.5f) * (1.0f / (float)Wp);
  // find_nearest_depth_4x4 (high offsets, strictly closer)
  int cy = (int)(uvy * (float)H), cx = (int)(uvx * (float)W);
  float nd = depth[idx2(H, W, cy, cx)];
  int ny = cy, nx = cx, oy = 0, ox = 0;
  // the high-quality 4x4 search order, unrolled (no indexed private arrays on Adreno)
#define NEAR(dy, dx)                                        \
  {                                                         \
    int sy = cy + (dy), sx = cx + (dx);                     \
    int on = sy >= 0 && sy < H && sx >= 0 && sx < W;        \
    float sd = depth[idx2(H, W, sy, sx)];                   \
    if (on && sd < nd) {                                    \
      nd = sd;                                              \
      ny = sy;                                              \
      nx = sx;                                              \
      oy = (dy);                                            \
      ox = (dx);                                            \
    }                                                       \
  }
  NEAR(1, 0) NEAR(0, 1) NEAR(0, -1) NEAR(-1, 0) NEAR(-1, 1) NEAR(1, 1) NEAR(-1, -1)
  NEAR(1, -1) NEAR(-1, 2) NEAR(0, 2) NEAR(1, 2) NEAR(2, 2) NEAR(2, 1) NEAR(2, 0) NEAR(2, -1)
  float2 m = bil2(motion, H, W, ((float)ny + 0.5f) / (float)H, ((float)nx + 0.5f) / (float)W, 1);
  float mth = sqrt(m.x * m.x + m.y * m.y) > 0.1f ? 1.0f : 0.0f;
  m *= mth;
  float rpy = uvy - m.x * iiy, rpx = uvx - m.y * iix;
  float ujy = uvy - jy * iiy, ujx = uvx - jx * iix;
  int dcy = ry >> 1, dcx = rx >> 1;
  float rdy = ((float)dcy + 0.5f) * (1.0f / (float)Hd) - m.x * iiy;
  float rdx = ((float)dcx + 0.5f) * (1.0f / (float)Wd) - m.y * iix;
  float rppy = upy - m.x * (1.0f / (float)Hp), rppx = upx - m.y * (1.0f / (float)Wp);
#ifdef ABL_NODEPTHCLIP
  float dis = 0.0f;
#else
  float dis = depth_clip(recon_depth, Hd, Wd, rdy, rdx, rs0, rs1, nd, dtv);
#endif
  float4 uc = karis4(bil4i(color, H, W, ujy, ujx, 1) * exposure);
  float4 wh = karis4(bil4i(history, Hh, Wh, rpy, rpx, 0) * exposure);
  // calculate_ycocg_derivative (not low/mid)
  float4 dt = bil4i(derivative_tm1, H, W, rpy, rpx, 0);
  float4 yc = ycocg_load(color, H, W, ry, rx, exposure);
  float d_c = ydelta(yc, dt);
  float d_n = ydelta(yc, ycocg_load(color, H, W, ry, rx - 1, exposure));
  float d_s = ydelta(yc, ycocg_load(color, H, W, ry, rx + 1, exposure));
  float d_e = ydelta(yc, ycocg_load(color, H, W, ry + 1, rx, exposure));
  float d_w = ydelta(yc, ycocg_load(color, H, W, ry - 1, rx, exposure));
  float s_sum = d_n + d_s + d_e + d_w;
  float s_max = fmax(fmax(d_n, d_s), fmax(d_e, d_w));
  float prev = dt.w;
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
  float uninit = (fabs(dt.x) + fabs(dt.y) + fabs(dt.z) + fabs(dt.w)) < 0.0001f ? 1.0f : 0.0f;
  vis = vis * (1.0f - disb);
  float4 st = (float4)(yc.x, yc.y, yc.z, filt), rs = (float4)(yc.x, yc.y, yc.z, 0.0f);
  float4 state;
  state.x = lerpt(lerpt(st.x, rs.x, disb), rs.x, uninit);
  state.y = lerpt(lerpt(st.y, rs.y, disb), rs.y, uninit);
  state.z = lerpt(lerpt(st.z, rs.z, disb), rs.z, uninit);
  state.w = lerpt(lerpt(st.w, rs.w, disb), rs.w, uninit);
  vis = lerpt(vis, 0.0f, uninit);
  // feedback, motion detector
  float4 fb = bilu8i(feedback_tm1, Hp, Wp, rppy, rppx, 0, u8f);
  fb = (float4)(lerpt(fb.x, 0.0f, disb), lerpt(fb.y, 0.0f, disb), lerpt(fb.z, 0.0f, disb), lerpt(fb.w, 0.0f, disb));
  float pmin = sqrt((1.0f / rs0) * (1.0f / rs0) + (1.0f / rs1) * (1.0f / rs1));
  float pmax = sqrt((200.0f / rs0) * (200.0f / rs0) + (200.0f / rs1) * (200.0f / rs1));
  float nm0 = m.x / rs0, nm1 = m.y / rs1;
  float mlen = clamp(sqrt(nm0 * nm0 + nm1 * nm1), pmin, pmax);
  float md = sqrt((mlen - pmin) * (1.0f / (pmax - pmin)));
  float4 i0 = (float4)(wh.x, wh.y, wh.z, uc.x), i1 = (float4)(uc.y, uc.z, md, fb.x),
         i2 = (float4)(fb.y, fb.z, fb.w, vis);
  int n = Hp * Wp, p = py * Wp + px;
  if (cnn_in) {
    cnn_in[p] = i0.x, cnn_in[n + p] = i0.y, cnn_in[2 * n + p] = i0.z, cnn_in[3 * n + p] = i0.w;
    cnn_in[4 * n + p] = i1.x, cnn_in[5 * n + p] = i1.y, cnn_in[6 * n + p] = i1.z, cnn_in[7 * n + p] = i1.w;
    cnn_in[8 * n + p] = i2.x, cnn_in[9 * n + p] = i2.y, cnn_in[10 * n + p] = i2.z, cnn_in[11 * n + p] = i2.w;
  }
  vstore4(convert_uchar4(clamp(rint(i0 * 255.0f), 0.0f, 255.0f)), 0, cnn_in_u8 + p * 12);
  vstore4(convert_uchar4(clamp(rint(i1 * 255.0f), 0.0f, 255.0f)), 0, cnn_in_u8 + p * 12 + 4);
  vstore4(convert_uchar4(clamp(rint(i2 * 255.0f), 0.0f, 255.0f)), 0, cnn_in_u8 + p * 12 + 8);
  if (py < H && px < W) {
    int q = py * W + px;
    write_imagef(derivative_out, (int2)(px, py), state);
    if (disocc_out) disocc_out[q] = dis;
    nearest_code[q] = (uchar)(((clampi(ox, -2, 2) + 2) << 3) | (clampi(oy, -2, 2) + 2));
  }
}

// ------------------------------------------------------------------------------------------------
// postprocess helpers
// Catmull-Rom as 5 bilinear taps in a cross (the reference's sample_catmull_rom), per axis weights
// (w0, w1 + w2, w3) at (tc - 1, tc + w2 / (w1 + w2), tc + 2); unrolled, no indexed private arrays.
#define CR_AXIS(uv, S, lw, mw, hw, lp, mp, hp)                  \
  {                                                             \
    float su = (uv) * (S);                                      \
    float tc = floor(su - 0.5f) + 0.5f;                         \
    float f1 = su - tc, f2 = f1 * f1, f3 = f2 * f1;             \
    float w0 = f2 - 0.5f * (f3 + f1);                           \
    float w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;                    \
    float w3 = 0.5f * (f3 - f2);                                \
    float w2 = 1.0f - w0 - w1 - w3;                             \
    float is = 1.0f / (S);                                      \
    lw = w0;                                                    \
    mw = w1 + w2;                                               \
    hw = w3;                                                    \
    lp = (tc - 1.0f) * is;                                      \
    mp = (tc + w2 / mw) * is;                                   \
    hp = (tc + 2.0f) * is;                                      \
  }
inline float4 catmull_rom(__read_only image2d_t t, int H, int W, float uvy, float uvx) {
  float lwy, mwy, hwy, lpy, mpy, hpy, lwx, mwx, hwx, lpx, mpx, hpx;
  CR_AXIS(uvy, (float)H, lwy, mwy, hwy, lpy, mpy, hpy)
  CR_AXIS(uvx, (float)W, lwx, mwx, hwx, lpx, mpx, hpx)
  // cross taps (y, x): (m, l), (l, m), (m, m), (h, m), (m, h)
  float4 s0 = bil4i(t, H, W, mpy, lpx, 1), s1 = bil4i(t, H, W, lpy, mpx, 1), s2 = bil4i(t, H, W, mpy, mpx, 1),
         s3 = bil4i(t, H, W, hpy, mpx, 1), s4 = bil4i(t, H, W, mpy, hpx, 1);
  float c0 = mwy * lwx, c1 = lwy * mwx, c2 = mwy * mwx, c3 = hwy * mwx, c4 = mwy * hwx;
  float4 acc = s0 * c0;
  acc = acc + s1 * c1;
  acc = acc + s2 * c2;
  acc = acc + s3 * c3;
  acc = acc + s4 * c4;
  float ws = c0;
  ws = ws + c1;
  ws = ws + c2;
  ws = ws + c3;
  ws = ws + c4;
  float4 mn = fmin(fmin(fmin(fmin(fmin((float4)MAX_HALF, s0), s1), s2), s3), s4);
  float4 mx = fmax(fmax(fmax(fmax(fmax((float4)(-MAX_HALF), s0), s1), s2), s3), s4);
  float4 o = acc * (1.0f / ws);
  if (o.x < 0.0f || o.y < 0.0f || o.z < 0.0f) o = fmax(fmin(o, mx), mn);
  return o;
}

// One thread per output pixel (Ho x Wo): the filtered + temporally accumulated output (linear, an RGBA32F
// image that is also next frame's history) and its reinhard-tonemapped RGBA8 display copy. lut: (6, mod_h * mod_w * taps).
__kernel __attribute__((reqd_work_group_size(32, 8, 1))) void postprocess(
    __read_only image2d_t color, __read_only image2d_t history, __global const float2* motion,
    __global const uchar* nearest_code, __global const uchar* kpn_u8, __read_only image2d_t temporal_u8,
    __constant float* offset_lut, int H, int W, int Ho, int Wo, int Hk, int Wk, int Kc, int Ht, int Wt,
    int mod_h, int mod_w, int taps, float exposure, float reset, __write_only image2d_t out_linear,
    __global uchar4* out_rgba) {
  __local float u8f[256], lut[6 * 64];  // lut: up to 64 (tile, tap) pairs, 36 at 2x2 tiles x 9 taps
  int lid = get_local_id(1) * 32 + get_local_id(0);
  u8f[lid] = U8F[lid];
  int nl = mod_h * mod_w * taps;
  for (int i = lid; i < 6 * nl; i += 256) lut[i] = offset_lut[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  int oy = get_global_id(1), ox = get_global_id(0);
  if (oy >= Ho || ox >= Wo) return;
  int p0 = oy * Wo + ox;
  float e = exposure, ie = 1.0f / e;
  float scy = (float)Ho / (float)H, scx = (float)Wo / (float)W;
  float isy = 1.0f / scy, isx = 1.0f / scx;
  // filter_color (dense 6x6 KPN, full-res preprocess)
#ifdef ABL_NOMOD
  int li = ((oy & 1) * 2 + (ox & 1));
#else
  int li = (oy % mod_h) * mod_w + (ox % mod_w);
#endif
  float kpsy = (float)Hk / (float)Ht, kpsx = (float)Wk / (float)Wt;
  float4 m1 = 0.0f, m2 = 0.0f, cc = 0.0f;
  float wsum = 0.0f, cv = 0.0f;
#ifdef ABL_NOFILTER
  for (int k = 0; k < 1; k++) {
#else
  for (int k = 0; k < taps; k++) {
#endif
    int j = li * taps + k;
    float t0 = lut[j], t1 = lut[nl + j], t2 = lut[2 * nl + j];
    float t3 = lut[3 * nl + j], t4 = lut[4 * nl + j], t5 = lut[5 * nl + j];
    int ly = (int)floor(((float)(oy + (int)t3) + 0.5f) * isy + 0.001f) + (int)t0;
    int lx = (int)floor(((float)(ox + (int)t4) + 0.5f) * isx + 0.001f) + (int)t1;
    ly = clampi(ly, 0, H - 1);
    lx = clampi(lx, 0, W - 1);
    float4 ct = fmin(read_imagef(color, NEAREST, (int2)(lx, ly)) * e, MAX_HALF);
    if (t3 == 0.0f && t4 == 0.0f) {
      cc = ct;
      cv = 1.0f;
    }
    int ky = clampi((int)floor(((float)ly + 0.5f + 0.001f) * kpsy), 0, Hk - 1);
    int kx = clampi((int)floor(((float)lx + 0.5f + 0.001f) * kpsx), 0, Wk - 1);
    int ch = clampi((int)t5, 0, Kc - 1);
    float raw = u8f[kpn_u8[(ky * Wk + kx) * Kc + ch]];
    float w = fmax(raw, EPS) * t2;
    m1 = m1 + ct * w;
    m2 = m2 + (ct * ct) * w;
    wsum = wsum + w;
  }
  float den = fmax(wsum, EPS);
  m1 = m1 / den;
  m2 = m2 / den;
  // sample_temporal_params (sharp theta); the temporal map is uint8 NHWC 4 channels, edge-clamped
  float iy = 1.0f / (float)Ho, ix = 1.0f / (float)Wo;
  float uvy = ((float)oy + 0.5f) * iy, uvx = ((float)ox + 0.5f) * ix;
#ifdef ABL_NOTEMPORAL
  float4 par = (float4)(0.5f);
#else
  float4 par = bilu8i(temporal_u8, Ht, Wt, uvy * ((float)H * (1.0f / (float)Ht)), uvx * ((float)W * (1.0f / (float)Wt)), 1, u8f);
#endif
  float theta = satf(par.x);
  float th2 = theta * theta, it = 1.0f - theta, it2 = it * it;
  theta = th2 / fmax(th2 + it2, 1e-06f);
  float alpha = par.y * 0.35f + 0.05f;
  float gamma = par.z * 2.0f;
  // load_motion via the nearest-depth offset code of the low-res pixel
  int icy = (int)floor((float)oy * isy), icx = (int)floor((float)ox * isx);
  int code = nearest_code[idx2(H, W, icy, icx)];
#ifdef ABL_NOMOTION
  float2 mv = (float2)(0.0f);
#else
  float2 mv = motion[idx2(H, W, icy + (code & 7) - 2, icx + ((code >> 3) & 7) - 2)];
#endif
  float mvy = mv.x * scy, mvx = mv.y * scx;
  float len = sqrt(fma1(mvy, mvy, mvx * mvx));
  float mth = len > 0.1f ? 1.0f : 0.0f;
  mvy *= mth;
  mvx *= mth;
  float rpy = fma1(-mvy, iy, uvy), rpx = fma1(-mvx, ix, uvx);
  float onscreen = (rpy >= 0.0f && rpx >= 0.0f && rpy <= 1.0f && rpx <= 1.0f) ? 1.0f : 0.0f;
#ifdef ABL_NOCATMULL
  float4 w = fmin(read_imagef(history, NEAREST, (int2)(ox, oy)) * e, MAX_HALF);
#else
  float4 w = fmin(catmull_rom(history, Ho, Wo, rpy, rpx) * e, MAX_HALF);
#endif
  // rectify_history (variance via fma), accumulate, karis inverse
  float4 var = fmax(fabs(fma4(-m1, m1, m2)), EPS);
  float4 sigma = sqrt(var) * gamma;
  float4 hcl = fmax(fmin(w, m1 + sigma), m1 - sigma);
  float4 rect;
  float tw = theta * onscreen * reset;
  rect.x = lerpt(lerpt(m1.x, hcl.x, reset), w.x, tw);
  rect.y = lerpt(lerpt(m1.y, hcl.y, reset), w.y, tw);
  rect.z = lerpt(lerpt(m1.z, hcl.z, reset), w.z, tw);
  rect.w = 0.0f;
#ifdef ABL_NORECT
  rect = w;
#endif
  float4 rm = karis4(rect), cm = karis4(cc);
  float a = alpha * cv * reset;
  float4 acc = clamp((float4)(lerpt(rm.x, cm.x, a), lerpt(rm.y, cm.y, a), lerpt(rm.z, cm.z, a), 0.0f), 0.0f,
                     1.0f - EPS);
  float lim = 65504.0f * (1.0f / (1.0f + 65504.0f));  // karis_forward of the MAX_HALF limit
  float4 cl = fmin(fmax(acc, 0.0f), lim);
  float inv = 1.0f / (1.0f - fmax(fmax(cl.x, cl.y), cl.z));
  float4 lin = cl * inv * ie;
  lin.w = 0.0f;
  int p = oy * Wo + ox;
  write_imagef(out_linear, (int2)(ox, oy), lin);
  float4 x = fmax(lin * e, 0.0f);
  x = clamp(x * (1.0f / (1.0f + x)), 0.0f, 1.0f);  // reinhard tonemap for display
#ifndef ABL_NORGBA
  out_rgba[p] = (uchar4)((uchar)rint(x.x * 255.0f), (uchar)rint(x.y * 255.0f), (uchar)rint(x.z * 255.0f), 255);
#endif
}

// bandwidth probe (nss_run startup): out[i] = in[i]
__kernel void copy4(__global const float4* in, __global float4* out) { out[get_global_id(0)] = in[get_global_id(0)]; }

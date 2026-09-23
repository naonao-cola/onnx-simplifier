// Super-resolution mode of the demo app (../vision_models/superres): an x4 SR model on the HTP (ORT +
// QNN EP, strict), uint8 NHWC low-res in -> uint8 NHWC 4x out, and everything around it in C++:
//   camera: the upright frame's centered 16:9 crop at the HR size (1920x1080, or 1080x1920 upright
//           in portrait; 1:1 sensor pixels from the 1920x1440 stream) = the "original"; its 4x4 box
//           average = the low-res input (what a game would render at 1/16 the pixels); the SR
//           output; optionally the low-res upscaled by CPU bicubic (PIL's a = -0.5, the README's
//           baseline), computed alongside the HTP inference, for the comparison.
//   images: the same from an RGBA bitmap already at the HR size.
// One session per orientation: <model>_270x480.onnx (landscape) and <model>_480x270.onnx
// (portrait), EP-context models compiled on first use. Built into libsr_demo.so (its own process).
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "htp_session.h"
#include "yuv_upright.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SrDemo", __VA_ARGS__)

namespace {
constexpr int SC = 4;                       // x4 models
constexpr int LW = 480, LH = 270;           // landscape low-res input
constexpr int kThreads = 4;
using demo::now_ms;

demo::Htp g_htp;
struct Orient {
  std::unique_ptr<Ort::Session> sess;
  std::string in, out;
};
Orient g_land, g_port;  // 270x480 / 480x270 low-res inputs
std::string g_dir, g_model, g_perf = "burst", g_err;
std::vector<uint8_t> g_hr, g_lr, g_sr;
std::vector<float> g_tmp;
std::mutex g_mu;

Orient& orient(bool portrait) {
  Orient& o = portrait ? g_port : g_land;
  if (!o.sess) {  // compiled/loaded on first use of this orientation
    const std::string stem = g_model + (portrait ? "_480x270" : "_270x480");
    o.sess = g_htp.session(g_dir, stem, g_perf, "SrDemo");
    Ort::AllocatorWithDefaultOptions a;
    o.in = o.sess->GetInputNameAllocated(0, a).get();
    o.out = o.sess->GetOutputNameAllocated(0, a).get();
    LOGI("%s loaded", stem.c_str());
  }
  return o;
}

struct Px {  // locked RGBA bitmap (null if the Java side passed none or the size is wrong)
  JNIEnv* e = nullptr;
  jobject b = nullptr;
  uint8_t* p = nullptr;
  int stride = 0;
  Px(JNIEnv* env, jobject bmp, int w, int h) : e(env), b(bmp) {
    AndroidBitmapInfo bi;
    if (!bmp || AndroidBitmap_getInfo(e, bmp, &bi) || (int)bi.width != w || (int)bi.height != h ||
        bi.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || AndroidBitmap_lockPixels(e, bmp, (void**)&p))
      p = nullptr;
    else
      stride = (int)bi.stride;
  }
  ~Px() {
    if (p) AndroidBitmap_unlockPixels(e, b);
  }
};

// packed RGB (w x h) -> RGBA bitmap rows
void to_rgba(const uint8_t* src, int w, int h, Px& d) {
  if (!d.p) return;
  demo::par_rows(h, kThreads, [&](long a, long b) {
    for (long y = a; y < b; ++y) {
      const uint8_t* s = src + (size_t)y * w * 3;
      uint8_t* o = d.p + (size_t)y * d.stride;
      for (int x = 0; x < w; ++x) {
        o[4 * x] = s[3 * x];
        o[4 * x + 1] = s[3 * x + 1];
        o[4 * x + 2] = s[3 * x + 2];
        o[4 * x + 3] = 255;
      }
    }
  });
}

// HR RGB -> 4x4 box-averaged LR RGB
void downscale(int hw, int hh) {
  const int lw = hw / SC, lh = hh / SC;
  demo::par_rows(lh, kThreads, [&](long a, long b) {
    for (long y = a; y < b; ++y)
      for (int x = 0; x < lw; ++x)
        for (int c = 0; c < 3; ++c) {
          int s = 0;
          for (int dy = 0; dy < SC; ++dy) {
            const uint8_t* r = g_hr.data() + ((size_t)(y * SC + dy) * hw + x * SC) * 3 + c;
            for (int dx = 0; dx < SC; ++dx) s += r[3 * dx];
          }
          g_lr[((size_t)y * lw + x) * 3 + c] = (uint8_t)((s + SC * SC / 2) / (SC * SC));
        }
  });
}

// Keys cubic, a = -0.5 (PIL BICUBIC), 4 taps; source coord (d + 0.5) / 4 - 0.5, edge-clamped.
struct Taps {
  int i[4];
  float w[4];
};
std::vector<Taps> taps(int n_out) {
  std::vector<Taps> t(n_out);
  const int n_in = n_out / SC;
  for (int d = 0; d < n_out; ++d) {
    const float s = (d + 0.5f) / SC - 0.5f;
    const int f = (int)std::floor(s);
    float sum = 0;
    for (int k = 0; k < 4; ++k) {
      const float x = std::fabs(s - (f - 1 + k));
      const float a = -0.5f;
      const float w = x <= 1 ? ((a + 2) * x - (a + 3)) * x * x + 1
                    : x < 2 ? ((a * x - 5 * a) * x + 8 * a) * x - 4 * a : 0.f;
      t[d].i[k] = std::min(n_in - 1, std::max(0, f - 1 + k));
      t[d].w[k] = w;
      sum += w;
    }
    for (float& w : t[d].w) w /= sum;
  }
  return t;
}

void bicubic(int hw, int hh, Px& d) {
  if (!d.p) return;
  const int lw = hw / SC, lh = hh / SC;
  const std::vector<Taps> tx = taps(hw), ty = taps(hh);
  g_tmp.resize((size_t)lh * hw * 3);
  demo::par_rows(lh, kThreads, [&](long a, long b) {  // horizontal: lh x hw
    for (long y = a; y < b; ++y)
      for (int x = 0; x < hw; ++x)
        for (int c = 0; c < 3; ++c) {
          float s = 0;
          for (int k = 0; k < 4; ++k) s += tx[x].w[k] * g_lr[((size_t)y * lw + tx[x].i[k]) * 3 + c];
          g_tmp[((size_t)y * hw + x) * 3 + c] = s;
        }
  });
  demo::par_rows(hh, kThreads, [&](long a, long b) {  // vertical, straight into the bitmap
    for (long y = a; y < b; ++y) {
      uint8_t* o = d.p + (size_t)y * d.stride;
      for (int x = 0; x < hw; ++x) {
        for (int c = 0; c < 3; ++c) {
          float s = 0;
          for (int k = 0; k < 4; ++k) s += ty[y].w[k] * g_tmp[((size_t)ty[y].i[k] * hw + x) * 3 + c];
          o[4 * x + c] = (uint8_t)std::min(255.f, std::max(0.f, s + 0.5f));
        }
        o[4 * x + 3] = 255;
      }
    }
  });
}

// times: 0 total, 1 pre (frame -> LR, and the HR original if shown), 2 htp, 3 post (SR -> bitmap), 4 bicubic (CPU)
void run_lr(int hw, int hh, JNIEnv* e, jobject lrBmp, jobject srBmp, jobject bicBmp, float* t, double t0,
            bool from_hr) {
  const int lw = hw / SC, lh = hh / SC;
  if (from_hr) downscale(hw, hh);
  {
    Px lp(e, lrBmp, lw, lh);
    to_rgba(g_lr.data(), lw, lh, lp);
  }
  const double t1 = now_ms();
  // the CPU bicubic reference runs alongside the HTP inference (only its own time is reported)
  Px bp(e, bicBmp, hw, hh);
  double tb = 0;
  std::thread bic;
  if (bp.p)
    bic = std::thread([&] {
      const double s = now_ms();
      bicubic(hw, hh, bp);
      tb = now_ms() - s;
    });
  struct Join {  // join before bp unlocks, also if the HTP run throws
    std::thread& t;
    ~Join() {
      if (t.joinable()) t.join();
    }
  } join{bic};
  Orient& o = orient(hh > hw);
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int64_t is[4] = {1, lh, lw, 3}, os[4] = {1, hh, hw, 3};
  Ort::Value in = Ort::Value::CreateTensor<uint8_t>(mi, g_lr.data(), (size_t)lw * lh * 3, is, 4);
  Ort::Value out = Ort::Value::CreateTensor<uint8_t>(mi, g_sr.data(), (size_t)hw * hh * 3, os, 4);
  const char* in_n[] = {o.in.c_str()};
  const char* out_n[] = {o.out.c_str()};
  o.sess->Run(Ort::RunOptions{nullptr}, in_n, &in, 1, out_n, &out, 1);
  const double t2 = now_ms();
  {
    Px sp(e, srBmp, hw, hh);
    to_rgba(g_sr.data(), hw, hh, sp);
  }
  const double t3 = now_ms();
  if (bic.joinable()) bic.join();
  const double t4 = now_ms();
  t[1] = (float)(t1 - t0);
  t[2] = (float)(t2 - t1);
  t[3] = (float)(t3 - t2);
  t[4] = (float)tb;
  t[0] = (float)(t4 - t0);
}

void sizes(int hw, int hh) {
  g_hr.resize((size_t)hw * hh * 3);
  g_lr.resize((size_t)hw * hh * 3 / (SC * SC));
  g_sr.resize((size_t)hw * hh * 3);
}

std::string str(JNIEnv* e, jstring s) {
  const char* c = e->GetStringUTFChars(s, nullptr);
  std::string r(c);
  e->ReleaseStringUTFChars(s, c);
  return r;
}
}  // namespace

// model: stem prefix in modelDir, e.g. sr_xlsr_int8 -> sr_xlsr_int8_270x480.onnx / _480x270.onnx.
// Re-init with another model drops the previous sessions first (one model at a time).
extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_SrEngine_nativeInit(JNIEnv* e, jclass, jstring jdir,
                                                                                      jstring jlib, jstring jmodel,
                                                                                      jstring jopts) {
  std::lock_guard<std::mutex> l(g_mu);
  try {
    auto o = demo::parse_opts(str(e, jopts), {{"htp_performance_mode", "burst"}});
    g_perf = o["htp_performance_mode"];
    g_dir = str(e, jdir);
    g_model = str(e, jmodel);
    g_htp.init(str(e, jlib), "sr");
    g_land.sess.reset();
    g_port.sess.reset();
    orient(false);  // compile/load the landscape graph now (the first frame shows its cost otherwise)
    return nullptr;
  } catch (const std::exception& ex) {
    return e->NewStringUTF(ex.what());
  }
}

// HR size for an upright w x h frame: 1920x1080 landscape, 1080x1920 portrait.
extern "C" JNIEXPORT void JNICALL Java_org_onnxsim_maskrcnndemo_SrEngine_nativeHrDims(JNIEnv* e, jclass, jint w, jint h,
                                                                                     jint rot, jintArray out) {
  const int rw = (rot % 180) ? h : w, rh = (rot % 180) ? w : h;
  jint d[2] = {rw >= rh ? LW * SC : LH * SC, rw >= rh ? LH * SC : LW * SC};
  e->SetIntArrayRegion(out, 0, 2, d);
}

// Camera frame (YUV_420_888 planes, clockwise rot to upright). hrBmp/lrBmp/srBmp/bicBmp: RGBA
// bitmaps of the HR (original), LR, SR and bicubic images, any of them may be null (not filled).
// Returns 0, -1 on error (nativeLastError).
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_SrEngine_nativeRunYuv(
    JNIEnv* e, jclass, jobject jy, jobject ju, jobject jv, jint ys, jint uvs, jint uvps, jint w, jint h, jint rot,
    jobject hrBmp, jobject lrBmp, jobject srBmp, jobject bicBmp, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  float t[5] = {0, 0, 0, 0, 0};
  try {
    const demo::YuvPlanes P{(const uint8_t*)e->GetDirectBufferAddress(jy), (const uint8_t*)e->GetDirectBufferAddress(ju),
                            (const uint8_t*)e->GetDirectBufferAddress(jv), ys, uvs, uvps, w, h};
    int RW, RH;
    demo::upright_dims(P, rot, &RW, &RH);
    const bool port = RH > RW;
    const int hw = port ? LH * SC : LW * SC, hh = port ? LW * SC : LH * SC;
    sizes(hw, hh);
    // centered crop of the upright frame with the HR aspect, mapped (nearest) onto hw x hh
    int cw = RW, ch = (int)((long)RW * hh / hw);
    if (ch > RH) {
      ch = RH;
      cw = (int)((long)RH * hw / hh);
    }
    const int cx = (RW - cw) / 2, cy = (RH - ch) / 2;
    std::vector<int> rxs(hw), rys(hh);
    for (int x = 0; x < hw; ++x) rxs[x] = cx + std::min(cw - 1, (int)(((long)x * cw + cw / 2) / hw));
    for (int y = 0; y < hh; ++y) rys[y] = cy + std::min(ch - 1, (int)(((long)y * ch + ch / 2) / hh));
    auto sensor = [&](int rx, int ry, int* sx, int* sy) {  // upright (rx, ry) -> sensor pixel
      switch (rot) {
        case 90: *sx = ry; *sy = P.h - 1 - rx; break;
        case 180: *sx = P.w - 1 - rx; *sy = P.h - 1 - ry; break;
        case 270: *sx = P.w - 1 - ry; *sy = rx; break;
        default: *sx = rx; *sy = ry;
      }
    };
    auto rgb = [](int yy, int uu, int vv, uint8_t* o) {  // JFIF BT.601, x1024 fixed point
      const int r = yy + ((1436 * vv + 512) >> 10), g = yy - ((352 * uu + 731 * vv + 512) >> 10),
                bb = yy + ((1815 * uu + 512) >> 10);
      o[0] = (uint8_t)std::min(255, std::max(0, r));
      o[1] = (uint8_t)std::min(255, std::max(0, g));
      o[2] = (uint8_t)std::min(255, std::max(0, bb));
    };
    // The low-res input straight from YUV: Y, U, V averaged over each 4x4 block of the HR crop, then
    // one conversion per low-res pixel (the conversion is affine, so this is the 4x4 RGB average up to
    // clamping). Converting all 2M HR pixels first cost ~22 ms on the phone; this is ~16x less work.
    const int lw = hw / SC, lh = hh / SC;
    demo::par_rows(lh, kThreads, [&](long a, long b) {
      for (long y = a; y < b; ++y)
        for (int x = 0; x < lw; ++x) {
          int sy_ = 0, su = 0, sv = 0;
          for (int dy = 0; dy < SC; ++dy)
            for (int dx = 0; dx < SC; ++dx) {
              int sx, sy;
              sensor(rxs[x * SC + dx], rys[y * SC + dy], &sx, &sy);
              sy_ += P.y[sy * P.ys + sx];
              const int ci = (sy >> 1) * P.uvs + (sx >> 1) * P.uvps;
              su += P.u[ci];
              sv += P.v[ci];
            }
          constexpr int N = SC * SC;
          rgb((sy_ + N / 2) / N, (su + N / 2) / N - 128, (sv + N / 2) / N - 128, &g_lr[((size_t)y * lw + x) * 3]);
        }
    });
    // the HR "original", only when it's on screen
    Px hp(e, hrBmp, hw, hh);
    if (hp.p)
      demo::par_rows(hh, kThreads, [&](long a, long b) {
        for (long oy = a; oy < b; ++oy) {
          uint8_t* o = hp.p + (size_t)oy * hp.stride;
          for (int ox = 0; ox < hw; ++ox) {
            int sx, sy;
            sensor(rxs[ox], rys[oy], &sx, &sy);
            const int ci = (sy >> 1) * P.uvs + (sx >> 1) * P.uvps;
            rgb(P.y[sy * P.ys + sx], P.u[ci] - 128, P.v[ci] - 128, o + 4 * ox);
            o[4 * ox + 3] = 255;
          }
        }
      });
    run_lr(hw, hh, e, lrBmp, srBmp, bicBmp, t, t0, false);
    e->SetFloatArrayRegion(jt, 0, 5, t);
    return 0;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

// Images mode: hrBmp is the RGBA original already at 1920x1080 or 1080x1920 (SrActivity crops and
// scales the test image); the rest as nativeRunYuv.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_SrEngine_nativeRun(JNIEnv* e, jclass, jobject hrBmp,
                                                                                  jobject lrBmp, jobject srBmp,
                                                                                  jobject bicBmp, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  float t[5] = {0, 0, 0, 0, 0};
  try {
    AndroidBitmapInfo bi;
    if (AndroidBitmap_getInfo(e, hrBmp, &bi) || bi.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
      throw std::runtime_error("original must be an RGBA_8888 bitmap");
    const int hw = (int)bi.width, hh = (int)bi.height;
    if (!((hw == LW * SC && hh == LH * SC) || (hw == LH * SC && hh == LW * SC)))
      throw std::runtime_error("original must be 1920x1080 or 1080x1920");
    sizes(hw, hh);
    {
      Px hp(e, hrBmp, hw, hh);
      if (!hp.p) throw std::runtime_error("lockPixels failed");
      demo::par_rows(hh, kThreads, [&](long a, long b) {
        for (long y = a; y < b; ++y) {
          const uint8_t* s = hp.p + (size_t)y * hp.stride;
          uint8_t* o = g_hr.data() + (size_t)y * hw * 3;
          for (int x = 0; x < hw; ++x) {
            o[3 * x] = s[4 * x];
            o[3 * x + 1] = s[4 * x + 1];
            o[3 * x + 2] = s[4 * x + 2];
          }
        }
      });
    }
    run_lr(hw, hh, e, lrBmp, srBmp, bicBmp, t, t0, true);
    e->SetFloatArrayRegion(jt, 0, 5, t);
    return 0;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_SrEngine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}

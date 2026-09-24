// Segment Anything mode of the demo app (SamActivity, its own process): EfficientViT-SAM-L0 from
// ../../vision_models/sam (PR #1876), both halves strict on the HTP from EP-context models.
//   encoder sam_l0_enc.onnx (= enc.fp16.onnx): uint8 NHWC [1, 512, 512, 3] -- the image with its
//           longest side resized to 512, padded bottom/right with the SAM mean pixel; normalization
//           is inside the graph -> image_embeddings [1, 256, 64, 64]. Once per frozen frame/image.
//   decoder sam_l0_dec.onnx (= dec.sim.onnx): embeddings + 2 prompt points in the 1024 prompt frame
//           (a point + a padding point, labels 1 / -1) -> iou_predictions [1, 4], low_res_masks
//           [1, 4, 256, 256]. Once per tap. The mask is slot 1 + argmax(iou[1:]) (sam.py's point
//           rule), thresholded at 0; one low-res pixel is 2x2 encoder pixels.
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "htp_session.h"
#include "yuv_upright.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SamDemo", __VA_ARGS__)

namespace {
using demo::now_ms;
constexpr int S = 512, PROMPT = 1024, LR = 256, EMB = 256 * 64 * 64;
constexpr uint8_t kPad[3] = {124, 116, 104};  // round(SAM pixel mean): ~0 after normalization

demo::Htp g_htp;
std::unique_ptr<Ort::Session> g_enc, g_dec;
std::vector<uint8_t> g_q(S * S * 3);
std::vector<float> g_emb(EMB), g_masks(4 * LR * LR);
bool g_have_emb = false;
std::string g_err;
std::mutex g_mu;

struct Fit {
  int fw, fh;
};
// longest side -> 512 (sam.py: r = 512 / max(h, w), int(h * r + 0.5))
Fit fit(int w, int h) {
  const float r = (float)S / std::max(w, h);
  return {std::max(1, (int)(w * r + 0.5f)), std::max(1, (int)(h * r + 0.5f))};
}
void pad_outside(const Fit& f) {
  for (int y = 0; y < S; ++y) {
    uint8_t* o = g_q.data() + (size_t)y * S * 3;
    for (int x = y < f.fh ? f.fw : 0; x < S; ++x) memcpy(o + 3 * x, kPad, 3);
  }
}

float encode(double t0, float* times) {
  const double t1 = now_ms();
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int64_t is[4] = {1, S, S, 3}, es[4] = {1, 256, 64, 64};
  Ort::Value in = Ort::Value::CreateTensor<uint8_t>(mi, g_q.data(), g_q.size(), is, 4);
  Ort::Value out = Ort::Value::CreateTensor<float>(mi, g_emb.data(), g_emb.size(), es, 4);
  const char* in_n[] = {"pixels_u8"};
  const char* out_n[] = {"image_embeddings"};
  g_enc->Run(Ort::RunOptions{nullptr}, in_n, &in, 1, out_n, &out, 1);
  g_have_emb = true;
  times[0] = (float)(t1 - t0);
  times[1] = (float)(now_ms() - t1);
  return times[1];
}

struct Locked {
  JNIEnv* e;
  jobject bmp;
  uint8_t* px = nullptr;
  AndroidBitmapInfo bi{};
  Locked(JNIEnv* e_, jobject b) : e(e_), bmp(b) {
    if (b && AndroidBitmap_getInfo(e, b, &bi) == 0 && bi.format == ANDROID_BITMAP_FORMAT_RGBA_8888 &&
        AndroidBitmap_lockPixels(e, b, (void**)&px) != 0)
      px = nullptr;
  }
  ~Locked() {
    if (px) AndroidBitmap_unlockPixels(e, bmp);
  }
};

demo::YuvPlanes planes(JNIEnv* e, jobject jy, jobject ju, jobject jv, int ys, int uvs, int uvps, int w, int h) {
  return {(const uint8_t*)e->GetDirectBufferAddress(jy), (const uint8_t*)e->GetDirectBufferAddress(ju),
          (const uint8_t*)e->GetDirectBufferAddress(jv), ys, uvs, uvps, w, h};
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_SamEngine_nativeInit(JNIEnv* e, jclass, jstring jdir,
                                                                                       jstring jlib, jstring jopts) {
  auto str = [&](jstring s) {
    const char* c = e->GetStringUTFChars(s, nullptr);
    std::string r(c);
    e->ReleaseStringUTFChars(s, c);
    return r;
  };
  std::lock_guard<std::mutex> l(g_mu);
  try {
    auto o = demo::parse_opts(str(jopts), {{"htp_performance_mode", "burst"}});
    const std::string dir = str(jdir);
    g_htp.init(str(jlib), "sam");
    g_enc = g_htp.session(dir, "sam_l0_enc", o["htp_performance_mode"], "SamDemo");
    g_dec = g_htp.session(dir, "sam_l0_dec", o["htp_performance_mode"], "SamDemo");
    return nullptr;
  } catch (const std::exception& ex) {
    return e->NewStringUTF(ex.what());
  }
}

// Upright display/encoder size (longest side 512) of a w x h sensor frame rotated by rot.
extern "C" JNIEXPORT void JNICALL Java_org_onnxsim_maskrcnndemo_SamEngine_nativeFitDims(JNIEnv* e, jclass, jint w,
                                                                                       jint h, jint rot, jintArray out) {
  Fit f = (rot % 180) ? fit(h, w) : fit(w, h);
  jint d[2] = {f.fw, f.fh};
  e->SetIntArrayRegion(out, 0, 2, d);
}

// Camera frame -> the upright display bitmap (fitDims size); with encode, also the encoder input,
// and runs the encoder. times: 0 pre, 1 encoder. Returns false on error (nativeLastError).
extern "C" JNIEXPORT jboolean JNICALL Java_org_onnxsim_maskrcnndemo_SamEngine_nativeYuv(
    JNIEnv* e, jclass, jobject jy, jobject ju, jobject jv, jint ys, jint uvs, jint uvps, jint w, jint h, jint rot,
    jobject disp, jboolean enc, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  try {
    const demo::YuvPlanes P = planes(e, jy, ju, jv, ys, uvs, uvps, w, h);
    int RW, RH;
    demo::upright_dims(P, rot, &RW, &RH);
    const Fit f = fit(RW, RH);
    Locked d(e, disp);
    if (!d.px || (int)d.bi.width != f.fw || (int)d.bi.height != f.fh) throw std::runtime_error("display bitmap size");
    if (enc) pad_outside(f);
    demo::yuv_upright(P, rot, f.fw, f.fh, 4, [&](int oy, int ox, uint8_t r, uint8_t g, uint8_t b) {
      uint8_t* p = d.px + (size_t)oy * d.bi.stride + 4 * ox;
      p[0] = r; p[1] = g; p[2] = b; p[3] = 255;
      if (enc) {
        uint8_t* q = g_q.data() + ((size_t)oy * S + ox) * 3;
        q[0] = r; q[1] = g; q[2] = b;
      }
    });
    if (enc) {
      float t[2];
      encode(t0, t);
      e->SetFloatArrayRegion(jt, 0, 2, t);
    }
    return JNI_TRUE;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return JNI_FALSE;
  }
}

// Images mode: an upright RGBA bitmap already fit to 512 (longest side) -> encoder.
extern "C" JNIEXPORT jboolean JNICALL Java_org_onnxsim_maskrcnndemo_SamEngine_nativeEncode(JNIEnv* e, jclass,
                                                                                          jobject bmp, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  try {
    Locked d(e, bmp);
    if (!d.px) throw std::runtime_error("bitmap must be RGBA_8888");
    const Fit f{std::min<int>(d.bi.width, S), std::min<int>(d.bi.height, S)};
    pad_outside(f);
    for (int y = 0; y < f.fh; ++y)
      for (int x = 0; x < f.fw; ++x) memcpy(g_q.data() + ((size_t)y * S + x) * 3, d.px + (size_t)y * d.bi.stride + 4 * x, 3);
    float t[2];
    encode(t0, t);
    e->SetFloatArrayRegion(jt, 0, 2, t);
    return JNI_TRUE;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return JNI_FALSE;
  }
}

// One tap at (x, y) in encoder/display pixels -> the chosen mask (256x256 bytes, 1 = inside) and
// its predicted IoU. Returns the slot (1..3), -1 on error. jt[0] = decoder ms.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_SamEngine_nativeDecode(JNIEnv* e, jclass, jfloat x,
                                                                                      jfloat y, jbyteArray jmask,
                                                                                      jfloatArray jiou, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  try {
    if (!g_have_emb) throw std::runtime_error("no image encoded yet");
    const double t0 = now_ms();
    const float k = (float)PROMPT / S;
    float pc[4] = {x * k, y * k, 0.f, 0.f}, pl[2] = {1.f, -1.f}, iou[4];
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t es[4] = {1, 256, 64, 64}, cs[3] = {1, 2, 2}, ls[2] = {1, 2}, is[2] = {1, 4}, ms[4] = {1, 4, LR, LR};
    Ort::Value in[3] = {Ort::Value::CreateTensor<float>(mi, g_emb.data(), g_emb.size(), es, 4),
                        Ort::Value::CreateTensor<float>(mi, pc, 4, cs, 3),
                        Ort::Value::CreateTensor<float>(mi, pl, 2, ls, 2)};
    Ort::Value out[2] = {Ort::Value::CreateTensor<float>(mi, iou, 4, is, 2),
                         Ort::Value::CreateTensor<float>(mi, g_masks.data(), g_masks.size(), ms, 4)};
    const char* in_n[] = {"image_embeddings", "point_coords", "point_labels"};
    const char* out_n[] = {"iou_predictions", "low_res_masks"};
    g_dec->Run(Ort::RunOptions{nullptr}, in_n, in, 3, out_n, out, 2);
    const int slot = 1 + (int)(std::max_element(iou + 1, iou + 4) - (iou + 1));
    std::vector<jbyte> m(LR * LR);
    const float* src = g_masks.data() + (size_t)slot * LR * LR;
    for (int i = 0; i < LR * LR; ++i) m[i] = src[i] > 0.f ? 1 : 0;
    e->SetByteArrayRegion(jmask, 0, LR * LR, m.data());
    e->SetFloatArrayRegion(jiou, 0, 4, iou);
    float t = (float)(now_ms() - t0);
    e->SetFloatArrayRegion(jt, 0, 1, &t);
    return slot;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_SamEngine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}

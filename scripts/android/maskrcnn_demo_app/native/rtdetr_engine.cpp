// RT-DETR mode of the demo app (RtDetrActivity, its own process): RT-DETR-r18 from
// ../../vision_models/rtdetr (PR #1867) as its phone runner runs it -- 4 strict HTP pieces around
// 3 multi-scale deformable attention calls on the HVX (the ../../msda_hvx skel, PR #1859):
//
//   pre (backbone uint8 + hybrid encoder W8A16, uint8 value maps) -> [msda0] -> mid0 -> [msda1]
//   -> mid1 -> [msda2] -> post -> logits (300, 80), boxes (300, 4: cx, cy, w, h in [0, 1])
//
// msda_hvx/dec_run.cpp is #included for its tensor store (rpcmem buffers the HTP writes and the DSP
// maps without a copy), piece runner and msda call; only the image source, the sessions (EP-context
// models, as the other modes) and the output path differ. NMS-free: the detections are HF's
// post_process_object_detection -- sigmoid, top-k over queries x classes.
// Input: the frame stretched to 640x640 (HF RTDetrImageProcessor: resize, no letterbox), uint8 RGB
// NHWC; /255 is in the graph.
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#define main rtdetr_dec_run_main  // the runner's own CLI entry point, unused here
#include "../../vision_models/rtdetr/msda_hvx/dec_run.cpp"
#undef main

#include <cmath>
#include <mutex>
#include <numeric>

#include "htp_session.h"
#include "yuv_upright.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "RtDetrDemo", __VA_ARGS__)

namespace {
constexpr int S = 640, Q = 300, NC = 80;
demo::Htp g_htp;
std::map<std::string, Piece> g_pc;
std::string g_err;
std::mutex g_mu;  // one frame at a time: the skel isn't reentrant (the demo's DSP mutex rule)
int g_flags = 260;  // MSDA_FLAGS: 4 threads, 16-query jobs (#1867's fastest)
float g_thresh = 0.4f;
bool g_open = false;

Piece piece(const std::string& dir, const std::string& stem, const std::string& name, const std::string& perf) {
  Piece P;
  P.name = name;
  P.s = g_htp.session(dir, stem, perf, "RtDetrDemo");
  Ort::AllocatorWithDefaultOptions a;
  for (size_t i = 0; i < P.s->GetInputCount(); ++i) {
    P.in.push_back(P.s->GetInputNameAllocated(i, a).get());
    auto ti = P.s->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    P.in_shape.push_back(ti.GetShape());
    P.in_type.push_back(ti.GetElementType());
  }
  for (size_t i = 0; i < P.s->GetOutputCount(); ++i) {
    P.out.push_back(P.s->GetOutputNameAllocated(i, a).get());
    auto ti = P.s->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
    P.out_shape.push_back(ti.GetShape());
    P.out_type.push_back(ti.GetElementType());
  }
  return P;
}

// The uint8 value maps' scale / zero point, from the pre piece's metadata (split.py --u8-values).
// Read from the source model: the EP-context model doesn't carry the custom metadata.
void read_value_quant(Piece& pre) {
  Ort::AllocatorWithDefaultOptions al;
  Ort::ModelMetadata md = pre.s->GetModelMetadata();
  for (int i = 0; i < 3; ++i) {
    const std::string v = "value" + std::to_string(i);
    auto sc = md.LookupCustomMetadataMapAllocated((v + "_scale").c_str(), al);
    auto zp = md.LookupCustomMetadataMapAllocated((v + "_zero_point").c_str(), al);
    if (sc && zp) vq[v] = {strtof(sc.get(), nullptr), atoi(zp.get())};
  }
}

// times: 0 total, 1 pre(process), 2 htp pieces, 3 msda (wall), 4 msda in-DSP, 5 post(process)
void infer(float* t, double t0) {
  const double t1 = now_ms();
  double htp = 0, dsp = 0, dsp_in = 0, a = t1;
  auto lap = [&](double* acc) { double b = now_ms(); *acc += b - a; a = b; };
  run(g_pc["pre"]);
  lap(&htp);
  std::string src = "pre";
  for (int i = 0; i < 3; ++i) {
    const std::string hn = i == 0 ? "pre/h" : src + "/h_out", rn = i == 0 ? "pre/ref" : src + "/ref_out";
    msda("pre/value" + std::to_string(i), src + "/off", src + "/w", rn, "msda" + std::to_string(i));
    lap(&dsp);
    dsp_in += last_dsp_us / 1000.0;
    const std::string nxt = i < 2 ? "mid" + std::to_string(i) : "post";
    run(g_pc[nxt], {{"msda", "msda" + std::to_string(i)}, {"h", hn}, {"ref", rn}});
    lap(&htp);
    src = nxt;
  }
  t[1] = (float)(t1 - t0);
  t[2] = (float)htp;
  t[3] = (float)dsp;
  t[4] = (float)dsp_in;
}

// Detections scored >= g_thresh, boxes in display pixels (dw x dh: the model's 640x640 is the
// display frame stretched).
int emit(int dw, int dh, float* t, JNIEnv* e, jfloatArray jb, jintArray jl, jfloatArray js) {
  const double t2 = now_ms();
  const float* lg = (const float*)get("post/logits").p;
  const float* bx = (const float*)get("post/boxes").p;
  const float lthr = std::log(g_thresh / (1.f - g_thresh));  // sigmoid(x) >= thr  <=>  x >= logit(thr)
  std::vector<int> idx;
  for (int i = 0; i < Q * NC; ++i)
    if (lg[i] >= lthr) idx.push_back(i);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
  const int n = std::min<int>((int)idx.size(), e->GetArrayLength(js));
  std::vector<float> b(4 * (size_t)n), s(n);
  std::vector<int> l(n);
  for (int k = 0; k < n; ++k) {
    const int q = idx[k] / NC, c = idx[k] % NC;
    const float* p = bx + 4 * q;
    b[4 * k] = (p[0] - p[2] / 2) * dw;
    b[4 * k + 1] = (p[1] - p[3] / 2) * dh;
    b[4 * k + 2] = (p[0] + p[2] / 2) * dw;
    b[4 * k + 3] = (p[1] + p[3] / 2) * dh;
    s[k] = 1.f / (1.f + std::exp(-lg[idx[k]]));
    l[k] = c + 1;  // Coco.name() is 1-based; HF RT-DETR's 80 labels are the COCO categories in order
  }
  e->SetFloatArrayRegion(jb, 0, 4 * n, b.data());
  e->SetFloatArrayRegion(js, 0, n, s.data());
  e->SetIntArrayRegion(jl, 0, n, l.data());
  const double t3 = now_ms();
  t[5] = (float)(t3 - t2);
  return n;
}

// display frame (dw x dh RGBA) -> model input (640x640 RGB uint8, stretched, nearest)
void stretch_into_input(const uint8_t* px, int stride, int dw, int dh) {
  Buf& img = get("pixel_values");
  uint8_t* q = (uint8_t*)img.p;
  std::vector<int> xs(S), ys(S);
  for (int i = 0; i < S; ++i) {
    xs[i] = std::min(dw - 1, (int)(((long)i * dw + dw / 2) / S));
    ys[i] = std::min(dh - 1, (int)(((long)i * dh + dh / 2) / S));
  }
  demo::par_rows(S, 4, [&](long a, long b) {
    for (long y = a; y < b; ++y) {
      const uint8_t* row = px + (size_t)ys[y] * stride;
      uint8_t* o = q + (size_t)y * S * 3;
      for (int x = 0; x < S; ++x) memcpy(o + 3 * x, row + 4 * xs[x], 3);
    }
  });
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
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_RtDetrEngine_nativeInit(JNIEnv* e, jclass,
                                                                                          jstring jdir, jstring jlib,
                                                                                          jstring jopts) {
  auto str = [&](jstring s) {
    const char* c = e->GetStringUTFChars(s, nullptr);
    std::string r(c);
    e->ReleaseStringUTFChars(s, c);
    return r;
  };
  std::lock_guard<std::mutex> l(g_mu);
  try {
    auto o = demo::parse_opts(str(jopts), {{"htp_performance_mode", "burst"}, {"flags", "260"}, {"thresh", "0.4"}});
    g_flags = std::stoi(o["flags"]);
    g_thresh = std::stof(o["thresh"]);
    setenv("MSDA_FLAGS", o["flags"].c_str(), 1);  // read by dec_run.cpp's msda()
    const std::string dir = str(jdir), lib = str(jlib);
    g_htp.init(lib, "rtdetr");
    // HTP sessions first, then our FastRPC session (opening it while QNN brings up its HTP device
    // stalled the Mask R-CNN demo's startup)
    g_pc["pre"] = piece(dir, "rtdetr_pre", "pre", o["htp_performance_mode"]);
    for (const char* n : {"mid0", "mid1", "post"})
      g_pc[n] = piece(dir, std::string("rtdetr_") + n, n, o["htp_performance_mode"]);
    {
      Ort::SessionOptions so;  // metadata only: a CPU session of the source model, then dropped
      so.DisablePerSessionThreads();
      Piece src;
      src.s = std::make_unique<Ort::Session>(*g_htp.env, (dir + "/rtdetr_pre.onnx").c_str(), so);
      read_value_quant(src);
    }
    struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
    if (msda_rpc_open("file:///libmsda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h_msda))
      throw std::runtime_error("msda_rpc_open failed (skel libmsda_rpc.so)");
    int32 prc = 0;
    msda_rpc_perf_vote(h_msda, 0, &prc);
    g_open = true;
    Piece& pre = g_pc["pre"];
    buf(pre.in[0], pre.in_shape[0], pre.in_type[0]);
    LOGI("init ok: %zu value quant, flags %d", vq.size(), g_flags);
    return nullptr;
  } catch (const std::exception& ex) {
    return e->NewStringUTF(ex.what());
  }
}

extern "C" JNIEXPORT void JNICALL Java_org_onnxsim_maskrcnndemo_RtDetrEngine_nativeFitDims(JNIEnv* e, jclass, jint w,
                                                                                          jint h, jint rot,
                                                                                          jintArray out) {
  const int RW = (rot % 180) ? h : w, RH = (rot % 180) ? w : h;
  const float r = (float)S / std::max(RW, RH);
  jint d[2] = {std::max(1, (int)(RW * r + 0.5f)), std::max(1, (int)(RH * r + 0.5f))};
  e->SetIntArrayRegion(out, 0, 2, d);
}

// Camera: YUV planes -> upright display bitmap (fitDims size) -> stretched model input -> the chain.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_RtDetrEngine_nativeRunYuv(
    JNIEnv* e, jclass, jobject jy, jobject ju, jobject jv, jint ys, jint uvs, jint uvps, jint w, jint h, jint rot,
    jobject disp, jfloatArray jb, jintArray jl, jfloatArray js, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  float t[6] = {0, 0, 0, 0, 0, 0};
  try {
    const demo::YuvPlanes P{(const uint8_t*)e->GetDirectBufferAddress(jy), (const uint8_t*)e->GetDirectBufferAddress(ju),
                            (const uint8_t*)e->GetDirectBufferAddress(jv), ys, uvs, uvps, w, h};
    Locked d(e, disp);
    if (!d.px) throw std::runtime_error("display bitmap");
    const int dw = d.bi.width, dh = d.bi.height;
    demo::yuv_upright(P, rot, dw, dh, 4, [&](int oy, int ox, uint8_t r, uint8_t g, uint8_t b) {
      uint8_t* p = d.px + (size_t)oy * d.bi.stride + 4 * ox;
      p[0] = r; p[1] = g; p[2] = b; p[3] = 255;
    });
    stretch_into_input(d.px, d.bi.stride, dw, dh);
    infer(t, t0);
    int n = emit(dw, dh, t, e, jb, jl, js);
    t[0] = (float)(now_ms() - t0);
    e->SetFloatArrayRegion(jt, 0, 6, t);
    return n;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

// Images: an upright RGBA bitmap (fit to 640, longest side) -> stretched model input -> the chain.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_RtDetrEngine_nativeRun(JNIEnv* e, jclass, jobject bmp,
                                                                                      jfloatArray jb, jintArray jl,
                                                                                      jfloatArray js, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  float t[6] = {0, 0, 0, 0, 0, 0};
  try {
    Locked d(e, bmp);
    if (!d.px) throw std::runtime_error("bitmap must be RGBA_8888");
    stretch_into_input(d.px, d.bi.stride, d.bi.width, d.bi.height);
    infer(t, t0);
    int n = emit(d.bi.width, d.bi.height, t, e, jb, jl, js);
    t[0] = (float)(now_ms() - t0);
    e->SetFloatArrayRegion(jt, 0, 6, t);
    return n;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_RtDetrEngine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}

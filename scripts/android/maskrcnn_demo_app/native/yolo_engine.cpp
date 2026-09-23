// YOLO mode of the demo app: one HTP session (ORT + QNN EP, strict, no CPU fallback) running a
// deploy-pipeline YOLO model (../../deploy/models/yolo26n.yaml, yolo11n.yaml: uint8 NHWC 640x640
// letterboxed RGB in, the head's decoded (1, 4+nc, N) float out), preprocessing and the head's
// post-processing done here in C++ (no ORT CPU graph):
//   post=end2end  YOLO26's NMS-free one-to-one head, x1,y1,x2,y2 + class scores: the two-stage
//                 top-k of Ultralytics' Detect.get_topk_index (== the deploy pipeline's
//                 yolo_end2end post graph): top max_det anchors by best class score, then the top
//                 max_det (anchor, class) pairs among them.
//   post=nms      YOLO11's head, cx,cy,w,h + class scores: per-class greedy NMS (iou 0.7, conf 0.25,
//                 at most max_det boxes), what the deploy pipeline's yolo_detect graph does with
//                 ONNX NonMaxSuppression.
// Built into its own libyolo_demo.so, loaded only by YoloActivity (its own process).
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "YoloDemo", __VA_ARGS__)

namespace {
constexpr int S = 640;           // model input side
constexpr uint8_t kPad = 114;    // Ultralytics letterbox fill
double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class F>
void par(long n, int threads, F f) {
  std::vector<std::thread> th;
  const long step = (n + threads - 1) / threads;
  for (int t = 1; t < threads; ++t) {
    long a = t * step, b = std::min(n, a + step);
    if (a < b) th.emplace_back([=] { f(a, b); });
  }
  f(0, std::min(n, step));
  for (auto& x : th) x.join();
}

std::unique_ptr<Ort::Env> g_env;
std::unique_ptr<Ort::Session> g_sess;
std::vector<Ort::ConstEpDevice> g_npu;
std::string g_in, g_out, g_post, g_err;
std::vector<uint8_t> g_q(S * S * 3);
std::vector<float> g_head;
int g_ch = 0, g_n = 0, g_maxdet = 300;
std::mutex g_mu;
float g_conf = 0.25f, g_iou = 0.7f;

// Letterbox geometry for an upright w x h frame: scale to fit 640x640, centered (as
// deploy/stages/images.py: r = min(640/h, 640/w), round, pad (640 - n) // 2).
struct Fit {
  int fw, fh, left, top;
};
Fit fit(int w, int h) {
  const float r = std::min((float)S / w, (float)S / h);
  Fit f;
  f.fw = std::max(1, (int)std::lround(w * r));
  f.fh = std::max(1, (int)std::lround(h * r));
  f.left = (S - f.fw) / 2;
  f.top = (S - f.fh) / 2;
  return f;
}
void pad_rows(int top, int fh) {
  memset(g_q.data(), kPad, (size_t)top * S * 3);
  memset(g_q.data() + (size_t)(top + fh) * S * 3, kPad, (size_t)(S - top - fh) * S * 3);
}

// times: 0 total, 1 pre, 2 htp, 3 post
void infer(float* times, double t0) {
  const double t1 = now_ms();
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int64_t ishape[4] = {1, S, S, 3};
  Ort::Value in = Ort::Value::CreateTensor<uint8_t>(mi, g_q.data(), g_q.size(), ishape, 4);
  int64_t oshape[3] = {1, g_ch, g_n};
  Ort::Value out = Ort::Value::CreateTensor<float>(mi, g_head.data(), g_head.size(), oshape, 3);
  const char* in_names[] = {g_in.c_str()};
  const char* out_names[] = {g_out.c_str()};
  g_sess->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, &out, 1);
  times[1] = (float)(t1 - t0);
  times[2] = (float)(now_ms() - t1);
}

struct Det {
  float x1, y1, x2, y2, s;
  int c;
};

std::vector<Det> post_end2end() {
  const int nc = g_ch - 4, N = g_n, k = std::min(g_maxdet, N);
  const float* h = g_head.data();
  std::vector<float> best(N, -1e30f);
  for (int c = 0; c < nc; ++c) {
    const float* row = h + (size_t)(4 + c) * N;
    for (int a = 0; a < N; ++a) best[a] = std::max(best[a], row[a]);
  }
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  // ties: ORT TopK keeps the lower index first
  auto by = [&](const std::vector<float>& v) {
    return [&v](int a, int b) { return v[a] > v[b] || (v[a] == v[b] && a < b); };
  };
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), by(best));
  std::vector<float> cand((size_t)k * nc);
  for (int i = 0; i < k; ++i)
    for (int c = 0; c < nc; ++c) cand[(size_t)i * nc + c] = h[(size_t)(4 + c) * N + idx[i]];
  std::vector<int> ci(cand.size());
  std::iota(ci.begin(), ci.end(), 0);
  std::partial_sort(ci.begin(), ci.begin() + k, ci.end(), by(cand));
  std::vector<Det> d;
  for (int j = 0; j < k; ++j) {
    const int a = idx[ci[j] / nc];
    const float s = cand[ci[j]];
    if (s < g_conf) break;  // sorted: the rest are lower (display threshold, not part of the top-k)
    d.push_back({h[a], h[(size_t)N + a], h[(size_t)2 * N + a], h[(size_t)3 * N + a], s, ci[j] % nc});
  }
  return d;
}

float iou(const Det& a, const Det& b) {
  const float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1), hh = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (w <= 0 || hh <= 0) return 0.f;
  const float i = w * hh, u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i;
  return u > 0 ? i / u : 0.f;
}

std::vector<Det> post_nms() {
  const int nc = g_ch - 4, N = g_n;
  const float* h = g_head.data();
  std::vector<Det> cand;
  for (int c = 0; c < nc; ++c) {
    const float* row = h + (size_t)(4 + c) * N;
    for (int a = 0; a < N; ++a)
      if (row[a] > g_conf) {
        const float cx = h[a], cy = h[(size_t)N + a], w = h[(size_t)2 * N + a] * 0.5f, hh = h[(size_t)3 * N + a] * 0.5f;
        cand.push_back({cx - w, cy - hh, cx + w, cy + hh, row[a], c});
      }
  }
  std::stable_sort(cand.begin(), cand.end(), [](const Det& a, const Det& b) { return a.s > b.s; });
  std::vector<Det> keep;
  for (const Det& d : cand) {
    bool ok = true;
    for (const Det& k : keep)
      if (k.c == d.c && iou(k, d) > g_iou) { ok = false; break; }
    if (ok) keep.push_back(d);
    if ((int)keep.size() >= g_maxdet) break;
  }
  return keep;
}

// Detections back into the upright frame's pixel coordinates (the display bitmap: fw x fh).
int emit(const Fit& f, float* times, double t0, JNIEnv* e, jfloatArray jb, jintArray jl, jfloatArray js) {
  const double t2 = now_ms();
  std::vector<Det> d = g_post == "nms" ? post_nms() : post_end2end();
  const int cap = std::min<int>(e->GetArrayLength(js), (int)d.size());
  std::vector<float> b(4 * (size_t)cap), s(cap);
  std::vector<int> l(cap);
  for (int i = 0; i < cap; ++i) {
    b[4 * i] = d[i].x1 - f.left;
    b[4 * i + 1] = d[i].y1 - f.top;
    b[4 * i + 2] = d[i].x2 - f.left;
    b[4 * i + 3] = d[i].y2 - f.top;
    s[i] = d[i].s;
    l[i] = d[i].c + 1;  // Coco.name() is 1-based (0 = background)
  }
  e->SetFloatArrayRegion(jb, 0, 4 * cap, b.data());
  e->SetFloatArrayRegion(js, 0, cap, s.data());
  e->SetIntArrayRegion(jl, 0, cap, l.data());
  const double t3 = now_ms();
  times[3] = (float)(t3 - t2);
  times[0] = (float)(t3 - t0);
  return cap;
}

void init(const std::string& dir, const std::string& lib_dir, const std::string& model, const std::string& opts) {
  std::unordered_map<std::string, std::string> o{{"post", model.rfind("yolo26", 0) == 0 ? "end2end" : "nms"},
                                                 {"htp_performance_mode", "burst"}};
  size_t p = 0;
  while (p < opts.size()) {
    size_t q = opts.find(';', p);
    std::string kv = opts.substr(p, q == std::string::npos ? std::string::npos : q - p);
    size_t eq = kv.find('=');
    if (eq != std::string::npos) o[kv.substr(0, eq)] = kv.substr(eq + 1);
    if (q == std::string::npos) break;
    p = q + 1;
  }
  g_post = o["post"];
  if (o.count("conf")) g_conf = std::stof(o["conf"]);
  if (!g_env) {
    std::string adsp = lib_dir + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
    Ort::ThreadingOptions to;
    to.SetGlobalIntraOpNumThreads(1);
    to.SetGlobalInterOpNumThreads(1);
    to.SetGlobalSpinControl(0);  // ORT_SPIN=0: pool spinning starves our own threads
    g_env = std::make_unique<Ort::Env>(to, ORT_LOGGING_LEVEL_WARNING, "yolo");
    std::string ep = lib_dir + "/libonnxruntime_providers_qnn.so";
    g_env->RegisterExecutionProviderLibrary("QNNExecutionProvider", ep.c_str());
    for (const auto& d : g_env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        g_npu.push_back(d);
    if (g_npu.empty()) throw std::runtime_error("no QNN NPU ep device");
  }
  g_sess.reset();
  Ort::SessionOptions so;
  so.DisablePerSessionThreads();
  so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  std::unordered_map<std::string, std::string> qo{{"backend_type", "htp"},
                                                  {"htp_performance_mode", o["htp_performance_mode"]}};
  so.AppendExecutionProvider_V2(*g_env, g_npu, qo);
  // EP-context model (context binary in its own file), compiled on the first launch and reused
  const std::string src = dir + "/" + model + ".onnx", ctx = dir + "/" + model + ".ctx0.onnx";
  if (!std::ifstream(ctx)) {
    const double t = now_ms();
    Ort::ModelCompilationOptions co(*g_env, so);
    co.SetInputModelPath(src.c_str());
    co.SetOutputModelPath(ctx.c_str());
    co.SetEpContextEmbedMode(false);
    Ort::Status st = Ort::CompileModel(*g_env, co);
    if (!st.IsOK()) throw std::runtime_error("CompileModel " + src + ": " + st.GetErrorMessage());
    LOGI("compiled %s in %.0f ms", ctx.c_str(), now_ms() - t);
  }
  const double t = now_ms();
  g_sess = std::make_unique<Ort::Session>(*g_env, ctx.c_str(), so);
  Ort::AllocatorWithDefaultOptions a;
  g_in = g_sess->GetInputNameAllocated(0, a).get();
  g_out = g_sess->GetOutputNameAllocated(0, a).get();
  auto sh = g_sess->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (sh.size() != 3) throw std::runtime_error("expected a (1, 4+nc, N) head output");
  g_ch = (int)sh[1];
  g_n = (int)sh[2];
  g_head.assign((size_t)g_ch * g_n, 0.f);
  LOGI("session %s in %.0f ms: %s -> %s (1,%d,%d), post %s", ctx.c_str(), now_ms() - t, g_in.c_str(), g_out.c_str(),
       g_ch, g_n, g_post.c_str());
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeInit(JNIEnv* e, jclass, jstring jdir,
                                                                                        jstring jlib, jstring jmodel,
                                                                                        jstring jopts) {
  auto str = [&](jstring s) {
    const char* c = e->GetStringUTFChars(s, nullptr);
    std::string r(c);
    e->ReleaseStringUTFChars(s, c);
    return r;
  };
  std::lock_guard<std::mutex> l(g_mu);
  try {
    init(str(jdir), str(jlib), str(jmodel), str(jopts));
    return nullptr;
  } catch (const std::exception& ex) {
    return e->NewStringUTF(ex.what());
  }
}

// Upright display size of a w x h sensor frame rotated by rot (the letterboxed content, <= 640).
extern "C" JNIEXPORT void JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeFitDims(JNIEnv* e, jclass, jint w,
                                                                                        jint h, jint rot, jintArray out) {
  Fit f = (rot % 180) ? fit(h, w) : fit(w, h);
  jint d[2] = {f.fw, f.fh};
  e->SetIntArrayRegion(out, 0, 2, d);
}

// Camera: YUV_420_888 planes -> rotated (clockwise rot), letterboxed RGB uint8 NHWC model input in
// one pass (nearest-neighbour; JFIF full-range BT.601, fixed point, as maskrcnn_engine.cpp), plus
// the upright letterboxed content as an RGBA display bitmap. Returns the detection count, -1 on
// error (nativeLastError), boxes in the display bitmap's pixels.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeRunYuv(
    JNIEnv* e, jclass, jobject jy, jobject ju, jobject jv, jint ys, jint uvs, jint uvps, jint w, jint h, jint rot,
    jobject disp, jfloatArray jb, jintArray jl, jfloatArray js, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  float times[4] = {0, 0, 0, 0};
  try {
    const uint8_t* py = (const uint8_t*)e->GetDirectBufferAddress(jy);
    const uint8_t* pu = (const uint8_t*)e->GetDirectBufferAddress(ju);
    const uint8_t* pv = (const uint8_t*)e->GetDirectBufferAddress(jv);
    const int RW = (rot % 180) ? h : w, RH = (rot % 180) ? w : h;
    const Fit f = fit(RW, RH);
    AndroidBitmapInfo bi;
    uint8_t* dp = nullptr;
    if (disp && AndroidBitmap_getInfo(e, disp, &bi) == 0 && (int)bi.width == f.fw && (int)bi.height == f.fh &&
        AndroidBitmap_lockPixels(e, disp, (void**)&dp) != 0)
      dp = nullptr;
    std::vector<int> rxs(f.fw), rys(f.fh);
    for (int x = 0; x < f.fw; ++x) rxs[x] = std::min(RW - 1, (int)(((long)x * RW + RW / 2) / f.fw));
    for (int y = 0; y < f.fh; ++y) rys[y] = std::min(RH - 1, (int)(((long)y * RH + RH / 2) / f.fh));
    pad_rows(f.top, f.fh);
    par(f.fh, 4, [&](long a, long b) {
      for (long oy = a; oy < b; ++oy) {
        uint8_t* o = g_q.data() + ((size_t)(f.top + oy) * S) * 3;
        memset(o, kPad, (size_t)f.left * 3);
        memset(o + (size_t)(f.left + f.fw) * 3, kPad, (size_t)(S - f.left - f.fw) * 3);
        o += (size_t)f.left * 3;
        uint8_t* d = dp ? dp + oy * bi.stride : nullptr;
        const int ry = rys[oy];
        for (int ox = 0; ox < f.fw; ++ox) {
          const int rx = rxs[ox];
          int sx, sy;
          switch (rot) {
            case 90: sx = ry; sy = h - 1 - rx; break;
            case 180: sx = w - 1 - rx; sy = h - 1 - ry; break;
            case 270: sx = w - 1 - ry; sy = rx; break;
            default: sx = rx; sy = ry;
          }
          const int yy = py[sy * ys + sx];
          const int ci = (sy >> 1) * uvs + (sx >> 1) * uvps;
          const int uu = pu[ci] - 128, vv = pv[ci] - 128;
          int r = yy + ((1436 * vv + 512) >> 10);
          int g = yy - ((352 * uu + 731 * vv + 512) >> 10);
          int bb = yy + ((1815 * uu + 512) >> 10);
          r = r < 0 ? 0 : (r > 255 ? 255 : r);
          g = g < 0 ? 0 : (g > 255 ? 255 : g);
          bb = bb < 0 ? 0 : (bb > 255 ? 255 : bb);
          o[3 * ox] = (uint8_t)r;
          o[3 * ox + 1] = (uint8_t)g;
          o[3 * ox + 2] = (uint8_t)bb;
          if (d) { d[4 * ox] = (uint8_t)r; d[4 * ox + 1] = (uint8_t)g; d[4 * ox + 2] = (uint8_t)bb; d[4 * ox + 3] = 255; }
        }
      }
    });
    if (dp) AndroidBitmap_unlockPixels(e, disp);
    infer(times, t0);
    int n = emit(f, times, t0, e, jb, jl, js);
    e->SetFloatArrayRegion(jt, 0, 4, times);
    return n;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

// Images mode: an upright RGBA bitmap already scaled to fit 640x640 (YoloActivity.decodeFit) is
// centered into the letterbox. Boxes come back in the bitmap's pixels.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeRun(JNIEnv* e, jclass, jobject bmp,
                                                                                    jfloatArray jb, jintArray jl,
                                                                                    jfloatArray js, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  float times[4] = {0, 0, 0, 0};
  try {
    AndroidBitmapInfo bi;
    uint8_t* px = nullptr;
    if (AndroidBitmap_getInfo(e, bmp, &bi) || bi.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        AndroidBitmap_lockPixels(e, bmp, (void**)&px))
      throw std::runtime_error("bitmap must be RGBA_8888");
    Fit f;
    f.fw = std::min<int>(bi.width, S);
    f.fh = std::min<int>(bi.height, S);
    f.left = (S - f.fw) / 2;
    f.top = (S - f.fh) / 2;
    pad_rows(f.top, f.fh);
    for (int y = 0; y < f.fh; ++y) {
      uint8_t* o = g_q.data() + ((size_t)(f.top + y) * S) * 3;
      memset(o, kPad, (size_t)f.left * 3);
      memset(o + (size_t)(f.left + f.fw) * 3, kPad, (size_t)(S - f.left - f.fw) * 3);
      o += (size_t)f.left * 3;
      const uint8_t* row = px + (size_t)y * bi.stride;
      for (int x = 0; x < f.fw; ++x) {
        o[3 * x] = row[4 * x];
        o[3 * x + 1] = row[4 * x + 1];
        o[3 * x + 2] = row[4 * x + 2];
      }
    }
    AndroidBitmap_unlockPixels(e, bmp);
    infer(times, t0);
    int n = emit(f, times, t0, e, jb, jl, js);
    e->SetFloatArrayRegion(jt, 0, 4, times);
    return n;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}

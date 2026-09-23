// Mask R-CNN demo engine: a JNI library the Android app drives frame by frame. It reuses
// ../../e2e_pipeline/e2e_run.cpp's pipeline machinery verbatim (#included below, its main renamed),
// so the app runs exactly the pipeline that PR #1841 measured: backbone and heads on the HTP (ORT +
// QNN EP), fused RPN and RoiAlign on the HVX DSP (our FastRPC skels), the rest on ORT CPU.
// Differences: the image comes from an RGBA bitmap (quantized straight to the backbone's uint8 NHWC
// input, same arithmetic as the pipeline's quant_in step), and results/timings go back over JNI.
#define main e2e_run_main
#include "../../e2e_pipeline/e2e_run.cpp"
#undef main

#include <android/bitmap.h>
#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MaskRcnnDemo", __VA_ARGS__)

namespace {
std::vector<std::unique_ptr<Step>> g_steps;
std::string g_err;
const float kMeanBgr[3] = {102.9801f, 115.9465f, 122.7717f};  // maskrcnn_e2e/eval_common.py

// timing buckets returned to Java: total, preprocess, backbone, rpn, roialign, heads, cpu(other)
enum { T_TOTAL, T_PRE, T_BACKBONE, T_RPN, T_ROI, T_HEADS, T_CPU, T_N };
int bucket(const Step& S) {
  if (S.op == "quant_in") return T_PRE;
  if (S.op == "rpn") return T_RPN;
  if (S.op == "roialign") return T_ROI;
  if (S.name == "backbone") return T_BACKBONE;
  if (S.name == "box_head" || S.name == "mask_head") return T_HEADS;
  return T_CPU;
}

// quant_in fused with the canvas: the pipeline's first step is
//   quant_in image image_u8 <scale> <zp>
// on eval_common.canvas's fp32 [3,800,1088] (BGR, mean-subtracted, zero-padded). Here the RGBA
// bitmap (already resized to fit 1088x800) goes straight to uint8 NHWC with the same float math:
// q = sat(rint((p - mean) / s) + z) inside the image, sat(rint(0 / s) + z) in the padding.
void quant_rgba(Step& S, const uint8_t* px, int w, int h, int stride) {
  const float s = std::stof(S.f[3]);
  const int z = std::stoi(S.f[4]);
  const int H = 800, W = 1088, C = 3;
  uint8_t* q = (uint8_t*)S.buf.get((size_t)H * W * C);
  float pv = __builtin_rintf(0.f / s) + (float)z;
  const uint8_t pad = (uint8_t)(pv < 0.f ? 0.f : (pv > 255.f ? 255.f : pv));
  par(H, envi("DQ_THREADS", 4), [&](long a, long b) {
    for (long y = a; y < b; ++y) {
      uint8_t* o = q + (long)y * W * C;
      long xx = 0;
      if (y < h) {
        const uint8_t* row = px + (long)y * stride;
        for (; xx < w && xx < W; ++xx) {
          for (int c = 0; c < 3; ++c) {  // BGR channel c <- RGBA byte (2 - c)
            float v = __builtin_rintf(((float)row[4 * xx + (2 - c)] - kMeanBgr[c]) / s) + (float)z;
            o[xx * C + c] = (uint8_t)(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
          }
        }
      }
      memset(o + xx * C, pad, (size_t)(W - xx) * C);
    }
  });
  put_raw(S.f[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, H, W, C}, q);
}

int init(const std::string& model_dir, const std::string& lib_dir, const std::string& pipe) {
  // The DSP loads skels (libQnnHtpV69Skel.so, librpn_rpc.so, libroialign_rpc.so) through this
  // process's FastRPC file listener, which searches ADSP_LIBRARY_PATH. Must be set before the
  // first FastRPC session opens.
  std::string adsp = lib_dir + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
  setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
  if (chdir(model_dir.c_str())) throw std::runtime_error("chdir " + model_dir);
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  int urc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  LOGI("unsigned PD request rc=%d", urc);

  Ort::ThreadingOptions to;
  to.SetGlobalIntraOpNumThreads(envi("ORT_THREADS", 4));
  to.SetGlobalInterOpNumThreads(1);
  env = std::make_unique<Ort::Env>(to, ORT_LOGGING_LEVEL_WARNING, "demo");

  std::ifstream pf(pipe);
  if (!pf) throw std::runtime_error("cannot open " + pipe);
  std::string line;
  bool need_htp = false, need_rpn = false, need_roi = false;
  while (std::getline(pf, line)) {
    if (line.empty()) continue;
    auto S = std::make_unique<Step>();
    std::istringstream ss(line);
    std::string w;
    while (ss >> w) S->f.push_back(w);
    S->op = S->f[0];
    S->name = S->op == "ort" || S->op == "ortpad" ? S->f[1]
              : S->op + ":" + S->f[S->op == "rpn" ? 4 : S->op == "roialign" ? 3 : 2];
    need_htp |= (S->op == "ort" && S->f[3] == "htp") || (S->op == "ortpad" && S->f[2] == "htp");
    need_rpn |= S->op == "rpn";
    need_roi |= S->op == "roialign";
    g_steps.push_back(std::move(S));
  }
  if (g_steps.empty() || g_steps[0]->op != "quant_in") throw std::runtime_error("pipe must start with quant_in");
  if (need_htp) {
    std::string ep = lib_dir + "/libonnxruntime_providers_qnn.so";
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", ep.c_str());
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
    LOGI("QNN EP registered, %zu NPU device(s)", npu.size());
  }
  if (need_rpn) {
    if (rpn_rpc_open("file:///librpn_rpc.so?rpn_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h_rpn))
      throw std::runtime_error("rpn_rpc_open failed");
    rpn_init();
    LOGI("rpn skel open");
  }
  if (need_roi) {
    if (roialign_rpc_open("file:///libroialign_rpc.so?roialign_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h_roi))
      throw std::runtime_error("roialign_rpc_open failed");
    LOGI("roialign skel open");
  }
  double t0 = now_ms();
  for (auto& S : g_steps) {
    if (S->op == "ort") {
      S->sess = wrap(make_session(S->f[2], S->f[3], S->f[4], &S->create_ms));
    } else if (S->op == "ortpad") {
      for (auto& e : split(S->f[5], ',')) {
        auto c = e.find(':');
        double ms = 0;
        S->buckets.emplace_back(std::stoi(e.substr(0, c)), wrap(make_session(e.substr(c + 1), S->f[2], S->f[3], &ms)));
        S->create_ms += ms;
      }
    }
    if (S->create_ms > 0) LOGI("session %s create_ms %.1f", S->name.c_str(), S->create_ms);
  }
  LOGI("setup_ms %.1f", now_ms() - t0);
  return 0;
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_Engine_nativeInit(JNIEnv* e, jclass, jstring jdir,
                                                                                    jstring jlib, jstring jpipe) {
  const char* d = e->GetStringUTFChars(jdir, nullptr);
  const char* l = e->GetStringUTFChars(jlib, nullptr);
  const char* p = e->GetStringUTFChars(jpipe, nullptr);
  std::string err;
  try {
    init(d, l, p);
  } catch (const std::exception& ex) {
    err = ex.what();
    if (err.empty()) err = "unknown error";
  }
  e->ReleaseStringUTFChars(jdir, d);
  e->ReleaseStringUTFChars(jlib, l);
  e->ReleaseStringUTFChars(jpipe, p);
  return err.empty() ? nullptr : e->NewStringUTF(err.c_str());
}

// Runs one frame. bmp: RGBA_8888, at most 1088x800 (resized to fit by the caller, top-left aligned).
// Fills boxes[4n] (model-input pixels), labels[n], scores[n], masks[n*784] (28x28 probabilities),
// times[T_N] (ms). Returns n (detections, unfiltered), or -1 on error (see nativeLastError).
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_Engine_nativeRun(JNIEnv* e, jclass, jobject bmp,
                                                                                jfloatArray jboxes, jintArray jlabels,
                                                                                jfloatArray jscores, jfloatArray jmasks,
                                                                                jfloatArray jtimes) {
  AndroidBitmapInfo info;
  void* px = nullptr;
  if (AndroidBitmap_getInfo(e, bmp, &info) || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
      AndroidBitmap_lockPixels(e, bmp, &px)) {
    g_err = "bad bitmap";
    return -1;
  }
  double t[T_N] = {0};
  int n = -1;
  try {
    store.clear();
    double t0 = now_ms();
    quant_rgba(*g_steps[0], (const uint8_t*)px, (int)info.width, (int)info.height, (int)info.stride);
    t[T_PRE] += now_ms() - t0;
    AndroidBitmap_unlockPixels(e, bmp);
    px = nullptr;
    for (size_t k = 1; k < g_steps.size(); ++k) {
      double a = now_ms();
      exec(*g_steps[k]);
      t[bucket(*g_steps[k])] += now_ms() - a;
    }
    t[T_TOTAL] = now_ms() - t0;
    Tensor& b = get("6568");   // boxes  fp32 [n,4]
    Tensor& lb = get("6570");  // labels int64 [n]
    Tensor& sc = get("6572");  // scores fp32 [n]
    Tensor& mk = get("6887");  // masks  fp32 [n,1,28,28]
    n = (int)sc.count();
    int cap = std::min<int>(n, e->GetArrayLength(jscores));
    e->SetFloatArrayRegion(jboxes, 0, 4 * cap, (const float*)b.data);
    std::vector<jint> li(cap);
    for (int i = 0; i < cap; ++i) li[i] = (jint)((const int64_t*)lb.data)[i];
    e->SetIntArrayRegion(jlabels, 0, cap, li.data());
    e->SetFloatArrayRegion(jscores, 0, cap, (const float*)sc.data);
    e->SetFloatArrayRegion(jmasks, 0, cap * 784, (const float*)mk.data);
    n = cap;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    n = -1;
  }
  if (px) AndroidBitmap_unlockPixels(e, bmp);
  float tf[T_N];
  for (int i = 0; i < T_N; ++i) tf[i] = (float)t[i];
  e->SetFloatArrayRegion(jtimes, 0, T_N, tf);
  return n;
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_Engine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}

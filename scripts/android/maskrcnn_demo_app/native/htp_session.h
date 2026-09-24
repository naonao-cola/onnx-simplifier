// One ORT environment with the QNN EP plugin registered, and strict HTP sessions from EP-context
// models, for the demo's single-model engines (yolo_engine.cpp, sam_engine.cpp). Each engine is
// its own library in its own app process, so each has its own copy of these globals.
#pragma once
#include <android/log.h>
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace demo {

inline double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline std::unordered_map<std::string, std::string> parse_opts(const std::string& opts,
                                                                std::unordered_map<std::string, std::string> o) {
  size_t p = 0;
  while (p < opts.size()) {
    size_t q = opts.find(';', p);
    std::string kv = opts.substr(p, q == std::string::npos ? std::string::npos : q - p);
    size_t eq = kv.find('=');
    if (eq != std::string::npos) o[kv.substr(0, eq)] = kv.substr(eq + 1);
    if (q == std::string::npos) break;
    p = q + 1;
  }
  return o;
}

struct Htp {
  std::unique_ptr<Ort::Env> env;
  std::vector<Ort::ConstEpDevice> npu;

  // lib_dir: the app's nativeLibraryDir (ORT, the QNN EP and QNN's HTP skel live there)
  void init(const std::string& lib_dir, const char* tag) {
    if (env) return;
    std::string adsp = lib_dir + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
    Ort::ThreadingOptions to;
    to.SetGlobalIntraOpNumThreads(1);
    to.SetGlobalInterOpNumThreads(1);
    to.SetGlobalSpinControl(0);  // ORT_SPIN=0: pool spinning starves the app's own threads
    env = std::make_unique<Ort::Env>(to, ORT_LOGGING_LEVEL_WARNING, tag);
    std::string ep = lib_dir + "/libonnxruntime_providers_qnn.so";
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", ep.c_str());
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
  }

  // Strict HTP session (no CPU fallback) from <dir>/<stem>.ctx0.onnx, the EP-context model (context
  // binary in its own file), compiled from <dir>/<stem>.onnx on the first launch and reused after.
  std::unique_ptr<Ort::Session> session(const std::string& dir, const std::string& stem, const std::string& perf,
                                        const char* log_tag) {
    Ort::SessionOptions so;
    so.DisablePerSessionThreads();
    so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    std::unordered_map<std::string, std::string> qo{{"backend_type", "htp"}, {"htp_performance_mode", perf}};
    so.AppendExecutionProvider_V2(*env, npu, qo);
    const std::string src = dir + "/" + stem + ".onnx", ctx = dir + "/" + stem + ".ctx0.onnx";
    if (!std::ifstream(ctx)) {
      const double t = now_ms();
      Ort::ModelCompilationOptions co(*env, so);
      co.SetInputModelPath(src.c_str());
      co.SetOutputModelPath(ctx.c_str());
      co.SetEpContextEmbedMode(false);
      Ort::Status st = Ort::CompileModel(*env, co);
      if (!st.IsOK()) throw std::runtime_error("CompileModel " + src + ": " + st.GetErrorMessage());
      __android_log_print(ANDROID_LOG_INFO, log_tag, "compiled %s in %.0f ms", ctx.c_str(), now_ms() - t);
    }
    const double t = now_ms();
    auto s = std::make_unique<Ort::Session>(*env, ctx.c_str(), so);
    __android_log_print(ANDROID_LOG_INFO, log_tag, "session %s in %.0f ms", ctx.c_str(), now_ms() - t);
    return s;
  }
};

}  // namespace demo

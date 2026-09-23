// Multi-input variant of qnn_run.cpp (same env/QNN options), for models with several inputs of
// mixed dtype (rest.onnx has 14; RoiAlign takes int64 batch indices).
//
// usage: qnn_run_multi <model.onnx> <manifest.txt> <mode> <iters> <out_prefix> [ctx.onnx]
//   mode: cpu | htp (strict, no CPU fallback) | htp-fallback
//   manifest: one input per line, "<name> <f32|i64|i32> <file.bin> <d0,d1,...>"
//   outputs: <out_prefix>_o<i>.bin (raw, native dtype) + one "out <i> <name> <dtype> <shape>" line
//   timing: prints every run, then "median_ms" over runs after the first two (warm-up).
//   ORT_PROFILE=<prefix>: enable ORT's per-node profiler, JSON written as <prefix>_*.json
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct In {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::vector<char> data;
};

static size_t esize(const std::string& t) { return t == "i64" ? 8 : 4; }

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s model manifest mode iters out_prefix [ctx]\n", argv[0]);
    return 2;
  }
  const std::string model = argv[1], manifest = argv[2], mode = argv[3];
  const int iters = atoi(argv[4]);
  const std::string out_prefix = argv[5];
  const std::string ctx = argc > 6 ? argv[6] : "";
  const char* qnn_lib = getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB") : "libonnxruntime_providers_qnn.so";
  const int log_level = getenv("ORT_LOG") ? atoi(getenv("ORT_LOG")) : ORT_LOGGING_LEVEL_WARNING;

  try {
    std::vector<In> ins;
    std::ifstream mf(manifest);
    std::string line;
    while (std::getline(mf, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      In in;
      std::string file, dims;
      ss >> in.name >> in.dtype >> file >> dims;
      std::istringstream ds(dims);
      std::string d;
      size_t n = 1;
      while (std::getline(ds, d, ',')) {
        in.shape.push_back(atoll(d.c_str()));
        n *= static_cast<size_t>(in.shape.back());
      }
      in.data.resize(n * esize(in.dtype));
      if (n) {
        std::ifstream f(file, std::ios::binary);
        f.read(in.data.data(), static_cast<std::streamsize>(in.data.size()));
        if (!f) throw std::runtime_error("short input file " + file);
      }
      ins.push_back(std::move(in));
    }

    Ort::Env env(static_cast<OrtLoggingLevel>(log_level), "qnn_run_multi");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(getenv("ORT_THREADS") ? atoi(getenv("ORT_THREADS")) : 1);
    so.SetLogSeverityLevel(log_level);
    if (getenv("ORT_PROFILE")) so.EnableProfiling(getenv("ORT_PROFILE"));

    if (mode != "cpu") {
      env.RegisterExecutionProviderLibrary("QNNExecutionProvider", qnn_lib);
      std::vector<Ort::ConstEpDevice> devs;
      for (const auto& d : env.GetEpDevices())
        if (std::string(d.EpName()) == "QNNExecutionProvider" &&
            d.Device().Type() == OrtHardwareDeviceType_NPU)
          devs.push_back(d);
      if (devs.empty()) throw std::runtime_error("no QNN NPU ep device");
      std::unordered_map<std::string, std::string> opts{{"backend_type", "htp"}};
      if (getenv("QNN_PERF")) opts["htp_performance_mode"] = getenv("QNN_PERF");
      if (getenv("QNN_EXTRA")) {  // k=v,k=v
        std::string s = getenv("QNN_EXTRA");
        size_t p = 0;
        while (p < s.size()) {
          size_t c = s.find(',', p);
          std::string kv = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
          size_t e = kv.find('=');
          if (e != std::string::npos) opts[kv.substr(0, e)] = kv.substr(e + 1);
          if (c == std::string::npos) break;
          p = c + 1;
        }
      }
      if (mode == "htp") so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      so.AppendExecutionProvider_V2(env, devs, opts);
    }

    std::string run_model = model;
    if (!ctx.empty() && mode != "cpu") {
      std::ifstream exists(ctx);
      if (!exists) {
        double t0 = now_ms();
        Ort::ModelCompilationOptions co(env, so);
        co.SetInputModelPath(model.c_str());
        co.SetOutputModelPath(ctx.c_str());
        co.SetEpContextEmbedMode(true);
        Ort::Status st = Ort::CompileModel(env, co);
        if (!st.IsOK()) throw std::runtime_error("CompileModel: " + st.GetErrorMessage());
        printf("compile_ms %.1f\n", now_ms() - t0);
      }
      run_model = ctx;
    }

    double t0 = now_ms();
    Ort::Session sess(env, run_model.c_str(), so);
    printf("session_create_ms %.1f\n", now_ms() - t0);

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> xs;
    std::vector<const char*> in_names;
    for (auto& in : ins) {
      ONNXTensorElementDataType t = in.dtype == "i64"   ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                                    : in.dtype == "i32" ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                                                        : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      xs.push_back(Ort::Value::CreateTensor(mem, in.data.data(), in.data.size(), in.shape.data(),
                                            in.shape.size(), t));
      in_names.push_back(in.name.c_str());
    }

    Ort::AllocatorWithDefaultOptions alloc;
    size_t nout = sess.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> hold;
    std::vector<const char*> out_names;
    for (size_t i = 0; i < nout; ++i) {
      hold.push_back(sess.GetOutputNameAllocated(i, alloc));
      out_names.push_back(hold.back().get());
    }

    std::vector<Ort::Value> outs;
    std::vector<double> ts;
    for (int it = 0; it < iters; ++it) {
      double a = now_ms();
      outs = sess.Run(Ort::RunOptions{nullptr}, in_names.data(), xs.data(), xs.size(),
                      out_names.data(), nout);
      ts.push_back(now_ms() - a);
      printf("run %d %.2f ms\n", it, ts.back());
    }
    if (ts.size() > 2) {
      std::vector<double> s(ts.begin() + 2, ts.end());
      std::sort(s.begin(), s.end());
      printf("median_ms %.2f (n=%zu, min %.2f)\n", s[s.size() / 2], s.size(), s.front());
    }
    if (getenv("ORT_PROFILE")) {
      auto pf = sess.EndProfilingAllocated(alloc);
      printf("profile %s\n", pf.get());
    }
    for (size_t i = 0; i < nout; ++i) {
      auto ti = outs[i].GetTensorTypeAndShapeInfo();
      size_t cnt = ti.GetElementCount();
      auto et = ti.GetElementType();
      size_t es = et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4;
      std::string p = out_prefix + "_o" + std::to_string(i) + ".bin";
      FILE* f = fopen(p.c_str(), "wb");
      if (cnt) fwrite(outs[i].GetTensorRawData(), es, cnt, f);
      fclose(f);
      std::string sh;
      for (auto d : ti.GetShape()) sh += std::to_string(d) + ",";
      printf("out %zu %s %s %s\n", i, out_names[i], es == 8 ? "i64" : "f32", sh.c_str());
    }
    printf("PASS mode=%s\n", mode.c_str());
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    return 1;
  }
  return 0;
}

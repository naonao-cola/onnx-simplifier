// Phone-side runner for the LLM measurements (ORT CPU or QNN HTP, same flags as
// htp_exploration/qnn_shell/qnn_run_multi.cpp).
//
//   llm_run enc <model.onnx> <mode> <iters> <nsets> <indir> <outdir> [ctx.onnx]
//     inputs <indir>/ids_<i>.bin, mask_<i>.bin (int32 [1,S]); times set 0 <iters> times, then runs
//     every set once and writes <outdir>/emb_<i>.bin (raw output 0).
//   llm_run dec <prefill.onnx> <decode.onnx> <mode> <ngen> <nprompts> <indir> <outdir> [pctx] [dctx]
//     prefill: input_ids int32 [1,P] (prompt, right-padded), last_idx int32 [1]
//              -> logits [1,V], k [L,H,P,D], v [L,H,P,D]
//     decode:  input_ids int32 [1,1], pos int32 [1], k_cache/v_cache [L,H,T,D]
//              -> logits [1,V], k_new/v_new [L,H,1,D]
//     The host keeps the KV cache and writes each step's new row at `pos`. Greedy decoding;
//     with <indir>/force_<i>.bin (int32 tokens) the forced tokens are fed instead of the argmax
//     (teacher forcing, for a per-step top-1 agreement rate). Writes gen_<i>.bin (int32 argmax per
//     step) and logits_<i>.bin (raw logits of the first SAVE_LOGITS steps, default 8).
//   mode: cpu | htp (strict, no CPU fallback) | htp-fallback
//   env: ORT_THREADS, QNN_PERF, QNN_EXTRA, ORT_LOG, PREFILL_ITERS (default 5)
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using ET = ONNXTensorElementDataType;

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static size_t esize(ET t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
    default: return 4;
  }
}
static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void write_file(const std::string& p, const void* d, size_t n) {
  FILE* f = fopen(p.c_str(), "wb");
  if (n) fwrite(d, 1, n, f);
  fclose(f);
}
static float h2f(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, o;
  if (e == 0) {
    if (m == 0) o = s << 31;
    else {  // subnormal
      e = 1;
      while (!(m & 1024)) m <<= 1, e--;
      m &= 1023;
      o = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 31) o = (s << 31) | 0x7f800000 | (m << 13);
  else o = (s << 31) | ((e + 112) << 23) | (m << 13);
  float f;
  memcpy(&f, &o, 4);
  return f;
}
static int argmax(const Ort::Value& v) {
  auto ti = v.GetTensorTypeAndShapeInfo();
  size_t n = ti.GetElementCount();
  const void* p = v.GetTensorRawData();
  int best = 0;
  float bv = -1e30f;
  for (size_t i = 0; i < n; ++i) {
    float x;
    switch (ti.GetElementType()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: x = h2f(((const uint16_t*)p)[i]); break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: x = ((const uint16_t*)p)[i]; break;  // monotonic in the real value
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: x = ((const uint8_t*)p)[i]; break;
      default: x = ((const float*)p)[i];
    }
    if (x > bv) bv = x, best = static_cast<int>(i);
  }
  return best;
}
static long vm_hwm_kb() {
  std::ifstream f("/proc/self/status");
  std::string l;
  while (std::getline(f, l))
    if (l.rfind("VmHWM:", 0) == 0) return atol(l.c_str() + 6);
  return -1;
}
static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Sess {
  Ort::Session s{nullptr};
  std::vector<std::string> in_names, out_names;
  std::vector<ET> in_types;
  std::vector<std::vector<int64_t>> in_shapes;
  int in_idx(const std::string& n) const {
    for (size_t i = 0; i < in_names.size(); ++i)
      if (in_names[i] == n) return static_cast<int>(i);
    throw std::runtime_error("no input " + n);
  }
  int out_idx(const std::string& n) const {
    for (size_t i = 0; i < out_names.size(); ++i)
      if (out_names[i] == n) return static_cast<int>(i);
    throw std::runtime_error("no output " + n);
  }
};

static Ort::Env* g_env;

static Ort::SessionOptions make_opts(const std::string& mode) {
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(getenv("ORT_THREADS") ? atoi(getenv("ORT_THREADS")) : 1);
  if (mode == "cpu") return so;
  static bool registered = false;
  const char* qnn_lib = getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB") : "libonnxruntime_providers_qnn.so";
  if (!registered) g_env->RegisterExecutionProviderLibrary("QNNExecutionProvider", qnn_lib), registered = true;
  std::vector<Ort::ConstEpDevice> devs;
  for (const auto& d : g_env->GetEpDevices())
    if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
      devs.push_back(d);
  if (devs.empty()) throw std::runtime_error("no QNN NPU ep device");
  std::unordered_map<std::string, std::string> opts{{"backend_type", "htp"}};
  if (getenv("QNN_PERF")) opts["htp_performance_mode"] = getenv("QNN_PERF");
  if (getenv("QNN_EXTRA")) {
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
  so.AppendExecutionProvider_V2(*g_env, devs, opts);
  return so;
}

static Sess open(const std::string& model, const std::string& mode, const std::string& ctx) {
  Ort::SessionOptions so = make_opts(mode);
  std::string run_model = model;
  if (!ctx.empty() && mode != "cpu") {
    std::ifstream exists(ctx);
    if (!exists) {
      double t0 = now_ms();
      Ort::ModelCompilationOptions co(*g_env, so);
      co.SetInputModelPath(model.c_str());
      co.SetOutputModelPath(ctx.c_str());
      co.SetEpContextEmbedMode(true);
      Ort::Status st = Ort::CompileModel(*g_env, co);
      if (!st.IsOK()) throw std::runtime_error("CompileModel: " + st.GetErrorMessage());
      printf("compile_ms %s %.1f\n", model.c_str(), now_ms() - t0);
    }
    run_model = ctx;
  }
  double t0 = now_ms();
  Sess r;
  r.s = Ort::Session(*g_env, run_model.c_str(), so);
  printf("session_create_ms %s %.1f\n", model.c_str(), now_ms() - t0);
  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < r.s.GetInputCount(); ++i) {
    r.in_names.push_back(r.s.GetInputNameAllocated(i, alloc).get());
    auto ti = r.s.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    r.in_types.push_back(ti.GetElementType());
    r.in_shapes.push_back(ti.GetShape());
  }
  for (size_t i = 0; i < r.s.GetOutputCount(); ++i) r.out_names.push_back(r.s.GetOutputNameAllocated(i, alloc).get());
  return r;
}

static std::vector<Ort::Value> run(Sess& s, std::vector<Ort::Value>& xs) {
  std::vector<const char*> in, out;
  for (auto& n : s.in_names) in.push_back(n.c_str());
  for (auto& n : s.out_names) out.push_back(n.c_str());
  return s.s.Run(Ort::RunOptions{nullptr}, in.data(), xs.data(), xs.size(), out.data(), out.size());
}

static Ort::MemoryInfo& mem() {
  static Ort::MemoryInfo m = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return m;
}
static Ort::Value tensor(void* p, size_t bytes, const std::vector<int64_t>& shape, ET t) {
  return Ort::Value::CreateTensor(mem(), p, bytes, shape.data(), shape.size(), t);
}

static int enc_main(int argc, char** argv) {
  // enc <model> <mode> <iters> <nsets> <indir> <outdir> [ctx]
  if (argc < 8) throw std::runtime_error("enc: bad args");
  std::string model = argv[2], mode = argv[3], indir = argv[6], outdir = argv[7], ctx = argc > 8 ? argv[8] : "";
  int iters = atoi(argv[4]), nsets = atoi(argv[5]);
  Sess s = open(model, mode, ctx);
  auto load = [&](int i) {
    std::vector<std::vector<char>> bufs;
    for (size_t k = 0; k < s.in_names.size(); ++k) {
      std::string nm = s.in_names[k] == "input_ids" ? "ids" : "mask";
      bufs.push_back(read_file(indir + "/" + nm + "_" + std::to_string(i) + ".bin"));
    }
    return bufs;
  };
  std::vector<double> ts;
  for (int i = 0; i < nsets; ++i) {
    auto bufs = load(i);
    std::vector<Ort::Value> xs;
    for (size_t k = 0; k < bufs.size(); ++k) xs.push_back(tensor(bufs[k].data(), bufs[k].size(), s.in_shapes[k], s.in_types[k]));
    int reps = i == 0 ? iters : 1;
    std::vector<Ort::Value> outs;
    for (int r = 0; r < reps; ++r) {
      double a = now_ms();
      outs = run(s, xs);
      if (i == 0 && r >= 2) ts.push_back(now_ms() - a);
    }
    auto ti = outs[0].GetTensorTypeAndShapeInfo();
    write_file(outdir + "/emb_" + std::to_string(i) + ".bin", outs[0].GetTensorRawData(),
               ti.GetElementCount() * esize(ti.GetElementType()));
  }
  printf("enc_median_ms %.3f (n=%zu, min %.3f)\n", median(ts), ts.size(), ts.empty() ? 0 : *std::min_element(ts.begin(), ts.end()));
  printf("vm_hwm_kb %ld\n", vm_hwm_kb());
  printf("PASS enc mode=%s\n", mode.c_str());
  return 0;
}

// f32 <-> f16 for caches whose prefill and decode dtypes differ (not expected, but cheap to support)
static uint16_t f2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 255, m = x & 0x7fffff;
  if (e == 255) return s | 0x7c00 | (m ? 0x200 : 0);
  int ne = static_cast<int>(e) - 112;
  if (ne >= 31) return s | 0x7c00;
  if (ne <= 0) {
    if (ne < -10) return s;
    m |= 0x800000;
    uint32_t sh = 14 - ne;
    uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (r & 1))) r++;
    return s | r;
  }
  uint32_t r = (ne << 10) | (m >> 13), rem = m & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
  return s | r;
}

static int dec_main(int argc, char** argv) {
  // dec <prefill> <decode> <mode> <ngen> <nprompts> <indir> <outdir> [pctx] [dctx]
  if (argc < 9) throw std::runtime_error("dec: bad args");
  std::string pre_m = argv[2], dec_m = argv[3], mode = argv[4], indir = argv[7], outdir = argv[8];
  std::string pctx = argc > 9 ? argv[9] : "", dctx = argc > 10 ? argv[10] : "";
  int ngen = atoi(argv[5]), nprompts = atoi(argv[6]);
  int save_logits = getenv("SAVE_LOGITS") ? atoi(getenv("SAVE_LOGITS")) : 8;
  int pre_iters = getenv("PREFILL_ITERS") ? atoi(getenv("PREFILL_ITERS")) : 5;
  Sess P = open(pre_m, mode, pctx), D = open(dec_m, mode, dctx);

  const int p_ids = P.in_idx("input_ids"), p_last = P.in_idx("last_idx");
  const int d_ids = D.in_idx("input_ids"), d_pos = D.in_idx("pos"), d_k = D.in_idx("k_cache"), d_v = D.in_idx("v_cache");
  const int P_len = static_cast<int>(P.in_shapes[p_ids][1]);
  const auto cshape = D.in_shapes[d_k];  // [L,H,T,D]
  const int64_t L = cshape[0], H = cshape[1], T = cshape[2], Dh = cshape[3];
  const ET ct = D.in_types[d_k];
  const size_t ce = esize(ct), row = static_cast<size_t>(Dh) * ce;
  printf("prefill_len %d cache [%lld,%lld,%lld,%lld] esize %zu\n", P_len, (long long)L, (long long)H, (long long)T,
         (long long)Dh, ce);
  std::vector<char> kc(static_cast<size_t>(L * H * T) * row), vc(kc.size());

  // copy rows [0, n) of src [L,H,S,D] (dtype st) into cache at positions [p0, p0+n)
  auto put = [&](std::vector<char>& cache, const Ort::Value& src, int64_t S, int64_t n, int64_t p0) {
    ET st = src.GetTensorTypeAndShapeInfo().GetElementType();
    const char* sp = static_cast<const char*>(src.GetTensorRawData());
    size_t se = esize(st);
    for (int64_t l = 0; l < L; ++l)
      for (int64_t h = 0; h < H; ++h)
        for (int64_t j = 0; j < n; ++j) {
          const char* s = sp + (((l * H + h) * S + j) * Dh) * se;
          char* d = cache.data() + (((l * H + h) * T + p0 + j) * Dh) * ce;
          if (st == ct) memcpy(d, s, row);
          else if (st == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && ct == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            for (int64_t e = 0; e < Dh; ++e) ((uint16_t*)d)[e] = f2h(((const float*)s)[e]);
          else if (st == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && ct == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            for (int64_t e = 0; e < Dh; ++e) ((float*)d)[e] = h2f(((const uint16_t*)s)[e]);
          else throw std::runtime_error("cache dtype mismatch between prefill and decode");
        }
  };

  std::vector<double> pre_ts, dec_ts;
  for (int pi = 0; pi < nprompts; ++pi) {
    std::vector<char> pb = read_file(indir + "/prompt_" + std::to_string(pi) + ".bin");
    std::vector<char> fb = read_file(indir + "/force_" + std::to_string(pi) + ".bin");
    int n = static_cast<int>(pb.size() / 4);
    if (n < 1 || n > P_len) throw std::runtime_error("prompt length");
    std::vector<int32_t> ids(P_len, 0);
    memcpy(ids.data(), pb.data(), pb.size());
    int32_t last = n - 1;
    std::vector<Ort::Value> xs;
    for (size_t k = 0; k < P.in_names.size(); ++k) {
      if (static_cast<int>(k) == p_ids) xs.push_back(tensor(ids.data(), ids.size() * 4, P.in_shapes[k], P.in_types[k]));
      else if (static_cast<int>(k) == p_last) xs.push_back(tensor(&last, 4, P.in_shapes[k], P.in_types[k]));
      else throw std::runtime_error("unexpected prefill input " + P.in_names[k]);
    }
    std::vector<Ort::Value> po;
    for (int r = 0; r < (pi == 0 ? pre_iters : 1); ++r) {
      double a = now_ms();
      po = run(P, xs);
      if (!(pi == 0 && r < 2 && pre_iters > 2)) pre_ts.push_back(now_ms() - a);
    }
    put(kc, po[P.out_idx("k")], P_len, n, 0);
    put(vc, po[P.out_idx("v")], P_len, n, 0);
    std::vector<int32_t> gen;
    std::vector<char> logits;
    auto keep_logits = [&](const Ort::Value& lv) {
      if (static_cast<int>(gen.size()) > save_logits) return;
      auto ti = lv.GetTensorTypeAndShapeInfo();
      const char* p = static_cast<const char*>(lv.GetTensorRawData());
      logits.insert(logits.end(), p, p + ti.GetElementCount() * esize(ti.GetElementType()));
    };
    gen.push_back(argmax(po[P.out_idx("logits")]));
    keep_logits(po[P.out_idx("logits")]);
    const int od_l = D.out_idx("logits"), od_k = D.out_idx("k_new"), od_v = D.out_idx("v_new");
    for (int s = 1; s < ngen && n + s - 1 < T; ++s) {
      int32_t tok = fb.size() >= static_cast<size_t>(s) * 4 ? reinterpret_cast<const int32_t*>(fb.data())[s - 1] : gen.back();
      int32_t pos = n + s - 1;
      std::vector<Ort::Value> dx;
      for (size_t k = 0; k < D.in_names.size(); ++k) {
        int ki = static_cast<int>(k);
        if (ki == d_ids) dx.push_back(tensor(&tok, 4, D.in_shapes[k], D.in_types[k]));
        else if (ki == d_pos) dx.push_back(tensor(&pos, 4, D.in_shapes[k], D.in_types[k]));
        else if (ki == d_k) dx.push_back(tensor(kc.data(), kc.size(), D.in_shapes[k], D.in_types[k]));
        else if (ki == d_v) dx.push_back(tensor(vc.data(), vc.size(), D.in_shapes[k], D.in_types[k]));
        else throw std::runtime_error("unexpected decode input " + D.in_names[k]);
      }
      double a = now_ms();
      auto o = run(D, dx);
      put(kc, o[od_k], 1, 1, pos);
      put(vc, o[od_v], 1, 1, pos);
      int nt = argmax(o[od_l]);
      dec_ts.push_back(now_ms() - a);  // one decode step incl. the host-side cache write
      gen.push_back(nt);
      keep_logits(o[od_l]);
    }
    write_file(outdir + "/gen_" + std::to_string(pi) + ".bin", gen.data(), gen.size() * 4);
    write_file(outdir + "/logits_" + std::to_string(pi) + ".bin", logits.data(), logits.size());
  }
  double dm = median(dec_ts);
  printf("prefill_median_ms %.2f (n=%zu)\n", median(pre_ts), pre_ts.size());
  printf("decode_median_ms %.3f (n=%zu) tok_s %.1f\n", dm, dec_ts.size(), dm > 0 ? 1000.0 / dm : 0);
  printf("vm_hwm_kb %ld\n", vm_hwm_kb());
  printf("PASS dec mode=%s\n", mode.c_str());
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s enc|dec ...\n", argv[0]);
    return 2;
  }
  try {
    const int lvl = getenv("ORT_LOG") ? atoi(getenv("ORT_LOG")) : ORT_LOGGING_LEVEL_WARNING;
    Ort::Env env(static_cast<OrtLoggingLevel>(lvl), "llm_run");
    g_env = &env;
    std::string cmd = argv[1];
    if (cmd == "enc") return enc_main(argc, argv);
    if (cmd == "dec") return dec_main(argc, argv);
    throw std::runtime_error("unknown command " + cmd);
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    return 1;
  }
}

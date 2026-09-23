// One process, the whole Mask R-CNN inference on the phone: runs a pipe_<stage>.txt from
// build_models.py over preprocessed images. ORT sessions run on the HTP (QNN EP plugin, strict: no
// CPU fallback) or the CPU; the fused RPN span and RoiAlign run on the CDSP through their own
// FastRPC skels (../tinygrad_hexagon_bridge/rpn_fused, roialign_fast), sharing rpcmem buffers.
//
// usage: e2e_run <pipe.txt> <warmup> <reps> <out_dir|-> <image.bin>...
//   per image: 1 untimed-for-median first run (reported as "cold" for the first image), <warmup>
//   more, then <reps> timed runs; prints the median per-step and total wall ms, and writes the four
//   final outputs of the last run to <out_dir>/<stem>_<k>.bin (+ .shape) for the host comparison.
// env: ORT_THREADS (CPU intra-op, default 4, one global pool shared by every CPU session),
//      QNN_EP_LIB, RPN_URI, ROI_URI, RPN_MODE (default 3 = phased), ROI_THREADS (default 104 =
//      4 threads + l2fetch prefetch), DQ_THREADS (default 4).
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

extern "C" {
#include "remote.h"
#include "rpcmem.h"
#include "roialign_rpc.h"
#include "rpn_rpc.h"
int rpn_glue_set_model(remote_handle64 h, int* post_cap, int* lv3);  // rpn_glue.c
}

#define RPN_NLVL 5
#define RPN_STATS (2 + 3 * RPN_NLVL + 4 * RPN_NLVL + 1 + RPN_NLVL + 4)

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> r;
  if (s == "-") return r;
  std::string cur;
  std::istringstream ss(s);
  while (std::getline(ss, cur, c)) r.push_back(cur);
  return r;
}
static int envi(const char* k, int d) { return getenv(k) ? atoi(getenv(k)) : d; }

// ---- tensor store -------------------------------------------------------------------------------
struct RpcBuf {  // grow-only rpcmem allocation (ION-backed; the DSP maps it instead of copying)
  void* p = nullptr;
  size_t cap = 0;
  void* get(size_t n) {
    if (n > cap) {
      if (p) rpcmem_free(p);
      cap = std::max<size_t>(n, 128);
      p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, cap);
      if (!p) throw std::runtime_error("rpcmem_alloc failed");
    }
    return p;
  }
  ~RpcBuf() { if (p) rpcmem_free(p); }
};

struct Tensor {
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
  void* data = nullptr;
  std::shared_ptr<Ort::Value> val;  // keeps an ORT output alive
  size_t count() const { size_t n = 1; for (auto d : shape) n *= (size_t)d; return n; }
};
static size_t esize(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
    default: return 4;
  }
}
// The demo app's engine (../maskrcnn_demo_app) pipelines frames across threads and defines this as
// `static thread_local` so each pipeline stage has its own tensor store; unchanged here.
#ifndef E2E_STORE_STORAGE
#define E2E_STORE_STORAGE static
#endif
E2E_STORE_STORAGE std::map<std::string, Tensor> store;
static Tensor& get(const std::string& n) {
  auto it = store.find(n);
  if (it == store.end()) throw std::runtime_error("missing tensor " + n);
  return it->second;
}
static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
static Ort::Value view(Tensor& t, const std::vector<int64_t>* shape = nullptr) {
  const auto& s = shape ? *shape : t.shape;
  size_t n = 1;
  for (auto d : s) n *= (size_t)d;
  return Ort::Value::CreateTensor(cpu_mem, t.data, n * esize(t.type), s.data(), s.size(), t.type);
}
static void put_value(const std::string& n, Ort::Value&& v) {
  Tensor t;
  auto info = v.GetTensorTypeAndShapeInfo();
  t.type = info.GetElementType();
  t.shape = info.GetShape();
  t.val = std::make_shared<Ort::Value>(std::move(v));
  t.data = t.count() ? t.val->GetTensorMutableRawData() : nullptr;
  store[n] = t;
}
static void put_raw(const std::string& n, ONNXTensorElementDataType ty, std::vector<int64_t> shape, void* data) {
  Tensor t;
  t.type = ty;
  t.shape = std::move(shape);
  t.data = data;
  store[n] = t;
}

// ---- ORT ---------------------------------------------------------------------------------------
static std::unique_ptr<Ort::Env> env;
static std::vector<Ort::ConstEpDevice> npu;

static std::unique_ptr<Ort::Session> make_session(const std::string& model, const std::string& ep, const std::string& opts,
                                                  double* create_ms) {
  Ort::SessionOptions so;
  so.DisablePerSessionThreads();
  so.SetLogSeverityLevel(envi("ORT_LOG", ORT_LOGGING_LEVEL_WARNING));
  std::string path = model;
  bool ctx = false;
  if (ep == "htp") {
    std::unordered_map<std::string, std::string> o{{"backend_type", "htp"}};
    for (auto& kv : split(opts, ';')) {
      auto e = kv.find('=');
      if (e == std::string::npos) continue;
      if (kv.substr(0, e) == "ctx") { ctx = kv.substr(e + 1) == "1"; continue; }
      o[kv.substr(0, e)] = kv.substr(e + 1);
    }
    so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    so.AppendExecutionProvider_V2(*env, npu, o);
    if (ctx) {
      path = model.substr(0, model.size() - 5) + ".ctx.onnx";
      std::ifstream exists(path);
      if (!exists) {
        Ort::ModelCompilationOptions co(*env, so);
        co.SetInputModelPath(model.c_str());
        co.SetOutputModelPath(path.c_str());
        co.SetEpContextEmbedMode(true);
        Ort::Status st = Ort::CompileModel(*env, co);
        if (!st.IsOK()) throw std::runtime_error("CompileModel " + model + ": " + st.GetErrorMessage());
      }
    }
  }
  double t0 = now_ms();
  auto s = std::make_unique<Ort::Session>(*env, path.c_str(), so);
  *create_ms = now_ms() - t0;
  return s;
}

struct Session {
  std::unique_ptr<Ort::Session> s;
  std::vector<std::string> in, out;
  std::vector<std::vector<int64_t>> in_shape;
};
static Session wrap(std::unique_ptr<Ort::Session> s) {
  Session r;
  Ort::AllocatorWithDefaultOptions a;
  for (size_t i = 0; i < s->GetInputCount(); ++i) {
    r.in.push_back(s->GetInputNameAllocated(i, a).get());
    r.in_shape.push_back(s->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < s->GetOutputCount(); ++i) r.out.push_back(s->GetOutputNameAllocated(i, a).get());
  r.s = std::move(s);
  return r;
}
static void run_session(Session& S, Ort::Value* in0 = nullptr) {
  std::vector<Ort::Value> xs;
  std::vector<const char*> in_names, out_names;
  for (size_t i = 0; i < S.in.size(); ++i) {
    xs.push_back(i == 0 && in0 ? std::move(*in0) : view(get(S.in[i])));
    in_names.push_back(S.in[i].c_str());
  }
  for (auto& o : S.out) out_names.push_back(o.c_str());
  auto ys = S.s->Run(Ort::RunOptions{nullptr}, in_names.data(), xs.data(), xs.size(), out_names.data(), out_names.size());
  for (size_t i = 0; i < ys.size(); ++i) put_value(S.out[i], std::move(ys[i]));
}

// ---- parallel helper ---------------------------------------------------------------------------
template <class F>
static void par(long n, int threads, F f) {
  std::vector<std::thread> th;
  long per = (n + threads - 1) / threads;
  for (int t = 0; t < threads; ++t) {
    long a = t * per, b = std::min(n, a + per);
    if (a >= b) break;
    th.emplace_back([=] { f(a, b); });
  }
  for (auto& x : th) x.join();
}

// ---- steps -------------------------------------------------------------------------------------
struct Step {
  std::string op, name;
  std::vector<std::string> f;  // raw fields
  Session sess;
  std::vector<std::pair<int, Session>> buckets;
  std::vector<char> pad;
  RpcBuf buf, buf2, buf3, buf4;
  double create_ms = 0;
  unsigned long long dsp_us = 0;
};

static remote_handle64 h_rpn = 0, h_roi = 0;
static int rpn_post_cap = 1000;
static std::vector<std::array<int, 3>> rpn_lv;  // A, H, W per level

static void rpn_init() {
  int lv[3 * RPN_NLVL];
  int rc = rpn_glue_set_model(h_rpn, &rpn_post_cap, lv);
  if (rc) throw std::runtime_error("rpn set_model rc=" + std::to_string(rc));
  for (int l = 0; l < RPN_NLVL; ++l) rpn_lv.push_back({lv[3 * l], lv[3 * l + 1], lv[3 * l + 2]});
}

static void exec(Step& S) {
  const auto& f = S.f;
  if (S.op == "ort") {
    run_session(S.sess);
  } else if (S.op == "ortpad") {
    Tensor& x = get(f[4]);
    int64_t n = x.shape[0];
    Session* B = nullptr;
    int b = 0;
    for (auto& kv : S.buckets)
      if (kv.first >= n && (!B || kv.first < b)) { B = &kv.second; b = kv.first; }
    if (!B) throw std::runtime_error("no bucket for n=" + std::to_string(n));
    size_t row = x.count() / std::max<int64_t>(n, 1) * esize(x.type);
    if (n == 0) row = 0;
    size_t need = (size_t)b * (row ? row : 1);
    if (S.pad.size() < need) S.pad.assign(need, 0);
    if (row) {
      memcpy(S.pad.data(), x.data, n * row);
      memset(S.pad.data() + n * row, 0, (b - n) * row);
    }
    Tensor tmp = x;
    tmp.data = S.pad.data();
    Ort::Value v = view(tmp, &B->in_shape[0]);
    run_session(*B, &v);
    for (auto& o : split(f[7], ',')) get(o).shape[0] = n;  // rows [0,n) of the padded result
  } else if (S.op == "quant_in") {
    Tensor& x = get(f[1]);
    const float s = std::stof(f[3]);
    const int z = std::stoi(f[4]);
    const int C = (int)x.shape[0], H = (int)x.shape[1], W = (int)x.shape[2];
    uint8_t* q = (uint8_t*)S.buf.get((size_t)H * W * C);
    const float* src = (const float*)x.data;
    // QuantizeLinear: saturate(round_half_even(x / s) + z). Row-parallel, each channel plane read
    // contiguously; __builtin_rintf is a single frintx (default rounding mode = half-to-even), and
    // the division is kept (x * (1/s) can round differently).
    par(H, envi("DQ_THREADS", 4), [&](long a, long b) {
      for (long y = a; y < b; ++y)
        for (int c = 0; c < C; ++c) {
          const float* in = src + ((long)c * H + y) * W;
          uint8_t* o = q + (long)y * W * C + c;
          for (int xx = 0; xx < W; ++xx) {
            float v = __builtin_rintf(in[xx] / s) + (float)z;
            v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
            o[(long)xx * C] = (uint8_t)v;
          }
        }
    });
    put_raw(f[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, H, W, C}, q);
  } else if (S.op == "dq") {
    Tensor& x = get(f[1]);
    const float s = std::stof(f[3]);
    const int z = std::stoi(f[4]);
    long n = (long)x.count();
    float* o = (float*)S.buf.get(n * 4);
    const uint8_t* q = (const uint8_t*)x.data;
    par(n, envi("DQ_THREADS", 4), [&](long a, long b) {
      for (long i = a; i < b; ++i) o[i] = (float)((int)q[i] - z) * s;  // DequantizeLinear
    });
    put_raw(f[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, x.shape, o);
  } else if (S.op == "rpn") {
    int src = std::stoi(f[1]);
    auto sc = split(f[2], ','), dl = split(f[3], ',');
    long stot = 0, dtot = 0, ntot = 0;
    for (auto& l : rpn_lv) { stot += l[0]; dtot += 4L * l[0]; ntot += 12L * l[1] * l[2]; }
    float* s = (float*)S.buf.get(stot * 4);
    long o = 0;
    for (auto& n : sc) { Tensor& t = get(n); memcpy(s + o, t.data, t.count() * 4); o += (long)t.count(); }
    if (o != stot) throw std::runtime_error("rpn: score count");
    float* d = (float*)S.buf2.get(src ? 16 : dtot * 4);
    uint8_t* nq = (uint8_t*)S.buf3.get(src ? ntot : 16);
    o = 0;
    for (auto& n : dl) {
      Tensor& t = get(n);
      size_t bytes = t.count() * esize(t.type);
      memcpy((src ? (char*)nq : (char*)d) + o, t.data, bytes);
      o += (long)bytes;
    }
    if (o != (src ? ntot : dtot * 4)) throw std::runtime_error("rpn: delta bytes");
    float* out = (float*)S.buf4.get(16L * rpn_post_cap);
    int st[RPN_STATS];
    unsigned long long du = 0;
    int rc = rpn_rpc_run(h_rpn, s, (int)stot, d, src ? 0 : (int)dtot, nq, src ? (int)ntot : 0, src, envi("RPN_MODE", 3),
                         4, 1, out, 4 * rpn_post_cap, st, RPN_STATS, &du);
    if (rc) throw std::runtime_error("rpn_rpc_run rc=" + std::to_string(rc));
    S.dsp_us = du;
    put_raw(f[4], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {st[0], 4}, out);
  } else if (S.op == "roialign") {
    Tensor& x = get(f[1]);
    Tensor& r = get(f[2]);
    int OH = std::stoi(f[4]), OW = std::stoi(f[5]), sr = std::stoi(f[6]);
    float scale = std::stof(f[7]);
    int H = (int)x.shape[1], W = (int)x.shape[2], C = (int)x.shape[3];
    int R = (int)r.shape[0];
    long on = (long)R * OH * OW * C;
    float* out = (float*)S.buf.get(std::max(on, 1L) * 4);
    S.dsp_us = 0;
    if (R) {
      float* rois = (float*)S.buf2.get((size_t)R * 16);
      memcpy(rois, r.data, (size_t)R * 16);
      unsigned long long du = 0;
      int rc = roialign_rpc_run(h_roi, (const float*)x.data, (int)x.count(), rois, 4 * R, H, W, C, OH, OW, sr, scale,
                                envi("ROI_THREADS", 104), out, (int)on, &du);
      if (rc) throw std::runtime_error("roialign_rpc_run rc=" + std::to_string(rc));
      S.dsp_us = du;
    }
    put_raw(f[3], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {R, C, OH, OW}, out);  // (R,OH,OW,C) rows
  } else {
    throw std::runtime_error("unknown op " + S.op);
  }
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s pipe.txt warmup reps out_dir image.bin...\n", argv[0]);
    return 2;
  }
  const std::string pipe = argv[1], out_dir = argv[4];
  const int warmup = atoi(argv[2]), reps = atoi(argv[3]);
  try {
    struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);

    Ort::ThreadingOptions to;
    to.SetGlobalIntraOpNumThreads(envi("ORT_THREADS", 4));
    to.SetGlobalInterOpNumThreads(1);
    env = std::make_unique<Ort::Env>(to, static_cast<OrtLoggingLevel>(envi("ORT_LOG", ORT_LOGGING_LEVEL_WARNING)), "e2e");

    std::vector<std::unique_ptr<Step>> steps;
    std::ifstream pf(pipe);
    std::string line;
    bool need_htp = false, need_rpn = false, need_roi = false;
    while (std::getline(pf, line)) {
      if (line.empty()) continue;
      auto S = std::make_unique<Step>();
      std::istringstream ss(line);
      std::string w;
      while (ss >> w) S->f.push_back(w);
      S->op = S->f[0];
      S->name = S->op == "ort" || S->op == "ortpad" ? S->f[1] : S->op + ":" + S->f[S->op == "rpn" ? 4 : S->op == "roialign" ? 3 : 2];
      need_htp |= (S->op == "ort" && S->f[3] == "htp") || (S->op == "ortpad" && S->f[2] == "htp");
      need_rpn |= S->op == "rpn";
      need_roi |= S->op == "roialign";
      steps.push_back(std::move(S));
    }
    if (need_htp) {
      env->RegisterExecutionProviderLibrary("QNNExecutionProvider",
                                            getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB") : "libonnxruntime_providers_qnn.so");
      for (const auto& d : env->GetEpDevices())
        if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
          npu.push_back(d);
      if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
    }
    if (need_rpn) {
      const char* u = getenv("RPN_URI") ? getenv("RPN_URI") : "file:///rpn_rpc.so?rpn_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp";
      if (rpn_rpc_open(u, &h_rpn)) throw std::runtime_error("rpn_rpc_open failed");
      rpn_init();
    }
    if (need_roi) {
      const char* u = getenv("ROI_URI") ? getenv("ROI_URI") : "file:///roialign_rpc.so?roialign_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp";
      if (roialign_rpc_open(u, &h_roi)) throw std::runtime_error("roialign_rpc_open failed");
    }
    double t_setup = now_ms();
    for (auto& S : steps) {
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
      if (S->create_ms > 0) printf("session %s create_ms %.1f\n", S->name.c_str(), S->create_ms);
    }
    printf("setup_ms %.1f\n", now_ms() - t_setup);

    std::vector<float> image;
    bool first = true;
    for (int im = 5; im < argc; ++im) {
      const std::string path = argv[im];
      std::string stem = path.substr(path.find_last_of('/') + 1);
      stem = stem.substr(0, stem.size() - 4);
      image.assign(3L * 800 * 1088, 0.f);
      std::ifstream in(path, std::ios::binary);
      in.read((char*)image.data(), image.size() * 4);
      if (!in) throw std::runtime_error("short image " + path);
      std::vector<std::vector<double>> st(steps.size());
      std::vector<std::vector<double>> dsp(steps.size());
      std::vector<double> tot;
      for (int it = 0; it < 1 + warmup + reps; ++it) {
        store.clear();
        put_raw("image", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {3, 800, 1088}, image.data());
        double t0 = now_ms();
        for (size_t k = 0; k < steps.size(); ++k) {
          double a = now_ms();
          exec(*steps[k]);
          if (it > warmup) { st[k].push_back(now_ms() - a); dsp[k].push_back(steps[k]->dsp_us / 1000.0); }
        }
        double t = now_ms() - t0;
        if (it == 0 && first) printf("%s cold_ms %.1f\n", stem.c_str(), t);
        if (it > warmup) tot.push_back(t);
      }
      first = false;
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; };
      printf("%s total_ms median %.2f min %.2f (n=%zu)\n", stem.c_str(), med(tot), *std::min_element(tot.begin(), tot.end()),
             tot.size());
      for (size_t k = 0; k < steps.size(); ++k) {
        double d = med(dsp[k]);
        printf("  step %-14s %8.2f ms%s\n", steps[k]->name.c_str(), med(st[k]),
               d > 0 ? (" (dsp " + std::to_string(d).substr(0, 6) + " ms)").c_str() : "");
      }
      if (out_dir != "-") {
        const char* fin[] = {"6568", "6570", "6572", "6887"};
        for (int k = 0; k < 4; ++k) {
          Tensor& t = get(fin[k]);
          std::string p = out_dir + "/" + stem + "_" + std::to_string(k);
          FILE* fo = fopen((p + ".bin").c_str(), "wb");
          if (t.count()) fwrite(t.data, esize(t.type), t.count(), fo);
          fclose(fo);
          FILE* fs = fopen((p + ".shape").c_str(), "w");
          fprintf(fs, "%d", (int)t.type);
          for (auto d : t.shape) fprintf(fs, " %lld", (long long)d);
          fprintf(fs, "\n");
          fclose(fs);
        }
      }
      fflush(stdout);
    }
    if (h_rpn) rpn_rpc_close(h_rpn);
    if (h_roi) roialign_rpc_close(h_roi);
    printf("PASS\n");
    fflush(stdout);
    _exit(0);  // skip static destructors: ORT's and ours tear down in an order that aborts on exit
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    return 1;
  }
  return 0;
}

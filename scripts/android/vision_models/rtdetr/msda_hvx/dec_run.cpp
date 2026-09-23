// RT-DETR on the phone, split around the HVX MSDA kernel (split.py): 4 HTP pieces (ORT + QNN EP,
// strict all-HTP) and 3 msda calls on the CDSP (../../../msda_hvx skel), chained in-process:
//
//   pre -> [msda0] -> mid0 -> [msda1] -> mid1 -> [msda2] -> post
//
// Every tensor lives in its own rpcmem (ION) buffer: ORT writes the HTP pieces' outputs straight into
// them (pre-bound output tensors) and the DSP maps them without a copy.
//
// usage: dec_run <piece dir> <pre model file> <warmup> <reps> <image dir>...
//   piece dir: <pre model file> (pre.sim.onnx: pixels f32 NCHW; pre.front8.onnx: uint8 NHWC image),
//              mid0.sim.onnx mid1.sim.onnx post.sim.onnx
//   image dir: pixels.f32 (1,3,640,640) or image.u8 (1,640,640,3), matching the pre model's input
//   writes <image dir>/logits.f32 (300,80), boxes.f32 (300,4); prints per-step and total medians.
// env: MSDA_URI, MSDA_FLAGS (threads | 256 * queries-per-job/16, default 4), QNN_PERF, QNN_EXTRA, ORT_LOG
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

extern "C" {
#include "msda_rpc.h"
#include "msda_shape.h"
#include "remote.h"
#include "rpcmem.h"
}

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static size_t esize(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return 2;
    default: return 4;
  }
}

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};
static std::map<std::string, Buf> store;

static Buf& buf(const std::string& name, std::vector<int64_t> shape,
                ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
  size_t n = esize(type);
  for (auto d : shape) n *= (size_t)d;
  Buf& b = store[name];
  if (!b.p) {
    b.p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, std::max<size_t>(n, 128));
    if (!b.p) throw std::runtime_error("rpcmem_alloc " + name);
    b.bytes = n;
    b.shape = shape;
    b.type = type;
  } else if (b.bytes != n) {
    throw std::runtime_error("shape mismatch for " + name);
  }
  return b;
}
static Buf& get(const std::string& name) {
  auto it = store.find(name);
  if (it == store.end()) throw std::runtime_error("missing tensor " + name);
  return it->second;
}

static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
static Ort::Value view(Buf& b) {
  return Ort::Value::CreateTensor(cpu_mem, b.p, b.bytes, b.shape.data(), b.shape.size(), b.type);
}

struct Piece {
  std::string name;
  std::unique_ptr<Ort::Session> s;
  std::vector<std::string> in, out;
  std::vector<std::vector<int64_t>> in_shape, out_shape;
  std::vector<ONNXTensorElementDataType> in_type, out_type;
};

static std::unique_ptr<Ort::Env> env;
static std::vector<Ort::ConstEpDevice> npu;

static Piece load(const std::string& path, const std::string& name, double* ms) {
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(1);
  so.SetLogSeverityLevel(getenv("ORT_LOG") ? atoi(getenv("ORT_LOG")) : ORT_LOGGING_LEVEL_WARNING);
  std::unordered_map<std::string, std::string> o{{"backend_type", "htp"}};
  if (getenv("QNN_PERF")) o["htp_performance_mode"] = getenv("QNN_PERF");
  if (getenv("QNN_EXTRA")) {
    std::string s = getenv("QNN_EXTRA");
    size_t p = 0;
    while (p < s.size()) {
      size_t c = s.find(',', p);
      std::string kv = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
      size_t e = kv.find('=');
      if (e != std::string::npos) o[kv.substr(0, e)] = kv.substr(e + 1);
      if (c == std::string::npos) break;
      p = c + 1;
    }
  }
  so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  so.AppendExecutionProvider_V2(*env, npu, o);
  Piece P;
  P.name = name;
  double t0 = now_ms();
  P.s = std::make_unique<Ort::Session>(*env, path.c_str(), so);
  *ms = now_ms() - t0;
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

// Runs a piece: input n reads store[alias[n]] (default: n), output n writes store[name + "/" + n].
static void run(Piece& P, const std::map<std::string, std::string>& alias = {}) {
  std::vector<Ort::Value> xs, ys;
  std::vector<const char*> in_names, out_names;
  for (auto& n : P.in) {
    auto it = alias.find(n);
    xs.push_back(view(get(it == alias.end() ? n : it->second)));
    in_names.push_back(n.c_str());
  }
  for (size_t i = 0; i < P.out.size(); ++i) {
    ys.push_back(view(buf(P.name + "/" + P.out[i], P.out_shape[i], P.out_type[i])));
    out_names.push_back(P.out[i].c_str());
  }
  P.s->Run(Ort::RunOptions{nullptr}, in_names.data(), xs.data(), xs.size(), out_names.data(), ys.data(), ys.size());
}

static remote_handle64 h_msda = 0;
static unsigned long long last_dsp_us = 0;

// RT-DETR-r18 decoder cross-attention: 1 value map, 3 levels, 8 heads x 32, 4 points, 300 queries,
// raw offsets + box reference points (MSDA_REF_BOX).
static void msda(const std::string& value, const std::string& off, const std::string& w, const std::string& ref,
                 const std::string& out) {
  msda_args_t a;
  memset(&a, 0, sizeof a);
  const int hw[3][2] = {{80, 80}, {40, 40}, {20, 20}};
  a.NV = 1; a.L = 3; a.M = 8; a.D = 32; a.P = 4; a.Q = 300; a.NO = 1; a.mode = MSDA_REF_BOX;
  a.NVR = 1; a.RL = 1; a.R = 1; a.RD = 4;
  int s = 0;
  for (int l = 0; l < 3; ++l) { a.H[l] = hw[l][0]; a.W[l] = hw[l][1]; a.start[l] = s; s += hw[l][0] * hw[l][1]; }
  a.S = s;
  Buf &v = get(value), &o = get(off), &at = get(w), &r = get(ref);
  Buf& y = buf(out, {a.Q, a.M * a.D});
  a.value = (const float*)v.p; a.loc = (const float*)o.p; a.attw = (const float*)at.p; a.ref = (const float*)r.p;
  if ((long)v.bytes / 4 < msda_n_value(&a) || (long)o.bytes / 4 < msda_n_loc(&a) || (long)at.bytes / 4 < msda_n_attw(&a) ||
      (long)r.bytes / 4 < msda_n_ref(&a))
    throw std::runtime_error("msda buffer too small");
  int32 shape[MSDA_SHAPE_LEN(MSDA_MAX_L)];
  const int ns = msda_shape_pack(&a, shape);
  uint64 us = 0;
  int flags = getenv("MSDA_FLAGS") ? atoi(getenv("MSDA_FLAGS")) : 4;
  int rc = msda_rpc_run(h_msda, a.value, (int)msda_n_value(&a), a.loc, (int)msda_n_loc(&a), a.ref, (int)msda_n_ref(&a),
                        a.attw, (int)msda_n_attw(&a), nullptr, 0, shape, ns, flags, (float*)y.p, (int)msda_n_out(&a), &us);
  if (rc) throw std::runtime_error("msda_rpc_run rc=" + std::to_string(rc));
  last_dsp_us = us;
}

static void read_into(const std::string& path, Buf& b) {
  std::ifstream f(path, std::ios::binary);
  f.read((char*)b.p, (std::streamsize)b.bytes);
  if (!f) throw std::runtime_error("short/missing " + path);
}
static void write_from(const std::string& path, Buf& b) {
  FILE* f = fopen(path.c_str(), "wb");
  fwrite(b.p, 1, b.bytes, f);
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s piece_dir pre_model warmup reps image_dir...\n", argv[0]);
    return 2;
  }
  const std::string pdir = argv[1], pre_model = argv[2];
  const int warmup = atoi(argv[3]), reps = atoi(argv[4]);
  try {
    struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
    const char* uri = getenv("MSDA_URI") ? getenv("MSDA_URI")
                                         : "file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    if (msda_rpc_open(uri, &h_msda)) throw std::runtime_error("msda_rpc_open failed");
    int32 prc = 0;
    msda_rpc_perf_vote(h_msda, 0, &prc);

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "dec_run");
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB")
                                                                                         : "libonnxruntime_providers_qnn.so");
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
    std::map<std::string, Piece> pc;
    double tot_create = 0, ms = 0;
    pc["pre"] = load(pdir + "/" + pre_model, "pre", &ms);
    tot_create += ms;
    for (auto n : {"mid0", "mid1", "post"}) {
      pc[n] = load(pdir + "/" + n + ".sim.onnx", n, &ms);
      tot_create += ms;
    }
    printf("sessions create_ms %.1f\n", tot_create);
    Piece& pre = pc["pre"];
    const bool u8 = pre.in_type[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    Buf& img = buf(pre.in[0], pre.in_shape[0], pre.in_type[0]);

    const char* step_names[] = {"pre", "msda0", "mid0", "msda1", "mid1", "msda2", "post"};
    const int nsteps = 7;
    for (int fi = 5; fi < argc; ++fi) {
      const std::string fd = argv[fi];
      read_into(fd + (u8 ? "/image.u8" : "/pixels.f32"), img);
      std::vector<std::vector<double>> st(nsteps), dsp(nsteps);
      std::vector<double> tot;
      for (int it = 0; it < warmup + reps; ++it) {
        int k = 0;
        double t0 = now_ms(), a = t0;
        auto mark = [&](double dsp_ms) {
          double b = now_ms();
          if (it >= warmup) { st[k].push_back(b - a); dsp[k].push_back(dsp_ms); }
          k++;
          a = b;
        };
        run(pre);
        mark(0);
        std::string src = "pre";
        for (int i = 0; i < 3; ++i) {
          const std::string hn = i == 0 ? "pre/h" : src + "/h_out", rn = i == 0 ? "pre/ref" : src + "/ref_out";
          msda("pre/value" + std::to_string(i), src + "/off", src + "/w", rn, "msda" + std::to_string(i));
          mark(last_dsp_us / 1000.0);
          const std::string nxt = i < 2 ? "mid" + std::to_string(i) : "post";
          run(pc[nxt], {{"msda", "msda" + std::to_string(i)}, {"h", hn}, {"ref", rn}});
          mark(0);
          src = nxt;
        }
        if (it >= warmup) tot.push_back(now_ms() - t0);
      }
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
      printf("%s total_ms median %.2f min %.2f (n=%zu)\n", fd.c_str(), med(tot), *std::min_element(tot.begin(), tot.end()),
             tot.size());
      double htp = 0, dsp_wall = 0, dsp_in = 0;
      for (int k = 0; k < nsteps; ++k) {
        double m = med(st[k]), d = med(dsp[k]);
        printf("  step %-6s %7.2f ms", step_names[k], m);
        if (d > 0) printf(" (dsp %.2f)", d);
        printf("\n");
        if (d > 0) { dsp_wall += m; dsp_in += d; } else { htp += m; }
      }
      printf("  htp pieces %.2f ms, msda calls %.2f ms (in-DSP %.2f, FastRPC %.2f)\n", htp, dsp_wall, dsp_in, dsp_wall - dsp_in);
      write_from(fd + "/logits.f32", get("post/logits"));
      write_from(fd + "/boxes.f32", get("post/boxes"));
      fflush(stdout);
    }
    msda_rpc_close(h_msda);
    printf("PASS\n");
    fflush(stdout);
    _exit(0);  // skip static destructors (ORT teardown order aborts on exit, as in pipe_run)
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    return 1;
  }
}

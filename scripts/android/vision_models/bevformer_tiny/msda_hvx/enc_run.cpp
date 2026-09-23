// BEVFormer-tiny's 3-layer encoder on the phone, split (split.py): 7 fp16 HTP pieces (ORT + QNN EP,
// strict all-HTP) and 6 deformable-sampling calls to the generic HVX MSDA skel
// (../../../msda_hvx/), chained in-process:
//
//   pre -> [tsa] -> mid0 -> [sca] -> post0 -> [tsa] -> mid1 -> [sca] -> post1 -> [tsa] -> mid2 -> [sca] -> post2
//
// Every tensor lives in one rpcmem (ION) buffer per name: ORT writes the HTP pieces' outputs straight
// into them (pre-bound output tensors) and the DSP maps them without a copy.
//
// usage: enc_run <piece dir> <warmup> <reps> <frame dir>...
//   piece dir: pre.onnx mid{0,1,2}.onnx post{0,1,2}.onnx (split.py export's *.sim.onnx, renamed)
//   frame dir: feats.f32 (6,256,15,25) prev_bev.f32 (2500,256) has_prev.f32 (1) can_bus.f32 (18)
//              tsa_ref.f32 (2,2500,1,2) ref_cam.f32 (6,2500,4,2) vis.u8 (6,2500)
//   writes <frame dir>/bev.f32 (2500,256) from the last run; prints per-step and total medians.
// env: MSDA_URI, MSDA_THREADS (default 4), QNN_PERF (htp_performance_mode), QNN_EXTRA (k=v,k=v), ORT_LOG
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

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};
static std::map<std::string, Buf> store;

static Buf& buf(const std::string& name, std::vector<int64_t> shape,
                ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
  size_t n = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? 1 : 4;
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
  std::vector<std::vector<int64_t>> out_shape;
};

static std::unique_ptr<Ort::Env> env;
static std::vector<Ort::ConstEpDevice> npu;

static Piece load(const std::string& dir, const std::string& name, double* ms) {
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
  P.s = std::make_unique<Ort::Session>(*env, (dir + "/" + name + ".onnx").c_str(), so);
  *ms = now_ms() - t0;
  Ort::AllocatorWithDefaultOptions a;
  for (size_t i = 0; i < P.s->GetInputCount(); ++i) P.in.push_back(P.s->GetInputNameAllocated(i, a).get());
  for (size_t i = 0; i < P.s->GetOutputCount(); ++i) {
    P.out.push_back(P.s->GetOutputNameAllocated(i, a).get());
    P.out_shape.push_back(P.s->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  return P;
}

// Runs a piece; `alias` maps a piece's input name to the store tensor that feeds it.
static void run(Piece& P, const std::map<std::string, std::string>& alias = {}) {
  std::vector<Ort::Value> xs, ys;
  std::vector<const char*> in_names, out_names;
  for (auto& n : P.in) {
    auto it = alias.find(n);
    xs.push_back(view(get(it == alias.end() ? n : it->second)));
    in_names.push_back(n.c_str());
  }
  for (size_t i = 0; i < P.out.size(); ++i) {
    ys.push_back(view(buf(P.out[i], P.out_shape[i])));
    out_names.push_back(P.out[i].c_str());
  }
  P.s->Run(Ort::RunOptions{nullptr}, in_names.data(), xs.data(), xs.size(), out_names.data(), ys.data(), ys.size());
}

static remote_handle64 h_msda = 0;
static unsigned long long last_dsp_us = 0;

// One TSA / SCA call: one level (H, W), 8 heads x 32, P points, ref points + pixel offsets (mode
// MSDA_REF_PIX), point p on ref entry p % R, NV value maps averaged over the visible ones.
static void msda(const std::string& value, size_t value_off_floats, const std::string& ref, const std::string& off,
                 const std::string& attw, const std::string& vis, int NV, int H, int W, int R, int NO, int P,
                 const std::string& out) {
  const int Q = 2500;
  Buf &v = get(value), &r = get(ref), &o = get(off), &a = get(attw), &s = get(vis);
  Buf& y = buf(out, {Q, 256});
  msda_args_t A;
  memset(&A, 0, sizeof A);
  A.NV = NV; A.L = 1; A.H[0] = H; A.W[0] = W; A.start[0] = 0; A.S = H * W; A.M = 8; A.D = 32; A.P = P; A.Q = Q;
  A.NO = NO; A.mode = MSDA_REF_PIX; A.vdtype = MSDA_F32; A.NVR = NV; A.RL = 1; A.R = R; A.RD = 2; A.vis = (const uint8_t*)s.p;
  int32 shape[MSDA_SHAPE_LEN(1)];
  const int ns = msda_shape_pack(&A, shape);
  if ((value_off_floats + msda_n_value(&A)) * 4 > v.bytes || msda_n_ref(&A) * 4 != (long)r.bytes ||
      msda_n_loc(&A) * 4 != (long)o.bytes || msda_n_attw(&A) * 4 != (long)a.bytes || msda_n_vis(&A) != (long)s.bytes)
    throw std::runtime_error("msda buffer sizes don't match the shape");
  uint64 us = 0;
  int flags = getenv("MSDA_THREADS") ? atoi(getenv("MSDA_THREADS")) : 4;
  int rc = msda_rpc_run(h_msda, (const float*)v.p + value_off_floats, (int)msda_n_value(&A), nullptr, 0, nullptr, 0,
                        nullptr, 0, (const float*)o.p,
                        (int)msda_n_loc(&A), (const float*)r.p, (int)msda_n_ref(&A), (const float*)a.p, (int)msda_n_attw(&A),
                        (const uint8*)s.p, (int)msda_n_vis(&A), shape, ns, flags, (float*)y.p, (int)msda_n_out(&A), &us);
  if (rc) throw std::runtime_error("msda_rpc_run rc=" + std::to_string(rc));
  last_dsp_us = us;
}

static void read_into(const std::string& path, Buf& b) {
  std::ifstream f(path, std::ios::binary);
  f.read((char*)b.p, (std::streamsize)b.bytes);
  if (!f) throw std::runtime_error("short/missing " + path);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s piece_dir warmup reps frame_dir...\n", argv[0]);
    return 2;
  }
  const std::string pdir = argv[1];
  const int warmup = atoi(argv[2]), reps = atoi(argv[3]);
  try {
    struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
    const char* uri = getenv("MSDA_URI") ? getenv("MSDA_URI")
                                         : "file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    if (msda_rpc_open(uri, &h_msda)) throw std::runtime_error("msda_rpc_open failed");
    int32 prc = 0;
    msda_rpc_perf_vote(h_msda, 0, &prc);

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "enc_run");
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB")
                                                                                         : "libonnxruntime_providers_qnn.so");
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
    std::vector<std::string> names = {"pre", "mid0", "post0", "mid1", "post1", "mid2", "post2"};
    std::map<std::string, Piece> pc;
    double tot_create = 0;
    for (auto& n : names) {
      double ms = 0;
      pc[n] = load(pdir, n, &ms);
      tot_create += ms;
    }
    printf("sessions create_ms %.1f\n", tot_create);

    // inputs
    buf("feats", {6, 256, 15, 25});
    buf("prev_bev", {2500, 256});
    buf("has_prev", {1});
    buf("can_bus", {18});
    buf("tsa_ref", {2, 2500, 1, 2});
    buf("ref_cam", {6, 2500, 4, 2});
    buf("vis", {6, 2500}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    Buf& ones = buf("tsa_vis", {2, 2500}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    memset(ones.p, 1, ones.bytes);

    const char* step_names[] = {"pre",     "tsa0",    "mid0",  "sca0", "post0", "tsa1", "mid1",
                                "sca1",    "post1",   "tsa2",  "mid2", "sca2",  "post2"};
    const int nsteps = 13;
    for (int fi = 4; fi < argc; ++fi) {
      const std::string fd = argv[fi];
      for (auto n : {"feats", "prev_bev", "has_prev", "can_bus", "tsa_ref", "ref_cam"}) read_into(fd + "/" + n + ".f32", get(n));
      read_into(fd + "/vis.u8", get("vis"));
      std::vector<std::vector<double>> st(nsteps), dsp(nsteps);
      std::vector<double> tot;
      for (int it = 0; it < warmup + reps; ++it) {
        int k = 0;
        double t0 = now_ms(), a;
        auto mark = [&](double dsp_ms) {
          double b = now_ms();
          if (it >= warmup) { st[k].push_back(b - a); dsp[k].push_back(dsp_ms); }
          k++;
          a = b;
        };
        a = t0;
        run(pc["pre"]);
        mark(0);
        std::string q = "q0";
        for (int i = 0; i < 3; ++i) {
          msda("tsa_v", 0, "tsa_ref", "tsa_off", "tsa_w", "tsa_vis", 2, 50, 50, 1, 2, 4, "tsa_out");
          mark(last_dsp_us / 1000.0);
          run(pc["mid" + std::to_string(i)], {{"q", q}});
          mark(0);
          msda("sca_v", (size_t)i * 6 * 375 * 256, "ref_cam", "sca_off", "sca_w", "vis", 6, 15, 25, 4, 1, 8, "sca_out");
          mark(last_dsp_us / 1000.0);
          run(pc["post" + std::to_string(i)]);
          mark(0);
          q = "q";
        }
        if (it >= warmup) tot.push_back(now_ms() - t0);
      }
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
      printf("%s encoder_ms median %.2f min %.2f (n=%zu)\n", fd.c_str(), med(tot), *std::min_element(tot.begin(), tot.end()),
             tot.size());
      double htp = 0, dsp_wall = 0, dsp_in = 0;
      for (int k = 0; k < nsteps; ++k) {
        double m = med(st[k]), d = med(dsp[k]);
        printf("  step %-6s %7.2f ms%s\n", step_names[k], m, d > 0 ? (" (dsp " + std::to_string(d).substr(0, 5) + ")").c_str() : "");
        if (d > 0) { dsp_wall += m; dsp_in += d; } else { htp += m; }
      }
      printf("  htp pieces %.2f ms, msda calls %.2f ms (in-DSP %.2f, FastRPC %.2f)\n", htp, dsp_wall, dsp_in, dsp_wall - dsp_in);
      Buf& bev = get("bev_embed");
      FILE* f = fopen((fd + "/bev.f32").c_str(), "wb");
      fwrite(bev.p, 1, bev.bytes, f);
      fclose(f);
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

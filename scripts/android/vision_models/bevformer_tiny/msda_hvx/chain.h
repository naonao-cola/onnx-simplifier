// Shared by enc_run.cpp and frame_run.cpp: rpcmem-backed named tensors, ORT/QNN HTP pieces, and one
// call of the generic HVX MSDA skel (../../../msda_hvx/) in BEVFormer's TSA / SCA shapes.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
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
  auto it = store.find(name);  // look up first: stage threads call this concurrently on existing names
  Buf& b = it != store.end() ? it->second : store[name];
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
  std::vector<ONNXTensorElementDataType> out_type;
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
    P.out_type.push_back(P.s->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
  }
  return P;
}

// Runs a piece; `alias` maps a piece's input name to the store tensor that feeds it, `oalias` an
// output name to the store tensor it is written into.
static void run(Piece& P, const std::map<std::string, std::string>& alias = {},
                const std::map<std::string, std::string>& oalias = {}) {
  std::vector<Ort::Value> xs, ys;
  std::vector<const char*> in_names, out_names;
  for (auto& n : P.in) {
    auto it = alias.find(n);
    xs.push_back(view(get(it == alias.end() ? n : it->second)));
    in_names.push_back(n.c_str());
  }
  for (size_t i = 0; i < P.out.size(); ++i) {
    auto it = oalias.find(P.out[i]);
    ys.push_back(view(buf(it == oalias.end() ? P.out[i] : it->second, P.out_shape[i], P.out_type[i])));
    out_names.push_back(P.out[i].c_str());
  }
  P.s->Run(Ort::RunOptions{nullptr}, in_names.data(), xs.data(), xs.size(), out_names.data(), ys.data(), ys.size());
}

static remote_handle64 h_msda = 0;
static unsigned long long last_dsp_us = 0;
static float tsa_scale[3];
static int32_t tsa_zp[3];

// One TSA / SCA call: one level (H, W), 8 heads x 32, P points, ref points + pixel offsets (mode
// MSDA_REF_PIX), point p on ref entry p % R, NV value maps averaged over the visible ones.
// uint8 value buffers (split.py export --tsa-u8) use layer `layer`'s scale / zero point.
// The skel isn't reentrant: callers on several threads (frame_run's encoder and split decoder) take turns.
static std::mutex msda_mu;
static void msda(const std::string& value, size_t value_off_floats, const std::string& ref, const std::string& off,
                 const std::string& attw, const std::string& vis, int NV, int H, int W, int R, int NO, int P,
                 const std::string& out, int layer, int Q = 2500) {
  std::lock_guard<std::mutex> lock(msda_mu);
  Buf &v = get(value), &r = get(ref), &o = get(off), &a = get(attw), &s = get(vis);
  Buf& y = buf(out, {Q, 256});
  msda_args_t A;
  memset(&A, 0, sizeof A);
  A.NV = NV; A.L = 1; A.H[0] = H; A.W[0] = W; A.start[0] = 0; A.S = H * W; A.M = 8; A.D = 32; A.P = P; A.Q = Q;
  const bool u8 = v.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  float vs[MSDA_MAX_NV];
  int32 vz[MSDA_MAX_NV];
  for (int i = 0; i < NV; i++) { vs[i] = tsa_scale[layer]; vz[i] = tsa_zp[layer]; }
  A.NO = NO; A.mode = MSDA_REF_PIX; A.vdtype = u8 ? MSDA_U8 : MSDA_F32; A.NVR = NV; A.RL = 1; A.R = R; A.RD = 2; A.vis = (const uint8_t*)s.p;
  int32 shape[MSDA_SHAPE_LEN(1)];
  const int ns = msda_shape_pack(&A, shape);
  if ((value_off_floats + msda_n_value(&A)) * (u8 ? 1 : 4) > v.bytes || msda_n_ref(&A) * 4 != (long)r.bytes ||
      msda_n_loc(&A) * 4 != (long)o.bytes || msda_n_attw(&A) * 4 != (long)a.bytes || msda_n_vis(&A) != (long)s.bytes)
    throw std::runtime_error("msda buffer sizes don't match the shape");
  uint64 us = 0;
  int flags = getenv("MSDA_THREADS") ? atoi(getenv("MSDA_THREADS")) : 4;
  int rc = msda_rpc_run(h_msda, u8 ? nullptr : (const float*)v.p + value_off_floats, u8 ? 0 : (int)msda_n_value(&A),
                        u8 ? (const uint8*)v.p + value_off_floats : nullptr, u8 ? (int)msda_n_value(&A) : 0,
                        u8 ? vs : nullptr, u8 ? NV : 0, u8 ? vz : nullptr, u8 ? NV : 0, (const float*)o.p,
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


// The encoder chain: pre -> 3 x ([tsa] -> mid -> [sca] -> post), reading feats / prev_bev / has_prev /
// can_bus / tsa_ref / ref_cam / vis / tsa_vis from the store and writing bev_embed (or `out` if given).
// mark(dsp_ms) runs after every step (dsp_ms = the kernel's in-DSP time for a sampling call, else 0).
template <class Mark>
static void encoder(std::map<std::string, Piece>& pc, Mark&& mark, const std::string& out = "bev_embed") {
  run(pc["pre"]);
  mark(0);
  std::string q = "q0";
  for (int i = 0; i < 3; ++i) {
    msda("tsa_v", 0, "tsa_ref", "tsa_off", "tsa_w", "tsa_vis", 2, 50, 50, 1, 2, 4, "tsa_out", i);
    mark(last_dsp_us / 1000.0);
    run(pc["mid" + std::to_string(i)], {{"q", q}});
    mark(0);
    msda("sca_v", (size_t)i * 6 * 375 * 256, "ref_cam", "sca_off", "sca_w", "vis", 6, 15, 25, 4, 1, 8, "sca_out", i);
    mark(last_dsp_us / 1000.0);
    run(pc["post" + std::to_string(i)], {}, i == 2 ? std::map<std::string, std::string>{{"bev_embed", out}}
                                                   : std::map<std::string, std::string>{});
    mark(0);
    q = "q";
  }
}

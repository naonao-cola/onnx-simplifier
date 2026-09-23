// Shared by s4d_run.cpp (one frame) and s4d_scene.cpp (a scene, instance bank on the phone, frames
// pipelined): named rpcmem tensors, ORT + QNN EP piece sessions, and the DFA skel call.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

extern "C" {
#include "dfa_rpc.h"
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
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
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
static std::mutex store_mu;  // the scene runner fills the store from two threads

static Buf& buf(const std::string& name, std::vector<int64_t> shape,
                ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
  size_t n = esize(type);
  for (auto d : shape) n *= (size_t)d;
  std::lock_guard<std::mutex> lk(store_mu);
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
  std::lock_guard<std::mutex> lk(store_mu);
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
  so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  // EP-context cache next to the model: the first run compiles, later runs load in ~100 ms
  const std::string ctx = path.substr(0, path.size() - 5) + ".ctx.onnx";
  struct stat st;
  const bool have_ctx = stat(ctx.c_str(), &st) == 0;
  if (!have_ctx) {
    so.AddConfigEntry("ep.context_enable", "1");
    so.AddConfigEntry("ep.context_file_path", ctx.c_str());
    so.AddConfigEntry("ep.context_embed_mode", "1");
  }
  so.AppendExecutionProvider_V2(*env, npu, o);
  Piece P;
  P.name = name;
  double t0 = now_ms();
  P.s = std::make_unique<Ort::Session>(*env, (have_ctx ? ctx : path).c_str(), so);
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

// input n reads store[alias[n]] (default: n); output n writes store[prefix + name + "/" + n]
static void run(Piece& P, const std::map<std::string, std::string>& alias = {}, const std::string& prefix = "") {
  std::vector<Ort::Value> xs, ys;
  std::vector<const char*> in_names, out_names;
  for (auto& n : P.in) {
    auto it = alias.find(n);
    xs.push_back(view(get(it == alias.end() ? n : it->second)));
    in_names.push_back(n.c_str());
  }
  for (size_t i = 0; i < P.out.size(); ++i) {
    ys.push_back(view(buf(prefix + P.name + "/" + P.out[i], P.out_shape[i], P.out_type[i])));
    out_names.push_back(P.out[i].c_str());
  }
  P.s->Run(Ort::RunOptions{nullptr}, in_names.data(), xs.data(), xs.size(), out_names.data(), ys.data(), ys.size());
}

static remote_handle64 h_dfa = 0;
static unsigned long long last_dsp_us = 0;
static float vscale[4];
static int32 vzp[4];
static const int32 HW[8] = {64, 176, 32, 88, 16, 44, 8, 22};

// one layer's DFA: pts / w from piece outputs `src`, value maps `vprefix`bb/v0..3 -> out (Q, 256)
static void dfa(const std::string& src, const std::string& out, const std::string& vprefix = "") {
  Buf &pts = get(src + "/pts"), &w = get(src + "/w");
  const int Q = (int)pts.shape[1];
  Buf& y = buf(out, {Q, 256});
  Buf* v[4];
  for (int l = 0; l < 4; ++l) v[l] = &get(vprefix + "bb/v" + std::to_string(l));
  uint64 us = 0;
  int flags = getenv("DFA_FLAGS") ? atoi(getenv("DFA_FLAGS")) : 4;
  if (w.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)  // the skel converts per block
    flags |= w.shape.size() == 3 ? 1 << 17 : 1 << 16;  // (Q, 8, 384): split.py's v2 layout
  int rc = dfa_rpc_run(h_dfa, (const uint8*)v[0]->p, (int)v[0]->bytes, (const uint8*)v[1]->p, (int)v[1]->bytes,
                       (const uint8*)v[2]->p, (int)v[2]->bytes, (const uint8*)v[3]->p, (int)v[3]->bytes, vscale, 4, vzp, 4,
                       HW, 8, (const float*)pts.p, (int)(pts.bytes / 4), (const float*)w.p, (int)(w.bytes / 4), Q, flags,
                       (float*)y.p, (int)(y.bytes / 4), &us);
  if (rc) throw std::runtime_error("dfa_rpc_run rc=" + std::to_string(rc));
  last_dsp_us = us;
}

static bool exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
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


// FastRPC unsigned PD + the DFA skel (h_dfa), then ORT with the QNN EP's NPU device.
static void init_dsp_and_ort() {
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  const char* uri = getenv("DFA_URI") ? getenv("DFA_URI")
                                      : "file:///dfa_rpc.so?dfa_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp";
  if (dfa_rpc_open(uri, &h_dfa)) throw std::runtime_error("dfa_rpc_open failed");
  int32 prc = 0;
  dfa_rpc_perf_vote(h_dfa, 0, &prc);
  env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "s4d_run");
  env->RegisterExecutionProviderLibrary("QNNExecutionProvider", getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB")
                                                                                       : "libonnxruntime_providers_qnn.so");
  for (const auto& d : env->GetEpDevices())
    if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
      npu.push_back(d);
  if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
}

// bb from bb.q8.onnx; pre0 / mid pieces from <name>.$S4D_MODELS.onnx when set (io16.py / split.py
// --w-layout variants), else <name>.sim.onnx. Returns the total session-create ms.
static double load_pieces(const std::string& sdir, const std::vector<std::string>& names, std::map<std::string, Piece>& pc) {
  const char* models = getenv("S4D_MODELS") && *getenv("S4D_MODELS") ? getenv("S4D_MODELS") : nullptr;
  double tot = 0, ms = 0;
  for (auto& n : names) {
    const bool has_w = n == "pre0" || n.rfind("mid", 0) == 0;  // the pieces that emit DFA inputs
    pc[n] = load(sdir + "/" + n + (n == "bb" ? ".q8" : has_w && models ? std::string(".") + models : ".sim") + ".onnx", n, &ms);
    tot += ms;
  }
  return tot;
}

// the value maps' uint8 scales / zero points, from bb.q8.onnx's metadata
static void read_vscales(Piece& bb) {
  Ort::AllocatorWithDefaultOptions al;
  Ort::ModelMetadata md = bb.s->GetModelMetadata();
  for (int l = 0; l < 4; ++l) {
    const std::string v = "v" + std::to_string(l);
    auto sc = md.LookupCustomMetadataMapAllocated((v + "_scale").c_str(), al);
    auto zp = md.LookupCustomMetadataMapAllocated((v + "_zero_point").c_str(), al);
    if (!sc || !zp) throw std::runtime_error("bb.q8.onnx has no " + v + " scale / zero point");
    vscale[l] = strtof(sc.get(), nullptr);
    vzp[l] = atoi(zp.get());
  }
}

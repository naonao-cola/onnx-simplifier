// One Sparse4D v3 frame on the phone, split around the HVX DFA (split.py): 8 HTP pieces (ORT + QNN
// EP, strict all-HTP) and 6 DFA calls on the CDSP (dfa_rpc skel), chained in-process:
//
//   bb -> pre0 -> [dfa0] -> mid0 -> [dfa1] -> mid1 -> ... -> mid4 -> [dfa5] -> post
//
// Every tensor lives in its own rpcmem buffer: ORT writes the pieces' outputs straight into them
// and the DSP maps them without a copy (the pattern of ../../rtdetr/msda_hvx/dec_run.cpp).
//
// usage: s4d_run <split dir> <frame dir> <warmup> <reps>
//   frame dir: rgb.u8 (6,256,704,3), proj.f32 (6,4,4), proj_n.f32 (6,3,4); a temporal frame also has
//              dt.f32 (1), temp_feat.f32 (600,256), temp_anchor.f32 (600,11) (the host instance bank)
//   writes cls.f32, box.f32, quality.f32, feat.f32; prints per-step and total medians.
// env: DFA_URI, DFA_FLAGS (threads | 256 * anchors-per-job/16, default 4), QNN_PERF, QNN_EXTRA, ORT_LOG
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

// input n reads store[alias[n]] (default: n); output n writes store[name + "/" + n]
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

static remote_handle64 h_dfa = 0;
static unsigned long long last_dsp_us = 0;
static float vscale[4];
static int32 vzp[4];
static const int32 HW[8] = {64, 176, 32, 88, 16, 44, 8, 22};

static void dfa(const std::string& src, const std::string& out) {
  Buf &pts = get(src + "/pts"), &w = get(src + "/w");
  const int Q = (int)pts.shape[1];
  Buf& y = buf(out, {Q, 256});
  Buf* v[4];
  for (int l = 0; l < 4; ++l) v[l] = &get("bb/v" + std::to_string(l));
  uint64 us = 0;
  int flags = getenv("DFA_FLAGS") ? atoi(getenv("DFA_FLAGS")) : 4;
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

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s split_dir frame_dir warmup reps\n", argv[0]);
    return 2;
  }
  const std::string sdir = argv[1], fd = argv[2];
  const int warmup = atoi(argv[3]), reps = atoi(argv[4]);
  try {
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

    const bool temporal = exists(fd + "/temp_feat.f32");
    const char* T = temporal ? "T" : "F";
    std::vector<std::string> names = {"bb", "pre0"};
    for (int k = 0; k < 5; ++k) names.push_back("mid" + std::to_string(k) + T);
    names.push_back(std::string("post") + T);
    std::map<std::string, Piece> pc;
    double tot_create = 0, ms = 0;
    for (auto& n : names) {
      pc[n] = load(sdir + "/" + n + (n == "bb" ? ".q8.onnx" : ".sim.onnx"), n, &ms);
      tot_create += ms;
    }
    printf("sessions create_ms %.1f (%s)\n", tot_create, temporal ? "temporal" : "first frame");
    {
      Ort::AllocatorWithDefaultOptions al;
      Ort::ModelMetadata md = pc["bb"].s->GetModelMetadata();
      for (int l = 0; l < 4; ++l) {
        const std::string v = "v" + std::to_string(l);
        auto sc = md.LookupCustomMetadataMapAllocated((v + "_scale").c_str(), al);
        auto zp = md.LookupCustomMetadataMapAllocated((v + "_zero_point").c_str(), al);
        if (!sc || !zp) throw std::runtime_error("bb.q8.onnx has no " + v + " scale / zero point");
        vscale[l] = strtof(sc.get(), nullptr);
        vzp[l] = atoi(zp.get());
      }
    }
    read_into(fd + "/rgb.u8", buf("rgb", {6, 256, 704, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8));
    read_into(fd + "/proj.f32", buf("proj", {6, 4, 4}));
    read_into(fd + "/proj_n.f32", buf("proj_n", {6, 3, 4}));
    // the initial instances / anchors are constants inside pre0's graph; mid0 reads them as inputs
    Buf& feat0 = buf("feat0", {900, 256});
    Buf& anchor0 = buf("anchor0", {900, 11});
    read_into(sdir + "/instance_feature.f32", feat0);
    read_into(sdir + "/anchor.f32", anchor0);
    if (temporal) {
      read_into(fd + "/dt.f32", buf("dt", {}));
      read_into(fd + "/temp_feat.f32", buf("temp_feat", {600, 256}));
      read_into(fd + "/temp_anchor.f32", buf("temp_anchor", {600, 11}));
    }
    std::vector<std::string> steps = {"bb", "pre0"};
    for (int k = 0; k < 6; ++k) {
      steps.push_back("dfa" + std::to_string(k));
      steps.push_back(k < 5 ? "mid" + std::to_string(k) : "post");
    }
    const int nsteps = (int)steps.size();
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
      run(pc["bb"]);
      mark(0);
      run(pc["pre0"]);
      mark(0);
      std::string src = "pre0", feat = "feat0", anchor = "anchor0";
      for (int i = 0; i < 6; ++i) {
        dfa(src, "agg" + std::to_string(i));
        mark(last_dsp_us / 1000.0);
        const std::string nxt = i < 5 ? "mid" + std::to_string(i) + T : std::string("post") + T;
        run(pc[nxt], {{"agg", "agg" + std::to_string(i)}, {"feat", feat}, {"anchor", anchor}});
        mark(0);
        feat = nxt + "/feat_out";
        anchor = nxt + "/anchor_out";
        src = nxt;
      }
      if (it >= warmup) tot.push_back(now_ms() - t0);
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    printf("total_ms median %.2f min %.2f (n=%zu)\n", med(tot), *std::min_element(tot.begin(), tot.end()), tot.size());
    double htp = 0, dsp_wall = 0, dsp_in = 0;
    for (int k = 0; k < nsteps; ++k) {
      double m = med(st[k]), d = med(dsp[k]);
      printf("  step %-6s %7.2f ms", steps[k].c_str(), m);
      if (d > 0) printf(" (dsp %.2f)", d);
      printf("\n");
      if (d > 0) { dsp_wall += m; dsp_in += d; } else { htp += m; }
    }
    printf("  htp pieces %.2f ms, dfa calls %.2f ms (in-DSP %.2f, FastRPC %.2f)\n", htp, dsp_wall, dsp_in, dsp_wall - dsp_in);
    const std::string post = std::string("post") + T;
    write_from(fd + "/cls.f32", get(post + "/cls"));
    write_from(fd + "/box.f32", get(post + "/anchor_out"));
    write_from(fd + "/quality.f32", get(post + "/quality"));
    write_from(fd + "/feat.f32", get(post + "/feat_out"));
    dfa_rpc_close(h_dfa);
    printf("PASS\n");
    fflush(stdout);
    _exit(0);  // skip static destructors (ORT teardown order aborts on exit)
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    fflush(stdout);
    return 1;
  }
}

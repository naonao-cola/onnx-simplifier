// MapTR-tiny on the phone, one frame end to end in-process (split.py's pieces): HTP pieces (ORT + QNN EP,
// strict all-HTP, fp16 inside) chained with deformable-sampling calls to the generic HVX MSDA skel
// (../../../msda_hvx/). Same machinery as ../../bevformer_tiny/msda_hvx/enc_run.cpp.
//
//   [backbone] -> pre -> {tsa} -> mid -> {sca} -> post -> decoder            (DEC=htp, default)
//   [backbone] -> cpu_tsa, prev -> {tsa} -> midc -> {sca} -> post -> ...      (piece dir has prev.onnx + tsa_*.f32)
//                                                      -> dvals -> 6 x (dpre<i> -> {dec<i>} -> dpost<i>)   (DEC=split)
//
// Every tensor lives in one rpcmem (ION) buffer per name: ORT writes the HTP pieces' outputs straight
// into them and the DSP maps them without a copy.
//
// usage: map_run <piece dir> <warmup> <reps> <frame dir>...
//   piece dir: [backbone.onnx] pre.onnx mid.onnx post.onnx, decoder.onnx (DEC=htp) or dvals.onnx
//              dpre{0..5}.onnx dpost{0..5}.onnx dq0.f32 dref0.f32 (DEC=split)
//   frame dir: img.f32 (6,3,480,800) if backbone.onnx, else feats.f32 (6,256,15,25); can_bus.f32 (18)
//              tsa_ref.f32 (1,20000,1,2) ref_cam.f32 (6,20000,4,2) vis.u8 (6,20000)
//   writes <frame dir>/{bev,cls,pts}.out.f32 from the last run; prints per-step and total medians.
// env: DEC (htp|split), MSDA_URI, MSDA_FLAGS (default 4), QNN_PERF, QNN_EXTRA (k=v,k=v), ORT_LOG
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <cmath>
#include <string>
#include <thread>
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

static size_t esize(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? 1 : t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ? 2 : 4;
}

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
  std::unique_ptr<Ort::Session> s;
  std::vector<std::string> in, out;
  std::vector<std::vector<int64_t>> out_shape;
  std::vector<ONNXTensorElementDataType> out_type;
};

static std::unique_ptr<Ort::Env> env;
static std::vector<Ort::ConstEpDevice> npu;

static Piece load(const std::string& path, double* ms) {
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
  double t0 = now_ms();
  P.s = std::make_unique<Ort::Session>(*env, path.c_str(), so);
  *ms = now_ms() - t0;
  Ort::AllocatorWithDefaultOptions a;
  for (size_t i = 0; i < P.s->GetInputCount(); ++i) P.in.push_back(P.s->GetInputNameAllocated(i, a).get());
  for (size_t i = 0; i < P.s->GetOutputCount(); ++i) {
    P.out.push_back(P.s->GetOutputNameAllocated(i, a).get());
    auto ti = P.s->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
    P.out_shape.push_back(ti.GetShape());
    P.out_type.push_back(ti.GetElementType());
  }
  return P;
}

// Runs a piece; `in_alias` / `out_alias` map a piece's input / output name to a store tensor.
static void run(Piece& P, const std::map<std::string, std::string>& in_alias = {},
                const std::map<std::string, std::string>& out_alias = {}) {
  std::vector<Ort::Value> xs, ys;
  std::vector<const char*> in_names, out_names;
  for (auto& n : P.in) {
    auto it = in_alias.find(n);
    xs.push_back(view(get(it == in_alias.end() ? n : it->second)));
    in_names.push_back(n.c_str());
  }
  for (size_t i = 0; i < P.out.size(); ++i) {
    auto it = out_alias.find(P.out[i]);
    ys.push_back(view(buf(it == out_alias.end() ? P.out[i] : it->second, P.out_shape[i], P.out_type[i])));
    out_names.push_back(P.out[i].c_str());
  }
  P.s->Run(Ort::RunOptions{nullptr}, in_names.data(), xs.data(), xs.size(), out_names.data(), ys.data(), ys.size());
}

static remote_handle64 h_msda = 0;
static unsigned long long last_dsp_us = 0;

// One call: one level (H, W), 8 heads x 32, P points, ref points + pixel offsets (MSDA_REF_PIX), point p
// on ref entry p % R, NV value maps averaged over the visible ones (vis "" = all).
static void msda(const std::string& value, size_t value_off_floats, const std::string& ref, const std::string& off,
                 const std::string& attw, const std::string& vis, int Q, int NV, int H, int W, int R, int P,
                 const std::string& out) {
  Buf &v = get(value), &r = get(ref), &o = get(off), &a = get(attw);
  Buf* s = vis.empty() ? nullptr : &get(vis);
  Buf& y = buf(out, {Q, 256});
  msda_args_t A;
  memset(&A, 0, sizeof A);
  A.NV = NV; A.L = 1; A.H[0] = H; A.W[0] = W; A.start[0] = 0; A.S = H * W; A.M = 8; A.D = 32; A.P = P; A.Q = Q;
  A.NO = 1; A.mode = MSDA_REF_PIX; A.vdtype = MSDA_F32; A.NVR = NV; A.RL = 1; A.R = R; A.RD = 2;
  A.vis = s ? (const uint8_t*)s->p : nullptr;
  int32 shape[MSDA_SHAPE_LEN(1)];
  const int ns = msda_shape_pack(&A, shape);
  if ((value_off_floats + msda_n_value(&A)) * 4 > v.bytes || msda_n_ref(&A) * 4 != (long)r.bytes ||
      msda_n_loc(&A) * 4 != (long)o.bytes || msda_n_attw(&A) * 4 != (long)a.bytes || (s && msda_n_vis(&A) != (long)s->bytes))
    throw std::runtime_error("msda buffer sizes don't match the shape (" + out + ")");
  uint64 us = 0;
  int flags = getenv("MSDA_FLAGS") ? atoi(getenv("MSDA_FLAGS")) : 4;
  int rc = msda_rpc_run(h_msda, (const float*)v.p + value_off_floats, (int)msda_n_value(&A), nullptr, 0, nullptr, 0,
                        nullptr, 0, (const float*)o.p, (int)msda_n_loc(&A), (const float*)r.p, (int)msda_n_ref(&A),
                        (const float*)a.p, (int)msda_n_attw(&A), s ? (const uint8*)s->p : nullptr,
                        s ? (int)msda_n_vis(&A) : 0, shape, ns, flags, (float*)y.p, (int)msda_n_out(&A), &us);
  if (rc) throw std::runtime_error("msda_rpc_run rc=" + std::to_string(rc) + " (" + out + ")");
  last_dsp_us = us;
}

static bool exists(const std::string& p) { return access(p.c_str(), R_OK) == 0; }

// CPU_TSA (piece dir has prev.onnx): TSA's value / offsets / weights are affine in c = can_bus_mlp(can_bus)
// (split.py tsa_consts): tsa_v = V0 + Wv c, off = A + Bo c, w = 0.5 * softmax_4(Aw + Bw c).
static std::map<std::string, std::vector<float>> K;
static void load_consts(const std::string& dir) {
  const std::pair<const char*, size_t> sz[] = {{"V0", 20000 * 256}, {"Wv", 256 * 256}, {"A", 20000 * 128}, {"Bo", 128 * 256},
                                               {"Aw", 20000 * 64}, {"Bw", 64 * 256}, {"m0w", 128 * 18}, {"m0b", 128},
                                               {"m2w", 256 * 128}, {"m2b", 256}, {"lnw", 256}, {"lnb", 256}};
  for (auto& [n, count] : sz) {
    auto& v = K[n];
    v.resize(count);
    std::ifstream f(dir + "/tsa_" + n + ".f32", std::ios::binary);
    f.read((char*)v.data(), (std::streamsize)(count * 4));
    if (!f) throw std::runtime_error(std::string("short/missing tsa_") + n + ".f32");
  }
}
static void matvec(const float* W, const float* x, const float* b, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float a = b ? b[r] : 0.f;
    for (int c = 0; c < cols; ++c) a += W[r * cols + c] * x[c];
    y[r] = a;
  }
}
static void cpu_tsa(const float* can_bus, float* tsa_v, float* off, float* w, int nthreads) {
  float h1[128], h2[256], c[256], dv[256], dof[128], dw[64];
  matvec(K["m0w"].data(), can_bus, K["m0b"].data(), h1, 128, 18);
  for (auto& x : h1) x = std::max(x, 0.f);
  matvec(K["m2w"].data(), h1, K["m2b"].data(), h2, 256, 128);
  float mean = 0, var = 0;
  for (auto& x : h2) { x = std::max(x, 0.f); mean += x; }
  mean /= 256;
  for (auto x : h2) var += (x - mean) * (x - mean);
  var /= 256;
  for (int i = 0; i < 256; ++i) c[i] = (h2[i] - mean) / std::sqrt(var + 1e-5f) * K["lnw"][i] + K["lnb"][i];
  matvec(K["Wv"].data(), c, nullptr, dv, 256, 256);
  matvec(K["Bo"].data(), c, nullptr, dof, 128, 256);
  matvec(K["Bw"].data(), c, nullptr, dw, 64, 256);
  const float *V0 = K["V0"].data(), *A = K["A"].data(), *Aw = K["Aw"].data();
  auto work = [&](int q0, int q1) {
    for (int q = q0; q < q1; ++q) {
      for (int i = 0; i < 256; ++i) tsa_v[q * 256 + i] = V0[q * 256 + i] + dv[i];
      for (int i = 0; i < 128; ++i) off[q * 128 + i] = A[q * 128 + i] + dof[i];
      for (int g = 0; g < 16; ++g) {  // (head, frame) groups of 4 points
        float l[4], mx = -INFINITY, sum = 0;
        for (int p = 0; p < 4; ++p) { l[p] = Aw[q * 64 + g * 4 + p] + dw[g * 4 + p]; mx = std::max(mx, l[p]); }
        for (int p = 0; p < 4; ++p) { l[p] = std::exp(l[p] - mx); sum += l[p]; }
        for (int p = 0; p < 4; ++p) w[q * 64 + g * 4 + p] = 0.5f * l[p] / sum;
      }
    }
  };
  std::vector<std::thread> ts;
  const int Q = 20000, step = (Q + nthreads - 1) / nthreads;
  for (int t = 0; t < nthreads; ++t) ts.emplace_back(work, t * step, std::min(Q, (t + 1) * step));
  for (auto& t : ts) t.join();
}

static void read_into(const std::string& path, Buf& b) {
  std::ifstream f(path, std::ios::binary);
  f.read((char*)b.p, (std::streamsize)b.bytes);
  if (!f) throw std::runtime_error("short/missing " + path);
}

static void write_from(const std::string& path, const Buf& b) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("can't write " + path);
  fwrite(b.p, 1, b.bytes, f);
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s piece_dir warmup reps frame_dir...\n", argv[0]);
    return 2;
  }
  const std::string pdir = argv[1];
  const int warmup = atoi(argv[2]), reps = atoi(argv[3]);
  const bool dec_split = getenv("DEC") && std::string(getenv("DEC")) == "split";
  const int Q = 20000, QD = 1000;
  try {
    struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
    const char* uri = getenv("MSDA_URI") ? getenv("MSDA_URI")
                                         : "file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    if (msda_rpc_open(uri, &h_msda)) throw std::runtime_error("msda_rpc_open failed");
    int32 prc = 0;
    msda_rpc_perf_vote(h_msda, 0, &prc);

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "map_run");
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB")
                                                                                         : "libonnxruntime_providers_qnn.so");
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
    const bool has_bb = exists(pdir + "/backbone.onnx");
    const bool cpu_tsa_path = exists(pdir + "/prev.onnx");
    std::vector<std::string> names = cpu_tsa_path ? std::vector<std::string>{"prev", "midc", "post"}
                                                  : std::vector<std::string>{"pre", "mid", "post"};
    if (has_bb) names.insert(names.begin(), "backbone");
    if (dec_split) {
      names.push_back("dvals");
      for (int i = 0; i < 6; ++i) { names.push_back("dpre" + std::to_string(i)); names.push_back("dpost" + std::to_string(i)); }
    } else {
      names.push_back("decoder");
    }
    std::map<std::string, Piece> pc;
    double tot_create = 0;
    for (auto& n : names) {
      double ms = 0;
      pc[n] = load(pdir + "/" + n + ".onnx", &ms);
      tot_create += ms;
    }
    printf("sessions %zu create_ms %.1f\n", names.size(), tot_create);

    if (has_bb) {  // the backbone's input may be the fp32 image or a uint8 NHWC one (int8 backbone)
      auto ti = pc["backbone"].s->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
      buf(pc["backbone"].in[0], ti.GetShape(), ti.GetElementType());
    } else {
      buf("feats", {6, 256, 15, 25});
    }
    buf("can_bus", {18});
    buf("tsa_ref", {1, Q, 1, 2});
    buf("ref_cam", {6, Q, 4, 2});
    buf("vis", {6, Q}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    if (cpu_tsa_path) {
      load_consts(pdir);
      buf("tsa_v", {Q, 256});
      buf("tsa_off", {Q, 8, 1, 8, 2});
      buf("tsa_w", {Q, 8, 1, 8});
    }
    if (dec_split) {
      read_into(pdir + "/dq0.f32", buf("dq0", {QD, 256}));
      read_into(pdir + "/dref0.f32", buf("dref0", {QD, 2}));
    }

    for (int fi = 4; fi < argc; ++fi) {
      const std::string fd = argv[fi];
      if (has_bb) {
        const std::string in = pc["backbone"].in[0];
        read_into(fd + "/" + (get(in).type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? "img_u8.u8" : "img.f32"), get(in));
      } else {
        read_into(fd + "/feats.f32", get("feats"));
      }
      for (auto n : {"can_bus", "tsa_ref", "ref_cam"}) read_into(fd + "/" + n + ".f32", get(n));
      read_into(fd + "/vis.u8", get("vis"));
      std::vector<std::string> step_names;
      std::map<std::string, std::vector<double>> st, dsp;
      std::vector<double> tot;
      for (int it = 0; it < warmup + reps; ++it) {
        double t0 = now_ms(), a = t0;
        auto mark = [&](const std::string& name, double dsp_ms) {
          double b = now_ms();
          if (it == 0) step_names.push_back(name);
          if (it >= warmup) { st[name].push_back(b - a); dsp[name].push_back(dsp_ms); }
          a = b;
        };
        if (has_bb) { run(pc["backbone"]); mark("backbone", 0); }
        if (cpu_tsa_path) {
          cpu_tsa((const float*)get("can_bus").p, (float*)get("tsa_v").p, (float*)get("tsa_off").p, (float*)get("tsa_w").p,
                  getenv("CPU_THREADS") ? atoi(getenv("CPU_THREADS")) : 4);
          mark("cpu_tsa", 0);
          run(pc["prev"]); mark("prev", 0);
        } else {
          run(pc["pre"]); mark("pre", 0);
        }
        msda("tsa_v", 0, "tsa_ref", "tsa_off", "tsa_w", "", Q, 1, 200, 100, 1, 8, "tsa_out");
        mark("tsa", last_dsp_us / 1000.0);
        if (cpu_tsa_path) { run(pc["midc"]); mark("midc", 0); } else { run(pc["mid"]); mark("mid", 0); }
        msda("sca_v", 0, "ref_cam", "sca_off", "sca_w", "vis", Q, 6, 15, 25, 4, 8, "sca_out");
        mark("sca", last_dsp_us / 1000.0);
        run(pc["post"]); mark("post", 0);
        if (dec_split) {
          run(pc["dvals"]); mark("dvals", 0);
          std::string q = "dq0", r = "dref0";
          for (int i = 0; i < 6; ++i) {
            const std::string s = std::to_string(i);
            run(pc["dpre" + s], {{"q", q}}); mark("dpre" + s, 0);
            // ref (QD, 2) is the kernel's ref (1, QD, 1, 2): the same floats
            msda("dv", (size_t)i * Q * 256, r, "doff", "dw", "", QD, 1, 200, 100, 1, 4, "msda_out");
            mark("dmsda" + s, last_dsp_us / 1000.0);
            const std::string nq = (i % 2) ? "qB" : "qA", nr = (i % 2) ? "refB" : "refA";
            if (i < 5) run(pc["dpost" + s], {{"ref_in", r}}, {{"q", nq}, {"ref", nr}});
            else run(pc["dpost" + s], {{"ref_in", r}});
            mark("dpost" + s, 0);
            q = nq; r = nr;
          }
        } else {
          run(pc["decoder"]); mark("decoder", 0);
        }
        if (it >= warmup) tot.push_back(now_ms() - t0);
      }
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
      printf("%s frame_ms median %.2f min %.2f (n=%zu)\n", fd.c_str(), med(tot), *std::min_element(tot.begin(), tot.end()),
             tot.size());
      double htp = 0, dsp_wall = 0, dsp_in = 0;
      for (auto& k : step_names) {
        double m = med(st[k]), d = med(dsp[k]);
        printf("  step %-9s %7.2f ms%s\n", k.c_str(), m, d > 0 ? (" (dsp " + std::to_string(d).substr(0, 5) + ")").c_str() : "");
        if (d > 0) { dsp_wall += m; dsp_in += d; } else { htp += m; }
      }
      printf("  htp pieces %.2f ms, msda calls %.2f ms (in-DSP %.2f, FastRPC %.2f)\n", htp, dsp_wall, dsp_in, dsp_wall - dsp_in);
      write_from(fd + "/bev.out.f32", get("bev"));
      write_from(fd + "/cls.out.f32", get("cls"));
      write_from(fd + "/pts.out.f32", get("pts"));
      fflush(stdout);
    }
    msda_rpc_close(h_msda);
    printf("PASS\n");
    fflush(stdout);
    _exit(0);  // skip static destructors (ORT teardown order aborts on exit, as in enc_run)
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    fflush(stdout);
    _exit(1);
  }
}

// Fast-BEV M0 / Fast-BEV++ end to end on the phone, one native process:
//   image encoder (HTP, int8, uint8 NHWC pixels in) -> view transform -> BEV net (HTP, int8)
//   -> decode + NMS (CPU, csrc/postproc.c)
// M0: the encoder writes into a 4-slot ring of rpcmem feature tables (each (rows + 1) x 64 uint8, the
//     last row the zero point); the 4 LUTs are computed here from the host's per-slot projection
//     matrices (the previous frames' change with ego motion); the fbgather DSP skel gathers the
//     (1, 200, 200, 1024) uint8 volume into an rpcmem buffer the BEV session reads.
//     With GATHER=htp the volume is gathered on the HTP instead (m0_viewbev.q8.c4.onnx).
// PP: encoder -> [feats, depth] -> view+BEV session (uint8 Gathers on the HTP, LUTs fixed per rig).
//
// usage: fastbev_run m0|pp <dir> <n_frames> [iters]
//   <dir>: models (see e2e_phone.py), io.txt ("name scale zero_point" per quantized tensor),
//   f<i>_img.bin (6, 256, 704, 3) uint8; m0: f<i>_proj.bin (4, 6, 3, 4) float32, f<i>_ages.txt
//   ("a0 a1 a2 a3 swap"); pp: f<i>_idx.bin, f<i>_didx.bin or one lut_idx.bin, lut_didx.bin (int32).
// Writes f<i>_dets.bin (float32 rows: x y z w l h yaw vx vy score label) and prints per-stage
// medians over `iters` passes of the whole sequence (after one warm-up pass).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

#include "onnxruntime_cxx_api.h"
extern "C" {
#include "fbgather_rpc.h"
#include "remote.h"
#include "rpcmem.h"
int nms_rotated(const float* boxes, int n, float thr, int max_keep, int* keep);
int nms_circle(const float* xy, int n, float thr, int max_keep, int* keep);
}

static double now_ms() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}
static void read_into(const std::string& p, void* dst, size_t n) {
  std::ifstream f(p, std::ios::binary);
  f.read((char*)dst, n);
  if (!f) throw std::runtime_error("short file " + p);
}

struct Q {
  float s;
  int z;
};
static std::map<std::string, Q> read_io(const std::string& p) {
  std::map<std::string, Q> m;
  std::ifstream f(p);
  std::string n;
  float s;
  int z;
  while (f >> n >> s >> z) m[n] = {s, z};
  return m;
}

struct RpcBuf {
  void* p = nullptr;
  size_t n = 0;
  explicit RpcBuf(size_t bytes) : n(bytes) {
    p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes);
    if (!p) throw std::runtime_error("rpcmem_alloc");
  }
  ~RpcBuf() { rpcmem_free(p); }
  template <class T>
  T* as() { return (T*)p; }
};

// ---------------------------------------------------------------- ORT
struct Net {
  std::unique_ptr<Ort::Session> s;
  std::vector<std::string> in, out;
};
static Ort::Env* g_env;
static Net load_net(const std::string& path, const std::string& ctx) {
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(1);
  so.AddConfigEntry("session.intra_op.allow_spinning", "0");
  so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  std::vector<Ort::ConstEpDevice> devs;
  for (const auto& d : g_env->GetEpDevices())
    if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
      devs.push_back(d);
  if (devs.empty()) throw std::runtime_error("no QNN NPU ep device");
  std::unordered_map<std::string, std::string> opts{{"backend_type", "htp"}, {"htp_performance_mode", "burst"},
                                                     {"htp_graph_finalization_optimization_mode", "3"}};
  so.AppendExecutionProvider_V2(*g_env, devs, opts);
  std::string run = path;
  if (!ctx.empty()) {
    std::ifstream exists(ctx);
    if (!exists) {
      Ort::ModelCompilationOptions co(*g_env, so);
      co.SetInputModelPath(path.c_str());
      co.SetOutputModelPath(ctx.c_str());
      co.SetEpContextEmbedMode(true);
      Ort::Status st = Ort::CompileModel(*g_env, co);
      if (!st.IsOK()) throw std::runtime_error("CompileModel: " + st.GetErrorMessage());
    }
    run = ctx;
  }
  Net n;
  n.s = std::make_unique<Ort::Session>(*g_env, run.c_str(), so);
  Ort::AllocatorWithDefaultOptions al;
  for (size_t i = 0; i < n.s->GetInputCount(); i++) n.in.push_back(n.s->GetInputNameAllocated(i, al).get());
  for (size_t i = 0; i < n.s->GetOutputCount(); i++) n.out.push_back(n.s->GetOutputNameAllocated(i, al).get());
  return n;
}
static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
template <class T>
static Ort::Value tensor(void* p, std::vector<int64_t> shape) {
  size_t n = 1;
  for (auto d : shape) n *= d;
  return Ort::Value::CreateTensor<T>(cpu_mem, (T*)p, n, shape.data(), shape.size());
}
static void run_net(Net& n, std::vector<Ort::Value>& ins, std::vector<Ort::Value>& outs) {
  std::vector<const char*> in, out;
  for (auto& s : n.in) in.push_back(s.c_str());
  for (auto& s : n.out) out.push_back(s.c_str());
  n.s->Run(Ort::RunOptions{nullptr}, in.data(), ins.data(), ins.size(), out.data(), outs.data(), outs.size());
}

// ---------------------------------------------------------------- M0 LUT (geometry.m0_lut)
static const int FH = 64, FW = 176, ROWS = 6 * FH * FW, NVOX = 200 * 200 * 4;
// proj: (6, 3, 4) = diag(1/4, 1/4, 1) @ lidar2img[:3]; last valid camera wins; `block` maps camera i
// to its feature block in the table (the upstream adjacent-frame camera swap, see data.py)
static void m0_lut(const float* proj, const int* block, int32_t* lut, int v0, int v1) {
  // one camera at a time over a strip of voxels, branch-free, so the compiler vectorizes it (NEON
  // fdiv / frintn = round half to even, like torch); later cameras overwrite (last valid wins)
  constexpr int S = 256;
  float px[S], py[S], pz[S];
  int32_t best[S];
  for (int s0 = v0; s0 < v1; s0 += S) {
    const int n = std::min(S, v1 - s0);
    for (int j = 0; j < n; j++) {
      const int v = s0 + j;
      px[j] = (v / 800) * 0.5f + -50.f;
      py[j] = ((v / 4) % 200) * 0.5f + -50.f;
      pz[j] = (v % 4) * 1.5f + -4.f;
      best[j] = ROWS;
    }
    for (int c = 0; c < 6; c++) {
      const float* P = proj + c * 12;
      const int32_t base = block[c] * FH * FW;
      for (int j = 0; j < n; j++) {
        const float u = ((P[0] * px[j] + P[1] * py[j]) + P[2] * pz[j]) + P[3];
        const float w = ((P[4] * px[j] + P[5] * py[j]) + P[6] * pz[j]) + P[7];
        const float d = ((P[8] * px[j] + P[9] * py[j]) + P[10] * pz[j]) + P[11];
        const float x = __builtin_roundevenf(u / d), y = __builtin_roundevenf(w / d);
        const bool ok = x >= 0.f && y >= 0.f && x < (float)FW && y < (float)FH && d > 0.f;
        best[j] = ok ? base + (int32_t)y * FW + (int32_t)x : best[j];
      }
    }
    memcpy(lut + s0, best, n * sizeof(int32_t));
  }
}
static void m0_lut_mt(const float* proj, const int* block, int32_t* lut, int nt) {
  std::vector<std::thread> th;
  const int per = (NVOX + nt - 1) / nt;
  for (int t = 0; t < nt; t++) th.emplace_back(m0_lut, proj, block, lut, t * per, std::min(NVOX, (t + 1) * per));
  for (auto& t : th) t.join();
}

// ---------------------------------------------------------------- decode (decode.py)
struct Det {
  float b[9], score;
  int label;
};
static inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }
// top k of uint8 values, ordered by value descending then index ascending (a stable sort of the
// scores, which are monotonic in the quantized logits): one histogram pass + one collecting pass
static void top_k_u8(const uint8_t* v, int n, int k, std::vector<int>& idx) {
  int hist[256] = {0};
  for (int i = 0; i < n; i++) hist[v[i]]++;
  k = std::min(k, n);
  int start[257], acc = 0, cut = 0;
  for (int b = 255; b >= 0; b--) {  // output position of each value's first element
    start[b] = acc;
    acc += hist[b];
    if (acc - hist[b] < k) cut = b;
  }
  idx.assign(k, 0);
  for (int i = 0; i < n; i++) {
    const int b = v[i];
    if (b < cut) continue;
    const int pos = start[b]++;
    if (pos < k) idx[pos] = i;
  }
}
static void m0_decode(const uint8_t* cls, const uint16_t* reg, const uint8_t* dir, Q qc, Q qr, std::vector<Det>& out) {
  static const float sizes[4][3] = {{0.8660f, 2.5981f, 1.f}, {0.5774f, 1.7321f, 1.f}, {1.f, 1.f, 1.f}, {0.4f, 0.4f, 1.f}};
  static const bool circle[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  static const float thr[10] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.5f, 0.5f, 0.2f};
  static const float radius[10] = {4, 12, 10, 10, 12, 0.85f, 0.85f, 0.175f, 0.175f, 1};
  static const float rescale[10] = {1.0f, 0.7f, 0.55f, 0.4f, 0.7f, 1.0f, 1.0f, 4.5f, 9.0f, 1.0f};
  const int NA = 100 * 100 * 8;
  static std::vector<uint8_t> mx(NA);
  for (int a = 0; a < NA; a++) mx[a] = *std::max_element(cls + a * 10, cls + a * 10 + 10);
  std::vector<int> top;
  top_k_u8(mx.data(), NA, 1000, top);
  struct Cand {
    float b[9], s[10];
    int dir;
  };
  std::vector<Cand> cand(top.size());
  for (size_t k = 0; k < top.size(); k++) {
    const int a = top[k], cell = a / 8, sz = (a / 2) % 4, rot = a % 2;
    const float ax = (cell % 100) - 49.5f, ay = (cell / 100) - 49.5f, az = -1.8f;
    const float aw = sizes[sz][0], al = sizes[sz][1], ah = sizes[sz][2], ar = rot ? 1.57f : 0.f;
    float d[9];
    for (int j = 0; j < 9; j++) d[j] = (reg[a * 9 + j] - qr.z) * qr.s;
    Cand& c = cand[k];
    const float diag = sqrtf(al * al + aw * aw), za = az + ah / 2;
    c.b[0] = d[0] * diag + ax;
    c.b[1] = d[1] * diag + ay;
    c.b[2] = d[2] * ah + za;
    c.b[3] = expf(d[3]) * aw;
    c.b[4] = expf(d[4]) * al;
    c.b[5] = expf(d[5]) * ah;
    c.b[6] = d[6] + ar;
    c.b[7] = d[7];
    c.b[8] = d[8];
    c.b[2] -= c.b[5] / 2;
    for (int j = 0; j < 10; j++) c.s[j] = sigm((cls[a * 10 + j] - qc.z) * qc.s);
    c.dir = dir[a * 2 + 1] > dir[a * 2];
  }
  std::vector<Det> all;
  std::vector<int> ord, keep;
  std::vector<float> arr;
  for (int i = 0; i < 10; i++) {
    ord.clear();
    for (size_t k = 0; k < cand.size(); k++)
      if (cand[k].s[i] > 0.05f) ord.push_back(k);
    if (ord.empty()) continue;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cand[a].s[i] > cand[b].s[i]; });
    keep.resize(ord.size());
    int nk;
    if (!circle[i]) {
      arr.resize(ord.size() * 5);
      for (size_t k = 0; k < ord.size(); k++) {
        const float* b = cand[ord[k]].b;
        float* o = &arr[k * 5];
        o[0] = b[0]; o[1] = b[1]; o[2] = b[3] * rescale[i]; o[3] = b[4] * rescale[i]; o[4] = b[6];
      }
      nk = nms_rotated(arr.data(), ord.size(), thr[i], ord.size(), keep.data());
    } else {
      arr.resize(ord.size() * 2);
      for (size_t k = 0; k < ord.size(); k++) { arr[2 * k] = cand[ord[k]].b[0]; arr[2 * k + 1] = cand[ord[k]].b[1]; }
      nk = nms_circle(arr.data(), ord.size(), radius[i], 83, keep.data());
    }
    for (int k = 0; k < nk; k++) {
      const Cand& c = cand[ord[keep[k]]];
      Det d;
      memcpy(d.b, c.b, sizeof d.b);
      float r = c.b[6] - (float)M_PI / 4;
      r = r - floorf(r / (float)M_PI) * (float)M_PI;
      d.b[6] = r + (float)M_PI / 4 + (float)M_PI * c.dir;
      d.score = c.s[i];
      d.label = i;
      all.push_back(d);
    }
  }
  std::stable_sort(all.begin(), all.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
  if (all.size() > 500) all.resize(500);
  out = all;
}

static void pp_decode(const uint8_t* heat, const uint16_t* reg, const uint16_t* hei, const uint16_t* dim,
                      const uint16_t* rot, const uint16_t* vel, const std::map<std::string, Q>& io,
                      std::vector<Det>& out) {
  static const float rescale[10] = {1.0f, 0.7f, 0.7f, 0.4f, 0.55f, 1.1f, 1.0f, 1.0f, 1.5f, 3.5f};
  const int HW = 128 * 128, N = HW * 10;
  static std::vector<uint8_t> chw(N);
  for (int p = 0; p < HW; p++)
    for (int c = 0; c < 10; c++) chw[c * HW + p] = heat[p * 10 + c];  // (cls, y, x) order like torch
  std::vector<int> top;
  top_k_u8(chw.data(), N, 500, top);
  auto dq = [&](const char* n, int v) { const Q& q = io.at(n); return (v - q.z) * q.s; };
  std::vector<Det> c;
  for (int t : top) {
    const int cls = t / HW, p = t % HW, y = p / 128, x = p % 128;
    Det d;
    d.score = sigm(dq("heatmap", chw[t]));
    d.label = cls;
    d.b[0] = (x + dq("reg", reg[p * 2])) * 0.8f - 51.2f;
    d.b[1] = (y + dq("reg", reg[p * 2 + 1])) * 0.8f - 51.2f;
    d.b[2] = dq("height", hei[p]);
    for (int j = 0; j < 3; j++) d.b[3 + j] = expf(dq("dim", dim[p * 3 + j]));
    d.b[6] = atan2f(dq("rot", rot[p * 2]), dq("rot", rot[p * 2 + 1]));
    d.b[7] = dq("vel", vel[p * 2]);
    d.b[8] = dq("vel", vel[p * 2 + 1]);
    if (d.score > 0.1f && fabsf(d.b[0]) <= 61.2f && fabsf(d.b[1]) <= 61.2f && fabsf(d.b[2]) <= 10.f) c.push_back(d);
  }
  std::stable_sort(c.begin(), c.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
  if (c.size() > 1000) c.resize(1000);
  std::vector<float> arr(c.size() * 5);
  for (size_t k = 0; k < c.size(); k++) {
    const float r = rescale[c[k].label];
    float* o = &arr[k * 5];
    o[0] = c[k].b[0]; o[1] = c[k].b[1]; o[2] = c[k].b[3] * r; o[3] = c[k].b[4] * r; o[4] = c[k].b[6];
  }
  std::vector<int> keep(c.size() + 1);
  const int nk = nms_rotated(arr.data(), c.size(), 0.2f, 500, keep.data());
  out.clear();
  for (int k = 0; k < nk; k++) {
    Det d = c[keep[k]];
    d.b[2] -= d.b[5] * 0.5f;
    out.push_back(d);
  }
}

static void write_dets(const std::string& p, const std::vector<Det>& d) {
  FILE* f = fopen(p.c_str(), "wb");
  for (auto& x : d) {
    float row[11];
    memcpy(row, x.b, 9 * sizeof(float));
    row[9] = x.score;
    row[10] = (float)x.label;
    fwrite(row, sizeof row, 1, f);
  }
  fclose(f);
}

static double med(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s m0|pp <dir> <n_frames> [iters]\n", argv[0]);
    return 2;
  }
  const std::string fam = argv[1], dir = argv[2];
  const int nf = atoi(argv[3]), iters = argc > 4 ? atoi(argv[4]) : 5;
  const int lut_threads = getenv("LUT_THREADS") ? atoi(getenv("LUT_THREADS")) : 4;
  const int dsp_threads = getenv("DSP_THREADS") ? atoi(getenv("DSP_THREADS")) : 4;
  const bool htp_gather = getenv("GATHER") && std::string(getenv("GATHER")) == "htp";
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "fastbev");
    g_env = &env;
    env.RegisterExecutionProviderLibrary("QNNExecutionProvider", "libonnxruntime_providers_qnn.so");
    auto io = read_io(dir + "/io.txt");
    double t0 = now_ms();
    Net enc = load_net(dir + "/" + fam + "_enc.q8.onnx", dir + "/" + fam + "_enc.q8.ctx.onnx");
    const std::string bev_name = fam == "pp" ? "pp_viewbev.q8" : htp_gather ? "m0_viewbev.q8.c4" : "m0_bev.q8";
    Net bev = load_net(dir + "/" + bev_name + ".onnx", dir + "/" + bev_name + ".ctx.onnx");
    printf("sessions ready in %.0f ms (%s)\n", now_ms() - t0, bev_name.c_str());
    std::vector<std::vector<uint8_t>> imgs(nf, std::vector<uint8_t>(6 * 256 * 704 * 3));
    for (int i = 0; i < nf; i++) read_into(dir + "/f" + std::to_string(i) + "_img.bin", imgs[i].data(), imgs[i].size());
    std::vector<std::string> stages;
    std::map<std::string, std::vector<double>> times;
    auto stamp = [&](const char* s, double& t) {
      double n = now_ms();
      times[s].push_back(n - t);
      if (std::find(stages.begin(), stages.end(), s) == stages.end()) stages.push_back(s);
      t = n;
    };
    if (fam == "m0") {
      std::vector<std::vector<float>> proj(nf, std::vector<float>(4 * 6 * 12));
      std::vector<std::array<int, 5>> ages(nf);
      for (int i = 0; i < nf; i++) {
        read_into(dir + "/f" + std::to_string(i) + "_proj.bin", proj[i].data(), proj[i].size() * 4);
        std::ifstream a(dir + "/f" + std::to_string(i) + "_ages.txt");
        for (int k = 0; k < 5; k++) a >> ages[i][k];
      }
      const size_t tb = (ROWS + 1) * 64 + 64;
      std::vector<std::unique_ptr<RpcBuf>> ring, luts;
      for (int k = 0; k < 4; k++) {
        ring.emplace_back(new RpcBuf(tb));
        memset(ring[k]->p, io.at("feats").z, tb);
        luts.emplace_back(new RpcBuf(NVOX * 4));
      }
      RpcBuf vol(NVOX * 256);
      remote_handle64 h = 0;
      if (!htp_gather) {
        struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
        remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
        if (fbgather_rpc_open("file:///fbgather_rpc.so?fbgather_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h))
          throw std::runtime_error("fbgather open failed");
        int rc;
        fbgather_rpc_perf_vote(h, 1, &rc);
      }
      std::vector<uint8_t> cls(100 * 100 * 80), dirc(100 * 100 * 16);
      std::vector<uint16_t> reg(100 * 100 * 72);
      static const int ident[6] = {0, 1, 2, 3, 4, 5}, swapped[6] = {0, 1, 2, 3, 5, 4};
      float slot0_proj[72];
      bool have_slot0 = false;
      for (int it = 0; it <= iters; it++) {
        for (int i = 0; i < nf; i++) {
          double t = now_ms(), tf = t;
          // the LUTs don't depend on the images: compute them on the CPU while the HTP runs the
          // encoder. Slot 0 (the current frame) depends only on the rig: recomputed only when its
          // projections change.
          double lut_ms = 0;
          std::thread lut_th([&] {
            double a = now_ms();
            for (int k = 0; k < 4; k++) {
              const float* P = proj[i].data() + k * 72;
              const int* blk = ages[i][k] > 0 && ages[i][4] ? swapped : ident;
              if (k == 0 && have_slot0 && !memcmp(P, slot0_proj, sizeof slot0_proj)) continue;
              m0_lut_mt(P, blk, luts[k]->as<int32_t>(), lut_threads);
              if (k == 0) { memcpy(slot0_proj, P, sizeof slot0_proj); have_slot0 = true; }
            }
            lut_ms = now_ms() - a;
          });
          {
            std::vector<Ort::Value> in, out;
            in.push_back(tensor<uint8_t>(imgs[i].data(), {6, 256, 704, 3}));
            out.push_back(tensor<uint8_t>(ring[i % 4]->p, {6, 64, 176, 64}));
            run_net(enc, in, out);
          }
          stamp("encoder (HTP)", t);
          lut_th.join();
          times["  (LUTs on the CPU, overlapped)"].push_back(lut_ms);
          stamp("LUTs (wait after encoder)", t);
          const int slot[4] = {(i - ages[i][0] + 400) % 4, (i - ages[i][1] + 400) % 4, (i - ages[i][2] + 400) % 4,
                               (i - ages[i][3] + 400) % 4};
          std::vector<Ort::Value> in, out;
          if (!htp_gather) {
            unsigned long long du = 0;
            int rc = fbgather_rpc_run(h, ring[slot[0]]->as<uint8_t>(), tb, ring[slot[1]]->as<uint8_t>(), tb,
                                      ring[slot[2]]->as<uint8_t>(), tb, ring[slot[3]]->as<uint8_t>(), tb,
                                      luts[0]->as<int32_t>(), NVOX, luts[1]->as<int32_t>(), NVOX, luts[2]->as<int32_t>(),
                                      NVOX, luts[3]->as<int32_t>(), NVOX, ROWS, dsp_threads, vol.as<uint8_t>(),
                                      NVOX * 256, &du);
            if (rc) throw std::runtime_error("fbgather rc " + std::to_string(rc));
            times["  (gather on the DSP itself)"].push_back(du / 1e3);
            stamp("gather (DSP RPC)", t);
            in.push_back(tensor<uint8_t>(vol.p, {1, 200, 200, 1024}));
          } else {
            for (int k = 0; k < 4; k++) in.push_back(tensor<uint8_t>(ring[slot[k]]->p, {ROWS + 1, 64}));
            for (int k = 0; k < 4; k++) in.push_back(tensor<int32_t>(luts[k]->p, {NVOX}));
          }
          out.push_back(tensor<uint8_t>(cls.data(), {1, 100, 100, 80}));
          out.push_back(tensor<uint16_t>(reg.data(), {1, 100, 100, 72}));
          out.push_back(tensor<uint8_t>(dirc.data(), {1, 100, 100, 16}));
          run_net(bev, in, out);
          stamp(htp_gather ? "gather + BEV net (HTP)" : "BEV net (HTP)", t);
          std::vector<Det> dets;
          m0_decode(cls.data(), reg.data(), dirc.data(), io.at("cls"), io.at("reg"), dets);
          stamp("decode + NMS (CPU)", t);
          times["total"].push_back(now_ms() - tf);
          if (it == 0) write_dets(dir + "/f" + std::to_string(i) + "_dets.bin", dets);
        }
        if (it == 0) times.clear();  // warm-up pass
      }
      if (h) fbgather_rpc_close(h);
    } else {
      // the LUTs depend only on calibration (a fixed rig: computed once); nuScenes' per-camera ego
      // poses make them differ slightly per frame, so the host may give one per frame
      std::vector<std::vector<int32_t>> idx(nf, std::vector<int32_t>(128 * 128 * 7)), didx = idx;
      for (int i = 0; i < nf; i++) {
        const std::string fi = dir + "/f" + std::to_string(i);
        const bool per = std::ifstream(fi + "_idx.bin").good();
        read_into(per ? fi + "_idx.bin" : dir + "/lut_idx.bin", idx[i].data(), idx[i].size() * 4);
        read_into(per ? fi + "_didx.bin" : dir + "/lut_didx.bin", didx[i].data(), didx[i].size() * 4);
      }
      std::vector<uint8_t> feats(6 * 16 * 44 * 64), depth(6 * 16 * 44 * 59), heat(128 * 128 * 10);
      std::vector<uint16_t> reg(128 * 128 * 2), hei(128 * 128), dim(128 * 128 * 3), rot(128 * 128 * 2),
          vel(128 * 128 * 2);
      for (int it = 0; it <= iters; it++) {
        for (int i = 0; i < nf; i++) {
          double t = now_ms(), tf = t;
          {
            std::vector<Ort::Value> in, out;
            in.push_back(tensor<uint8_t>(imgs[i].data(), {6, 256, 704, 3}));
            out.push_back(tensor<uint8_t>(feats.data(), {6, 16, 44, 64}));
            out.push_back(tensor<uint8_t>(depth.data(), {6, 16, 44, 59}));
            run_net(enc, in, out);
          }
          stamp("encoder (HTP)", t);
          std::vector<Ort::Value> in, out;
          in.push_back(tensor<uint8_t>(feats.data(), {6 * 16 * 44, 64}));  // the graph appends the zero row
          in.push_back(tensor<uint8_t>(depth.data(), {6 * 16 * 44 * 59}));
          in.push_back(tensor<int32_t>(idx[i].data(), {128 * 128 * 7}));
          in.push_back(tensor<int32_t>(didx[i].data(), {128 * 128 * 7}));
          out.push_back(tensor<uint8_t>(heat.data(), {1, 128, 128, 10}));
          out.push_back(tensor<uint16_t>(reg.data(), {1, 128, 128, 2}));
          out.push_back(tensor<uint16_t>(hei.data(), {1, 128, 128, 1}));
          out.push_back(tensor<uint16_t>(dim.data(), {1, 128, 128, 3}));
          out.push_back(tensor<uint16_t>(rot.data(), {1, 128, 128, 2}));
          out.push_back(tensor<uint16_t>(vel.data(), {1, 128, 128, 2}));
          run_net(bev, in, out);
          stamp("gather + BEV net (HTP)", t);
          std::vector<Det> dets;
          pp_decode(heat.data(), reg.data(), hei.data(), dim.data(), rot.data(), vel.data(), io, dets);
          stamp("decode + NMS (CPU)", t);
          times["total"].push_back(now_ms() - tf);
          if (it == 0) write_dets(dir + "/f" + std::to_string(i) + "_dets.bin", dets);
        }
        if (it == 0) times.clear();
      }
    }
    for (auto& s : stages) printf("stage %-28s median %7.2f ms\n", s.c_str(), med(times[s]));
    for (const char* s : {"  (LUTs on the CPU, overlapped)", "  (gather on the DSP itself)"})
      if (times.count(s)) printf("stage %-28s median %7.2f ms\n", s, med(times[s]));
    printf("total per frame median %.2f ms (%.1f FPS), n=%zu\n", med(times["total"]), 1000.0 / med(times["total"]),
           times["total"].size());
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    return 1;
  }
  return 0;
}

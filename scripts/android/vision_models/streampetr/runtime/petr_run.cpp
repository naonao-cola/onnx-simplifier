// StreamPETR's whole frame in one phone process: the int8 image piece and the fp16 head piece on the
// HTP (ORT QNN EP), and HostState (model.py) in C++ -- memory queue, ego-motion transforms, sine /
// nerf encodings, box post-processing and top-128 propagation -- instead of Python/torch on the host.
//   petr_run <dir> <img.onnx> <head.onnx> seq|pipe <warmup> <reps> <out_dir> <nframes>
// <head.onnx> is export.py's head_pe (uint8 feat, per-frame pe_in / cone). <dir> from prep.py. Frames run in order `warmup + reps` times (a frame with prev = 0 resets the
// memory); the last pass's cls / box per frame go to <out_dir>/f<i>_{cls,box}.bin (428 x 10 f32), and
// with DUMP=1 also the head inputs/outputs (f<i>_mem_*.bin, f<i>_{reg,dec}.bin) for check_run.py.
// pipe: a second thread runs frame t+1's image piece and position-embedding geometry (both independent of
// the memory queue) while this one does frame t's pre / head / post.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>

#include <unistd.h>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr int NQ = 300, NP = 128, NQ_ALL = NQ + NP, MEM = 512, EMB = 256, T = 4224, NCLS = 10, TOPK = 128;
constexpr float PC[6] = {-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
constexpr float POS[6] = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};
constexpr int NCAM = 6, FH = 16, FW = 44, NDEP = 64;

static void read_file(const std::string& p, void* dst, size_t n) {
  std::ifstream f(p, std::ios::binary);
  f.read((char*)dst, (std::streamsize)n);
  if (!f) throw std::runtime_error("short/missing " + p);
}
static void write_file(const std::string& p, const void* src, size_t n) {
  std::ofstream f(p, std::ios::binary);
  f.write((const char*)src, (std::streamsize)n);
}

struct Mat4 { float m[16]; };
static Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      float s = 0;
      for (int k = 0; k < 4; k++) s += a.m[i * 4 + k] * b.m[k * 4 + j];
      r.m[i * 4 + j] = s;
    }
  return r;
}
static void xform(const Mat4& p, const float* in, float* out) {  // transform_reference_points
  for (int i = 0; i < 3; i++) out[i] = p.m[i * 4] * in[0] + p.m[i * 4 + 1] * in[1] + p.m[i * 4 + 2] * in[2] + p.m[i * 4 + 3];
}
static Mat4 eye() { Mat4 r{}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
static float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
static float inv_sigmoid(float x) {
  x = std::min(std::max(x, 0.f), 1.f);
  const float e = 1e-5f;
  return std::log(std::max(x, e) / std::max(1 - x, e));
}

struct Frame { Mat4 pose, inv, l2i[NCAM], intr[NCAM]; double ts; int prev; std::vector<uint8_t> img; };

// 4x4 inverse (Gauss-Jordan in double, like torch.inverse up to rounding)
static Mat4 inverse(const Mat4& a) {
  double m[4][8];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 8; j++) m[i][j] = j < 4 ? a.m[i * 4 + j] : (j - 4 == i);
  for (int c = 0; c < 4; c++) {
    int p = c;
    for (int r = c + 1; r < 4; r++) if (std::fabs(m[r][c]) > std::fabs(m[p][c])) p = r;
    for (int j = 0; j < 8; j++) std::swap(m[c][j], m[p][j]);
    const double d = m[c][c];
    for (int j = 0; j < 8; j++) m[c][j] /= d;
    for (int r = 0; r < 4; r++)
      if (r != c) {
        const double f = m[r][c];
        for (int j = 0; j < 8; j++) m[r][j] -= f * m[c][j];
      }
  }
  Mat4 r;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) r.m[i * 4 + j] = (float)m[i][j + 4];
  return r;
}

// model.py position_embedding: pe_in (T, 192) = inverse_sigmoid(normalized coords3d), cone (T, 8)
static void position_embedding(const Frame& f, const float* coords_d, float* pe_in, float* cone) {
  Mat4 i2l[NCAM];
  for (int n = 0; n < NCAM; n++) i2l[n] = inverse(f.l2i[n]);
  for (int t = 0; t < NCAM * FH * FW; t++) {
    const int n = t / (FH * FW), y = (t / FW) % FH, x = t % FW;
    const float u = 16.f * x + 8.f, v = 16.f * y + 8.f;
    const Mat4& M = i2l[n];
    float c3[NDEP][3];
    for (int d = 0; d < NDEP; d++) {
      const float z = coords_d[d], s = std::max(z, 1e-5f), in[4] = {u * s, v * s, z, 1.f};
      for (int c = 0; c < 3; c++) {
        const float w = M.m[c * 4] * in[0] + M.m[c * 4 + 1] * in[1] + M.m[c * 4 + 2] * in[2] + M.m[c * 4 + 3] * in[3];
        c3[d][c] = (w - POS[c]) / (POS[3 + c] - POS[c]);
        pe_in[(size_t)t * NDEP * 3 + d * 3 + c] = inv_sigmoid(c3[d][c]);
      }
    }
    const Mat4& K = f.intr[t % NCAM];  // upstream's repeat: token t takes camera t % N's focal lengths
    cone[t * 8] = std::fabs(K.m[0]) / 1e3f;
    cone[t * 8 + 1] = std::fabs(K.m[5]) / 1e3f;
    for (int c = 0; c < 3; c++) { cone[t * 8 + 2 + c] = c3[NDEP - 1][c]; cone[t * 8 + 5 + c] = c3[34][c]; }
  }
}

// model.py HostState: the memory queue (up to MEM + TOPK rows between post and the next pre)
struct Host {
  std::vector<float> emb, ref, velo, refn_prop;
  std::vector<double> ts;
  std::vector<Mat4> pose;
  const float *refpts, *pseudo;
  float dim3[128], dim1[256];
  Host(const float* consts) : refpts(consts), pseudo(consts + NQ * 3) {
    for (int k = 0; k < 128; k++) dim3[k] = std::pow(10000.f, 2.f * (k / 2) / 128.f);
    for (int k = 0; k < 256; k++) dim1[k] = std::pow(10000.f, 2.f * (k / 2) / 256.f);
    reset();
  }
  void reset() {
    emb.assign(MEM * EMB, 0); ref.assign(MEM * 3, 0); velo.assign(MEM * 2, 0); ts.assign(MEM, 0);
    pose.assign(MEM, Mat4{});
  }
  // pre_update_memory + the host part of temporal alignment -> the head's mem_* inputs
  void pre(const Frame& f, float* mem_emb, float* pe3d, float* mtime, float* motion) {
    if (!f.prev) {
      reset();
      for (int i = 0; i < NP; i++) {
        for (int c = 0; c < 3; c++) ref[i * 3 + c] += pseudo[i * 3 + c] * (PC[3 + c] - PC[c]) + PC[c];
        Mat4 e = eye();
        for (int k = 0; k < 16; k++) pose[i].m[k] += e.m[k];
      }
    } else {
      ts.resize(MEM); pose.resize(MEM); ref.resize(MEM * 3); emb.resize(MEM * EMB); velo.resize(MEM * 2);
      for (int i = 0; i < MEM; i++) {
        ts[i] += f.ts;
        pose[i] = mul(f.inv, pose[i]);
        float r[3];
        xform(f.inv, &ref[i * 3], r);
        memcpy(&ref[i * 3], r, sizeof r);
      }
    }
    memcpy(mem_emb, emb.data(), sizeof(float) * MEM * EMB);
    refn_prop.resize(NP * 3);
    for (int i = 0; i < MEM; i++) {
      float rn[3];
      for (int c = 0; c < 3; c++) rn[c] = (ref[i * 3 + c] - PC[c]) / (PC[3 + c] - PC[c]);
      if (i < NP) memcpy(&refn_prop[i * 3], rn, sizeof rn);
      // pos2posemb3d: (y, x, z) blocks of 128, sin / cos interleaved
      const int order[3] = {1, 0, 2};
      for (int b = 0; b < 3; b++) {
        const float p = rn[order[b]] * (float)(2 * M_PI);
        for (int k = 0; k < 128; k += 2) {
          pe3d[i * 384 + b * 128 + k] = std::sin(p / dim3[k]);
          pe3d[i * 384 + b * 128 + k + 1] = std::cos(p / dim3[k + 1]);
        }
      }
      // pos2posemb1d on the float64 timestamps
      const double pt = ts[i] * (2 * M_PI);
      for (int k = 0; k < 256; k += 2) {
        mtime[i * 256 + k] = (float)std::sin(pt / (double)dim1[k]);
        mtime[i * 256 + k + 1] = (float)std::cos(pt / (double)dim1[k + 1]);
      }
      // nerf_positional_encoding([velo, ts, pose[:3, :]]) with 6 bands: [sin b0, cos b0, sin b1, ...] x 15
      float m[15];
      m[0] = velo[i * 2]; m[1] = velo[i * 2 + 1]; m[2] = (float)ts[i];
      for (int k = 0; k < 12; k++) m[3 + k] = pose[i].m[k];
      for (int b = 0; b < 6; b++) {
        const float band = (float)(1 << b);
        for (int k = 0; k < 15; k++) {
          motion[i * 180 + b * 30 + k] = std::sin(m[k] * band);
          motion[i * 180 + b * 30 + 15 + k] = std::cos(m[k] * band);
        }
      }
    }
  }
  // box coordinates + post_update_memory; box (428, 10) = reg with the centers decoded
  void post(const Frame& f, const float* cls, const float* reg, const float* dec, float* box) {
    std::vector<float> score(NQ_ALL);
    for (int q = 0; q < NQ_ALL; q++) {
      const float* r = refpts;
      const float* rq = q < NQ ? &r[q * 3] : &refn_prop[(q - NQ) * 3];
      memcpy(&box[q * NCLS], &reg[q * NCLS], sizeof(float) * NCLS);
      for (int c = 0; c < 3; c++) box[q * NCLS + c] = sigmoid(reg[q * NCLS + c] + inv_sigmoid(rq[c])) * (PC[3 + c] - PC[c]) + PC[c];
      float s = -1e30f;
      for (int c = 0; c < NCLS; c++) s = std::max(s, cls[q * NCLS + c]);
      score[q] = sigmoid(s);  // model.py ranks sigmoid(cls).max: fp32 sigmoid ties close logits, ties -> lower index
    }
    std::vector<int> top(NQ_ALL);
    std::iota(top.begin(), top.end(), 0);
    std::partial_sort(top.begin(), top.begin() + TOPK, top.end(),
                      [&](int a, int b) { return score[a] > score[b] || (score[a] == score[b] && a < b); });
    std::vector<float> nemb((TOPK + ts.size()) * EMB), nref((TOPK + ts.size()) * 3), nvelo((TOPK + ts.size()) * 2);
    std::vector<double> nts(TOPK + ts.size());
    std::vector<Mat4> npose(TOPK + ts.size());
    for (int k = 0; k < TOPK; k++) {
      const int q = top[k];
      memcpy(&nemb[k * EMB], &dec[q * EMB], sizeof(float) * EMB);
      xform(f.pose, &box[q * NCLS], &nref[k * 3]);
      nvelo[k * 2] = box[q * NCLS + 8];
      nvelo[k * 2 + 1] = box[q * NCLS + 9];
      nts[k] = -f.ts;
      npose[k] = f.pose;
    }
    for (size_t i = 0; i < ts.size(); i++) {
      memcpy(&nemb[(TOPK + i) * EMB], &emb[i * EMB], sizeof(float) * EMB);
      xform(f.pose, &ref[i * 3], &nref[(TOPK + i) * 3]);
      nvelo[(TOPK + i) * 2] = velo[i * 2];
      nvelo[(TOPK + i) * 2 + 1] = velo[i * 2 + 1];
      nts[TOPK + i] = ts[i] - f.ts;
      npose[TOPK + i] = mul(f.pose, pose[i]);
    }
    emb.swap(nemb); ref.swap(nref); velo.swap(nvelo); ts.swap(nts); pose.swap(npose);
  }
};

struct Piece {
  std::unique_ptr<Ort::Session> s;
  std::vector<std::string> in, out;
};
static std::unique_ptr<Ort::Env> env;
static std::vector<Ort::ConstEpDevice> npu;
static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

static Piece load(const std::string& path) {
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(1);
  std::unordered_map<std::string, std::string> o{{"backend_type", "htp"}, {"htp_performance_mode", "burst"}};
  so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  so.AppendExecutionProvider_V2(*env, npu, o);
  Piece P;
  P.s = std::make_unique<Ort::Session>(*env, path.c_str(), so);
  Ort::AllocatorWithDefaultOptions a;
  for (size_t i = 0; i < P.s->GetInputCount(); ++i) P.in.push_back(P.s->GetInputNameAllocated(i, a).get());
  for (size_t i = 0; i < P.s->GetOutputCount(); ++i) P.out.push_back(P.s->GetOutputNameAllocated(i, a).get());
  return P;
}

template <class T>
static Ort::Value tensor(T* p, size_t n, std::vector<int64_t> shape) {
  return Ort::Value::CreateTensor<T>(cpu_mem, p, n, shape.data(), shape.size());
}

static void run(Piece& P, std::vector<Ort::Value>& xs, std::vector<Ort::Value>& ys) {
  std::vector<const char*> in, out;
  for (auto& n : P.in) in.push_back(n.c_str());
  for (auto& n : P.out) out.push_back(n.c_str());
  P.s->Run(Ort::RunOptions{nullptr}, in.data(), xs.data(), xs.size(), out.data(), ys.data(), ys.size());
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 9) {
    fprintf(stderr, "usage: %s dir img.onnx head.onnx seq|pipe warmup reps out_dir nframes\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1], mode = argv[4], odir = argv[7];
  const int warmup = atoi(argv[5]), reps = atoi(argv[6]), nf = atoi(argv[8]);
  const bool dump = getenv("DUMP") != nullptr;
  try {
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "petr_run");
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", "libonnxruntime_providers_qnn.so");
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU) npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
    double t0 = now_ms();
    Piece img = load(dir + "/" + argv[2]), head = load(dir + "/" + argv[3]);
    printf("sessions %.0f ms\n", now_ms() - t0);

    std::vector<float> consts((NQ + NP) * 3 + NDEP);
    read_file(dir + "/consts.bin", consts.data(), consts.size() * 4);
    const float* coords_d = consts.data() + (NQ + NP) * 3;
    std::vector<Frame> fr(nf);
    for (int i = 0; i < nf; i++) {
      const std::string d = dir + "/f" + std::to_string(i);
      char meta[16 * 4 * 2 + 8 + 4 + 2 * NCAM * 64];
      read_file(d + "/meta.bin", meta, sizeof meta);
      memcpy(fr[i].pose.m, meta, 64); memcpy(fr[i].inv.m, meta + 64, 64);
      memcpy(&fr[i].ts, meta + 128, 8); memcpy(&fr[i].prev, meta + 136, 4);
      for (int n = 0; n < NCAM; n++) {
        memcpy(fr[i].l2i[n].m, meta + 140 + n * 64, 64);
        memcpy(fr[i].intr[n].m, meta + 140 + NCAM * 64 + n * 64, 64);
      }
      fr[i].img.resize((size_t)6 * 256 * 704 * 3);
      read_file(d + "/img.u8", fr[i].img.data(), fr[i].img.size());
    }

    Host host(consts.data());
    std::vector<uint8_t> fq[2] = {std::vector<uint8_t>((size_t)T * EMB), std::vector<uint8_t>((size_t)T * EMB)};
    std::vector<float> pe_in_s[2] = {std::vector<float>((size_t)T * NDEP * 3), std::vector<float>((size_t)T * NDEP * 3)};
    std::vector<float> cone_s[2] = {std::vector<float>((size_t)T * 8), std::vector<float>((size_t)T * 8)};
    std::vector<float> mem_emb(MEM * EMB), pe3d(MEM * 384), mtime(MEM * 256), motion(MEM * 180);
    std::vector<float> cls(NQ_ALL * NCLS), reg(NQ_ALL * NCLS), dec(NQ_ALL * EMB), box(NQ_ALL * NCLS);

    auto run_img = [&](const Frame& f, std::vector<uint8_t>& out) {
      std::vector<Ort::Value> xs, ys;
      xs.push_back(tensor<uint8_t>(const_cast<uint8_t*>(f.img.data()), f.img.size(), {6, 256, 704, 3}));
      ys.push_back(tensor<uint8_t>(out.data(), out.size(), {6, 16, 44, 256}));
      run(img, xs, ys);
    };
    const int total = (warmup + reps) * nf;
    std::vector<double> t_img, t_geo, t_pre, t_deq, t_head, t_post, t_frame;
    std::mutex mu;
    std::condition_variable cv;
    int produced = -1, consumed = -1;  // pipe: frame index whose image piece output is ready / has been taken
    std::vector<double> img_ms(total), geo_ms(total);
    std::thread producer;
    if (mode == "pipe")
      producer = std::thread([&] {
        for (int k = 0; k < total; k++) {
          {
            std::unique_lock<std::mutex> l(mu);
            cv.wait(l, [&] { return consumed >= k - 2; });  // two slots
          }
          double a = now_ms();
          run_img(fr[k % nf], fq[k % 2]);
          double b = now_ms();
          position_embedding(fr[k % nf], coords_d, pe_in_s[k % 2].data(), cone_s[k % 2].data());
          img_ms[k] = b - a;
          geo_ms[k] = now_ms() - b;
          { std::lock_guard<std::mutex> l(mu); produced = k; }
          cv.notify_all();
        }
      });
    double t_start = now_ms(), t_meas = 0;
    for (int k = 0; k < total; k++) {
      const Frame& f = fr[k % nf];
      const bool meas = k >= warmup * nf;
      if (k == warmup * nf) t_meas = now_ms();
      double a = now_ms();
      host.pre(f, mem_emb.data(), pe3d.data(), mtime.data(), motion.data());
      double b = now_ms(), c0 = b, c1;
      std::vector<float>&pe_in = pe_in_s[k % 2], &cone = cone_s[k % 2];
      if (mode == "pipe") {
        std::unique_lock<std::mutex> l(mu);
        cv.wait(l, [&] { return produced >= k; });
        c0 = now_ms();
      } else {
        run_img(f, fq[k % 2]);
        double gg = now_ms();
        img_ms[k] = gg - c0;
        position_embedding(f, coords_d, pe_in.data(), cone.data());
        geo_ms[k] = now_ms() - gg;
      }
      c1 = now_ms();
      double d = now_ms();
      std::vector<Ort::Value> xs, ys;
      xs.push_back(tensor<uint8_t>(fq[k % 2].data(), fq[k % 2].size(), {NCAM, FH, FW, EMB}));
      xs.push_back(tensor(pe_in.data(), pe_in.size(), {T, NDEP * 3}));
      xs.push_back(tensor(cone.data(), cone.size(), {T, 8}));
      xs.push_back(tensor(mem_emb.data(), mem_emb.size(), {MEM, EMB}));
      xs.push_back(tensor(pe3d.data(), pe3d.size(), {MEM, 384}));
      xs.push_back(tensor(mtime.data(), mtime.size(), {MEM, 256}));
      xs.push_back(tensor(motion.data(), motion.size(), {MEM, 180}));
      ys.push_back(tensor(cls.data(), cls.size(), {NQ_ALL, NCLS}));
      ys.push_back(tensor(reg.data(), reg.size(), {NQ_ALL, NCLS}));
      ys.push_back(tensor(dec.data(), dec.size(), {NQ_ALL, EMB}));
      run(head, xs, ys);
      if (mode == "pipe") {  // the head has read slot k % 2
        { std::lock_guard<std::mutex> l(mu); consumed = k; }
        cv.notify_all();
      }
      double e = now_ms();
      host.post(f, cls.data(), reg.data(), dec.data(), box.data());
      double g = now_ms();
      if (meas) {
        t_pre.push_back(b - a);
        t_img.push_back(img_ms[k]);
        t_geo.push_back(geo_ms[k]);
        t_deq.push_back(c0 - b);  // pipe: waiting for the image piece
        t_head.push_back(e - d);
        t_post.push_back(g - e);
        t_frame.push_back(g - a);
      }
      if (k >= total - nf) {
        const std::string p = odir + "/f" + std::to_string(k % nf);
        write_file(p + "_cls.bin", cls.data(), cls.size() * 4);
        write_file(p + "_box.bin", box.data(), box.size() * 4);
        if (dump) {
          write_file(p + "_reg.bin", reg.data(), reg.size() * 4);
          write_file(p + "_dec.bin", dec.data(), dec.size() * 4);
          write_file(p + "_mem_emb.bin", mem_emb.data(), mem_emb.size() * 4);
          write_file(p + "_mem_pe3d.bin", pe3d.data(), pe3d.size() * 4);
          write_file(p + "_mem_time.bin", mtime.data(), mtime.size() * 4);
          write_file(p + "_mem_motion.bin", motion.data(), motion.size() * 4);
          write_file(p + "_pe_in.bin", pe_in.data(), pe_in.size() * 4);
          write_file(p + "_cone.bin", cone.data(), cone.size() * 4);
        }
      }
    }
    double t_end = now_ms();
    if (producer.joinable()) producer.join();
    const int nmeas = reps * nf;
    printf("%s: img %.2f geometry %.2f pre %.2f wait %.2f head %.2f post %.2f ms (medians) | per frame %.2f ms (%.1f FPS), "
           "frame latency median %.2f ms, %d frames\n",
           mode.c_str(), median(t_img), median(t_geo), median(t_pre), median(t_deq), median(t_head), median(t_post),
           (t_end - t_meas) / nmeas, 1000.0 * nmeas / (t_end - t_meas), median(t_frame) + (mode == "pipe" ? median(t_img) + median(t_geo) : 0),
           nmeas);
    (void)t_start;
    printf("PASS\n");
    fflush(stdout);
  } catch (const std::exception& e) {
    fprintf(stderr, "FAIL %s\n", e.what());
    fflush(stderr);
    _exit(1);
  }
  // skip static destructors: ORT's QNN EP library tears down a mutex the global MemoryInfo still uses
  _exit(0);
}

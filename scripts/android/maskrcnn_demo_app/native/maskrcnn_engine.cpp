// Mask R-CNN demo engine: a JNI library the Android app drives frame by frame. It reuses
// ../../e2e_pipeline/e2e_run.cpp's pipeline machinery verbatim (#included below, its main renamed),
// so the app runs exactly the pipeline that PR #1841 measured: backbone and heads on the HTP (ORT +
// QNN EP), fused RPN and RoiAlign on the HVX DSP (our FastRPC skels), the rest on ORT CPU.
//
// On top of that, selectable per launch (the options string, "key=value;..."), each off by default
// so the plain e2e pipeline stays available for A/B:
//   quant=lut        image quantize through a per-channel 256-entry table (same values as the
//                    float math, which only depends on the pixel byte and the channel)
//   merge=a,b        replace the named ORT steps (pure ScatterND chains: the RoiAlign level merges,
//                    seg2/seg4 in pipe_e_opt.txt) with a native multithreaded row scatter
//   pipeline=<step>  two-stage cross-frame pipelining: stage A = steps before <step>, stage B = the
//                    rest, on its own thread; frame N+1's stage A runs while frame N's stage B does
//   env.NAME=value   set an environment knob before init (e.g. env.ORT_THREADS=6, env.ORT_SPIN=1:
//                    ORT's pool spinning, off by default here, see e2e_pipeline/README.md)
//   par_load=1       create the HTP sessions in parallel threads (the DSP skels always open on
//                    their own thread, overlapped with session creation)
//   warmup=0         skip the warm-up inference at the end of init (default 1: one gray frame plus
//                    every ortpad bucket, so the first camera frame is not the cold run)
// The image comes from an RGBA bitmap (quantized straight to the backbone's uint8 NHWC input) and
// results/timings go back over JNI.
#define E2E_STORE_STORAGE static thread_local
#define main e2e_run_main
#include "../../e2e_pipeline/e2e_run.cpp"
#undef main

#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include <cerrno>
#include <condition_variable>
#include <functional>
#include <mutex>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MaskRcnnDemo", __VA_ARGS__)

namespace {
const float kMeanBgr[3] = {102.9801f, 115.9465f, 122.7717f};  // maskrcnn_e2e/eval_common.py
const int kInH = 800, kInW = 1088;                                // the backbone's input

// timing buckets returned to Java: total, preprocess, backbone, rpn, roialign, heads, cpu(other),
// stage A, stage B, pipeline wait (stage A blocked handing off to stage B)
enum { T_TOTAL, T_PRE, T_BACKBONE, T_RPN, T_ROI, T_HEADS, T_CPU, T_STAGE_A, T_STAGE_B, T_WAIT, T_N };

std::vector<std::unique_ptr<Step>> g_steps;
std::map<std::string, std::string> g_opt;
std::string g_err;
bool g_lut = false;
std::vector<std::string> g_merge;
size_t g_split = 0;  // first stage-B step index; 0 = no pipelining
uint8_t g_lut_tab[3][256];

int bucket(const Step& S) {
  if (S.op == "quant_in") return T_PRE;
  if (S.op == "rpn") return T_RPN;
  if (S.op == "roialign") return T_ROI;
  if (S.name == "backbone") return T_BACKBONE;
  if (S.name == "box_head" || S.name == "mask_head") return T_HEADS;
  return T_CPU;
}

// ---- image -> backbone input ------------------------------------------------------------------
// quant_in fused with eval_common.canvas: q = sat(rint((p - mean) / s) + z) inside the image,
// sat(rint(0 / s) + z) in the zero padding. With quant=lut the per-pixel float math is replaced by
// a table built with exactly that expression, so results are bit-identical.
uint8_t qval(float v, float s, int z) {
  float q = __builtin_rintf(v / s) + (float)z;
  return (uint8_t)(q < 0.f ? 0.f : (q > 255.f ? 255.f : q));
}
void quant_rgba(Step& S, const uint8_t* px, int w, int h, int stride) {
  const float s = std::stof(S.f[3]);
  const int z = std::stoi(S.f[4]);
  const int H = kInH, W = kInW, C = 3;
  uint8_t* q = (uint8_t*)S.buf.get((size_t)H * W * C);
  const uint8_t pad = qval(0.f, s, z);
  par(H, envi("DQ_THREADS", 4), [&](long a, long b) {
    for (long y = a; y < b; ++y) {
      uint8_t* o = q + (long)y * W * C;
      long xx = 0;
      if (y < h) {
        const uint8_t* row = px + (long)y * stride;
        const long n = std::min<long>(w, W);
        if (g_lut) {
          for (; xx < n; ++xx) {  // BGR channel c <- RGBA byte (2 - c)
            o[xx * 3 + 0] = g_lut_tab[0][row[4 * xx + 2]];
            o[xx * 3 + 1] = g_lut_tab[1][row[4 * xx + 1]];
            o[xx * 3 + 2] = g_lut_tab[2][row[4 * xx + 0]];
          }
        } else {
          for (; xx < n; ++xx)
            for (int c = 0; c < 3; ++c) o[xx * C + c] = qval((float)row[4 * xx + (2 - c)] - kMeanBgr[c], s, z);
        }
      }
      memset(o + xx * C, pad, (size_t)(W - xx) * C);
    }
  });
  put_raw(S.f[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, H, W, C}, q);
}

// Camera fast path: Camera2 YUV_420_888 -> gravity-up, letterboxed, quantized uint8 NHWC backbone
// input in one pass, plus the same upright region as an RGBA display bitmap. rot (0/90/180/270) is
// the clockwise rotation that makes the sensor frame upright; the upright frame (RW x RH) is scaled
// to fit 1088x800 (eval_common.canvas: top-left aligned, zero-padded), nearest-neighbour sampling.
// YUV -> RGB is JFIF full-range BT.601 (what camera YUV_420_888 is), fixed point.
struct Yuv {
  const uint8_t *y, *u, *v;
  long ycap, ucap, vcap;  // plane buffer sizes (direct ByteBuffer capacities)
  int ys, uvs, uvps, w, h, rot;
  uint8_t* disp;  // RGBA, fw x fh, stride disp_stride (may be null)
  int disp_stride;
};
void fit_dims(int w, int h, int rot, int* fw, int* fh) {
  const int RW = (rot % 180) ? h : w, RH = (rot % 180) ? w : h;
  const float ratio = std::min(1088.f / RW, 800.f / RH);
  *fw = std::max(1, (int)(RW * ratio));
  *fh = std::max(1, (int)(RH * ratio));
}
void quant_yuv(Step& S, const Yuv& Y) {
  const float s = std::stof(S.f[3]);
  const int z = std::stoi(S.f[4]);
  const int H = kInH, W = kInW, C = 3;
  uint8_t* q = (uint8_t*)S.buf.get((size_t)H * W * C);
  const uint8_t pad = qval(0.f, s, z);
  const uint8_t *py = Y.y, *pu = Y.u, *pv = Y.v;
  int fw, fh;
  fit_dims(Y.w, Y.h, Y.rot, &fw, &fh);
  const int RW = (Y.rot % 180) ? Y.h : Y.w, RH = (Y.rot % 180) ? Y.w : Y.h;
  // output (ox,oy) -> upright (rx,ry) = (ox*RW/fw, oy*RH/fh) -> sensor (sx,sy)
  std::vector<int> rxs(fw), rys(fh);
  for (int x = 0; x < fw; ++x) rxs[x] = std::min(RW - 1, (int)(((long)x * RW + RW / 2) / fw));
  for (int y = 0; y < fh; ++y) rys[y] = std::min(RH - 1, (int)(((long)y * RH + RH / 2) / fh));
  par(H, envi("DQ_THREADS", 4), [&](long a, long b) {
    for (long oy = a; oy < b; ++oy) {
      uint8_t* o = q + oy * W * C;
      int ox = 0;
      if (oy < fh) {
        uint8_t* d = Y.disp ? Y.disp + oy * Y.disp_stride : nullptr;
        const int ry = rys[oy];
        for (; ox < fw; ++ox) {
          const int rx = rxs[ox];
          int sx, sy;
          switch (Y.rot) {
            case 90: sx = ry; sy = Y.h - 1 - rx; break;
            case 180: sx = Y.w - 1 - rx; sy = Y.h - 1 - ry; break;
            case 270: sx = Y.w - 1 - ry; sy = rx; break;
            default: sx = rx; sy = ry;
          }
          const int yy = py[sy * Y.ys + sx];
          const int ci = (sy >> 1) * Y.uvs + (sx >> 1) * Y.uvps;
          const int uu = pu[ci] - 128, vv = pv[ci] - 128;
          // x1024 fixed point: 1.402, 0.344136, 0.714136, 1.772
          int r = yy + ((1436 * vv + 512) >> 10);
          int g = yy - ((352 * uu + 731 * vv + 512) >> 10);
          int bb = yy + ((1815 * uu + 512) >> 10);
          r = r < 0 ? 0 : (r > 255 ? 255 : r);
          g = g < 0 ? 0 : (g > 255 ? 255 : g);
          bb = bb < 0 ? 0 : (bb > 255 ? 255 : bb);
          o[ox * 3 + 0] = g_lut_tab[0][bb];
          o[ox * 3 + 1] = g_lut_tab[1][g];
          o[ox * 3 + 2] = g_lut_tab[2][r];
          if (d) { d[4 * ox] = (uint8_t)r; d[4 * ox + 1] = (uint8_t)g; d[4 * ox + 2] = (uint8_t)bb; d[4 * ox + 3] = 255; }
        }
      }
      memset(o + ox * C, pad, (size_t)(W - ox) * C);
    }
  });
  put_raw(S.f[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, H, W, C}, q);
}

// ---- native level merge ---------------------------------------------------------------------
// "scatter_rows <name> base,idx1,upd1,...,idxK,updK out": the ScatterND chain the RoiAlign level
// merges are (out = base; out[idx_k[i]] = upd_k[i], k in order, later writes win), done as row
// copies. If the index sets cover every row of base, base itself is never read (the ORT segment
// copies it first); otherwise it is copied.
void scatter_rows(Step& S) {
  auto in = split(S.f[2], ',');
  Tensor& base = get(in[0]);
  const long R = base.shape.empty() ? 0 : (long)base.shape[0];
  const size_t row = R ? base.count() / (size_t)R * esize(base.type) : 0;
  char* out = (char*)S.buf.get(std::max<size_t>(R * row, 1));
  std::vector<std::pair<const int64_t*, std::pair<long, const char*>>> parts;
  std::vector<char> hit(R, 0);
  long covered = 0;
  for (size_t k = 1; k + 1 < in.size(); k += 2) {
    Tensor& idx = get(in[k]);
    Tensor& upd = get(in[k + 1]);
    const long n = idx.shape.empty() ? 0 : (long)idx.shape[0];
    const int64_t* ix = (const int64_t*)idx.data;
    for (long i = 0; i < n; ++i) {
      int64_t r = ix[i] < 0 ? ix[i] + R : ix[i];
      if (r < 0 || r >= R) throw std::runtime_error("scatter_rows: index out of range");
      if (!hit[r]) { hit[r] = 1; ++covered; }
    }
    parts.push_back({ix, {n, (const char*)upd.data}});
  }
  if (covered < R) memcpy(out, base.data, R * row);
  // rows are independent; later parts win on duplicates, so split the row *targets* across threads
  // and apply the parts in order within each thread
  par(R, envi("MERGE_THREADS", 4), [&](long a, long b) {
    for (auto& p : parts)
      for (long i = 0; i < p.second.first; ++i) {
        int64_t r = p.first[i] < 0 ? p.first[i] + R : p.first[i];
        if (r >= a && r < b) memcpy(out + r * row, p.second.second + i * row, row);
      }
  });
  put_raw(S.f[3], base.type, base.shape, out);
}

// Our DSP skels aren't reentrant: two stages calling the same skel at once (e.g. box RoiAlign in
// stage A while stage B runs the mask RoiAlign) fails with rc=78. Serialize each skel's calls.
std::mutex g_rpn_mu, g_roi_mu;
void exec_step(Step& S) {
  if (S.op == "scatter_rows") {
    scatter_rows(S);
  } else if (S.op == "roialign") {
    std::lock_guard<std::mutex> l(g_roi_mu);
    exec(S);
  } else if (S.op == "rpn") {
    std::lock_guard<std::mutex> l(g_rpn_mu);
    exec(S);
  } else {
    exec(S);
  }
}

// ---- two-stage pipeline ---------------------------------------------------------------------
// Stage-A steps whose outputs stage B reads own buffers (quant_in/dq/rpn/roialign: RpcBuf members)
// that frame N+1's stage A would overwrite while stage B still reads frame N's. Each stage-A step
// therefore alternates between two buffer sets by frame parity (swapped in place, only stage A's
// thread touches stage-A steps). Stage A hands frame N+1 over only once stage B has finished frame
// N, so frame N+2 (same parity as N) never overwrites buffers stage B is still reading.
struct Side { RpcBuf b[4]; };
std::vector<Side> g_side;
std::vector<int> g_side_cur;  // which parity the step's live buffers belong to
void swap_buf(RpcBuf& a, RpcBuf& b) { std::swap(a.p, b.p); std::swap(a.cap, b.cap); }
void use_parity(size_t k, int parity) {
  if (g_side_cur[k] == parity) return;
  Step& S = *g_steps[k];
  swap_buf(S.buf, g_side[k].b[0]);
  swap_buf(S.buf2, g_side[k].b[1]);
  swap_buf(S.buf3, g_side[k].b[2]);
  swap_buf(S.buf4, g_side[k].b[3]);
  g_side_cur[k] = parity;
}

struct Packet {
  long id = -1;
  std::map<std::string, Tensor> store;
  double t[T_N] = {0};
};
struct Done {
  long id = -1;
  std::vector<float> boxes, scores, masks;
  std::vector<int> labels;
  double t[T_N] = {0};
};

std::mutex g_mu;
std::condition_variable g_cv;
bool g_has_in = false, g_stop = false, g_b_busy = false;
Packet g_in;
Done g_done;
bool g_has_done = false;
std::string g_b_err;
std::thread g_b;

void collect(Done& d) {
  Tensor& b = get("6568");   // boxes  fp32 [n,4]
  Tensor& lb = get("6570");  // labels int64 [n]
  Tensor& sc = get("6572");  // scores fp32 [n]
  Tensor& mk = get("6887");  // masks  fp32 [n,1,28,28]
  size_t n = sc.count();
  d.boxes.assign((const float*)b.data, (const float*)b.data + 4 * n);
  d.scores.assign((const float*)sc.data, (const float*)sc.data + n);
  d.masks.assign((const float*)mk.data, (const float*)mk.data + 784 * n);
  d.labels.resize(n);
  for (size_t i = 0; i < n; ++i) d.labels[i] = (int)((const int64_t*)lb.data)[i];
}

void run_steps(size_t a, size_t b, double* t) {
  for (size_t k = a; k < b; ++k) {
    double s = now_ms();
    exec_step(*g_steps[k]);
    t[bucket(*g_steps[k])] += now_ms() - s;
  }
}

void stage_b_loop() {
  for (;;) {
    Packet p;
    {
      std::unique_lock<std::mutex> l(g_mu);
      g_cv.wait(l, [] { return g_has_in || g_stop; });
      if (g_stop) return;
      p = std::move(g_in);
      g_has_in = false;
      g_b_busy = true;
    }
    g_cv.notify_all();
    Done d;
    std::string err;
    try {
      store = std::move(p.store);
      double s = now_ms();
      run_steps(g_split, g_steps.size(), p.t);
      p.t[T_STAGE_B] = now_ms() - s;
      collect(d);
      store.clear();
    } catch (const std::exception& e) {
      err = e.what();
    }
    d.id = p.id;
    std::copy(p.t, p.t + T_N, d.t);
    {
      std::lock_guard<std::mutex> l(g_mu);
      if (!err.empty()) g_b_err = err;
      g_done = std::move(d);
      g_has_done = true;
      g_b_busy = false;
    }
    g_cv.notify_all();
  }
}

// One full inference on a gray (letterbox-colored) frame, then every ortpad bucket once on zeros:
// the first run of each HTP graph and ORT session is several times slower than the steady state,
// and a gray frame may not reach every bucket (the mask head's 100 needs > 32 detections).
void warmup() {
  double t[T_N] = {0};
  store.clear();
  Step& S0 = *g_steps[0];
  const float s = std::stof(S0.f[3]);
  const int z = std::stoi(S0.f[4]);
  uint8_t* q = (uint8_t*)S0.buf.get((size_t)kInH * kInW * 3);
  memset(q, qval(0.f, s, z), (size_t)kInH * kInW * 3);
  put_raw(S0.f[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, kInH, kInW, 3}, q);
  run_steps(1, g_steps.size(), t);
  store.clear();
  for (auto& S : g_steps)
    for (auto& kv : S->buckets) {
      Session& B = kv.second;
      Tensor x;
      x.type = B.s->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
      x.shape = B.in_shape[0];
      std::vector<char> zero(x.count() * esize(x.type), 0);
      x.data = zero.data();
      Ort::Value v = view(x);
      run_session(B, &v);
    }
  store.clear();
}

void init(const std::string& model_dir, const std::string& lib_dir, const std::string& pipe, const std::string& opts) {
  for (auto& kv : split(opts, ';')) {
    auto e = kv.find('=');
    if (e != std::string::npos) g_opt[kv.substr(0, e)] = kv.substr(e + 1);
  }
  // "env.NAME=value" sets an environment knob e2e_run.cpp reads (ORT_THREADS, DQ_THREADS,
  // ROI_THREADS, RPN_MODE, MERGE_THREADS), before anything reads it
  for (auto& kv : g_opt)
    if (kv.first.rfind("env.", 0) == 0) setenv(kv.first.substr(4).c_str(), kv.second.c_str(), 1);
  g_lut = g_opt["quant"] == "lut";
  g_merge = split(g_opt.count("merge") ? g_opt["merge"] : "-", ',');
  // The DSP loads skels (libQnnHtpV69Skel.so, librpn_rpc.so, libroialign_rpc.so) through this
  // process's FastRPC file listener, which searches ADSP_LIBRARY_PATH. Must be set before the
  // first FastRPC session opens.
  std::string adsp = lib_dir + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
  setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
  if (chdir(model_dir.c_str())) throw std::runtime_error("chdir " + model_dir + ": " + strerror(errno));
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  int urc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  LOGI("unsigned PD request rc=%d, opts '%s'", urc, opts.c_str());

  Ort::ThreadingOptions to;
  to.SetGlobalIntraOpNumThreads(envi("ORT_THREADS", 4));
  to.SetGlobalInterOpNumThreads(1);
  to.SetGlobalSpinControl(envi("ORT_SPIN", 0));
  const double t_init = now_ms();
  env = std::make_unique<Ort::Env>(to, ORT_LOGGING_LEVEL_WARNING, "demo");

  std::ifstream pf(pipe);
  if (!pf) throw std::runtime_error("cannot open " + pipe + ": " + strerror(errno));
  std::string line;
  bool need_htp = false, need_rpn = false, need_roi = false;
  while (std::getline(pf, line)) {
    if (line.empty()) continue;
    auto S = std::make_unique<Step>();
    std::istringstream ss(line);
    std::string w;
    while (ss >> w) S->f.push_back(w);
    S->op = S->f[0];
    S->name = S->op == "ort" || S->op == "ortpad" ? S->f[1]
              : S->op + ":" + S->f[S->op == "rpn" ? 4 : S->op == "roialign" || S->op == "mask_sel" ? 3 : 2];
    if (S->op == "ort" && std::find(g_merge.begin(), g_merge.end(), S->name) != g_merge.end()) {
      // "ort seg2 model cpu - IN OUT" -> "scatter_rows seg2 IN OUT"
      S->f = {"scatter_rows", S->name, S->f[5], S->f[6]};
      S->op = "scatter_rows";
    }
    need_htp |= (S->op == "ort" && S->f[3] == "htp") || (S->op == "ortpad" && S->f[2] == "htp");
    need_rpn |= S->op == "rpn";
    need_roi |= S->op == "roialign";
    g_steps.push_back(std::move(S));
  }
  if (g_steps.empty() || g_steps[0]->op != "quant_in") throw std::runtime_error("pipe must start with quant_in");
  std::string missing;
  for (auto& S : g_steps) {
    std::vector<std::string> files;
    if (S->op == "ort") files.push_back(S->f[2]);
    if (S->op == "ortpad")
      for (auto& e : split(S->f[5], ',')) files.push_back(e.substr(e.find(':') + 1));
    for (auto& f : files)
      if (access(f.c_str(), R_OK)) missing += " " + f + " (" + strerror(errno) + ")";
  }
  for (const char* f : {"levels.txt", "model.txt"})
    if (access(f, R_OK)) missing += std::string(" ") + f + " (" + strerror(errno) + ")";
  if (!missing.empty()) throw std::runtime_error("model files not readable in " + model_dir + ":" + missing);
  if (g_opt.count("pipeline")) {
    for (size_t k = 1; k < g_steps.size(); ++k)
      if (g_steps[k]->name == g_opt["pipeline"]) g_split = k;
    if (!g_split) throw std::runtime_error("pipeline split step not found: " + g_opt["pipeline"]);
  }
  {
    const float s = std::stof(g_steps[0]->f[3]);
    const int z = std::stoi(g_steps[0]->f[4]);
    for (int c = 0; c < 3; ++c)
      for (int p = 0; p < 256; ++p) g_lut_tab[c][p] = qval((float)p - kMeanBgr[c], s, z);
  }

  if (need_htp) {
    std::string ep = lib_dir + "/libonnxruntime_providers_qnn.so";
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", ep.c_str());
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
    LOGI("QNN EP registered, %zu NPU device(s)", npu.size());
  }
  const double t_ep = now_ms();
  // The DSP skels open (and the RPN model constants upload) on their own thread while the ORT
  // sessions are created -- but only after the first HTP session exists: opening our FastRPC
  // sessions while QNN sets up its HTP device makes that setup fail ("Failed to create device ...
  // INVALID_CONFIG", "Unable to acquire runtime HTP arch"), and the session then lands on the
  // (disabled) CPU EP.
  std::string dsp_err;
  auto dsp_open = [&] {
    try {
      const double t = now_ms();
      if (need_rpn) {
        if (rpn_rpc_open("file:///librpn_rpc.so?rpn_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h_rpn))
          throw std::runtime_error("rpn_rpc_open failed");
        rpn_init();
      }
      if (need_roi &&
          roialign_rpc_open("file:///libroialign_rpc.so?roialign_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h_roi))
        throw std::runtime_error("roialign_rpc_open failed");
      LOGI("dsp skels open in %.1f ms", now_ms() - t);
    } catch (const std::exception& ex) {
      dsp_err = ex.what();
    }
  };
  std::thread dsp;
  double t0 = now_ms();
  // one job per session; ortpad buckets pre-sized so parallel jobs never touch a shared vector
  std::vector<std::pair<bool, std::function<void()>>> jobs;  // (on the HTP, job)
  std::vector<std::vector<double>> bucket_ms(g_steps.size());
  for (size_t k = 0; k < g_steps.size(); ++k) {
    Step* S = g_steps[k].get();
    double* bms = nullptr;
    if (S->op == "ort") {
      jobs.emplace_back(S->f[3] == "htp", [S] { S->sess = wrap(make_session(S->f[2], S->f[3], S->f[4], &S->create_ms)); });
    } else if (S->op == "ortpad") {
      auto bs = split(S->f[5], ',');
      S->buckets.resize(bs.size());
      bucket_ms[k].assign(bs.size(), 0);
      bms = bucket_ms[k].data();
      for (size_t i = 0; i < bs.size(); ++i) {
        const std::string e = bs[i];
        jobs.emplace_back(S->f[2] == "htp", [S, i, e, bms] {
          auto c = e.find(':');
          S->buckets[i] = {std::stoi(e.substr(0, c)), wrap(make_session(e.substr(c + 1), S->f[2], S->f[3], &bms[i]))};
        });
      }
    }
  }
  const bool par_load = g_opt["par_load"] == "1";
  std::vector<std::thread> th;
  std::vector<std::string> errs(jobs.size());
  size_t first_htp = jobs.size();
  for (size_t j = 0; j < jobs.size() && first_htp == jobs.size(); ++j)
    if (jobs[j].first) first_htp = j;
  auto run = [&](size_t j) {
    try { jobs[j].second(); } catch (const std::exception& ex) { errs[j] = ex.what(); }
  };
  if (first_htp < jobs.size()) run(first_htp);  // sets up the QNN HTP device
  dsp = std::thread(dsp_open);
  for (size_t j = 0; j < jobs.size(); ++j) {
    if (j == first_htp) continue;
    if (par_load && jobs[j].first) th.emplace_back(run, j);
    else run(j);
  }
  for (auto& t : th) t.join();
  dsp.join();
  for (auto& e : errs)
    if (!e.empty()) throw std::runtime_error(e);
  if (!dsp_err.empty()) throw std::runtime_error(dsp_err);
  for (size_t k = 0; k < g_steps.size(); ++k) {
    Step& S = *g_steps[k];
    for (double ms : bucket_ms[k]) S.create_ms += ms;
    if (S.create_ms > 0) LOGI("session %s create_ms %.1f", S.name.c_str(), S.create_ms);
  }
  LOGI("setup_ms %.1f, %zu steps, split at %zu (%s)", now_ms() - t0, g_steps.size(), g_split,
       g_split ? g_steps[g_split]->name.c_str() : "none");
  g_side.resize(g_steps.size());
  g_side_cur.assign(g_steps.size(), 0);
  const double t_sess = now_ms();
  if (!g_opt.count("warmup") || g_opt["warmup"] != "0") warmup();
  LOGI("init phases: env+ep %.1f, sessions+dsp %.1f, warmup %.1f ms (par_load %d)", t_ep - t_init, t_sess - t_ep,
       now_ms() - t_sess, (int)par_load);
  if (g_split) g_b = std::thread(stage_b_loop);
}

long g_frame_parity = 0;

// Runs stage A (or the whole pipeline when not pipelining) for frame `id` on the calling thread.
// Returns the finished result's frame id into `out` if one is ready (pipelined: usually the
// previous frame), -1 if none yet.
long run_frame(const std::function<void(Step&)>& fill, long id, Done& out) {
  double t[T_N] = {0};
  if (!g_split) {
    store.clear();
    double t0 = now_ms();
    fill(*g_steps[0]);
    t[T_PRE] += now_ms() - t0;
    run_steps(1, g_steps.size(), t);
    t[T_TOTAL] = now_ms() - t0;
    t[T_STAGE_A] = t[T_TOTAL];
    out.id = id;
    collect(out);
    std::copy(t, t + T_N, out.t);
    store.clear();
    return id;
  }
  const int parity = (int)(g_frame_parity++ & 1);
  for (size_t k = 0; k < g_split; ++k) use_parity(k, parity);
  store.clear();
  double t0 = now_ms();
  fill(*g_steps[0]);
  t[T_PRE] += now_ms() - t0;
  run_steps(1, g_split, t);
  t[T_STAGE_A] = now_ms() - t0;
  Packet p;
  p.id = id;
  p.store = std::move(store);
  store.clear();
  std::copy(t, t + T_N, p.t);
  p.t[T_TOTAL] = t0;  // stage B turns this into end-to-end latency
  double w0 = now_ms();
  std::unique_lock<std::mutex> l(g_mu);
  // Hand off only once stage B is idle (finished the previous frame): the next stage A then reuses
  // this parity's buffers only after stage B is done reading them. Overlap stays one frame deep:
  // stage A of frame N+1 runs while stage B works on frame N.
  g_cv.wait(l, [] { return !g_has_in && !g_b_busy; });
  p.t[T_WAIT] = now_ms() - w0;
  g_in = std::move(p);
  g_has_in = true;
  g_cv.notify_all();
  if (!g_b_err.empty()) throw std::runtime_error("stage B: " + g_b_err);
  if (!g_has_done) return -1;
  out = std::move(g_done);
  g_has_done = false;
  out.t[T_TOTAL] = now_ms() - out.t[T_TOTAL];  // from its stage-A start to now (latest possible)
  return out.id;
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_Engine_nativeInit(JNIEnv* e, jclass, jstring jdir,
                                                                                    jstring jlib, jstring jpipe,
                                                                                    jstring jopts) {
  const char* d = e->GetStringUTFChars(jdir, nullptr);
  const char* l = e->GetStringUTFChars(jlib, nullptr);
  const char* p = e->GetStringUTFChars(jpipe, nullptr);
  const char* o = e->GetStringUTFChars(jopts, nullptr);
  std::string err;
  try {
    init(d, l, p, o);
  } catch (const std::exception& ex) {
    err = ex.what();
    if (err.empty()) err = "unknown error";
  }
  e->ReleaseStringUTFChars(jdir, d);
  e->ReleaseStringUTFChars(jlib, l);
  e->ReleaseStringUTFChars(jpipe, p);
  e->ReleaseStringUTFChars(jopts, o);
  return err.empty() ? nullptr : e->NewStringUTF(err.c_str());
}

// Submits frame `id` (bmp: RGBA_8888, at most 1088x800, resized to fit by the caller, top-left
// aligned) and returns the id of a finished frame whose outputs were written into the arrays:
// boxes[4n] (model-input pixels), labels[n], scores[n], masks[n*784] (28x28 probabilities),
// times[T_N] (ms). Without pipelining that is `id` itself. -1: nothing finished yet (pipeline
// filling); -2: error (see nativeLastError). ndet[0] receives n.
void put_result(JNIEnv* e, long got, Done& d, jfloatArray jboxes, jintArray jlabels, jfloatArray jscores,
                jfloatArray jmasks, jfloatArray jtimes, jintArray jndet);

extern "C" JNIEXPORT jlong JNICALL Java_org_onnxsim_maskrcnndemo_Engine_nativeRun(
    JNIEnv* e, jclass, jobject bmp, jlong id, jfloatArray jboxes, jintArray jlabels, jfloatArray jscores,
    jfloatArray jmasks, jfloatArray jtimes, jintArray jndet) {
  AndroidBitmapInfo info;
  void* px = nullptr;
  if (AndroidBitmap_getInfo(e, bmp, &info) || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
      AndroidBitmap_lockPixels(e, bmp, &px)) {
    g_err = "bad bitmap";
    return -2;
  }
  bool released = false;
  Done d;
  long got = -2;
  try {
    got = run_frame(
        [&](Step& S) {
          quant_rgba(S, (const uint8_t*)px, (int)info.width, (int)info.height, (int)info.stride);
          AndroidBitmap_unlockPixels(e, bmp);
          released = true;
        },
        (long)id, d);
  } catch (const std::exception& ex) {
    g_err = ex.what();
    got = -2;
  }
  if (!released) AndroidBitmap_unlockPixels(e, bmp);
  put_result(e, got, d, jboxes, jlabels, jscores, jmasks, jtimes, jndet);
  return got;
}

void put_result(JNIEnv* e, long got, Done& d, jfloatArray jboxes, jintArray jlabels, jfloatArray jscores,
                jfloatArray jmasks, jfloatArray jtimes, jintArray jndet) {
  if (got >= 0) {
    int cap = std::min<int>((int)d.scores.size(), e->GetArrayLength(jscores));
    e->SetFloatArrayRegion(jboxes, 0, 4 * cap, d.boxes.data());
    e->SetIntArrayRegion(jlabels, 0, cap, d.labels.data());
    e->SetFloatArrayRegion(jscores, 0, cap, d.scores.data());
    e->SetFloatArrayRegion(jmasks, 0, 784 * cap, d.masks.data());
    float tf[T_N];
    for (int i = 0; i < T_N; ++i) tf[i] = (float)d.t[i];
    e->SetFloatArrayRegion(jtimes, 0, T_N, tf);
    jint n = cap;
    e->SetIntArrayRegion(jndet, 0, 1, &n);
  }
}

// Camera fast path (see quant_yuv). y/u/v: the YUV_420_888 planes as direct ByteBuffers; disp: an
// RGBA_8888 bitmap of the fitted upright size (nativeFitDims) that receives the displayed frame.
extern "C" JNIEXPORT jlong JNICALL Java_org_onnxsim_maskrcnndemo_Engine_nativeRunYuv(
    JNIEnv* e, jclass, jobject jy, jobject ju, jobject jv, jint ys, jint uvs, jint uvps, jint w, jint h, jint rot,
    jobject disp, jlong id, jfloatArray jboxes, jintArray jlabels, jfloatArray jscores, jfloatArray jmasks,
    jfloatArray jtimes, jintArray jndet) {
  Yuv Y{(const uint8_t*)e->GetDirectBufferAddress(jy), (const uint8_t*)e->GetDirectBufferAddress(ju),
        (const uint8_t*)e->GetDirectBufferAddress(jv), (long)e->GetDirectBufferCapacity(jy),
        (long)e->GetDirectBufferCapacity(ju), (long)e->GetDirectBufferCapacity(jv), ys, uvs, uvps, w, h, rot,
        nullptr, 0};
  AndroidBitmapInfo info;
  void* px = nullptr;
  if (!Y.y || !Y.u || !Y.v || AndroidBitmap_getInfo(e, disp, &info) ||
      info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || AndroidBitmap_lockPixels(e, disp, &px)) {
    g_err = "bad yuv planes or display bitmap";
    return -2;
  }
  int fw, fh;
  fit_dims(w, h, rot, &fw, &fh);
  if ((int)info.width != fw || (int)info.height != fh) {
    AndroidBitmap_unlockPixels(e, disp);
    g_err = "display bitmap size mismatch";
    return -2;
  }
  Y.disp = (uint8_t*)px;
  Y.disp_stride = (int)info.stride;
  bool released = false;
  Done d;
  long got = -2;
  try {
    got = run_frame(
        [&](Step& S) {
          quant_yuv(S, Y);
          AndroidBitmap_unlockPixels(e, disp);
          released = true;
        },
        (long)id, d);
  } catch (const std::exception& ex) {
    g_err = ex.what();
    got = -2;
  }
  if (!released) AndroidBitmap_unlockPixels(e, disp);
  put_result(e, got, d, jboxes, jlabels, jscores, jmasks, jtimes, jndet);
  return got;
}

extern "C" JNIEXPORT void JNICALL Java_org_onnxsim_maskrcnndemo_Engine_nativeFitDims(JNIEnv* e, jclass, jint w, jint h,
                                                                                    jint rot, jintArray out) {
  jint d[2];
  fit_dims(w, h, rot, &d[0], &d[1]);
  e->SetIntArrayRegion(out, 0, 2, d);
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_Engine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}

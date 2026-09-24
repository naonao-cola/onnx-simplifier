// A whole Sparse4D v3 scene on the phone: the split pipeline of s4d_run.cpp frame after frame, with
// the temporal instance bank (model.InstanceBank: top-600 cache, confidence decay, ego-motion anchor
// projection) kept in-process instead of on the host.
//
// The next frame's bb + pre0 don't read the bank, so they can overlap the current frame's decoder
// chain, which alternates between the HTP pieces and the DFA calls on the CDSP. PIPELINE=1 runs them
// freely on a second thread (they then compete with the mids for the HTP); PIPELINE=2 starts bb when
// DFA call 0 starts and pre0 when call 1 starts, the HTP being idle then. The decoder chain itself is
// sequential (each mid needs the previous DFA, the next frame's mid0 needs this frame's bank), so it
// sets the frame rate. Two slots (s0/, s1/) hold a frame's inputs, value maps and DFA inputs.
//
// usage: s4d_scene <split dir> <scene dir> <frames> <passes>
//   scene dir/f<k>: rgb.u8 (6,256,704,3), proj.f32 (6,4,4), proj_n.f32 (6,3,4),
//                   meta.f64 (33: timestamp, T_global 4x4, T_global_inv 4x4)
//   writes f<k>/cls.f32, box.f32, quality.f32 (the last pass); prints per-frame and total times.
//   Each pass restarts the scene from an empty bank; pass 0 is warmup.
// env: as s4d_run (DFA_FLAGS, S4D_MODELS, QNN_PERF, ...), PIPELINE=0 | 1 | 2
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>

#include "s4d_common.h"

static constexpr int N = 900, NT = 600, C = 256, A = 11;
enum { X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ };

struct Bank {  // model.InstanceBank (decay 0.6, max_dt 2, default_dt 0.5)
  bool has = false;
  std::vector<float> feat = std::vector<float>(NT * C), anchor = std::vector<float>(NT * A), conf = std::vector<float>(NT);
  double ts = 0, Tg[16] = {0};
};

// bank.get: project the cached anchors into the current frame -> temp_feat / temp_anchor / dt buffers
static bool bank_get(Bank& b, const double* meta, Buf& temp_feat, Buf& temp_anchor, Buf& dtb) {
  float* dt_out = (float*)dtb.p;
  if (!b.has) { *dt_out = 0.5f; return false; }
  const float dt = (float)(meta[0] - b.ts);
  if (std::fabs(dt) > 2.0f) throw std::runtime_error("gap > max_time_interval (upstream drops the cache)");
  const double* Tinv = meta + 17;
  float T[16];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += Tinv[i * 4 + k] * b.Tg[k * 4 + j];
      T[i * 4 + j] = (float)s;
    }
  float* ta = (float*)temp_anchor.p;
  for (int n = 0; n < NT; ++n) {  // model.anchor_projection
    const float* a = &b.anchor[n * A];
    float* o = ta + n * A;
    const float c[3] = {a[X] + a[VX] * dt, a[Y] + a[VY] * dt, a[Z] + a[VZ] * dt};
    for (int i = 0; i < 3; ++i) {
      o[X + i] = T[i * 4] * c[0] + T[i * 4 + 1] * c[1] + T[i * 4 + 2] * c[2] + T[i * 4 + 3];
      o[VX + i] = T[i * 4] * a[VX] + T[i * 4 + 1] * a[VY] + T[i * 4 + 2] * a[VZ];
    }
    o[W] = a[W]; o[L] = a[L]; o[H] = a[H];
    // upstream writes the projected [cos, sin] into the [sin, cos] slots (its "TODO: Fix bug")
    o[SIN_YAW] = T[0] * a[COS_YAW] + T[1] * a[SIN_YAW];
    o[COS_YAW] = T[4] * a[COS_YAW] + T[5] * a[SIN_YAW];
  }
  b.anchor.assign(ta, ta + NT * A);  // get() replaces the cache with the projection
  memcpy(temp_feat.p, b.feat.data(), temp_feat.bytes);
  *dt_out = dt != 0.0f ? dt : 0.5f;
  return true;
}

// bank.cache: keep the 600 most confident instances of this frame's output
static void bank_cache(Bank& b, const float* cls, const float* feat, const float* anchor, const double* meta) {
  std::vector<float> conf(N);
  for (int n = 0; n < N; ++n) {
    const float m = *std::max_element(cls + n * 10, cls + n * 10 + 10);
    conf[n] = 1.0f / (1.0f + std::exp(-m));
  }
  if (b.has)
    for (int n = 0; n < NT; ++n) conf[n] = std::max(b.conf[n] * 0.6f, conf[n]);
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + NT, idx.end(), [&](int i, int j) { return conf[i] > conf[j]; });
  for (int k = 0; k < NT; ++k) {
    b.conf[k] = conf[idx[k]];
    memcpy(&b.feat[k * C], feat + idx[k] * C, C * sizeof(float));
    memcpy(&b.anchor[k * A], anchor + idx[k] * A, A * sizeof(float));
  }
  b.ts = meta[0];
  memcpy(b.Tg, meta + 1, sizeof b.Tg);
  b.has = true;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s split_dir scene_dir frames passes\n", argv[0]);
    return 2;
  }
  const std::string sdir = argv[1], scene = argv[2];
  const int nf = atoi(argv[3]), passes = atoi(argv[4]);
  // 1: the next frame's front runs freely on a second thread (it competes with the mids for the HTP);
  // 2: bb runs during this frame's DFA call 0 and pre0 during call 1, joined before the next mid
  const int pipe = getenv("PIPELINE") ? atoi(getenv("PIPELINE")) : 0;
  try {
    init_dsp_and_ort();
    std::vector<std::string> names = {"bb", "pre0", "postF", "postT"};
    for (int k = 0; k < 5; ++k)
      for (const char* t : {"F", "T"}) names.push_back("mid" + std::to_string(k) + t);
    std::map<std::string, Piece> pc;
    printf("sessions create_ms %.1f, %s\n", load_pieces(sdir, names, pc),
           pipe == 2 ? "front during the DFA calls" : pipe ? "front on a free thread" : "sequential");
    read_vscales(pc["bb"]);
    Buf& feat0 = buf("feat0", {N, C});
    Buf& anchor0 = buf("anchor0", {N, A});
    read_into(sdir + "/instance_feature.f32", feat0);
    read_into(sdir + "/anchor.f32", anchor0);
    // per-frame inputs, read once (the timing covers the models, not the phone's flash)
    std::vector<std::vector<uint8_t>> rgb(nf);
    std::vector<std::vector<float>> proj(nf), projn(nf);
    std::vector<std::vector<double>> meta(nf);
    for (int f = 0; f < nf; ++f) {
      const std::string d = scene + "/f" + std::to_string(f) + "/";
      auto slurp = [&](const std::string& p, size_t bytes, void* dst) {
        std::ifstream s(d + p, std::ios::binary);
        s.read((char*)dst, (std::streamsize)bytes);
        if (!s) throw std::runtime_error("short/missing " + d + p);
      };
      rgb[f].resize(6 * 256 * 704 * 3); slurp("rgb.u8", rgb[f].size(), rgb[f].data());
      proj[f].resize(96); slurp("proj.f32", 96 * 4, proj[f].data());
      projn[f].resize(72); slurp("proj_n.f32", 72 * 4, projn[f].data());
      meta[f].resize(33); slurp("meta.f64", 33 * 8, meta[f].data());
    }
    // front: frame f's inputs into slot f % 2, then bb (part 0) and pre0 (part 1)
    auto front_part = [&](int f, int part) {
      const std::string s = "s" + std::to_string(f % 2) + "/";
      if (part == 0) {
        memcpy(buf(s + "rgb", {6, 256, 704, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8).p, rgb[f].data(), rgb[f].size());
        memcpy(buf(s + "proj", {6, 4, 4}).p, proj[f].data(), 96 * 4);
        memcpy(buf(s + "proj_n", {6, 3, 4}).p, projn[f].data(), 72 * 4);
        run(pc["bb"], {{"rgb", s + "rgb"}}, s);
      } else {
        run(pc["pre0"], {{"proj", s + "proj"}, {"proj_n", s + "proj_n"}}, s);
      }
    };
    auto front = [&](int f) { front_part(f, 0); front_part(f, 1); };
    Buf& temp_feat = buf("temp_feat", {NT, C});
    Buf& temp_anchor = buf("temp_anchor", {NT, A});
    Buf& dtb = buf("dt", {});
    std::vector<double> frame_ms, pass_ms;
    std::vector<double> dfa_dsp(nf);
    for (int pass = 0; pass < passes; ++pass) {
      Bank bank;
      double p0 = now_ms();
      std::thread next;
      if (pipe != 1) front(0);
      else next = std::thread(front, 0);
      for (int f = 0; f < nf; ++f) {
        double t0 = now_ms();
        if (next.joinable()) next.join();
        if (pipe == 1 && f + 1 < nf) next = std::thread(front, f + 1);
        const std::string s = "s" + std::to_string(f % 2) + "/";
        const bool temporal = bank_get(bank, meta[f].data(), temp_feat, temp_anchor, dtb);
        const char* T = temporal ? "T" : "F";
        std::string src = s + "pre0", feat = "feat0", anchor = "anchor0";
        double dsp = 0, dfa_wall = 0, htp_wall = 0, join_wait = 0;
        for (int i = 0; i < 6; ++i) {
          const std::string agg = s + "agg" + std::to_string(i);
          if (pipe == 2 && i < 2 && f + 1 < nf) next = std::thread(front_part, f + 1, i);  // HTP idle during the DFA
          double a0 = now_ms();
          dfa(src, agg, s);
          dsp += last_dsp_us / 1000.0;
          dfa_wall += now_ms() - a0;
          a0 = now_ms();
          if (next.joinable() && pipe == 2) next.join();
          join_wait += now_ms() - a0;
          a0 = now_ms();
          const std::string nxt = i < 5 ? "mid" + std::to_string(i) + T : std::string("post") + T;
          std::map<std::string, std::string> al{{"agg", agg}, {"feat", feat}, {"anchor", anchor},
                                                {"proj", s + "proj"}, {"proj_n", s + "proj_n"}};
          run(pc[nxt], al, s);
          htp_wall += now_ms() - a0;
          feat = s + nxt + "/feat_out";
          anchor = s + nxt + "/anchor_out";
          src = s + nxt;
        }
        const std::string post = s + "post" + T;
        bank_cache(bank, (float*)get(post + "/cls").p, (float*)get(post + "/feat_out").p,
                   (float*)get(post + "/anchor_out").p, meta[f].data());
        if (!pipe && f + 1 < nf) front(f + 1);  // sequential: the next front counts in this frame
        const double ms = now_ms() - t0;
        if (pass > 0) frame_ms.push_back(ms);
        dfa_dsp[f] = dsp;
        if (pass == passes - 1 && f == nf / 2)
          printf("frame %d: dfa calls %.2f ms (in-DSP %.2f), mid/post pieces %.2f ms, front joins %.2f ms\n", f, dfa_wall,
                 dsp, htp_wall, join_wait);
        if (pass == passes - 1) {
          const std::string d = scene + "/f" + std::to_string(f) + "/";
          write_from(d + "cls.f32", get(post + "/cls"));
          write_from(d + "box.f32", get(post + "/anchor_out"));
          write_from(d + "quality.f32", get(post + "/quality"));
        }
      }
      if (next.joinable()) next.join();
      if (pass > 0) pass_ms.push_back(now_ms() - p0);
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    printf("frame_ms median %.2f (n=%zu), scene_ms median %.2f for %d frames -> %.2f ms/frame\n", med(frame_ms),
           frame_ms.size(), med(pass_ms), nf, med(pass_ms) / nf);
    double d = 0;
    for (double x : dfa_dsp) d += x;
    printf("dfa in-DSP per frame (last pass) %.2f ms\n", d / nf);
    dfa_rpc_close(h_dfa);
    printf("PASS\n");
    fflush(stdout);
    _exit(0);
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    fflush(stdout);
    return 1;
  }
}

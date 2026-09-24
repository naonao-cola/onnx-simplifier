// Game-upscaling mode of the demo app: replays a short stretch of Arm's NSS test sequence (Bistro,
// 960x540 renders + depth + motion + camera, game_seq.py) through
//   NSS  (../vision_models/nss): 540p -> 1080p temporal super sampling, the CNN on the HTP, the
//        pre/post-processing as OpenCL kernels on the Adreno (nss_kernels.cl), and optionally
//   NFRU (../vision_models/nfru): one generated frame between each two NSS outputs, the network on the
//        HTP, flow / splats / blend as OpenCL kernels (nfru_kernels.cl).
// The same kernels and uint8 NHWC int8 networks as nss_run.cpp / nfru_run.cpp; everything stays in GPU
// buffers/images, the networks read/write host-mapped buffers shared with ORT, and only the displayed
// RGBA8 frames are copied out (into the Java bitmaps).
// NFRU takes NSS's linear output through its own colour pipeline (exposure exp(2), reinhard -- the
// same tonemap NSS displays with), motion = -NSS motion (NFRU's mv point back, NSS's forward; checked
// against the camera: README), and the forward motion hint of m1 = NSS motion of p1 (an approximation:
// the NSS sequence has no forward vectors). Built into libgame_demo.so (its own process).
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <fcntl.h>
#include <onnxruntime_cxx_api.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cl_dl.h"
#include "game_cl.h"  // NSS_CL, NFRU_CL: the kernel sources (build_app.sh)
#include "htp_session.h"
#include "nss_lut.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "GameDemo", __VA_ARGS__)

namespace {
using demo::now_ms;

// NSS geometry ("high": 960x540 -> 1920x1080; the CNN at 544 x 960)
constexpr int H = 540, W = 960, Hp = 544, Wp = 960, Ho = 1080, Wo = 1920, Hd = 270, Wd = 480;
constexpr int Hk = 136, Wk = 240, Kc = 36, Ht = 544, Wt = 960;
constexpr float EXPO = 7.389056205749512f;  // exp(2): both models' test colour pipelines
constexpr float NFRU_PSC = 0.35356706380844116f, NFRU_PZP = 172.0f, T = 0.5f;

const char* UNPACK_CL = R"CL(
// the sequence's float16 planar (row, col) motion -> NSS's float2 buffer + a planar float copy for NFRU
__kernel void unpack_motion(__global const half* m, int n, __global float2* nss, __global float* raw) {
  int i = get_global_id(0);
  if (i >= n) return;
  float a = vload_half(i, m), b = vload_half(n + i, m);
  nss[i] = (float2)(a, b);
  raw[i] = a;
  raw[n + i] = b;
}
)CL";

struct Arg {
  size_t size;
  const void* p;
};
template <typename T>
Arg A(const T& v) {
  return Arg{sizeof(T), &v};
}

struct Seq {  // game_seq.bin, memory-mapped
  const uint8_t* base = nullptr;
  size_t size = 0;
  int frames = 0;
  static constexpr size_t kScal = 32 * 4, kCol = (size_t)H * W * 8, kMot = (size_t)H * W * 4, kDep = (size_t)H * W * 4;
  static constexpr size_t kFrame = kScal + kCol + kMot + kDep;
  const float* scal(int t) const { return (const float*)(base + 16 + kFrame * t); }
  const uint8_t* col(int t) const { return base + 16 + kFrame * t + kScal; }
  const uint8_t* mot(int t) const { return col(t) + kCol; }
  const uint8_t* dep(int t) const { return mot(t) + kMot; }
};

struct Lv {
  int h, w, hd, wd;
};

struct Engine {
  demo::Htp htp;
  std::unique_ptr<Ort::Session> nss_cnn, nfru_net;
  Seq seq;
  cl_context ctx = nullptr;
  cl_command_queue q = nullptr;
  cl_program p_nss = nullptr, p_nfru = nullptr, p_game = nullptr;
  std::map<std::string, cl_kernel> kern;
  cl_mem nullmem = nullptr;
  // NSS state
  cl_mem b_color, b_m16, b_motion, b_recon, b_lut, b_hist[2], b_deriv[2], b_in_u8, b_code, b_kpn, b_tmp, i_tmp,
      b_rgba, b_lr;
  // NFRU, per rendered frame (two slots)
  std::vector<Lv> lv;
  int fH = 0, fW = 0;
  struct Frame {
    cl_mem rgb, pyr[6], depth, sy, mv;
    float vp[16];
  } fr[2];
  struct BmLv {
    cl_mem vp, sw, hint, vec, won, med, out;
  } bl[4];
  cl_mem m_mm0, m_mm1, m_mm3, b_hint, b_flow, b_dyn, b_packed, b_holes_t, b_holes_m1, b_pflow, b_mvt, b_flowt, b_net,
      b_par, b_gen;
  float vp_m3[16];
  int last_t = -2;
  std::string err;

  cl_kernel K(cl_program p, const char* n) {
    std::string key = std::to_string((uintptr_t)p) + n;
    auto it = kern.find(key);
    if (it != kern.end()) return it->second;
    cl_int e;
    cl_kernel k = p_clCreateKernel(p, n, &e);
    CK(e);
    return kern[key] = k;
  }
  cl_mem B(size_t bytes, cl_mem_flags fl = CL_MEM_READ_WRITE) {
    cl_int e;
    cl_mem m = p_clCreateBuffer(ctx, fl, std::max<size_t>(bytes, 4), nullptr, &e);
    CK(e);
    return m;
  }
  cl_mem IMG(int w, int h, cl_channel_type type = CL_FLOAT) {
    cl_image_format f{CL_RGBA, type};
    cl_image_desc d{};
    d.image_type = CL_MEM_OBJECT_IMAGE2D;
    d.image_width = w;
    d.image_height = h;
    cl_int e;
    cl_mem m = p_clCreateImage(ctx, CL_MEM_READ_WRITE, &f, &d, nullptr, &e);
    CK(e);
    return m;
  }
  cl_program build(const char* src, const char* name) {
    cl_int e;
    size_t sl = strlen(src);
    cl_program p = p_clCreateProgramWithSource(ctx, 1, &src, &sl, &e);
    CK(e);
    cl_device_id dev;
    cl_platform_id plat;
    CK(p_clGetPlatformIDs(1, &plat, nullptr));
    CK(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    if (p_clBuildProgram(p, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
      std::vector<char> log(1 << 16);
      p_clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
      throw std::runtime_error(std::string("build ") + name + ": " + log.data());
    }
    return p;
  }
  void run(cl_program p, const char* name, std::vector<Arg> args, size_t gx, size_t gy = 0, const size_t* lws = nullptr) {
    cl_kernel k = K(p, name);
    for (cl_uint i = 0; i < args.size(); i++) CK(p_clSetKernelArg(k, i, args[i].size, args[i].p));
    size_t g[2] = {gx, gy};
    if (lws) g[0] = (gx + lws[0] - 1) / lws[0] * lws[0], g[1] = (gy + lws[1] - 1) / lws[1] * lws[1];
    CK(p_clEnqueueNDRangeKernel(q, k, gy ? 2 : 1, nullptr, g, lws, 0, nullptr, nullptr));
  }

  void init(const std::string& dir, const std::string& lib_dir, const std::string& perf) {
    // the sequence
    int fd = open((dir + "/game_seq.bin").c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("no " + dir + "/game_seq.bin (deploy.sh GAME=...)");
    struct stat st;
    fstat(fd, &st);
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("mmap game_seq.bin");
    seq.base = (const uint8_t*)m;
    seq.size = st.st_size;
    const int32_t* hd = (const int32_t*)m;
    if (hd[0] != 0x5153534E || hd[2] != H || hd[3] != W) throw std::runtime_error("game_seq.bin: bad header");
    seq.frames = hd[1];
    if (seq.size < 16 + Seq::kFrame * seq.frames) throw std::runtime_error("game_seq.bin: truncated");
    // networks on the HTP
    htp.init(lib_dir, "GameDemo");
    nss_cnn = htp.session(dir, "game_nss_cnn", perf, "GameDemo");
    nfru_net = htp.session(dir, "game_nfru_net", perf, "GameDemo");
    // OpenCL (high GPU clocks hint, cl_qcom_perf_hint)
    load_cl();
    cl_platform_id plat;
    CK(p_clGetPlatformIDs(1, &plat, nullptr));
    cl_device_id dev;
    CK(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0x40C2, 0x40C3, 0};
    cl_int e;
    ctx = p_clCreateContext(props, 1, &dev, nullptr, nullptr, &e);
    CK(e);
    q = p_clCreateCommandQueue(ctx, dev, 0, &e);
    CK(e);
    double t0 = now_ms();
    p_nss = build(NSS_CL, "nss_kernels.cl");
    p_nfru = build(NFRU_CL, "nfru_kernels.cl");
    p_game = build(UNPACK_CL, "unpack");
    LOGI("cl build %.0f ms", now_ms() - t0);
    const cl_mem_flags mapped = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
    // NSS
    b_color = IMG(W, H, CL_HALF_FLOAT);
    b_m16 = B((size_t)H * W * 4);
    b_motion = B((size_t)H * W * 8);
    b_recon = B(Hd * Wd * 4);
    b_lut = B(6 * 64 * 4 * 4, CL_MEM_READ_ONLY);
    b_hist[0] = IMG(Wo, Ho), b_hist[1] = IMG(Wo, Ho), b_deriv[0] = IMG(W, H), b_deriv[1] = IMG(W, H);
    b_in_u8 = B((size_t)Hp * Wp * 12, mapped), b_code = B(H * W), b_kpn = B((size_t)Hk * Wk * Kc, mapped);
    b_tmp = B((size_t)Ht * Wt * 4, mapped), b_rgba = B((size_t)Ho * Wo * 4, mapped), b_lr = B((size_t)H * W * 4, mapped);
    {
      cl_image_format f{CL_RGBA, CL_UNSIGNED_INT8};
      cl_image_desc d{};
      d.image_type = CL_MEM_OBJECT_IMAGE2D;
      d.image_width = Wt;
      d.image_height = Ht;
      d.image_row_pitch = (size_t)Wt * 4;
      d.buffer = b_tmp;
      i_tmp = p_clCreateImage(ctx, CL_MEM_READ_ONLY, &f, &d, nullptr, &e);
      CK(e);
    }
    // NFRU
    lv = {{Ho, Wo, Ho, Wo}};
    for (int i = 1; i < 6; i++) {
      int hd2 = lv.back().h / 2, wd2 = lv.back().w / 2;
      lv.push_back({hd2 + hd2 % 2, wd2 + wd2 % 2, hd2, wd2});
    }
    fH = lv[2].h, fW = lv[2].w;
    for (auto& f : fr) {
      f.rgb = IMG(Wo, Ho);
      for (int i = 0; i < 6; i++) f.pyr[i] = B((size_t)lv[i].h * lv[i].w);
      f.depth = B(H * W * 4), f.sy = B(2 * H * W * 4), f.mv = B(2 * H * W * 4);
    }
    for (int l = 0; l < 4; l++) {
      const Lv& L = lv[5 - l];
      bl[l] = {B(2 * L.h * L.w * 4), B(L.h * L.w), B(L.h * L.w), B(2 * L.h * L.w * 4),
               B(L.h * L.w),         B(2 * L.h * L.w * 4), B(2 * L.hd * L.wd * 4)};
    }
    m_mm0 = B(64, CL_MEM_READ_ONLY), m_mm1 = B(64, CL_MEM_READ_ONLY), m_mm3 = B(64, CL_MEM_READ_ONLY);
    b_hint = B(2 * fH * fW * 4), b_flow = B(2 * fH * fW * 4), b_dyn = B(H * W * 4);
    b_packed = B(H * W * 4), b_holes_t = B(H * W * 4), b_holes_m1 = B(H * W * 4), b_pflow = B(fH * fW * 4);
    b_mvt = B(2 * H * W * 4), b_flowt = B(2 * fH * fW * 4);
    b_net = B(fH * fW * 16, mapped), b_par = B(fH * fW * 4, mapped), b_gen = B((size_t)Ho * Wo * 4, mapped);
  }

  void zero_state() {  // the start of a sequence: NSS's zero history / derivative / feedback
    std::vector<char> z((size_t)Ho * Wo * 16, 0);
    size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)Wo, (size_t)Ho, 1}, rg2[3] = {(size_t)W, (size_t)H, 1};
    CK(p_clEnqueueWriteImage(q, b_hist[0], CL_TRUE, org, reg, 0, 0, z.data(), 0, nullptr, nullptr));
    CK(p_clEnqueueWriteImage(q, b_deriv[0], CL_TRUE, org, rg2, 0, 0, z.data(), 0, nullptr, nullptr));
    CK(p_clEnqueueWriteBuffer(q, b_tmp, CL_TRUE, 0, (size_t)Ht * Wt * 4, z.data(), 0, nullptr, nullptr));
  }

  // the renderer's frame t -> GPU (colour image, motion, depth): what a game would already have there
  void upload(int t, Frame& f) {
    size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)W, (size_t)H, 1};
    CK(p_clEnqueueWriteImage(q, b_color, CL_FALSE, org, reg, 0, 0, seq.col(t), 0, nullptr, nullptr));
    CK(p_clEnqueueWriteBuffer(q, b_m16, CL_FALSE, 0, Seq::kMot, seq.mot(t), 0, nullptr, nullptr));
    CK(p_clEnqueueWriteBuffer(q, f.depth, CL_FALSE, 0, Seq::kDep, seq.dep(t), 0, nullptr, nullptr));
    memcpy(f.vp, seq.scal(t) + 6, 64);
    CK(p_clFinish(q));
  }

  double nss_htp_ms = 0, nfru_htp_ms = 0;

  // NSS for frame t (slot f): the output's linear image is b_hist[nxt]
  cl_mem nss(int t, Frame& f) {
    const float* s = seq.scal(t);
    float jy = s[0], jx = s[1], rs0 = H, rs1 = W, e = EXPO, reset = t == 0 ? 0.0f : 1.0f;
    cl_float4 dtv = {{s[2], s[3], s[4], s[5]}};
    NssLut L = nss_offset_lut(H, W, Ho, Wo, jy, jx, 9);
    CK(p_clEnqueueWriteBuffer(q, b_lut, CL_FALSE, 0, L.lut.size() * 4, L.lut.data(), 0, nullptr, nullptr));
    int cur = t & 1, nxt = cur ^ 1, n = H * W;
    run(p_game, "unpack_motion", {A(b_m16), A(n), A(b_motion), A(f.sy)}, n);
    int gi = Hd * Wd;
    run(p_nss, "depth_scatter_init", {A(b_recon)}, gi);
    run(p_nss, "depth_scatter", {A(b_motion), A(f.depth), A(H), A(W), A(b_recon), A(Hd), A(Wd)}, Wd, Hd);
    const size_t lws[2] = {32, 8};
    run(p_nss, "preprocess",
        {A(b_color), A(b_hist[cur]), A(b_motion), A(f.depth), A(i_tmp), A(b_deriv[cur]), A(b_recon), A(H), A(W),
         A(Hp), A(Wp), A(Ho), A(Wo), A(Hd), A(Wd), A(jy), A(jx), A(e), A(rs0), A(rs1), A(dtv), A(nullmem),
         A(b_in_u8), A(b_deriv[nxt]), A(nullmem), A(b_code)},
        Wp, Hp, lws);
    run(p_nfru, "tonemap8", {A(b_color), A(e), A(H), A(W), A(b_lr)}, W, H);  // the native render, for display
    CK(p_clFinish(q));
    cl_int er;
    void* in_p = p_clEnqueueMapBuffer(q, b_in_u8, CL_TRUE, CL_MAP_READ, 0, (size_t)Hp * Wp * 12, 0, nullptr, nullptr, &er);
    CK(er);
    void* kpn_p = p_clEnqueueMapBuffer(q, b_kpn, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, (size_t)Hk * Wk * Kc, 0,
                                       nullptr, nullptr, &er);
    CK(er);
    void* tmp_p = p_clEnqueueMapBuffer(q, b_tmp, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, (size_t)Ht * Wt * 4, 0,
                                       nullptr, nullptr, &er);
    CK(er);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t in_shape[] = {1, Hp, Wp, 12}, kpn_shape[] = {1, Hk, Wk, Kc}, tmp_shape[] = {1, Ht, Wt, 4};
    Ort::Value x = Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)in_p, (size_t)Hp * Wp * 12, in_shape, 4);
    Ort::Value ys[2] = {Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)kpn_p, (size_t)Hk * Wk * Kc, kpn_shape, 4),
                        Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)tmp_p, (size_t)Ht * Wt * 4, tmp_shape, 4)};
    const char* in_names[] = {"x"};
    const char* out_names[] = {"kpn", "temporal"};
    double h0 = now_ms();
    nss_cnn->Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names, ys, 2);
    nss_htp_ms = now_ms() - h0;
    CK(p_clEnqueueUnmapMemObject(q, b_in_u8, in_p, 0, nullptr, nullptr));
    CK(p_clEnqueueUnmapMemObject(q, b_kpn, kpn_p, 0, nullptr, nullptr));
    CK(p_clEnqueueUnmapMemObject(q, b_tmp, tmp_p, 0, nullptr, nullptr));
    int mh = L.mod_h, mw = L.mod_w, taps = L.taps;
    run(p_nss, "postprocess",
        {A(b_color), A(b_hist[cur]), A(b_motion), A(b_code), A(b_kpn), A(i_tmp), A(b_lut), A(H), A(W), A(Ho), A(Wo),
         A(Hk), A(Wk), A(Kc), A(Ht), A(Wt), A(mh), A(mw), A(taps), A(e), A(reset), A(b_hist[nxt]), A(b_rgba)},
        Wo, Ho, lws);
    return b_hist[nxt];
  }

  // NFRU's per-frame work on an NSS output: colour + luma pyramid + normalized motion (-NSS motion)
  void nfru_frame(cl_mem lin, Frame& f) {
    float e = EXPO;
    run(p_nfru, "colour_luma_img", {A(lin), A(e), A(Ho), A(Wo), A(f.rgb), A(f.pyr[0])}, Wo, Ho);
    for (int i = 1; i < 6; i++) {
      int blur = (i - 1) >= 1 && (i - 1) <= 4, quad = i == 5;
      run(p_nfru, "pyr_down",
          {A(f.pyr[i - 1]), A(lv[i - 1].h), A(lv[i - 1].w), A(blur), A(quad), A(f.pyr[i]), A(lv[i].h), A(lv[i].w),
           A(lv[i].hd), A(lv[i].wd)},
          lv[i].w, lv[i].h);
    }
    int nm = H * W;
    float m1 = -1.0f, fh = (float)H, fw = (float)W;
    run(p_nfru, "norm_mv", {A(f.sy), A(nm), A(m1), A(fh), A(fw), A(f.mv)}, nm);
  }

  static void matmul_inv(const float* a, const float* b, float* out) {  // out = a @ inv(b), row-major 4x4
    double m[4][8];
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 8; j++) m[i][j] = j < 4 ? b[i * 4 + j] : (j - 4 == i);
    for (int c = 0; c < 4; c++) {
      int p = c;
      for (int r = c + 1; r < 4; r++)
        if (std::fabs(m[r][c]) > std::fabs(m[p][c])) p = r;
      for (int j = 0; j < 8; j++) std::swap(m[c][j], m[p][j]);
      double d = m[c][c];
      for (int j = 0; j < 8; j++) m[c][j] /= d;
      for (int r = 0; r < 4; r++)
        if (r != c) {
          double f = m[r][c];
          for (int j = 0; j < 8; j++) m[r][j] -= f * m[c][j];
        }
    }
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) {
        double s = 0;
        for (int k = 0; k < 4; k++) s += a[i * 4 + k] * m[k][4 + j];
        out[i * 4 + j] = (float)s;
      }
  }

  // NFRU: the frame half way between m1 and p1 (both already through nfru_frame) -> b_gen (RGBA8)
  void nfru(Frame& m1, Frame& p1, const float* dp, unsigned seed) {
    float mm0[16], mm1[16], mm3[16];
    matmul_inv(m1.vp, p1.vp, mm0);  // MotionMat[0] = VP_m1 @ inv(VP_p1)
    matmul_inv(p1.vp, m1.vp, mm1);  // MotionMat[1] = VP_p1 @ inv(VP_m1)
    matmul_inv(vp_m3, m1.vp, mm3);  // VP_m3 @ inv(VP_m1)
    CK(p_clEnqueueWriteBuffer(q, m_mm0, CL_FALSE, 0, 64, mm0, 0, nullptr, nullptr));
    CK(p_clEnqueueWriteBuffer(q, m_mm1, CL_FALSE, 0, 64, mm1, 0, nullptr, nullptr));
    CK(p_clEnqueueWriteBuffer(q, m_mm3, CL_FALSE, 0, 64, mm3, 0, nullptr, nullptr));
    // the motion hint of m1 towards p1: NSS motion of p1 (f.sy holds the raw NSS motion)
    run(p_nfru, "hint_mv", {A(p1.sy), A(m1.depth), A(H), A(W), A(b_hint), A(fH), A(fW)}, fW, fH);
    const size_t lws[2] = {16, 16};
    for (int l = 0; l < 4; l++) {
      const Lv& L = lv[5 - l];
      BmLv& b = bl[l];
      cl_mem srch = p1.pyr[5 - l], tmpl = m1.pyr[5 - l], vp = nullmem, sw = srch, hint = nullmem, hmv = nullmem,
             won = nullmem;
      if (l) {
        const Lv& P = lv[6 - l];
        run(p_nfru, "bm_upsample", {A(bl[l - 1].out), A(P.hd), A(P.wd), A(b.vp), A(L.h), A(L.w)}, L.w, L.h);
        run(p_nfru, "bm_warp", {A(srch), A(L.h), A(L.w), A(b.vp), A(L.h), A(L.w), A(b.sw)}, L.w, L.h);
        vp = b.vp, sw = b.sw;
      }
      if (l == 3) {
        run(p_nfru, "bm_warp", {A(srch), A(L.h), A(L.w), A(b_hint), A(L.h), A(L.w), A(b.hint)}, L.w, L.h);
        hint = b.hint, hmv = b_hint, won = b.won;
      }
      run(p_nfru, "bm_match", {A(sw), A(tmpl), A(hint), A(L.h), A(L.w), A(vp), A(b.vec), A(b.won)}, L.w, L.h, lws);
      run(p_nfru, "bm_median", {A(b.vec), A(L.h), A(L.w), A(b.med)}, L.w, L.h);
      run(p_nfru, "bm_jbf",
          {A(b.med), A(tmpl), A(L.h), A(L.w), A(won), A(hmv), A(L.h), A(L.w), A(b.out), A(L.hd), A(L.wd)}, L.wd, L.hd);
    }
    int nf = fH * fW, nm = H * W;
    float m4 = -4.0f, ffh = (float)fH, ffw = (float)fW, t = T, t1 = 1.0f - T;
    run(p_nfru, "norm_mv", {A(bl[3].out), A(nf), A(m4), A(ffh), A(ffw), A(b_flow)}, nf);
    run(p_nfru, "dyn_mask", {A(m1.depth), A(m1.mv), A(m_mm3), A(H), A(W), A(b_dyn)}, W, H);
    for (cl_mem z : {b_packed, b_holes_t, b_holes_m1}) run(p_nfru, "zero_i32", {A(z)}, nm);
    run(p_nfru, "zero_i32", {A(b_pflow)}, nf);
    run(p_nfru, "warp_mv",
        {A(p1.depth), A(m1.depth), A(p1.mv), A(b_dyn), A(m_mm1), A(H), A(W), A(t), A(b_packed), A(b_holes_t),
         A(b_holes_m1)},
        W, H);
    run(p_nfru, "fill_mv", {A(b_packed), A(H), A(W), A(b_mvt)}, W, H);
    run(p_nfru, "warp_flow", {A(m1.depth), A(H), A(W), A(b_flow), A(fH), A(fW), A(t1), A(b_pflow)}, fW, fH);
    run(p_nfru, "fill_mv", {A(b_pflow), A(fH), A(fW), A(b_flowt)}, fW, fH);
    cl_float4 dpv = {{dp[0], dp[1], dp[2], dp[3]}};
    run(p_nfru, "preprocess",
        {A(b_flowt), A(b_mvt), A(H), A(W), A(m1.rgb), A(p1.rgb), A(Ho), A(Wo), A(m1.depth), A(p1.depth),
         A(b_holes_t), A(b_holes_m1), A(m_mm1), A(m_mm0), A(dpv), A(t), A(seed), A(fH), A(fW), A(nullmem), A(b_net)},
        fW, fH);
    CK(p_clFinish(q));
    cl_int er;
    void* in_p = p_clEnqueueMapBuffer(q, b_net, CL_TRUE, CL_MAP_READ, 0, nf * 16, 0, nullptr, nullptr, &er);
    CK(er);
    void* out_p = p_clEnqueueMapBuffer(q, b_par, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, nf * 4, 0, nullptr,
                                       nullptr, &er);
    CK(er);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t in_shape[] = {1, fH, fW, 16}, out_shape[] = {1, fH, fW, 4};
    Ort::Value x = Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)in_p, nf * 16, in_shape, 4);
    Ort::Value y = Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)out_p, nf * 4, out_shape, 4);
    const char* in_names[] = {"x"};
    const char* out_names[] = {"params"};
    double h0 = now_ms();
    nfru_net->Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names, &y, 1);
    nfru_htp_ms = now_ms() - h0;
    CK(p_clEnqueueUnmapMemObject(q, b_net, in_p, 0, nullptr, nullptr));
    CK(p_clEnqueueUnmapMemObject(q, b_par, out_p, 0, nullptr, nullptr));
    float psc = NFRU_PSC, pzp = NFRU_PZP;
    run(p_nfru, "postprocess",
        {A(b_flowt), A(fH), A(fW), A(b_mvt), A(H), A(W), A(b_par), A(fH), A(fW), A(psc), A(pzp), A(m1.rgb), A(p1.rgb),
         A(Ho), A(Wo), A(t), A(nullmem), A(b_gen)},
        Wo, Ho);
  }
};

Engine* g = nullptr;
std::mutex g_mu;
std::string g_err;

struct Px {  // locked RGBA bitmap (null if none passed or the size is wrong)
  JNIEnv* e;
  jobject b;
  uint8_t* p = nullptr;
  int stride = 0;
  Px(JNIEnv* env, jobject bmp, int w, int h) : e(env), b(bmp) {
    AndroidBitmapInfo bi;
    if (!bmp || AndroidBitmap_getInfo(e, bmp, &bi) || (int)bi.width != w || (int)bi.height != h ||
        bi.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || AndroidBitmap_lockPixels(e, bmp, (void**)&p))
      p = nullptr;
    else
      stride = (int)bi.stride;
  }
  ~Px() {
    if (p) AndroidBitmap_unlockPixels(e, b);
  }
};

void copy_out(cl_command_queue q, cl_mem m, int w, int h, Px& d) {  // mapped RGBA8 buffer -> bitmap
  if (!d.p) return;
  cl_int er;
  void* p = p_clEnqueueMapBuffer(q, m, CL_TRUE, CL_MAP_READ, 0, (size_t)w * h * 4, 0, nullptr, nullptr, &er);
  CK(er);
  for (int y = 0; y < h; y++) memcpy(d.p + (size_t)y * d.stride, (const uint8_t*)p + (size_t)y * w * 4, (size_t)w * 4);
  CK(p_clEnqueueUnmapMemObject(q, m, p, 0, nullptr, nullptr));
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_GameEngine_nativeInit(JNIEnv* e, jclass, jstring jdir,
                                                                                        jstring jlib, jstring jopts) {
  std::lock_guard<std::mutex> lk(g_mu);
  const char* d = e->GetStringUTFChars(jdir, nullptr);
  const char* l = e->GetStringUTFChars(jlib, nullptr);
  const char* o = e->GetStringUTFChars(jopts, nullptr);
  std::string dir = d, lib = l, opts = o;
  e->ReleaseStringUTFChars(jdir, d);
  e->ReleaseStringUTFChars(jlib, l);
  e->ReleaseStringUTFChars(jopts, o);
  try {
    auto kv = demo::parse_opts(opts, {{"htp_performance_mode", "burst"}});
    if (!g) {
      g = new Engine();
      g->init(dir, lib, kv["htp_performance_mode"]);
    }
    return nullptr;
  } catch (const std::exception& ex) {
    delete g;
    g = nullptr;
    return e->NewStringUTF(ex.what());
  }
}

extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_GameEngine_nativeFrames(JNIEnv*, jclass) {
  return g ? g->seq.frames : 0;
}

// Frame t of the replay: NSS (-> nss, and the native render -> lr) and, if nfru and t > 0, the frame
// generated between t - 1 and t (-> gen). times: see GameEngine.T_*. Returns 1 if gen was produced,
// 0 if not, -1 on error.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_GameEngine_nativeStep(JNIEnv* e, jclass, jint t,
                                                                                     jboolean nfru, jobject lrBmp,
                                                                                     jobject nssBmp, jobject genBmp,
                                                                                     jfloatArray jtimes) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!g) {
    g_err = "not initialized";
    return -1;
  }
  float tm[6] = {0};
  int made = 0;
  try {
    Engine& E = *g;
    Engine::Frame &cur = E.fr[t & 1], &prev = E.fr[(t + 1) & 1];
    if (t == 0) E.zero_state();
    double u0 = now_ms();
    E.upload(t, cur);
    tm[3] = (float)(now_ms() - u0);
    double n0 = now_ms();
    cl_mem lin = E.nss(t, cur);
    CK(p_clFinish(E.q));
    tm[0] = (float)(now_ms() - n0);
    tm[1] = (float)E.nss_htp_ms;
    // NFRU needs the previous NSS output through nfru_frame; run it for every frame while NFRU is on
    const bool have_prev = E.last_t == t - 1;
    if (nfru) {
      double f0 = now_ms();
      E.nfru_frame(lin, cur);
      if (have_prev) {
        E.nfru(prev, cur, E.seq.scal(t) + 2, 12345u + t);
        made = 1;
      }
      CK(p_clFinish(E.q));
      tm[2] = (float)(now_ms() - f0);
      tm[4] = (float)E.nfru_htp_ms;
      E.last_t = t;
    } else {
      E.last_t = -2;
    }
    memcpy(E.vp_m3, have_prev ? prev.vp : cur.vp, 64);  // the frame before m1, for the next window
    double c0 = now_ms();
    Px lr(e, lrBmp, W, H), ns(e, nssBmp, Wo, Ho), gn(e, genBmp, Wo, Ho);
    copy_out(E.q, E.b_lr, W, H, lr);
    copy_out(E.q, E.b_rgba, Wo, Ho, ns);
    if (made) copy_out(E.q, E.b_gen, Wo, Ho, gn);
    tm[5] = (float)(now_ms() - c0);
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
  jsize n = e->GetArrayLength(jtimes);
  e->SetFloatArrayRegion(jtimes, 0, std::min<jsize>(n, 6), tm);
  return made;
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_GameEngine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}

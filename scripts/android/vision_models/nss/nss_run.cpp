// NSS v1 ("high") on the phone: pre/post-processing as OpenCL kernels on the Adreno GPU
// (nss_kernels.cl), the CNN on the HTP (ORT + QNN EP, uint8 NHWC I/O). One process, closed loop: the
// kernels' output is the next frame's history, the CNN's temporal output the next frame's feedback.
//
//   nss_run <dir> <cnn.onnx> <cnn_ctx.onnx> <frames> [iters]
//
// <dir> holds nss_kernels.cl and per-frame inputs from `nss_gpu.py phone` (frameNNN.bin: colour, motion,
// depth float32 planar; frameNNN.txt: jitter y x, exposure, render size y x, depth params x4, reset,
// LUT modulo h w, taps, then the LUT floats). Writes outNNN.bin (the tonemapped RGBA8 output) and prints
// per-stage GPU times (OpenCL profiling events) and HTP / wall times per frame.
#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <dlfcn.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// OpenCL entry points, resolved from the vendor's libOpenCL.so (no link-time dependency)
#define CLFN(name) static decltype(&::name) p_##name;
#define CLFNS(X)                                                                                            \
  X(clGetPlatformIDs) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext) X(clCreateCommandQueue)       \
  X(clCreateBuffer) X(clCreateProgramWithSource) X(clBuildProgram) X(clGetProgramBuildInfo)                 \
  X(clCreateKernel) X(clSetKernelArg) X(clEnqueueNDRangeKernel) X(clEnqueueWriteBuffer)                     \
  X(clEnqueueReadBuffer) X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) X(clFinish)                       \
  X(clGetEventProfilingInfo) X(clReleaseEvent) X(clWaitForEvents)
CLFNS(CLFN)

static void load_cl() {
  const char* paths[] = {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so"};
  void* h = nullptr;
  for (auto p : paths)
    if ((h = dlopen(p, RTLD_NOW | RTLD_LOCAL))) break;
  if (!h) throw std::runtime_error(std::string("dlopen libOpenCL.so: ") + dlerror());
#define CLLOAD(name)                                                    \
  p_##name = reinterpret_cast<decltype(p_##name)>(dlsym(h, #name));     \
  if (!p_##name) throw std::runtime_error("dlsym " #name);
  CLFNS(CLLOAD)
}

#define CK(x)                                                                                 \
  do {                                                                                        \
    cl_int e_ = (x);                                                                          \
    if (e_ != CL_SUCCESS) throw std::runtime_error(std::string(#x) + " = " + std::to_string(e_)); \
  } while (0)

static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("open " + p);
  return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

struct Arg {
  size_t size;
  const void* p;
};
template <typename T>
static Arg A(const T& v) {
  return Arg{sizeof(T), &v};
}

int main(int argc, char** argv) try {
  if (argc < 5) {
    fprintf(stderr, "usage: nss_run <dir> <cnn.onnx> <cnn_ctx.onnx> <frames> [iters]\n");
    return 2;
  }
  std::string dir = argv[1], model = argv[2], ctxp = argv[3];
  int frames = atoi(argv[4]), iters = argc > 5 ? atoi(argv[5]) : 1;
  const int H = 540, W = 960, Hp = 544, Wp = 960, Ho = 1080, Wo = 1920, Hd = 270, Wd = 480;
  const int Hk = 136, Wk = 240, Kc = 36, Ht = 544, Wt = 960;

  // ---------------- OpenCL
  load_cl();
  cl_platform_id plat;
  CK(p_clGetPlatformIDs(1, &plat, nullptr));
  cl_device_id dev;
  CK(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
  char dname[256] = {0};
  p_clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, nullptr);
  cl_int err;
  cl_context ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  CK(err);
  cl_command_queue q = p_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
  CK(err);
  auto src = read_file(dir + "/nss_kernels.cl");
  const char* sp = src.data();
  size_t sl = src.size();
  cl_program prog = p_clCreateProgramWithSource(ctx, 1, &sp, &sl, &err);
  CK(err);
  double tb = now_ms();
  // Correctly rounded divide/sqrt where the driver offers it (Adreno rejects the option: spec accuracy)
  if (p_clBuildProgram(prog, 1, &dev, "-cl-fp32-correctly-rounded-divide-sqrt", nullptr, nullptr) != CL_SUCCESS &&
      p_clBuildProgram(prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
    std::vector<char> log(1 << 16);
    p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
    fprintf(stderr, "%s\n", log.data());
    return 1;
  }
  printf("device %s, cl build %.1f ms\n", dname, now_ms() - tb);
  auto K = [&](const char* n) {
    cl_kernel k = p_clCreateKernel(prog, n, &err);
    CK(err);
    return k;
  };
  cl_kernel k_init = K("depth_scatter_init"), k_ds = K("depth_scatter"), k_pre = K("preprocess"),
            k_post = K("postprocess"), k_fb = K("temporal_to_feedback");
  auto B = [&](size_t bytes, cl_mem_flags fl = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR) {
    cl_mem m = p_clCreateBuffer(ctx, fl, bytes, nullptr, &err);
    CK(err);
    return m;
  };
  cl_mem b_color = B(3 * H * W * 4), b_motion = B(2 * H * W * 4), b_depth = B(H * W * 4);
  cl_mem b_recon = B(Hd * Wd * 4), b_lut = B(6 * 64 * 4 * 4);
  cl_mem b_hist[2] = {B(3 * Ho * Wo * 4), B(3 * Ho * Wo * 4)};
  cl_mem b_deriv[2] = {B(4 * H * W * 4), B(4 * H * W * 4)};
  cl_mem b_fb = B(4 * Hp * Wp * 4);  // feedback_tm1 as float planar (from the previous temporal output)
  cl_mem b_in_u8 = B(Hp * Wp * 12), b_code = B(H * W);
  cl_mem b_kpn = B(Hk * Wk * Kc), b_tmp = B(Ht * Wt * 4), b_rgba = B(Ho * Wo * 4);
  cl_mem nullmem = nullptr;
  std::vector<char> zeros(3 * Ho * Wo * 4, 0);
  auto zero_state = [&]() {  // the gym's zero history buffers at the start of a sequence
    CK(p_clEnqueueWriteBuffer(q, b_hist[0], CL_TRUE, 0, 3 * Ho * Wo * 4, zeros.data(), 0, nullptr, nullptr));
    CK(p_clEnqueueWriteBuffer(q, b_deriv[0], CL_TRUE, 0, 4 * H * W * 4, zeros.data(), 0, nullptr, nullptr));
    CK(p_clEnqueueWriteBuffer(q, b_fb, CL_TRUE, 0, 4 * Hp * Wp * 4, zeros.data(), 0, nullptr, nullptr));
  };
  auto set = [&](cl_kernel k, std::vector<Arg> args) {
    for (cl_uint i = 0; i < args.size(); i++) CK(p_clSetKernelArg(k, i, args[i].size, args[i].p));
  };
  auto ev_ms = [&](cl_event e) {
    cl_ulong a, b;
    p_clWaitForEvents(1, &e);
    p_clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr);
    p_clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof b, &b, nullptr);
    p_clReleaseEvent(e);
    return (b - a) * 1e-6;
  };

  // ---------------- HTP (ORT + QNN EP)
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nss_run");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(1);
  env.RegisterExecutionProviderLibrary("QNNExecutionProvider", "libonnxruntime_providers_qnn.so");
  std::vector<Ort::ConstEpDevice> devs;
  for (const auto& d : env.GetEpDevices())
    if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
      devs.push_back(d);
  if (devs.empty()) throw std::runtime_error("no QNN NPU ep device");
  std::unordered_map<std::string, std::string> opts{{"backend_type", "htp"}, {"htp_performance_mode", "burst"}};
  so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  so.AppendExecutionProvider_V2(env, devs, opts);
  {
    std::ifstream exists(ctxp);
    if (!exists) {
      Ort::ModelCompilationOptions co(env, so);
      co.SetInputModelPath(model.c_str());
      co.SetOutputModelPath(ctxp.c_str());
      co.SetEpContextEmbedMode(true);
      Ort::Status st = Ort::CompileModel(env, co);
      if (!st.IsOK()) throw std::runtime_error("CompileModel: " + st.GetErrorMessage());
    }
  }
  Ort::Session sess(env, ctxp.c_str(), so);
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char* in_names[] = {"x"};
  const char* out_names[] = {"kpn", "temporal"};
  int64_t in_shape[] = {1, Hp, Wp, 12}, kpn_shape[] = {1, Hk, Wk, Kc}, tmp_shape[] = {1, Ht, Wt, 4};

  printf("frame upload_ms ds_ms pre_ms htp_ms post_ms fb_ms gpu_wall_ms frame_ms\n");
  std::vector<double> tot;
  for (int it = 0; it < iters; it++) {
    for (int t = 0; t < frames; t++) {
      char nb[64];
      snprintf(nb, sizeof nb, "/frame%03d", t);
      auto bin = read_file(dir + nb + ".bin");
      std::ifstream tf(dir + nb + ".txt");
      float jy, jx, e, rs0, rs1, dp[4], reset;
      int mh, mw, taps;
      tf >> jy >> jx >> e >> rs0 >> rs1 >> dp[0] >> dp[1] >> dp[2] >> dp[3] >> reset >> mh >> mw >> taps;
      std::vector<float> lut(6 * mh * mw * taps);
      for (auto& v : lut) tf >> v;
      if (t == 0) zero_state();
      cl_float4 dtv = {{dp[0], dp[1], dp[2], dp[3]}};
      int cur = t & 1, nxt = cur ^ 1;

      double f0 = now_ms();
      size_t o = 0;
      CK(p_clEnqueueWriteBuffer(q, b_color, CL_FALSE, 0, 3 * H * W * 4, bin.data() + o, 0, nullptr, nullptr));
      o += 3 * H * W * 4;
      CK(p_clEnqueueWriteBuffer(q, b_motion, CL_FALSE, 0, 2 * H * W * 4, bin.data() + o, 0, nullptr, nullptr));
      o += 2 * H * W * 4;
      CK(p_clEnqueueWriteBuffer(q, b_depth, CL_FALSE, 0, H * W * 4, bin.data() + o, 0, nullptr, nullptr));
      CK(p_clEnqueueWriteBuffer(q, b_lut, CL_FALSE, 0, lut.size() * 4, lut.data(), 0, nullptr, nullptr));
      CK(p_clFinish(q));
      double up_ms = now_ms() - f0;

      double g0 = now_ms();
      cl_event e_init, e_ds, e_pre, e_post;
      size_t gi = Hd * Wd;
      set(k_init, {A(b_recon)});
      CK(p_clEnqueueNDRangeKernel(q, k_init, 1, nullptr, &gi, nullptr, 0, nullptr, &e_init));
      set(k_ds, {A(b_motion), A(b_depth), A(H), A(W), A(b_recon), A(Hd), A(Wd)});
      size_t gds[2] = {(size_t)Wd, (size_t)Hd};
      CK(p_clEnqueueNDRangeKernel(q, k_ds, 2, nullptr, gds, nullptr, 0, nullptr, &e_ds));
      set(k_pre, {A(b_color), A(b_hist[cur]), A(b_motion), A(b_depth), A(b_fb), A(b_deriv[cur]), A(b_recon),
                  A(H), A(W), A(Hp), A(Wp), A(Ho), A(Wo), A(Hd), A(Wd), A(jy), A(jx), A(e), A(rs0), A(rs1),
                  A(dtv), A(nullmem), A(b_in_u8), A(b_deriv[nxt]), A(nullmem), A(b_code)});
      size_t gp[2] = {(size_t)Wp, (size_t)Hp};
      CK(p_clEnqueueNDRangeKernel(q, k_pre, 2, nullptr, gp, nullptr, 0, nullptr, &e_pre));
      CK(p_clFinish(q));
      double pre_wall = now_ms() - g0;

      // HTP: the CNN reads the mapped uint8 input and writes into the mapped output buffers
      double h0 = now_ms();
      void* in_p = p_clEnqueueMapBuffer(q, b_in_u8, CL_TRUE, CL_MAP_READ, 0, Hp * Wp * 12, 0, nullptr, nullptr, &err);
      CK(err);
      void* kpn_p = p_clEnqueueMapBuffer(q, b_kpn, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, Hk * Wk * Kc, 0,
                                         nullptr, nullptr, &err);
      CK(err);
      void* tmp_p = p_clEnqueueMapBuffer(q, b_tmp, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, Ht * Wt * 4, 0,
                                         nullptr, nullptr, &err);
      CK(err);
      Ort::Value x = Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)in_p, Hp * Wp * 12, in_shape, 4);
      Ort::Value ys[2] = {Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)kpn_p, Hk * Wk * Kc, kpn_shape, 4),
                          Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)tmp_p, Ht * Wt * 4, tmp_shape, 4)};
      double r0 = now_ms();
      sess.Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names, ys, 2);
      double htp_ms = now_ms() - r0;
      CK(p_clEnqueueUnmapMemObject(q, b_in_u8, in_p, 0, nullptr, nullptr));
      CK(p_clEnqueueUnmapMemObject(q, b_kpn, kpn_p, 0, nullptr, nullptr));
      CK(p_clEnqueueUnmapMemObject(q, b_tmp, tmp_p, 0, nullptr, nullptr));
      double htp_wall = now_ms() - h0;

      double p0 = now_ms();
      int ntaps = taps;
      set(k_post, {A(b_color), A(b_hist[cur]), A(b_motion), A(b_code), A(b_kpn), A(b_tmp), A(b_lut), A(H), A(W),
                   A(Ho), A(Wo), A(Hk), A(Wk), A(Kc), A(Ht), A(Wt), A(mh), A(mw), A(ntaps), A(e), A(reset),
                   A(b_hist[nxt]), A(b_rgba)});
      size_t go[2] = {(size_t)Wo, (size_t)Ho};
      CK(p_clEnqueueNDRangeKernel(q, k_post, 2, nullptr, go, nullptr, 0, nullptr, &e_post));
      // next frame's feedback_tm1: the temporal output as float planar
      cl_event e_fb;
      int nfb = Hp * Wp;
      size_t gf = nfb;
      set(k_fb, {A(b_tmp), A(b_fb), A(nfb)});
      CK(p_clEnqueueNDRangeKernel(q, k_fb, 1, nullptr, &gf, nullptr, 0, nullptr, &e_fb));
      CK(p_clFinish(q));
      double post_wall = now_ms() - p0;
      double frame_ms = now_ms() - f0 - up_ms;
      double ds = ev_ms(e_init) + ev_ms(e_ds), pre = ev_ms(e_pre), post = ev_ms(e_post), fbm = ev_ms(e_fb);
      printf("%3d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", t, up_ms, ds, pre, htp_ms, post, fbm,
             pre_wall + post_wall, frame_ms);
      (void)htp_wall;
      if (it > 0 || iters == 1) tot.push_back(frame_ms);
      if (it == iters - 1) {
        void* rp = p_clEnqueueMapBuffer(q, b_rgba, CL_TRUE, CL_MAP_READ, 0, Ho * Wo * 4, 0, nullptr, nullptr, &err);
        CK(err);
        snprintf(nb, sizeof nb, "/out%03d.bin", t);
        std::ofstream(dir + nb, std::ios::binary).write((const char*)rp, Ho * Wo * 4);
        CK(p_clEnqueueUnmapMemObject(q, b_rgba, rp, 0, nullptr, nullptr));
        CK(p_clFinish(q));
      }
    }
  }
  std::sort(tot.begin(), tot.end());
  printf("frame_ms median %.2f (n=%zu)\n", tot[tot.size() / 2], tot.size());
  return 0;
} catch (const std::exception& ex) {
  fprintf(stderr, "error: %s\n", ex.what());
  return 1;
}

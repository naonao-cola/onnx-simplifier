// NFRU v1 frame generation on the phone: pre/post-processing as OpenCL kernels on the Adreno GPU
// (nfru_kernels.cl), the network on the HTP (ORT + QNN EP, uint8 NHWC I/O). One process; every rendered
// frame is processed once (colour, luma pyramid, motion normalization) and serves as p1 of one window and
// m1 of the next -- each window generates the frame half way between them.
//
//   nfru_run <dir> <net.onnx> <net_ctx.onnx> <windows> [iters]
//
// <dir> holds nfru_kernels.cl, rendered frames fNNN.bin (float32: linear rgb 3 x 1080 x 1920, depth
// 540 x 960, motion (row, col) 2 x 540 x 960 in pixels (to the frame two back), the "sy" motion hint to the
// next frame 2 x 540 x 960) and per-window wNNN.txt (motion matrices m1->p1 / p1->m1 / m3<-m1 x 16, depth
// params x 4, seed) from `nfru.py phone`. Writes outNNN.bin (the generated frame, RGBA8) and prints per-stage
// GPU times (OpenCL profiling events) and HTP / wall times per generated frame.
#include <onnxruntime_cxx_api.h>

#include "cl_dl.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
    fprintf(stderr, "usage: nfru_run <dir> <net.onnx> <net_ctx.onnx> <windows> [iters]\n");
    return 2;
  }
  std::string dir = argv[1], model = argv[2], ctxp = argv[3];
  int windows = atoi(argv[4]), iters = argc > 5 ? atoi(argv[5]) : 1;
  const int cH = 1080, cW = 1920, H = 540, W = 960;  // colour, depth / motion
  const float T = 0.5f, EXPO = 7.389056205749512f, PSC = 0.35356706380844116f, PZP = 172.0f;
  // luma pyramid, finest first: 1080 x 1920 ... 34 x 60 (padded to even; true size (hd, wd))
  struct Lv {
    int h, w, hd, wd;
  };
  std::vector<Lv> lv{{cH, cW, cH, cW}};
  for (int i = 1; i < 6; i++) {
    int hd = lv.back().h / 2, wd = lv.back().w / 2;
    lv.push_back({hd + hd % 2, wd + wd % 2, hd, wd});
  }
  const int fH = lv[2].h, fW = lv[2].w;  // the flow / network resolution (270 x 480)

  // ---------------- OpenCL
  load_cl();
  cl_platform_id plat;
  CK(p_clGetPlatformIDs(1, &plat, nullptr));
  cl_device_id dev;
  CK(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
  char dname[256] = {0};
  p_clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, nullptr);
  cl_int err;
  // cl_qcom_perf_hint: this context asks for high GPU clocks (an app-level hint, no system setting)
  const char* perf = getenv("NFRU_PERF") ? getenv("NFRU_PERF") : "high";
  cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0x40C2 /* PERF_HINT */,
                                   std::string(perf) == "low"      ? 0x40C5
                                   : std::string(perf) == "normal" ? 0x40C4
                                                                   : 0x40C3,
                                   0};
  if (std::string(perf) == "none") props[2] = 0;
  cl_context ctx = p_clCreateContext(props, 1, &dev, nullptr, nullptr, &err);
  CK(err);
  cl_command_queue q = p_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
  CK(err);
  auto src = read_file(dir + "/nfru_kernels.cl");
  const char* sp = src.data();
  size_t sl = src.size();
  cl_program prog = p_clCreateProgramWithSource(ctx, 1, &sp, &sl, &err);
  CK(err);
  double tb = now_ms();
  std::string extra = getenv("NFRU_CLFLAGS") ? getenv("NFRU_CLFLAGS") : "";
  std::string o1 = "-cl-fp32-correctly-rounded-divide-sqrt " + extra;
  if (p_clBuildProgram(prog, 1, &dev, o1.c_str(), nullptr, nullptr) != CL_SUCCESS &&
      p_clBuildProgram(prog, 1, &dev, extra.c_str(), nullptr, nullptr) != CL_SUCCESS) {
    std::vector<char> log(1 << 16);
    p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
    fprintf(stderr, "%s\n", log.data());
    return 1;
  }
  printf("device %s, perf hint %s, cl build %.1f ms\n", dname, perf, now_ms() - tb);
  std::map<std::string, cl_kernel> kern;
  auto K = [&](const char* n) {
    auto it = kern.find(n);
    if (it != kern.end()) return it->second;
    cl_kernel k = p_clCreateKernel(prog, n, &err);
    CK(err);
    return kern[n] = k;
  };
  auto B = [&](size_t bytes, cl_mem_flags fl = CL_MEM_READ_WRITE) {
    cl_mem m = p_clCreateBuffer(ctx, fl, std::max<size_t>(bytes, 4), nullptr, &err);
    CK(err);
    return m;
  };
  cl_mem nullmem = nullptr;
  auto IMG = [&](int w, int h) {  // RGBA32F: the tonemapped colour, read through the texture path
    cl_image_format f{CL_RGBA, CL_FLOAT};
    cl_image_desc d{};
    d.image_type = CL_MEM_OBJECT_IMAGE2D;
    d.image_width = w;
    d.image_height = h;
    cl_mem m = p_clCreateImage(ctx, CL_MEM_READ_WRITE, &f, &d, nullptr, &err);
    CK(err);
    return m;
  };
  // per rendered frame (two slots: m1 / p1): colour, pyramid, depth, normalized motion, the motion hint
  struct Frame {
    cl_mem lin, rgb, pyr[6], depth, mv_raw, mv, sy;
  } fr[2];
  for (auto& f : fr) {
    f.lin = B((size_t)3 * cH * cW * 4);
    f.rgb = IMG(cW, cH);
    for (int i = 0; i < 6; i++) f.pyr[i] = B((size_t)lv[i].h * lv[i].w);
    f.depth = B(H * W * 4);
    f.mv_raw = B(2 * H * W * 4);
    f.mv = B(2 * H * W * 4);
    f.sy = B(2 * H * W * 4);
  }
  cl_mem m_mm0 = B(64, CL_MEM_READ_ONLY), m_mm1 = B(64, CL_MEM_READ_ONLY), m_mm3 = B(64, CL_MEM_READ_ONLY);
  cl_mem b_hint = B(2 * fH * fW * 4), b_flow = B(2 * fH * fW * 4), b_dyn = B(H * W * 4);
  cl_mem b_packed = B(H * W * 4), b_holes_t = B(H * W * 4), b_holes_m1 = B(H * W * 4), b_pflow = B(fH * fW * 4);
  cl_mem b_mvt = B(2 * H * W * 4), b_flowt = B(2 * fH * fW * 4);
  // block matcher scratch per level (coarsest first = pyramid levels 5..2)
  struct BmLv {
    cl_mem vp, sw, hint, vec, won, med, out;
  } bl[4];
  for (int l = 0; l < 4; l++) {
    const Lv& L = lv[5 - l];
    bl[l] = {B(2 * L.h * L.w * 4), B(L.h * L.w), B(L.h * L.w), B(2 * L.h * L.w * 4),
             B(L.h * L.w),         B(2 * L.h * L.w * 4), B(2 * L.hd * L.wd * 4)};
  }
  // the network's input / output: host-visible (mapped for ORT); the generated frame RGBA8 likewise
  const cl_mem_flags mapped = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
  cl_mem b_net = B(fH * fW * 16, mapped), b_par = B(fH * fW * 4, mapped), b_rgba = B(cH * cW * 4, mapped);

  auto set = [&](cl_kernel k, std::vector<Arg> args) {
    for (cl_uint i = 0; i < args.size(); i++) CK(p_clSetKernelArg(k, i, args[i].size, args[i].p));
  };
  // enqueue a kernel (1-D if gy == 0), timed into the named stage; bm_match needs 16 x 16 work-groups
  std::vector<std::pair<std::string, cl_event>> evs;
  const bool detail = getenv("NFRU_DETAIL") != nullptr;  // per-kernel times too
  std::vector<std::string> names;
  auto run = [&](const char* stage, const char* name, std::vector<Arg> args, size_t gx, size_t gy = 0) {
    cl_kernel k = K(name);
    set(k, args);
    size_t g[2] = {gx, gy}, l[2] = {16, 16};
    const bool tiled = std::string(name) == "bm_match";
    if (tiled) g[0] = (gx + 15) / 16 * 16, g[1] = (gy + 15) / 16 * 16;
    cl_event e;
    CK(p_clEnqueueNDRangeKernel(q, k, gy ? 2 : 1, nullptr, g, tiled ? l : nullptr, 0, nullptr, &e));
    evs.push_back({stage, e});
    if (detail) names.push_back(name);
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
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nfru_run");
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
  const char* out_names[] = {"params"};
  int64_t in_shape[] = {1, fH, fW, 16}, out_shape[] = {1, fH, fW, 4};

  // one rendered frame: upload (not timed as NFRU work), then colour + luma pyramid + motion normalization
  auto upload = [&](int t, Frame& f) {
    char nb[64];
    snprintf(nb, sizeof nb, "/f%03d.bin", t);
    auto bin = read_file(dir + nb);
    size_t o = 0;
    auto put = [&](cl_mem m, size_t n) {
      CK(p_clEnqueueWriteBuffer(q, m, CL_FALSE, 0, n, bin.data() + o, 0, nullptr, nullptr));
      o += n;
    };
    put(f.lin, (size_t)3 * cH * cW * 4);
    put(f.depth, H * W * 4);
    put(f.mv_raw, 2 * H * W * 4);
    put(f.sy, 2 * H * W * 4);
    CK(p_clFinish(q));
  };
  auto frame_work = [&](Frame& f) {
    run("colour", "colour_luma", {A(f.lin), A(EXPO), A(cH), A(cW), A(f.rgb), A(f.pyr[0])}, cW, cH);
    for (int i = 1; i < 6; i++) {
      int blur = (i - 1) >= 1 && (i - 1) <= 4, quad = i == 5;
      run("pyramid", "pyr_down",
          {A(f.pyr[i - 1]), A(lv[i - 1].h), A(lv[i - 1].w), A(blur), A(quad), A(f.pyr[i]), A(lv[i].h), A(lv[i].w),
           A(lv[i].hd), A(lv[i].wd)},
          lv[i].w, lv[i].h);
    }
    int nm = H * W;
    float one = 1.0f, fh = (float)H, fw = (float)W;
    run("colour", "norm_mv", {A(f.mv_raw), A(nm), A(one), A(fh), A(fw), A(f.mv)}, nm);
  };

  printf("window colour_ms pyr_ms bm_ms motion_ms pre_ms htp_ms post_ms gpu_sum_ms gen_ms\n");
  std::vector<double> tot, tot_nf;
  for (int it = 0; it < iters; it++) {
    upload(0, fr[0]);
    frame_work(fr[0]);
    CK(p_clFinish(q));
    for (auto& e : evs) p_clReleaseEvent(e.second);
    evs.clear();
    names.clear();
    for (int w = 0; w < windows; w++) {
      Frame &m1 = fr[w & 1], &p1 = fr[(w + 1) & 1];
      upload(w + 1, p1);
      char nb[64];
      snprintf(nb, sizeof nb, "/w%03d.txt", w);
      std::ifstream tf(dir + nb);
      float mm0[16], mm1[16], mm3[16];
      cl_float4 dp;
      unsigned seed;
      for (float& v : mm0) tf >> v;
      for (float& v : mm1) tf >> v;
      for (float& v : mm3) tf >> v;
      for (int i = 0; i < 4; i++) tf >> dp.s[i];
      tf >> seed;
      CK(p_clEnqueueWriteBuffer(q, m_mm0, CL_FALSE, 0, 64, mm0, 0, nullptr, nullptr));
      CK(p_clEnqueueWriteBuffer(q, m_mm1, CL_FALSE, 0, 64, mm1, 0, nullptr, nullptr));
      CK(p_clEnqueueWriteBuffer(q, m_mm3, CL_FALSE, 0, 64, mm3, 0, nullptr, nullptr));
      CK(p_clFinish(q));

      double f0 = now_ms();
      frame_work(p1);  // the new rendered frame
      // optical flow: block matching of p1 (search) against m1 (template), coarse to fine, with the
      // rendered-motion hint at the finest level
      run("bm", "hint_mv", {A(m1.sy), A(m1.depth), A(H), A(W), A(b_hint), A(fH), A(fW)}, fW, fH);
      for (int l = 0; l < 4; l++) {
        const Lv& L = lv[5 - l];
        BmLv& b = bl[l];
        cl_mem srch = p1.pyr[5 - l], tmpl = m1.pyr[5 - l], vp = nullmem, sw = srch, hint = nullmem, hmv = nullmem,
               won = nullmem;
        if (l) {
          const Lv& P = lv[6 - l];
          run("bm", "bm_upsample", {A(bl[l - 1].out), A(P.hd), A(P.wd), A(b.vp), A(L.h), A(L.w)}, L.w, L.h);
          run("bm", "bm_warp", {A(srch), A(L.h), A(L.w), A(b.vp), A(L.h), A(L.w), A(b.sw)}, L.w, L.h);
          vp = b.vp;
          sw = b.sw;
        }
        if (l == 3) {
          run("bm", "bm_warp", {A(srch), A(L.h), A(L.w), A(b_hint), A(L.h), A(L.w), A(b.hint)}, L.w, L.h);
          hint = b.hint;
          hmv = b_hint;
          won = b.won;
        }
        run("bm", "bm_match", {A(sw), A(tmpl), A(hint), A(L.h), A(L.w), A(vp), A(b.vec), A(b.won)}, L.w, L.h);
        run("bm", "bm_median", {A(b.vec), A(L.h), A(L.w), A(b.med)}, L.w, L.h);
        run("bm", "bm_jbf",
            {A(b.med), A(tmpl), A(L.h), A(L.w), A(won), A(hmv), A(L.h), A(L.w), A(b.out), A(L.hd), A(L.wd)}, L.wd,
            L.hd);
      }
      int nf = fH * fW, nm = H * W;
      float m4 = -4.0f, ffh = (float)fH, ffw = (float)fW;
      run("bm", "norm_mv", {A(bl[3].out), A(nf), A(m4), A(ffh), A(ffw), A(b_flow)}, nf);
      // motion: the dynamic mask (m1 vs m3), depth-aware splats of the rendered motion and the flow to t
      run("motion", "dyn_mask", {A(m1.depth), A(m1.mv), A(m_mm3), A(H), A(W), A(b_dyn)}, W, H);
      for (cl_mem z : {b_packed, b_holes_t, b_holes_m1}) run("motion", "zero_i32", {A(z)}, nm);
      run("motion", "zero_i32", {A(b_pflow)}, nf);
      float t = T, t1 = 1.0f - T;
      run("motion", "warp_mv",
          {A(p1.depth), A(m1.depth), A(p1.mv), A(b_dyn), A(m_mm1), A(H), A(W), A(t), A(b_packed), A(b_holes_t),
           A(b_holes_m1)},
          W, H);
      run("motion", "fill_mv", {A(b_packed), A(H), A(W), A(b_mvt)}, W, H);
      run("motion", "warp_flow", {A(m1.depth), A(H), A(W), A(b_flow), A(fH), A(fW), A(t1), A(b_pflow)}, fW, fH);
      run("motion", "fill_mv", {A(b_pflow), A(fH), A(fW), A(b_flowt)}, fW, fH);
      run("pre", "preprocess",
          {A(b_flowt), A(b_mvt), A(H), A(W), A(m1.rgb), A(p1.rgb), A(cH), A(cW), A(m1.depth), A(p1.depth),
           A(b_holes_t), A(b_holes_m1), A(m_mm1), A(m_mm0), A(dp), A(t), A(seed), A(fH), A(fW), A(nullmem),
           A(b_net)},
          fW, fH);
      CK(p_clFinish(q));
      double pre_wall = now_ms() - f0;

      // HTP: the network reads the mapped uint8 input and writes the mapped uint8 logits
      void* in_p = p_clEnqueueMapBuffer(q, b_net, CL_TRUE, CL_MAP_READ, 0, nf * 16, 0, nullptr, nullptr, &err);
      CK(err);
      void* out_p =
          p_clEnqueueMapBuffer(q, b_par, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, nf * 4, 0, nullptr, nullptr, &err);
      CK(err);
      Ort::Value x = Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)in_p, nf * 16, in_shape, 4);
      Ort::Value y = Ort::Value::CreateTensor<uint8_t>(mem, (uint8_t*)out_p, nf * 4, out_shape, 4);
      double r0 = now_ms();
      sess.Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names, &y, 1);
      double htp_ms = now_ms() - r0;
      CK(p_clEnqueueUnmapMemObject(q, b_net, in_p, 0, nullptr, nullptr));
      CK(p_clEnqueueUnmapMemObject(q, b_par, out_p, 0, nullptr, nullptr));

      double p0 = now_ms();
      float psc = PSC, pzp = PZP;
      run("post", "postprocess",
          {A(b_flowt), A(fH), A(fW), A(b_mvt), A(H), A(W), A(b_par), A(fH), A(fW), A(psc), A(pzp), A(m1.rgb),
           A(p1.rgb), A(cH), A(cW), A(t), A(nullmem), A(b_rgba)},
          cW, cH);
      CK(p_clFinish(q));
      double post_wall = now_ms() - p0, gen_ms = now_ms() - f0;
      std::map<std::string, double> st;
      double sum = 0;
      std::string det;
      for (size_t i = 0; i < evs.size(); i++) {
        double ms = ev_ms(evs[i].second);
        st[evs[i].first] += ms;
        sum += ms;
        if (detail) det += " " + names[i] + "=" + std::to_string(ms).substr(0, 5);
      }
      evs.clear();
      names.clear();
      if (detail) printf("   %s\n", det.c_str());
      printf("%3d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", w, st["colour"], st["pyramid"], st["bm"],
             st["motion"], st["pre"], htp_ms, st["post"], sum, gen_ms);
      (void)pre_wall;
      (void)post_wall;
      if (it > 0 || iters == 1) tot.push_back(gen_ms);
      if (it == iters - 1) {
        void* rp = p_clEnqueueMapBuffer(q, b_rgba, CL_TRUE, CL_MAP_READ, 0, cH * cW * 4, 0, nullptr, nullptr, &err);
        CK(err);
        snprintf(nb, sizeof nb, "/out%03d.bin", w);
        std::ofstream(dir + nb, std::ios::binary).write((const char*)rp, cH * cW * 4);
        CK(p_clEnqueueUnmapMemObject(q, b_rgba, rp, 0, nullptr, nullptr));
        CK(p_clFinish(q));
      }
    }
  }
  std::sort(tot.begin(), tot.end());
  printf("gen_ms median %.2f (n=%zu)\n", tot[tot.size() / 2], tot.size());
  return 0;
} catch (const std::exception& ex) {
  fprintf(stderr, "error: %s\n", ex.what());
  return 1;
}

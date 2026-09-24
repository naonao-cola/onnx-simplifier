// BEVFormer-tiny's 3-layer encoder on the phone, split (split.py): 7 fp16 HTP pieces (ORT + QNN EP,
// strict all-HTP) and 6 deformable-sampling calls to the generic HVX MSDA skel
// (../../../msda_hvx/), chained in-process:
//
//   pre -> [tsa] -> mid0 -> [sca] -> post0 -> [tsa] -> mid1 -> [sca] -> post1 -> [tsa] -> mid2 -> [sca] -> post2
//
// Every tensor lives in one rpcmem (ION) buffer per name: ORT writes the HTP pieces' outputs straight
// into them (pre-bound output tensors) and the DSP maps them without a copy.
//
// usage: enc_run <piece dir> <warmup> <reps> <frame dir>...
//   piece dir: pre.onnx mid{0,1,2}.onnx post{0,1,2}.onnx (split.py export's *.sim.onnx, renamed); with
//              tsa_vq.txt (split.py export --tsa-u8: "scale zero_point" per layer) the pieces emit the
//              TSA value maps as uint8 and the kernel reads them as such
//   frame dir: feats.f32 (6,256,15,25) prev_bev.f32 (2500,256) has_prev.f32 (1) can_bus.f32 (18)
//              tsa_ref.f32 (2,2500,1,2) ref_cam.f32 (6,2500,4,2) vis.u8 (6,2500)
//   writes <frame dir>/bev.f32 (2500,256) from the last run; prints per-step and total medians.
// env: MSDA_URI, MSDA_THREADS (default 4), QNN_PERF (htp_performance_mode), QNN_EXTRA (k=v,k=v), ORT_LOG
#include "chain.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s piece_dir warmup reps frame_dir...\n", argv[0]);
    return 2;
  }
  const std::string pdir = argv[1];
  const int warmup = atoi(argv[2]), reps = atoi(argv[3]);
  try {
    struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
    const char* uri = getenv("MSDA_URI") ? getenv("MSDA_URI")
                                         : "file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    if (msda_rpc_open(uri, &h_msda)) throw std::runtime_error("msda_rpc_open failed");
    int32 prc = 0;
    msda_rpc_perf_vote(h_msda, 0, &prc);

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "enc_run");
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB")
                                                                                         : "libonnxruntime_providers_qnn.so");
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");
    std::vector<std::string> names = {"pre", "mid0", "post0", "mid1", "post1", "mid2", "post2"};
    std::map<std::string, Piece> pc;
    double tot_create = 0;
    for (auto& n : names) {
      double ms = 0;
      pc[n] = load(pdir, n, &ms);
      tot_create += ms;
    }
    printf("sessions create_ms %.1f\n", tot_create);
    {
      std::ifstream vq(pdir + "/tsa_vq.txt");
      for (int i = 0; vq && i < 3; i++) vq >> tsa_scale[i] >> tsa_zp[i];
    }

    // inputs
    buf("feats", {6, 256, 15, 25});
    buf("prev_bev", {2500, 256});
    buf("has_prev", {1});
    buf("can_bus", {18});
    buf("tsa_ref", {2, 2500, 1, 2});
    buf("ref_cam", {6, 2500, 4, 2});
    buf("vis", {6, 2500}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    Buf& ones = buf("tsa_vis", {2, 2500}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    memset(ones.p, 1, ones.bytes);

    const char* step_names[] = {"pre",     "tsa0",    "mid0",  "sca0", "post0", "tsa1", "mid1",
                                "sca1",    "post1",   "tsa2",  "mid2", "sca2",  "post2"};
    const int nsteps = 13;
    for (int fi = 4; fi < argc; ++fi) {
      const std::string fd = argv[fi];
      for (auto n : {"feats", "prev_bev", "has_prev", "can_bus", "tsa_ref", "ref_cam"}) read_into(fd + "/" + n + ".f32", get(n));
      read_into(fd + "/vis.u8", get("vis"));
      std::vector<std::vector<double>> st(nsteps), dsp(nsteps);
      std::vector<double> tot;
      for (int it = 0; it < warmup + reps; ++it) {
        int k = 0;
        double t0 = now_ms(), a;
        auto mark = [&](double dsp_ms) {
          double b = now_ms();
          if (it >= warmup) { st[k].push_back(b - a); dsp[k].push_back(dsp_ms); }
          k++;
          a = b;
        };
        a = t0;
        encoder(pc, mark);
        if (it >= warmup) tot.push_back(now_ms() - t0);
      }
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
      printf("%s encoder_ms median %.2f min %.2f (n=%zu)\n", fd.c_str(), med(tot), *std::min_element(tot.begin(), tot.end()),
             tot.size());
      double htp = 0, dsp_wall = 0, dsp_in = 0;
      for (int k = 0; k < nsteps; ++k) {
        double m = med(st[k]), d = med(dsp[k]);
        printf("  step %-6s %7.2f ms%s\n", step_names[k], m, d > 0 ? (" (dsp " + std::to_string(d).substr(0, 5) + ")").c_str() : "");
        if (d > 0) { dsp_wall += m; dsp_in += d; } else { htp += m; }
      }
      printf("  htp pieces %.2f ms, msda calls %.2f ms (in-DSP %.2f, FastRPC %.2f)\n", htp, dsp_wall, dsp_in, dsp_wall - dsp_in);
      Buf& bev = get("bev_embed");
      FILE* f = fopen((fd + "/bev.f32").c_str(), "wb");
      fwrite(bev.p, 1, bev.bytes, f);
      fclose(f);
      fflush(stdout);
    }
    msda_rpc_close(h_msda);
    printf("PASS\n");
    fflush(stdout);
    _exit(0);  // skip static destructors (ORT teardown order aborts on exit, as in pipe_run)
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    return 1;
  }
}

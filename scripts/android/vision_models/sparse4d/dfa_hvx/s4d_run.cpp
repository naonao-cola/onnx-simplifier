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
// env: DFA_URI, DFA_FLAGS (threads | 256 * anchors-per-job/16, default 4), S4D_MODELS (io16.py variant of the
//      pre0 / mid pieces, e.g. w16), QNN_PERF, QNN_EXTRA, ORT_LOG
#include "s4d_common.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s split_dir frame_dir warmup reps\n", argv[0]);
    return 2;
  }
  const std::string sdir = argv[1], fd = argv[2];
  const int warmup = atoi(argv[3]), reps = atoi(argv[4]);
  try {
    init_dsp_and_ort();

    const bool temporal = exists(fd + "/temp_feat.f32");
    const char* T = temporal ? "T" : "F";
    std::vector<std::string> names = {"bb", "pre0"};
    for (int k = 0; k < 5; ++k) names.push_back("mid" + std::to_string(k) + T);
    names.push_back(std::string("post") + T);
    std::map<std::string, Piece> pc;
    const double tot_create = load_pieces(sdir, names, pc);
    printf("sessions create_ms %.1f (%s)\n", tot_create, temporal ? "temporal" : "first frame");
    read_vscales(pc["bb"]);
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

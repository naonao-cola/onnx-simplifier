/* Phone-CPU baseline: rpn_region.onnx (capture_rpn_fused.py -- exactly the 604 rest.onnx nodes the
 * fused DSP call replaces: 10 inputs -> proposal list) in ONNX Runtime's CPU EP on the phone, one
 * Run, via the C API of the stock onnxruntime-android arm64 libonnxruntime.so (same harness style
 * as ../proposal_decode/ort_pd_bench.c). Checks ORT's output against the captured proposals.
 *   ort_rpn_bench IMG_DIR...   (cwd: rpn_region.onnx, rpn_region_io.txt) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"
static const OrtApi* g;
#define CK(x) do { OrtStatus* s_ = (x); if (s_) { fprintf(stderr, "%s\n", g->GetErrorMessage(s_)); exit(1); } } while (0)
static void* slurp(const char* p, size_t* n) {
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void* b = malloc(*n); if (fread(b, 1, *n, f) != *n) exit(1); fclose(f); return b;
}
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
int main(int argc, char** argv) {
  g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env; CK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "b", &env));
  OrtMemoryInfo* mi; CK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  char names[16][96], files[16][64], out_name[96], out_file[64]; int nd[16]; int64_t dims[16][4]; int ni = 0;
  FILE* f = fopen("rpn_region_io.txt", "r"); char line[512];
  while (fgets(line, sizeof line, f)) {
    if (!strncmp(line, "OUT ", 4)) { sscanf(line + 4, "%95s %63s", out_name, out_file); continue; }
    int pos;
    if (sscanf(line, "%95s %63s %d%n", names[ni], files[ni], &nd[ni], &pos) != 3) continue;
    char* r = line + pos;
    for (int i = 0; i < nd[ni]; i++) { long v; int adv; sscanf(r, "%ld%n", &v, &adv); dims[ni][i] = v; r += adv; }
    ni++;
  }
  fclose(f);
  const char* inp[16]; const char* outp[1] = {out_name};
  for (int i = 0; i < ni; i++) inp[i] = names[i];
  for (int ti = 0; ti < 2; ti++) {
    OrtSessionOptions* so; CK(g->CreateSessionOptions(&so));
    CK(g->SetIntraOpNumThreads(so, ti == 0 ? 1 : 0));
    OrtSession* s; CK(g->CreateSession(env, "rpn_region.onnx", so, &s));
    for (int im = 1; im < argc; im++) {
      OrtValue* in[16]; void* bufs[16]; char p[512];
      for (int i = 0; i < ni; i++) {
        size_t n; snprintf(p, sizeof p, "%s/%s", argv[im], files[i]); bufs[i] = slurp(p, &n);
        CK(g->CreateTensorWithDataAsOrtValue(mi, bufs[i], n, dims[i], nd[i], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[i]));
      }
      size_t rn; snprintf(p, sizeof p, "%s/%s", argv[im], out_file); float* ref = slurp(p, &rn);
      double ts[40]; int reps = 31, bad = 0;
      for (int k = 0; k <= reps; k++) {
        OrtValue* y = 0; double t0 = now_ms();
        CK(g->Run(s, NULL, inp, (const OrtValue* const*)in, ni, outp, 1, &y));
        if (k) ts[k - 1] = now_ms() - t0;
        if (k == reps) {
          float* d; CK(g->GetTensorMutableData(y, (void**)&d));
          OrtTensorTypeAndShapeInfo* ti2; size_t cnt; CK(g->GetTensorTypeAndShape(y, &ti2)); CK(g->GetTensorShapeElementCount(ti2, &cnt));
          g->ReleaseTensorTypeAndShapeInfo(ti2);
          bad = cnt * 4 != rn || memcmp(d, ref, rn) != 0;
        }
        g->ReleaseValue(y);
      }
      qsort(ts, reps, sizeof ts[0], cmpd);
      printf("ORT CPU (phone) rpn_region %s intra_op_threads=%s: median=%.3f ms min=%.3f ms output_matches_capture=%s\n",
             argv[im], ti ? "default" : "1", ts[reps / 2], ts[0], bad ? "NO" : "yes");
      for (int i = 0; i < ni; i++) { g->ReleaseValue(in[i]); free(bufs[i]); }
      free(ref);
    }
    g->ReleaseSession(s); g->ReleaseSessionOptions(so);
  }
  return 0;
}

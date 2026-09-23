/* Phone-CPU baseline: time each real single-node TopK model (gen_topk_test_data.py) in ONNX
 * Runtime's CPU EP on the phone itself, via the C API from the stock onnxruntime-android AAR's arm64
 * libonnxruntime.so -- the op's actual current execution path in this pipeline. Also checks ORT's
 * on-phone output against the captured full-graph output (same tie order on ARM as on x86). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"
static const OrtApi* g;
#define CK(x) do { OrtStatus* s_ = (x); if (s_) { fprintf(stderr, "%s\n", g->GetErrorMessage(s_)); exit(1); } } while (0)
static void* slurp(const char* p, size_t* n) { FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); } fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* b = malloc(*n); if (fread(b, 1, *n, f) != *n) exit(1); fclose(f); return b; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }
int main(void) {
  g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env; CK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "b", &env));
  OrtMemoryInfo* mi; CK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  FILE* cf = fopen("calls.txt", "r"); long n, k; int axis, i = 0, bad = 0;
  double tot[2] = {0, 0};
  while (fscanf(cf, "%ld %ld %d", &n, &k, &axis) == 3) {
    char p[64]; size_t sz;
    snprintf(p, sizeof p, "call%d_x.bin", i); void* x = slurp(p, &sz);
    snprintf(p, sizeof p, "call%d_vals.bin", i); float* rv = slurp(p, &sz);
    snprintf(p, sizeof p, "call%d_idx.bin", i); long long* ri = slurp(p, &sz);
    int64_t xs2[2] = {1, n}, xs1[1] = {n}, ks[1] = {1}, kv = k;
    OrtValue* in[2];
    CK(g->CreateTensorWithDataAsOrtValue(mi, x, n * 4, axis ? xs2 : xs1, axis ? 2 : 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]));
    CK(g->CreateTensorWithDataAsOrtValue(mi, &kv, 8, ks, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[1]));
    const char* inames[2] = {"X", "K"}; const char* onames[2] = {"V", "I"};
    for (int ti = 0; ti < 2; ti++) {
      OrtSessionOptions* so; CK(g->CreateSessionOptions(&so));
      CK(g->SetIntraOpNumThreads(so, ti == 0 ? 1 : 0));
      snprintf(p, sizeof p, "call%d.onnx", i);
      OrtSession* s; CK(g->CreateSession(env, p, so, &s));
      double ts[50];
      for (int r = 0; r < 51; r++) {
        OrtValue* y[2] = {NULL, NULL}; double t0 = now_ms();
        CK(g->Run(s, NULL, inames, (const OrtValue* const*)in, 2, onames, 2, y));
        double dt = now_ms() - t0;
        if (r == 0) {
          float* v; long long* ix;
          CK(g->GetTensorMutableData(y[0], (void**)&v)); CK(g->GetTensorMutableData(y[1], (void**)&ix));
          if (memcmp(v, rv, k * 4) || memcmp(ix, ri, k * 8)) { printf("call%d: phone ORT output differs from captured\n", i); bad = 1; }
        } else ts[r - 1] = dt;
        g->ReleaseValue(y[0]); g->ReleaseValue(y[1]);
      }
      qsort(ts, 50, sizeof ts[0], cmpd); tot[ti] += ts[25];
      printf("call%d n=%ld k=%ld ort_%s_ms(median)=%.4f\n", i, n, k, ti ? "default" : "1thr", ts[25]);
      g->ReleaseSession(s); g->ReleaseSessionOptions(so);
    }
    i++;
  }
  printf("TOTAL ort_1thr_ms=%.3f ort_default_ms=%.3f %s\n", tot[0], tot[1], bad ? "MISMATCH" : "outputs-match-capture");
  return bad;
}

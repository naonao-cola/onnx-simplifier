/* Phone-CPU baseline: time pd_region.onnx (make_pd_ort_model.py -- exactly the rest.onnx nodes the
 * kernel replaces, all 5 levels, one Run) in ONNX Runtime's CPU EP on the phone, via the C API of the
 * stock onnxruntime-android arm64 libonnxruntime.so (same harness style as
 * ../roialign_fast/ort_roialign_bench.c). Checks ORT's outputs against the captured reference. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"
static const OrtApi* g;
#define CK(x) do { OrtStatus* s_ = (x); if (s_) { fprintf(stderr, "%s\n", g->GetErrorMessage(s_)); exit(1); } } while (0)
static void* slurp(const char* p, size_t* n) { FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); } fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* b = malloc(*n); if (fread(b, 1, *n, f) != *n) exit(1); fclose(f); return b; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
int main(void) {
  g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env; CK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "b", &env));
  OrtMemoryInfo* mi; CK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  char in_names[16][64], out_names[8][64], out_ref[8][64]; long out_n[8];
  OrtValue* in[16]; int ni = 0, no = 0;
  FILE* f = fopen("pd_region_io.txt", "r"); char line[256];
  while (fgets(line, sizeof line, f)) {
    char a[64], b[64], c[64]; int pos;
    if (!strncmp(line, "OUT ", 4)) { sscanf(line + 4, "%63s %63s %ld", out_names[no], out_ref[no], &out_n[no]); no++; continue; }
    int nd; int64_t dims[4];
    if (sscanf(line, "%63s %63s %63s %d%n", a, b, c, &nd, &pos) != 4) continue;
    char* rest = line + pos;
    for (int i = 0; i < nd; i++) { long v; int adv; sscanf(rest, "%ld%n", &v, &adv); dims[i] = v; rest += adv; }
    size_t n; void* buf = slurp(c, &n);
    strcpy(in_names[ni], a);
    CK(g->CreateTensorWithDataAsOrtValue(mi, buf, n, dims, nd, !strcmp(b, "i64") ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[ni]));
    ni++;
  }
  const char* inp[16]; const char* outp[8];
  for (int i = 0; i < ni; i++) inp[i] = in_names[i];
  for (int i = 0; i < no; i++) outp[i] = out_names[i];
  for (int ti = 0; ti < 2; ti++) {
    OrtSessionOptions* so; CK(g->CreateSessionOptions(&so));
    CK(g->SetIntraOpNumThreads(so, ti == 0 ? 1 : 0));
    OrtSession* s; CK(g->CreateSession(env, "pd_region.onnx", so, &s));
    double ts[40]; int reps = 31, bad = 0;
    for (int k = 0; k <= reps; k++) {
      OrtValue* y[8] = {0}; double t0 = now_ms();
      CK(g->Run(s, NULL, inp, (const OrtValue* const*)in, ni, outp, no, y));
      if (k) ts[k - 1] = now_ms() - t0;
      if (k == reps) for (int o = 0; o < no; o++) {
        float* d; CK(g->GetTensorMutableData(y[o], (void**)&d));
        size_t n; float* r = slurp(out_ref[o], &n);
        bad |= memcmp(d, r, out_n[o] * 4) != 0; free(r);
      }
      for (int o = 0; o < no; o++) g->ReleaseValue(y[o]);
    }
    qsort(ts, reps, sizeof ts[0], cmpd);
    printf("ORT CPU (phone) pd_region intra_op_threads=%s: median=%.3f ms min=%.3f ms outputs_match_ref=%s\n",
           ti ? "default" : "1", ts[reps / 2], ts[0], bad ? "NO" : "yes");
    g->ReleaseSession(s); g->ReleaseSessionOptions(so);
  }
  return 0;
}

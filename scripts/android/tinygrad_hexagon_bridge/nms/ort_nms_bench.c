/* Phone-CPU baseline: time each real single-node NonMaxSuppression model
 * (make_nms_single_node_models.py) in ONNX Runtime's CPU EP on the phone itself, via the C API
 * from the stock onnxruntime-android AAR's arm64 libonnxruntime.so -- the op's actual current
 * execution path in this pipeline. Also checks each output against the host ORT reference
 * ({level,class}_ref.bin from gen_nms_test_data.py), so the phone's ORT build is confirmed to
 * select the same boxes. Prints per-group sums of per-call medians, 1 thread and default threads. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"
static const OrtApi* g;
#define CK(x) do { OrtStatus* s_ = (x); if (s_) { fprintf(stderr, "%s\n", g->GetErrorMessage(s_)); exit(1); } } while (0)
static void* slurp(const char* p, size_t* n) { FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); } fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* b = malloc(*n + 16); if (*n && fread(b, 1, *n, f) != *n) exit(1); fclose(f); return b; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }
int main(void) {
  g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env; CK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "b", &env));
  OrtMemoryInfo* mi; CK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  size_t n_; int* ref[2] = {slurp("level_ref.bin", &n_), slurp("class_ref.bin", &n_)}; long ro[2] = {0, 0};
  FILE* cf = fopen("ort_calls.txt", "r"); char grp[16]; int idx, n, nsel, bad = 0;
  double tot[2][2] = {{0, 0}, {0, 0}}; int calls[2] = {0, 0};
  while (fscanf(cf, "%15s %d %d %d", grp, &idx, &n, &nsel) == 4) {
    int gi = strcmp(grp, "level") ? 1 : 0; char p[64]; size_t bn, sn;
    snprintf(p, sizeof p, "%s_call%d_boxes.bin", grp, idx); void* b = slurp(p, &bn);
    snprintf(p, sizeof p, "%s_call%d_scores.bin", grp, idx); void* s = slurp(p, &sn);
    int64_t bs[3] = {1, n, 4}, ss[3] = {1, 1, n};
    OrtValue* in[2];
    CK(g->CreateTensorWithDataAsOrtValue(mi, b, bn, bs, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]));
    CK(g->CreateTensorWithDataAsOrtValue(mi, s, sn, ss, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]));
    const char* inames[2] = {"boxes", "scores"}; const char* onames[1] = {"sel"};
    for (int ti = 0; ti < 2; ti++) {
      OrtSessionOptions* so; CK(g->CreateSessionOptions(&so));
      CK(g->SetIntraOpNumThreads(so, ti == 0 ? 1 : 0));
      snprintf(p, sizeof p, "%s_call%d.onnx", grp, idx);
      OrtSession* sess; CK(g->CreateSession(env, p, so, &sess));
      double ts[15];
      for (int k = 0; k < 16; k++) {
        OrtValue* y = NULL; double t0 = now_ms();
        CK(g->Run(sess, NULL, inames, (const OrtValue* const*)in, 2, onames, 1, &y));
        if (k) ts[k - 1] = now_ms() - t0;
        if (k == 0 && ti == 0) {
          OrtTensorTypeAndShapeInfo* info; CK(g->GetTensorTypeAndShape(y, &info));
          size_t cnt; CK(g->GetTensorShapeElementCount(info, &cnt)); g->ReleaseTensorTypeAndShapeInfo(info);
          int64_t* d; CK(g->GetTensorMutableData(y, (void**)&d));
          int ok = (int)(cnt / 3) == nsel;
          for (int j = 0; ok && j < nsel; j++) ok = d[3 * j + 2] == ref[gi][ro[gi] + j];
          if (!ok) { printf("%s call%d: phone ORT output differs from host ORT reference\n", grp, idx); bad = 1; }
        }
        g->ReleaseValue(y);
      }
      qsort(ts, 15, sizeof ts[0], cmpd); tot[gi][ti] += ts[7];
      g->ReleaseSession(sess); g->ReleaseSessionOptions(so);
    }
    ro[gi] += nsel; calls[gi]++;
    g->ReleaseValue(in[0]); g->ReleaseValue(in[1]); free(b); free(s);
  }
  for (int gi = 0; gi < 2; gi++)
    printf("phone ORT %s: %d calls, sum of medians 1thr=%.3fms default=%.3fms\n", gi ? "class" : "level", calls[gi], tot[gi][0], tot[gi][1]);
  puts(bad ? "phone ORT != host ORT: FAIL" : "phone ORT == host ORT reference on all calls");
  return bad;
}

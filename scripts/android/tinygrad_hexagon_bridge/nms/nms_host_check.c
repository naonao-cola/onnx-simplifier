/* Host (x86) check of nms_kernel.h against ONNX Runtime's real outputs: both kernels, every call.
 * Build: clang-19 -O2 -ffp-contract=off -o hostcheck nms_host_check.c; run: ./hostcheck DATA_DIR */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nms_kernel.h"
static void* slurp(const char* p) {
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  void* b = aligned_alloc(128, (n + 127) / 128 * 128 + 128); if (fread(b, 1, n, f) != (size_t)n) exit(1); fclose(f); return b;
}
int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  const char* groups[2] = {"level", "class"};
  int bad = 0;
  for (int g = 0; g < 2; g++) {
    char p[512];
    snprintf(p, sizeof p, "%s/%s_boxes.bin", dir, groups[g]); float* boxes = slurp(p);
    snprintf(p, sizeof p, "%s/%s_scores.bin", dir, groups[g]); float* scores = slurp(p);
    snprintf(p, sizeof p, "%s/%s_ref.bin", dir, groups[g]); int* ref = slurp(p);
    snprintf(p, sizeof p, "%s/%s_calls.txt", dir, groups[g]); FILE* cf = fopen(p, "r");
    int n, mo, nsel, i = 0, match[2] = {0, 0}; float thr; long bo = 0, ro = 0;
    while (fscanf(cf, "%d %f %d %d", &n, &thr, &mo, &nsel) == 4) {
      int *sel = malloc(4 * (n + 1)), *ord = malloc(4 * (n + 1)), *tmp = malloc(4 * (n + 1));
      float* soa = aligned_alloc(128, 4 * NMS_SOA * ((n + 63) & ~63) + 128);
      for (int k = 0; k < 2; k++) {
        int m = k ? nms_hvx(boxes + 4 * bo, scores + bo, n, thr, mo, sel, ord, tmp, soa)
                  : nms_scalar(boxes + 4 * bo, scores + bo, n, thr, mo, sel, ord, tmp);
        int ok = m == nsel && !memcmp(sel, ref + ro, 4 * m);
        match[k] += ok;
        if (!ok) { printf("%s call%d n=%d %s MISMATCH got %d want %d\n", groups[g], i, n, k ? "hvx" : "scalar", m, nsel); bad = 1; }
      }
      free(sel); free(ord); free(tmp); free(soa);
      bo += n; ro += nsel; i++;
    }
    printf("%s: %d calls, exact match scalar=%d/%d hvx-path=%d/%d\n", groups[g], i, match[0], i, match[1], i);
  }
  puts(bad ? "FAIL" : "PASS"); return bad;
}

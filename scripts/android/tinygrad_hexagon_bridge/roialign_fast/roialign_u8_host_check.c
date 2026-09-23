/* Host (x86) check of roialign_u8_kernel.h on capture_merged_io.py's real captures: builds the job
 * list from the per-level RoIs + destination rows, runs the merged uint8 kernel (optionally with the
 * locality sort), and compares every output byte with QuantizeLinear(ORT fp32 RoiAlign) merged by
 * ScatterND -- the exact bytes the e2e pipeline's CPU steps produce today.
 * Exact-match criterion: identical bytes. Reported: fraction of exact bytes, max |diff| in LSBs, and
 * how many rows have any diff.
 * Build: clang-19 -O2 -o u8check roialign_u8_host_check.c -lm ; run: ./u8check CAPTURE_DIR [sort] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "roialign_u8_kernel.h"

static void* slurp(const char* dir, const char* name, long* n) {
  char p[1024];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void* b = aligned_alloc(128, (*n + 127) / 128 * 128 + 128);
  if (fread(b, 1, *n, f) != (size_t)*n) { perror(p); exit(1); }
  fclose(f);
  return b;
}

typedef struct { int H, W, C, z; float s, sp; } map_meta;

int main(int argc, char** argv) {
  const char* dir = argv[1];
  int sort = argc > 2 && !strcmp(argv[2], "sort");
  char path[1024];
  snprintf(path, sizeof path, "%s/meta.txt", dir);
  FILE* mf = fopen(path, "r");
  if (!mf) { perror(path); return 1; }
  map_meta mm[4];
  int nlev[2][4] = {{0}}, N[2] = {0}, OH[2] = {0}, sr[2] = {0}, zo[2] = {0};
  float so[2] = {0};
  char line[256];
  while (fgets(line, sizeof line, mf)) {
    int k, a, b, c, d; float x, y;
    char tag[16];
    if (sscanf(line, "map %d %d %d %d %f %d %f", &k, &a, &b, &c, &x, &d, &y) == 7) {
      mm[k] = (map_meta){a, b, c, d, x, y};
    } else if (sscanf(line, "%15s %d %d", tag, &k, &a) == 3 && strstr(tag, "_level")) {
      nlev[tag[0] == 'm'][k] = a;
    } else if (sscanf(line, "%15s %d %d %d %d %f %d", tag, &a, &b, &c, &d, &x, &k) == 7) {
      int t = tag[0] == 'm';
      N[t] = a; OH[t] = b; sr[t] = d; so[t] = x; zo[t] = k;
    }
  }
  int bad = 0;
  for (int t = 0; t < 2; t++) {
    const char* tag = t ? "mask" : "box";
    const int C = mm[0].C;
    ru8_level_t lv[4];
    for (int k = 0; k < 4; k++) {
      long n; char nm[64];
      snprintf(nm, sizeof nm, "l%d_u8.bin", k);
      lv[k] = (ru8_level_t){(const uint8_t*)slurp(dir, nm, &n), mm[k].H, mm[k].W, mm[k].sp, mm[k].z, 0, 0};
      ru8_requant_params(mm[k].s, so[t], sr[t] * sr[t], &lv[k].mult, &lv[k].shift);
    }
    ru8_job_t* jobs = malloc(sizeof(ru8_job_t) * (N[t] + 1));
    int nj = 0;
    for (int k = 0; k < 4; k++) {
      if (!nlev[t][k]) continue;
      long n; char nm[64];
      snprintf(nm, sizeof nm, "%s_rois%d.bin", tag, k); float* rois = slurp(dir, nm, &n);
      snprintf(nm, sizeof nm, "%s_rows%d.bin", tag, k); int32_t* rows = slurp(dir, nm, &n);
      for (int i = 0; i < nlev[t][k]; i++) {
        jobs[nj].level = k; jobs[nj].row = rows[i];
        memcpy(jobs[nj].box, rois + 4 * i, 16);
        nj++;
      }
    }
    if (sort) ru8_sort_jobs(jobs, nj);
    long rowb = (long)OH[t] * OH[t] * C, total = rowb * N[t];
    uint8_t* out = aligned_alloc(128, total + 128);
    memset(out, 0xAA, total);
    ru8_run_jobs(lv, C, jobs, nj, OH[t], OH[t], sr[t], zo[t], 0, out);
    long n; char nm[64];
    snprintf(nm, sizeof nm, "%s_ref_u8.bin", tag);
    uint8_t* ref = slurp(dir, nm, &n);
    if (n != total) { printf("%s: size mismatch %ld vs %ld\n", tag, n, total); return 1; }
    long exact = 0; int maxd = 0; long rows_diff = 0; long hist[4] = {0};
    for (long r = 0; r < N[t]; r++) {
      int rd = 0;
      for (long i = 0; i < rowb; i++) {
        int d = abs((int)out[r * rowb + i] - (int)ref[r * rowb + i]);
        if (!d) exact++; else rd = 1;
        if (d > maxd) maxd = d;
        hist[d > 3 ? 3 : d]++;
      }
      rows_diff += rd;
    }
    printf("%s: N=%d jobs=%d%s exact=%.6f%% (%ld/%ld) maxdiff=%d LSB | diff histogram 0:%ld 1:%ld 2:%ld >=3:%ld | rows with any diff %ld\n",
           tag, N[t], nj, sort ? " sorted" : "", 100.0 * exact / total, exact, total, maxd, hist[0], hist[1], hist[2], hist[3], rows_diff);
    if (maxd > 1) bad = 1;
  }
  puts(bad ? "FAIL" : "PASS");
  return bad;
}

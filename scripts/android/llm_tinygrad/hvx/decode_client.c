/* ARM64 client for the whole-model DSP decode (decode_impl.c): uploads the int8 packed weights and fp32 tables once,
 * then runs every prompt token by token through one FastRPC call per token (the prompt too: no separate prefill),
 * free-running and teacher-forced like eval_decoder.py. Writes <out>/{free,forced}/gen_<i>.bin (int32[32]) and
 * logits_<i>.bin (fp32 [9][V]); prints per-token wall / DSP / GEMV time.
 * Usage: decode_client <uri> <a16> <nthreads> <blk4> <vec> <turbo> <out> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "decode_rpc.h"

#define H 576
#define V 49152
#define NGEN 32
#define NPROMPT 10

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static long fsize(const char* p) { struct stat s; return stat(p, &s) ? -1 : (long)s.st_size; }
static void* slurp(const char* p, long* n) {
  *n = fsize(p);
  void* b = *n > 0 ? malloc(*n) : 0;
  FILE* f = fopen(p, "rb");
  if (!b || !f || fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "load %s failed\n", p); exit(1); }
  fclose(f);
  return b;
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "usage: %s uri a16 nthreads blk4 vec turbo out\n", argv[0]); return 1; }
  const char* uri = argv[1];
  int a16 = atoi(argv[2]), nt = atoi(argv[3]), b4 = atoi(argv[4]), vec = atoi(argv[5]), turbo = atoi(argv[6]);
  const char* out = argv[7];
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (decode_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  long nw, nf, ne;
  unsigned char* W = slurp("W.bin", &nw);
  unsigned char* F = slurp("F.bin", &nf);
  __fp16* E = slurp("emb.f16.bin", &ne);
  double t0 = now_ms();
  if (decode_rpc_alloc(h, (int)nw, (int)nf)) { puts("alloc failed"); return 1; }
  const long CH = 8 << 20;
  unsigned char* buf = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, CH);
  for (int part = 0; part < 2; part++) {
    long n = part ? nf : nw;
    unsigned char* src = part ? F : W;
    for (long o = 0; o < n; o += CH) {
      long c = n - o < CH ? n - o : CH;
      memcpy(buf, src + o, c);
      if (decode_rpc_load(h, part, (int)o, buf, (int)c)) { puts("load failed"); return 1; }
    }
  }
  rpcmem_free(buf);
  int vrc = -1;
  if (decode_rpc_setup(h, a16, nt, b4, vec, turbo, &vrc)) { puts("setup failed"); return 1; }
  printf("upload %.0f ms (%ld MB), a16=%d threads=%d blk4=%d vec=%d turbo=%d vote=%d\n", now_ms() - t0, (nw + nf) >> 20, a16, nt, b4, vec, turbo, vrc);
  float* emb = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, H * sizeof(float));
  float* lg = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, V * sizeof(float));
  static double wall[NPROMPT * (NGEN + 32) * 2], dsp[sizeof wall / sizeof wall[0]], gv[sizeof wall / sizeof wall[0]];
  int nt_ = 0, nl_ = 0, nw_ = 0;
  static double prof_sum[8], wl[sizeof wall / sizeof wall[0]];
  double pre_ms = 0; int pre_n = 0;
  for (int forced = 0; forced < 2; forced++) {
    char d[256];
    snprintf(d, sizeof d, "%s/%s", out, forced ? "forced" : "free");
    mkdir(out, 0755); mkdir(d, 0755);
    for (int i = 0; i < NPROMPT; i++) {
      char p[256];
      long npb, nfb;
      snprintf(p, sizeof p, "prompt_%d.bin", i);
      int* pr = slurp(p, &npb);
      snprintf(p, sizeof p, "force_%d.bin", i);
      int* fo = slurp(p, &nfb);
      int n = (int)(npb / 4), gen[NGEN];
      float* keep = malloc(sizeof(float) * 9 * V);
      int nkeep = 0;
      for (int s = -n + 1; s < NGEN; s++) { /* s<=0: prompt tokens; s>=1: generated */
        int pos = n - 1 + s, tok;
        if (s <= 0) tok = pr[n - 1 + s];
        else tok = forced ? fo[s - 1] : gen[s - 1];
        for (int k = 0; k < H; k++) emb[k] = (float)E[(size_t)tok * H + k];
        uint64 du = 0, gu = 0, pf[8] = {0};
        int want = s >= 0 && nkeep < 9, am = -1; /* logits back only for the steps the accuracy check keeps */
        double t = now_ms();
        int rc = decode_rpc_step(h, pos, emb, H, lg, want ? V : 0, &am, &du, &gu, pf, 8);
        if (s >= 1) for (int k = 0; k < 8; k++) prof_sum[k] += pf[k];
        t = now_ms() - t;
        if (rc) { printf("step rc=0x%x at prompt %d pos %d\n", rc, i, pos); return 1; }
        if (s < 0) { pre_ms += t; pre_n++; continue; }
        gen[s] = am;
        if (want) memcpy(keep + (size_t)nkeep++ * V, lg, sizeof(float) * V);
        if (s >= 1) { wall[nt_] = t; dsp[nt_] = du / 1e3; gv[nt_] = gu / 1e3; nl_ += !want; if (!want) wl[nw_++] = t; nt_++; }
      }
      snprintf(p, sizeof p, "%s/gen_%d.bin", d, i);
      FILE* f = fopen(p, "wb"); fwrite(gen, 4, NGEN, f); fclose(f);
      snprintf(p, sizeof p, "%s/logits_%d.bin", d, i);
      f = fopen(p, "wb"); fwrite(keep, sizeof(float), (size_t)nkeep * V, f); fclose(f);
      free(keep); free(pr); free(fo);
    }
  }
  qsort(wl, nw_, sizeof(double), cmpd);
  printf("generation steps without logits back (argmax on the DSP) %d: wall median %.2f ms (%.1f tok/s)\n", nw_, wl[nw_ / 2], 1e3 / wl[nw_ / 2]);
  qsort(wall, nt_, sizeof(double), cmpd); qsort(dsp, nt_, sizeof(double), cmpd); qsort(gv, nt_, sizeof(double), cmpd);
  printf("decode steps %d: wall median %.2f ms (%.1f tok/s), min %.2f | DSP median %.2f ms | GEMV median %.2f ms | prompt steps %.2f ms avg\n",
         nt_, wall[nt_ / 2], 1e3 / wall[nt_ / 2], wall[0], dsp[nt_ / 2], gv[nt_ / 2], pre_ms / (pre_n ? pre_n : 1));
  printf("per-token phases (us, thread 0 incl. waits): norm+quant %.0f | qkv deq+rope %.0f | attention %.0f | att quant %.0f | "
         "o/down deq+norm %.0f | swiglu+quant %.0f | head deq %.0f\n", prof_sum[0] / nt_, prof_sum[1] / nt_, prof_sum[2] / nt_,
         prof_sum[3] / nt_, prof_sum[4] / nt_, prof_sum[5] / nt_, prof_sum[6] / nt_);
  decode_rpc_close(h);
  return 0;
}

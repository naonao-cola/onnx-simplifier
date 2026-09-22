/* From-scratch native ARM64 FastRPC client: proves TVM's Python/C++ RPC stack is NOT required
 * to drive a custom Hexagon skel. Talks to mini_rpc.so (our own, non-TVM skel, built by
 * ourselves via qaic) purely through libcdsprpc.so's remote_handle64_open/invoke/close, using
 * the qaic-generated stub (mini_rpc_stub.c) for marshaling. No tvm.rpc, no tvm.contrib.hexagon,
 * no MinRPC, no libhexagon_rpc_skel.so anywhere in this binary or its dependency graph. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mini_rpc.h"

#define GEMM_A_LEN 3481600
#define GEMM_B_LEN 16384
#define GEMM_C_LEN 55705600
#define GEMM_A_PATH "/data/local/tmp/native_transport/gemm_a.bin"
#define GEMM_B_PATH "/data/local/tmp/native_transport/gemm_bp.bin"
#define GEMM_C_PATH "/data/local/tmp/native_transport/gemm_c_out.bin"

#define REQUANT_A_LEN 8000000
#define REQUANT_C_LEN 2000000
#define REQUANT_A_PATH "/data/local/tmp/native_transport/requant_a.bin"
#define REQUANT_C_PATH "/data/local/tmp/native_transport/requant_c_out.bin"

#define SUBGRAPH_A_LEN 3527056
#define SUBGRAPH_C_LEN 3481600
#define SUBGRAPH_A_PATH "/data/local/tmp/native_transport/subgraph_image_padded.bin"
#define SUBGRAPH_C_PATH "/data/local/tmp/native_transport/subgraph_maxpool_out.bin"

#define SMALL_SUBGRAPH_A_LEN 5776
#define SMALL_SUBGRAPH_C_LEN 4096
#define SMALL_SUBGRAPH_A_PATH "/data/local/tmp/native_transport/small_image_padded.bin"
#define SMALL_SUBGRAPH_C_PATH "/data/local/tmp/native_transport/small_maxpool_out.bin"

static unsigned char* read_file_exact(const char* path, int expect_len) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  unsigned char* buf = malloc(expect_len);
  size_t n = fread(buf, 1, expect_len, f);
  fclose(f);
  if ((int)n != expect_len) { free(buf); return NULL; }
  return buf;
}

int main(int argc, char** argv) {
  const char* uri = argv[1];  /* e.g. "file:///data/local/tmp/native_transport/mini_rpc.so?mini_rpc_skel_handle_invoke&_dom=cdsp" */

  /* The missing piece: unsigned .so loading on the CDSP is refused (AEE_ECONNREFUSED) unless
   * this process explicitly opts in per-session first. TVM's own launcher/session code
   * (apps/hexagon_launcher/launcher_android.cc, src/runtime/hexagon/rpc/android/session.cc)
   * does exactly this before its first hexagon_rpc_open() call. */
  struct remote_rpc_control_unsigned_module unsigned_pd;
  unsigned_pd.domain = CDSP_DOMAIN_ID;
  unsigned_pd.enable = 1;
  int urc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_pd, sizeof(unsigned_pd));
  printf("enable_unsigned_pd rc=%d\n", urc);

  remote_handle64 h = 0;
  int rc = mini_rpc_open(uri, &h);
  if (rc != 0) {
    printf("open failed rc=%d\n", rc);
    return 1;
  }
  printf("open OK, handle=0x%llx\n", (unsigned long long)h);

  int32_t result = -1;
  rc = mini_rpc_run_add(h, 40, 2, &result);
  printf("run_add(40,2) rc=%d result=%d\n", rc, result);

  unsigned char a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  unsigned char b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  unsigned char c[8] = {0};
  rc = mini_rpc_run_kernel(h, a, 8, b, 8, c, 8);
  printf("run_kernel (placeholder byte-add) rc=%d out=", rc);
  for (int i = 0; i < 8; i++) printf("%d ", c[i]);
  printf("\n");

  /* Real hex_gemm_kernel.py-generated vrmpy GEMM at the flagship Mask R-CNN backbone shape
   * (cin=64,cout=256,m=54400) -- only runs if gen_gemm_test_data.py's output files were pushed
   * to the device first; skipped gracefully otherwise so the base PoC still works standalone. */
  unsigned char* ga = read_file_exact(GEMM_A_PATH, GEMM_A_LEN);
  unsigned char* gb = read_file_exact(GEMM_B_PATH, GEMM_B_LEN);
  if (ga && gb) {
    unsigned char* gc = malloc(GEMM_C_LEN);
    printf("running real hex_gemm kernel (cin=64,cout=256,m=54400)...\n");
    int grc = mini_rpc_run_kernel(h, ga, GEMM_A_LEN, gb, GEMM_B_LEN, gc, GEMM_C_LEN);
    printf("run_gemm_kernel rc=%d\n", grc);
    if (grc == 0) {
      FILE* out = fopen(GEMM_C_PATH, "wb");
      fwrite(gc, 1, GEMM_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes) -- verify against a numpy reference host-side\n", GEMM_C_PATH, GEMM_C_LEN);
    }
    free(gc);
  } else {
    printf("gemm test data not found at %s / %s, skipping real-kernel test\n", GEMM_A_PATH, GEMM_B_PATH);
  }
  free(ga);
  free(gb);

  /* Real hex_requantize_kernel.py-generated requantize, n=2,000,000 (a real-scale, RPC-safe
   * slice of the stem conv's output) -- only runs if gen_requantize_test_data.py's output was
   * pushed first; skipped gracefully otherwise. */
  unsigned char* ra = read_file_exact(REQUANT_A_PATH, REQUANT_A_LEN);
  if (ra) {
    unsigned char* rc_out = malloc(REQUANT_C_LEN);
    printf("running real hex_requantize kernel (n=2000000)...\n");
    int rrc = mini_rpc_run_kernel(h, ra, REQUANT_A_LEN, ra, 0, rc_out, REQUANT_C_LEN);
    printf("run_requantize_kernel rc=%d\n", rrc);
    if (rrc == 0) {
      FILE* out = fopen(REQUANT_C_PATH, "wb");
      fwrite(rc_out, 1, REQUANT_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes) -- verify against the numpy reference host-side\n", REQUANT_C_PATH, REQUANT_C_LEN);
    }
    free(rc_out);
  } else {
    printf("requantize test data not found at %s, skipping\n", REQUANT_A_PATH);
  }
  free(ra);

  /* Stage 2: the fused stem7x7 -> bias_add -> requantize -> layout_transform -> maxpool
   * subgraph, real backbone.onnx weights/bias/scales, real quantized+padded image in -- only
   * runs if the image was pushed first; skipped gracefully otherwise. Times the on-device call
   * (the only thing crossing the RPC boundary is this input and the final maxpool output --
   * every intermediate activation stays on-device, see subgraph_driver.c). */
  /* Diagnostic: a tiny-scale (ih=iw=32) copy of the same pipeline, to isolate whether a
   * full-scale failure is a real per-process memory-size limit vs. a logic bug -- run before
   * the full-scale test so its result is available even if the full-scale one crashes. */
  unsigned char* ssa = read_file_exact(SMALL_SUBGRAPH_A_PATH, SMALL_SUBGRAPH_A_LEN);
  if (ssa) {
    unsigned char* ssc = malloc(SMALL_SUBGRAPH_C_LEN);
    printf("running SMALL-scale (32x32) subgraph diagnostic...\n");
    int ssrc = mini_rpc_run_kernel(h, ssa, SMALL_SUBGRAPH_A_LEN, ssa, 0, ssc, SMALL_SUBGRAPH_C_LEN);
    printf("run_small_subgraph rc=%d\n", ssrc);
    if (ssrc == 0) {
      FILE* out = fopen(SMALL_SUBGRAPH_C_PATH, "wb");
      fwrite(ssc, 1, SMALL_SUBGRAPH_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes)\n", SMALL_SUBGRAPH_C_PATH, SMALL_SUBGRAPH_C_LEN);
    }
    free(ssc);
  } else {
    printf("small subgraph test data not found at %s, skipping\n", SMALL_SUBGRAPH_A_PATH);
  }
  free(ssa);

  unsigned char* sa = read_file_exact(SUBGRAPH_A_PATH, SUBGRAPH_A_LEN);
  if (sa) {
    unsigned char* sc = malloc(SUBGRAPH_C_LEN);
    printf("running real stem7x7->bias_add->requantize->maxpool subgraph...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int src = mini_rpc_run_kernel(h, sa, SUBGRAPH_A_LEN, sa, 0, sc, SUBGRAPH_C_LEN);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("run_subgraph rc=%d wall_ms=%.3f\n", src, ms);
    if (src == 0) {
      FILE* out = fopen(SUBGRAPH_C_PATH, "wb");
      fwrite(sc, 1, SUBGRAPH_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes) -- verify against the ORT reference host-side\n", SUBGRAPH_C_PATH, SUBGRAPH_C_LEN);
    }
    free(sc);
  } else {
    printf("subgraph test data not found at %s, skipping\n", SUBGRAPH_A_PATH);
  }
  free(sa);

  mini_rpc_close(h);
  printf("closed OK\n");
  return (result == 42 && rc == 0) ? 0 : 2;
}

/* FastRPC skel around a whole tinygrad-generated graph (qdq_net.py: k<n>.c + graph.h): one call runs every kernel in order
 * on one HVX + HMX worker thread (runtime from hmx_gemm/hmx_runtime.h). a = the graph input, b = the constants blob
 * (graph.h's offsets), c = the output; t[0] = the kernels' time over `iters` inferences (the copies in/out excluded). */
#include <stdlib.h>
#include <string.h>
#include "tg_hmx_rpc.h"
#include "hmx_runtime.h"
#include "graph.h"
extern unsigned long long HAP_perf_get_time_us(void);

unsigned char* __hmx_vtcm;
unsigned int __hmx_gen;

int tg_hmx_rpc_open(const char* uri, remote_handle64* h) { *h = (remote_handle64)(uintptr_t)malloc(1); return 0; }
int tg_hmx_rpc_close(remote_handle64 h) { free((void*)(uintptr_t)h); return 0; }

typedef struct { hmx_rt_t* rt; int iters; unsigned char** B; uint64* t; int* codes; } job_t;
#define STACK_SIZE (256 * 1024)
static char g_stack[STACK_SIZE] __attribute__((aligned(128)));

static void worker(void* p) {
  job_t* j = (job_t*)p;
  j->codes[2] = qurt_hvx_lock(QURT_HVX_MODE_128B);
  j->codes[3] = j->codes[2] ? -1 : HAP_compute_res_hmx_lock(j->rt->ctx);
  if (j->codes[3] == 0) {
    unsigned long long t0 = HAP_perf_get_time_us();
    for (int it = 0; it < j->iters; it++) g_run(j->B);
    j->t[0] = HAP_perf_get_time_us() - t0;
    HAP_compute_res_hmx_unlock(j->rt->ctx);
  }
  if (j->codes[2] == 0) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

int tg_hmx_rpc_run(remote_handle64 h, int iters, const uint8* a, int aLen, const uint8* b, int bLen, uint8* c, int cLen,
                   uint64* t, int tLen, int* codes, int codesLen) {
  if (tLen < 1 || codesLen < 7 || bLen < G_BLOB_BYTES || aLen > (int)G_BYTES[G_INPUT] || cLen > (int)G_BYTES[G_OUTPUT]) return AEE_EBADPARM;
  memset(t, 0, tLen * sizeof(uint64)); memset(codes, 0, codesLen * sizeof(int));
  codes[0] = hmx_rt_power((void*)tg_hmx_rpc_run, 1);
  if (codes[0]) return 0;
  hmx_rt_t rt;
  if (hmx_rt_acquire(&rt, G_VTCM_KB * 1024)) { codes[1] = -1; return 0; }
  codes[1] = (int)rt.ctx; codes[4] = (int)rt.vtcm_bytes;
  /* every buffer 128-byte aligned (no memalign in the DSP runtime's libc: one block, aligned by hand) */
  size_t total = 128;
  for (int i = 0; i < G_NBUF; i++) total += G_BYTES[i];
  char* raw = malloc(total);
  unsigned char* B[G_NBUF];
  if (!raw) { codes[6] = -1; hmx_rt_release(&rt); return 0; }
  unsigned char* p = (unsigned char*)(((uintptr_t)raw + 127) & ~(uintptr_t)127);
  for (int i = 0; i < G_NBUF; i++) {
    B[i] = p, p += G_BYTES[i];
    if (G_OFF[i] >= 0) memcpy(B[i], b + G_OFF[i], G_BYTES[i]);
  }
  memcpy(B[G_INPUT], a, aLen);
  __hmx_vtcm = rt.vtcm; __hmx_gen++;
  job_t j = {&rt, iters, B, t, codes};
  qurt_thread_attr_t ta; qurt_thread_attr_init(&ta);
  qurt_thread_attr_set_stack_addr(&ta, g_stack); qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
  qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
  qurt_thread_t tid; int st;
  codes[5] = qurt_thread_create(&tid, &ta, worker, &j);
  if (codes[5] == 0) qurt_thread_join(tid, &st);
  memcpy(c, B[G_OUTPUT], cLen);
  codes[6] = (int)(total >> 10);  /* KB of DSP heap */
  free(raw);
  hmx_rt_release(&rt);
  return 0;
}

/* DSP side of the TopK FastRPC skel: topk_kernel.h over a batch of independent calls. Same
 * structure as ../roialign_fast/roialign_impl.c (HAP_perf_get_time_us timing, qurt_hvx_lock per
 * thread), plus two things measured to matter here:
 *   - Scratch is allocated once per batch slot and reused: a fresh 2.6 MB malloc/free per call cost
 *     ~1-1.5 ms on the DSP heap, plus first-touch page cost in the kernel itself.
 *   - nthreads > 1: calls run one after another, but the streaming collect of a big call is split
 *     across nthreads QuRT threads (disjoint index ranges, survivors compacted back in range
 *     order, so index order -- and therefore ORT's tie order -- is preserved).
 *     nthreads <= -1: the calls themselves run concurrently, one thread per call; with
 *     nthreads = -N (N > 1) the collect of calls with n >= HYBRID_SPLIT_MIN (only the biggest FPN
 *     level here) is additionally split N ways from inside its call's thread.
 * phase_us (4 per call): scratch allocation, threshold (sample + quickselect), collect, emit. */
#include <stdlib.h>
#include "topk_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "topk_kernel.h"

extern unsigned long long HAP_perf_get_time_us(void);

AEEResult topk_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
  void* ctx = (void*)(uintptr_t)h;
  HAP_power_request_t req = {0};
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = 1;
  int r1 = HAP_power_set(ctx, &req);
  HAP_power_request_t d = {0};
  d.type = HAP_power_set_DCVS_v2;
  d.dcvs_v2.dcvs_enable = turbo ? FALSE : TRUE;
  d.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  d.dcvs_v2.set_latency = TRUE;
  d.dcvs_v2.latency = 40;
  d.dcvs_v2.set_dcvs_params = turbo ? TRUE : FALSE;
  d.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
  d.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_TURBO;
  d.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
  int r2 = HAP_power_set(ctx, &d);
  *rc = r1 ? r1 : r2;
  return 0;
}

int topk_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int topk_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

#define MAX_CALLS 8
#define MAX_THREADS 8
#define STACK_SIZE 16384
#define SPLIT_MIN 32768        /* below this, one thread collects the whole call */
#define HYBRID_SPLIT_MIN 65536 /* concurrent mode: only one call may split (one part-stack pool) */
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8)));      /* call threads */
static char part_stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8))); /* collect parts */
static uint32_t* g_scr[MAX_CALLS];
static long g_scr_words[MAX_CALLS];

static uint32_t* scratch_for(int slot, int n) {
  long need = TK_SCRATCH_WORDS(n);
  if (g_scr_words[slot] < need) {
    free(g_scr[slot]);
    g_scr[slot] = malloc(need * 4);
    g_scr_words[slot] = g_scr[slot] ? need : 0;
  }
  return g_scr[slot];
}

typedef struct {
  const float* x; float* ov; int64_t* oi; int n, k, variant, slot, nsplit, survivors, rc;
  unsigned ph[4];
} job_t;

typedef struct { const float* x; int i0, i1, variant, c; uint32_t t; uint32_t *ck, *ci; } part_t;

static void part_main(void* arg) {
  part_t* p = arg;
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  p->c = tk_collect_range(p->x, p->i0, p->i1, p->t, p->ck + p->i0, p->ci + p->i0, p->variant);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

static int spawn(qurt_thread_t* tid, char* stack, void (*fn)(void*), void* arg) {
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
  qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
  return qurt_thread_create(tid, &attr, fn, arg) == QURT_EOK ? 0 : -2;
}

/* Survivors of the whole call, split across nsplit threads; compacted into ck/ci in index order. */
static int collect_split(const float* x, int n, uint32_t t, uint32_t* ck, uint32_t* ci, int variant, int nsplit) {
  part_t parts[MAX_THREADS];
  qurt_thread_t tids[MAX_THREADS];
  int per = ((n + nsplit - 1) / nsplit + 31) & ~31, started = 0;
  for (int s = 0; s < nsplit; s++) {
    int i0 = s * per, i1 = i0 + per > n ? n : i0 + per;
    if (i0 >= i1) break;
    parts[s] = (part_t){x, i0, i1, variant, 0, t, ck, ci};
    if (spawn(&tids[s], part_stacks[s], part_main, &parts[s])) return -2;
    started++;
  }
  int c = 0;
  for (int s = 0; s < started; s++) {
    int st;
    qurt_thread_join(tids[s], &st);
    if (parts[s].c && c != parts[s].i0) {
      memmove(ck + c, ck + parts[s].i0, parts[s].c * 4);
      memmove(ci + c, ci + parts[s].i0, parts[s].c * 4);
    }
    c += parts[s].c;
  }
  return c;
}

static void run_job(job_t* j) {
  unsigned long long t0 = HAP_perf_get_time_us();
  uint32_t* scr = scratch_for(j->slot, j->n);
  unsigned long long t1 = HAP_perf_get_time_us();
  if (!scr) { j->rc = -3; return; }
  int n = j->n, k = j->k;
  uint32_t *ck = scr, *ci = scr + n, *tk = scr + 2L * n, *ti = scr + 3L * n;
  uint32_t *sk = scr + 4L * n, *tmp = sk + TK_SAMPLE, *hs = tmp + TK_SAMPLE;
  int r, m, c;
  uint32_t t = tk_threshold(j->x, n, k, sk, tmp, &r, &m);
  unsigned long long t2 = HAP_perf_get_time_us();
  for (;;) {
    c = (j->nsplit > 1 && n >= SPLIT_MIN) ? collect_split(j->x, n, t, ck, ci, j->variant, j->nsplit)
                                          : tk_collect_range(j->x, 0, n, t, ck, ci, j->variant);
    if (c < 0) { j->rc = c; return; }
    if (c >= k || t == 0) break;
    t = tk_retry(sk, tmp, m, &r);
  }
  unsigned long long t3 = HAP_perf_get_time_us();
  tk_emit(j->x, ck, ci, tk, ti, hs, c, k, j->ov, j->oi);
  unsigned long long t4 = HAP_perf_get_time_us();
  j->ph[0] = t1 - t0; j->ph[1] = t2 - t1; j->ph[2] = t3 - t2; j->ph[3] = t4 - t3;
  j->survivors = c;
  j->rc = 0;
}

static void job_main(void* arg) {
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  run_job((job_t*)arg);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

AEEResult topk_rpc_run(remote_handle64 h, const float* x, int xLen, const int32* ns, int nsLen,
                       const int32* ks, int ksLen, int32 variant, int32 nthreads, float* vals,
                       int valsLen, int64* idx, int idxLen, int32* surv, int survLen, uint32* ph, int phLen, uint64* dsp_us) {
  int nc = nsLen;
  if (nc < 1 || nc > MAX_CALLS || ksLen != nc || survLen < nc || phLen < 4 * nc) return -1;
  int nsplit = nthreads > 1 ? (nthreads > MAX_THREADS ? MAX_THREADS : nthreads) : 1;
  job_t jobs[MAX_CALLS];
  long xo = 0, oo = 0;
  for (int c = 0; c < nc; c++) {
    if (ks[c] < 0 || ks[c] > ns[c]) return -1;
    jobs[c] = (job_t){x + xo, vals + oo, (int64_t*)idx + oo, ns[c], ks[c], variant, c, nsplit, 0, 0, {0}};
    xo += ns[c];
    oo += ks[c];
  }
  if (xo > xLen || oo > valsLen || oo > idxLen) return -1;
  unsigned long long t0 = HAP_perf_get_time_us();
  if (nthreads >= 0) {
    int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
    for (int c = 0; c < nc; c++) run_job(&jobs[c]);
    if (locked) qurt_hvx_unlock();
  } else {
    qurt_thread_t tids[MAX_CALLS];
    int started = 0;
    for (int c = 0; c < nc; c++) {
      jobs[c].nsplit = (nthreads < -1 && ns[c] >= HYBRID_SPLIT_MIN) ? (-nthreads > MAX_THREADS ? MAX_THREADS : -nthreads) : 1;
      if (spawn(&tids[c], stacks[c], job_main, &jobs[c])) return -2;
      started++;
    }
    for (int c = 0; c < started; c++) { int st; qurt_thread_join(tids[c], &st); }
  }
  *dsp_us = HAP_perf_get_time_us() - t0;
  for (int c = 0; c < nc; c++) {
    if (jobs[c].rc) return jobs[c].rc;
    surv[c] = jobs[c].survivors;
    for (int q = 0; q < 4; q++) ph[4 * c + q] = jobs[c].ph[q];
  }
  return 0;
}

/* DSP side of the fused RPN FastRPC skel: rpn_kernel.h -- per-level TopK -> decode -> filter -> NMS
 * -> cap, then concat -> TopK -> gather -- in ONE call, from the objectness scores and deltas to
 * the proposal list. Only the inputs cross the RPC boundary on the way in and 1000 boxes on the way
 * out; every intermediate stays on the DSP.
 *
 * Threading reuses what each standalone skel measured to be best (../topk/topk_impl.c,
 * ../proposal_decode/pd_impl.c, ../nms/nms_impl.c), adapted to the fused shape of the work: the 5
 * levels are independent until the merge, so each level is one QuRT thread (holding an HVX
 * context) running its whole chain; P2's 163,200-score TopK collect is the one step big enough to
 * also split across threads (mode 2, the TopK skel's "hybrid"). Model state and every working set
 * (P2's 2.6 MB TopK scratch included) are allocated once in set_model and reused: the TopK work
 * found a fresh multi-MB malloc per call cost ~1-1.5 ms on the DSP heap. */
#include <stdlib.h>
#include <string.h>
#include "rpn_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
extern unsigned long long HAP_perf_get_time_us(void);
#define RPN_NOW() HAP_perf_get_time_us()
#include "rpn_kernel.h"

#define RPN_STATS (2 + 3 * RPN_NLVL + 4 * RPN_NLVL + 1 + RPN_NLVL + 4)
#define DEC_THREADS 6 /* ../proposal_decode's best: decode split by box across 6 threads */
#define MAX_SPLIT 8
#define STACK_SIZE 32768
#define PART_STACK 16384

static rpn_model g_model;
static float* g_anchors;
static const float* g_an[RPN_NLVL];
static pd_prep g_prep[RPN_NLVL];
static rpn_level_ws* g_ws[RPN_NLVL];
static uint32_t* g_tks[RPN_NLVL];
static rpn_merge_ws* g_mw;
static void* g_raw[RPN_NLVL + 1];
static int g_ready;
static char stacks[RPN_NLVL > 6 ? RPN_NLVL : 6][STACK_SIZE] __attribute__((aligned(8)));
static char part_stacks[MAX_SPLIT][PART_STACK] __attribute__((aligned(8)));

AEEResult rpn_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
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

int rpn_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int rpn_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

static void* alloc128(void** raw, size_t bytes) { /* rpn_level_ws.soa needs 128-byte alignment */
  free(*raw);
  *raw = malloc(bytes + 128);
  return *raw ? (void*)(((uintptr_t)*raw + 127) & ~(uintptr_t)127) : 0;
}

AEEResult rpn_rpc_set_model(remote_handle64 h, const float* anchors, int anchorsLen, const unsigned char* model,
                            int modelLen, int32* grid_mask) {
  if (modelLen != (int)sizeof g_model) return -1;
  memcpy(&g_model, model, sizeof g_model);
  long tot = 0;
  for (int l = 0; l < RPN_NLVL; l++) {
    if (g_model.lv[l].k > RPN_KMAX || g_model.lv[l].k > g_model.lv[l].A) return -1;
    tot += 4L * g_model.lv[l].A;
  }
  if (anchorsLen != tot) return -1;
  free(g_anchors);
  g_anchors = malloc(4 * tot);
  if (!g_anchors) return -1;
  memcpy(g_anchors, anchors, 4 * tot);
  *grid_mask = 0;
  long off = 0;
  for (int l = 0; l < RPN_NLVL; l++) {
    const rpn_level_cfg* L = &g_model.lv[l];
    g_an[l] = g_anchors + off;
    off += 4L * L->A;
    pd_prepare(&L->P, &g_prep[l]);
    pd_prepare_anchors(g_an[l], L->H, L->W, &g_prep[l]);
    *grid_mask |= g_prep[l].grid << l;
    if (!(g_ws[l] = alloc128(&g_raw[l], sizeof(rpn_level_ws)))) return -1;
    free(g_tks[l]);
    if (!(g_tks[l] = malloc(4 * TK_SCRATCH_WORDS(L->A)))) return -1;
  }
  if (!(g_mw = alloc128(&g_raw[RPN_NLVL], sizeof(rpn_merge_ws)))) return -1;
  g_ready = 1;
  return 0;
}

/* P2's TopK collect split across threads: disjoint index ranges, survivors compacted back in range
 * order, so index order (and with it ORT's tie order) is preserved -- ../topk/topk_impl.c's
 * collect_split. */
typedef struct { const float* x; int i0, i1, c; uint32_t t; uint32_t *ck, *ci; } part_t;
static void part_main(void* arg) {
  part_t* p = arg;
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  p->c = tk_collect_range(p->x, p->i0, p->i1, p->t, p->ck + p->i0, p->ci + p->i0, TK_VEC_ROT);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}
static int spawn(qurt_thread_t* tid, char* stack, int size, void (*fn)(void*), void* arg) {
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, size);
  qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
  return qurt_thread_create(tid, &attr, fn, arg) == QURT_EOK ? 0 : -2;
}
static int collect_split(void* ctx, const float* x, int n, uint32_t t, uint32_t* ck, uint32_t* ci) {
  int nsplit = *(int*)ctx;
  part_t parts[MAX_SPLIT];
  qurt_thread_t tids[MAX_SPLIT];
  int per = ((n + nsplit - 1) / nsplit + 31) & ~31, started = 0, c = 0;
  for (int s = 0; s < nsplit; s++) {
    int i0 = s * per, i1 = i0 + per > n ? n : i0 + per;
    if (i0 >= i1) break;
    parts[s] = (part_t){x, i0, i1, 0, t, ck, ci};
    if (spawn(&tids[s], part_stacks[s], PART_STACK, part_main, &parts[s])) return -2;
    started++;
  }
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

typedef struct {
  int l, src, nms, nsplit, rc, step; /* step: 0 whole chain, 1 TopK only, 2 decode+filter+NMS, 3 filter+NMS */
  const float *scores, *deltas;
  const uint8_t* nchw;
  unsigned us;
} job_t;

static void run_level(job_t* j) {
  unsigned long long t0 = HAP_perf_get_time_us();
  const rpn_level_cfg* L = &g_model.lv[j->l];
  int split = j->nsplit > 1 && L->A >= 65536;
  const float* dl = j->src ? 0 : j->deltas;
  const uint8_t* nq = j->src ? j->nchw : 0;
  if (j->step == 0)
    j->rc = rpn_level(L, &g_prep[j->l], g_an[j->l], j->scores, dl, nq, g_ws[j->l], g_tks[j->l],
                      split ? collect_split : rpn_collect_plain, split ? &j->nsplit : 0, j->nms);
  else if (j->step == 1)
    j->rc = rpn_level_topk(L, j->scores, g_ws[j->l], g_tks[j->l], split ? collect_split : rpn_collect_plain,
                           split ? &j->nsplit : 0);
  else {
    if (j->step == 2) rpn_level_decode(L, &g_prep[j->l], g_an[j->l], dl, nq, g_ws[j->l], 0, L->k);
    rpn_level_filter_nms(L, g_ws[j->l], j->nms);
    j->rc = 0;
  }
  j->us = (unsigned)(HAP_perf_get_time_us() - t0);
}
static void level_main(void* arg) {
  job_t* j = arg;
  if (j->step == 2) { /* decode is scalar: run it before taking an HVX context, so the level threads
                       * don't hold all the contexts while the others still need one for NMS */
    const rpn_level_cfg* L = &g_model.lv[j->l];
    unsigned long long t0 = HAP_perf_get_time_us();
    rpn_level_decode(L, &g_prep[j->l], g_an[j->l], j->src ? 0 : j->deltas, j->src ? j->nchw : 0, g_ws[j->l], 0, L->k);
    g_ws[j->l]->ph[1] = (unsigned)(HAP_perf_get_time_us() - t0);
    j->step = 3;
  }
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  run_level(j);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

/* phase B: decode boxes [b0, b1) of the global 0..sum(k) range (level l owns [base, base + k));
 * scalar code, so these threads take no HVX context */
typedef struct { int b0, b1; job_t* jobs; } dec_t;
static void dec_main(void* arg) {
  dec_t* d = arg;
  int base = 0;
  for (int l = 0; l < RPN_NLVL; l++) {
    const rpn_level_cfg* L = &g_model.lv[l];
    job_t* j = &d->jobs[l];
    int lo = d->b0 > base ? d->b0 : base, hi = d->b1 < base + L->k ? d->b1 : base + L->k;
    if (lo < hi)
      rpn_level_decode(L, &g_prep[l], g_an[l], j->src ? 0 : j->deltas, j->src ? j->nchw : 0, g_ws[l], lo - base, hi - base);
    base += L->k;
  }
  qurt_thread_exit(0);
}

/* run the level jobs on one thread each (with an HVX context), join */
static int run_levels_threaded(job_t* jobs, int step) {
  qurt_thread_t tids[RPN_NLVL];
  int started = 0;
  for (int l = 0; l < RPN_NLVL; l++) {
    jobs[l].step = step;
    if (spawn(&tids[l], stacks[l], STACK_SIZE, level_main, &jobs[l])) break;
    started++;
  }
  for (int l = 0; l < started; l++) { int st; qurt_thread_join(tids[l], &st); }
  if (started < RPN_NLVL) return -2;
  for (int l = 0; l < RPN_NLVL; l++) if (jobs[l].rc) return jobs[l].rc;
  return 0;
}
static int decode_split(job_t* jobs, int nthreads) {
  int total = 0;
  for (int l = 0; l < RPN_NLVL; l++) total += g_model.lv[l].k;
  dec_t parts[DEC_THREADS];
  qurt_thread_t tids[DEC_THREADS];
  int per = (total + nthreads - 1) / nthreads, started = 0;
  for (int t = 0; t < nthreads; t++) {
    parts[t] = (dec_t){t * per, (t + 1) * per < total ? (t + 1) * per : total, jobs};
    if (parts[t].b0 >= parts[t].b1) break;
    if (spawn(&tids[t], stacks[t], STACK_SIZE, dec_main, &parts[t])) return -2;
    started++;
  }
  for (int t = 0; t < started; t++) { int st; qurt_thread_join(tids[t], &st); }
  return 0;
}

AEEResult rpn_rpc_run(remote_handle64 h, const float* scores, int scoresLen, const float* deltas, int deltasLen,
                      const unsigned char* nchw, int nchwLen, int32 src, int32 mode, int32 nsplit, int32 nms,
                      float* proposals, int proposalsLen, int32* stats, int statsLen, uint64* dsp_us) {
  if (!g_ready || statsLen < RPN_STATS || proposalsLen < 4 * g_model.post_cap) return -1;
  long so = 0, dof = 0, no = 0;
  job_t jobs[RPN_NLVL];
  for (int l = 0; l < RPN_NLVL; l++) {
    const rpn_level_cfg* L = &g_model.lv[l];
    jobs[l] = (job_t){l, src, nms, (mode == 2 || mode >= 3) ? (nsplit > MAX_SPLIT ? MAX_SPLIT : nsplit) : 1, 0, 0,
                      scores + so, deltas + dof, (const uint8_t*)nchw + no, 0};
    so += L->A;
    dof += 4L * L->A;
    no += 12L * L->H * L->W;
  }
  if (scoresLen < so || (src ? nchwLen < no : deltasLen < dof)) return -1;
  unsigned long long t0 = HAP_perf_get_time_us(), ta = t0, tb = t0, tc = t0;
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0; /* this thread: mode 0 levels, and the merge */
  if (mode == 0) {
    for (int l = 0; l < RPN_NLVL; l++) { jobs[l].step = 0; run_level(&jobs[l]); }
  } else {
    if (locked) qurt_hvx_unlock(); /* free a context for the worker threads while this one waits */
    int rc = 0;
    if (mode <= 2) {
      rc = run_levels_threaded(jobs, 0);
    } else { /* phased: each step with its standalone kernel's best threading */
      rc = run_levels_threaded(jobs, 1);                         /* A: TopK, thread per level (+P2 split) */
      ta = HAP_perf_get_time_us();
      if (!rc && mode == 3) {
        rc = decode_split(jobs, DEC_THREADS);                   /* B: decode split by box */
        tb = HAP_perf_get_time_us();
        if (!rc) rc = run_levels_threaded(jobs, 3);             /* C: filter + NMS, thread per level */
      } else if (!rc) {
        tb = ta;
        rc = run_levels_threaded(jobs, 2);                      /* B+C: decode + filter + NMS per level */
      }
      tc = HAP_perf_get_time_us();
    }
    locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
    if (rc) { if (locked) qurt_hvx_unlock(); return rc; }
  }
  for (int l = 0; l < RPN_NLVL; l++) if (jobs[l].rc) { if (locked) qurt_hvx_unlock(); return jobs[l].rc; }
  unsigned long long tm = HAP_perf_get_time_us();
  int total, kp = rpn_merge(&g_model, g_ws, g_mw, proposals, &total);
  unsigned long long te = HAP_perf_get_time_us();
  if (locked) qurt_hvx_unlock();
  *dsp_us = te - t0;
  if (kp < 0) return -3;
  int* s = stats;
  *s++ = kp;
  *s++ = total;
  for (int l = 0; l < RPN_NLVL; l++) { *s++ = g_ws[l]->nf; *s++ = g_ws[l]->ns; *s++ = g_ws[l]->m; }
  for (int l = 0; l < RPN_NLVL; l++) for (int q = 0; q < 4; q++) *s++ = (int)g_ws[l]->ph[q];
  *s++ = (int)(te - tm);
  for (int l = 0; l < RPN_NLVL; l++) *s++ = (int)jobs[l].us;
  *s++ = (int)(ta - t0); *s++ = (int)(tb - ta); *s++ = (int)(tc - tb); *s++ = (int)(te - tm);
  return 0;
}

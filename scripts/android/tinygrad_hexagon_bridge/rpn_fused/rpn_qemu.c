/* The whole fused RPN span (rpn_kernel.h) under qemu-hexagon-static, per image: runs both delta
 * sources with both NMS kernels and checks each level's NMS selection and the final proposal list
 * byte for byte against ORT; prints QEMU's executed-instruction count (control register 21).
 * (plus one run with the provably no-op w/h filter computed instead of skipped)
 * Build: clang-19 --target=hexagon -mcpu=hexagonv73 -O2 -ffp-contract=off -static -nostdlib
 *        -ffreestanding -fuse-ld=lld -o rpnq rpn_qemu.c
 * Run:   qemu-hexagon-static rpnq DATA/ DATA/IMG/     (both with the trailing slash)
 * DATA/model.bin and DATA/lN_anchors.bin are written by rpn_host_check. No -mhvx: qemu 8.2 can't
 * execute the qfloat HVX ops nms_hvx uses on the DSP (same as ../nms/nms_qemu.c), so this runs
 * nms_kernel.h's portable classification + the same exact recheck, and TopK's vector code lowered
 * to scalar Hexagon; the qfloat path itself is checked on the phone. Freestanding, raw trap0
 * syscalls, as in ../topk/topk_qemu.c. */
#include <stddef.h>
void* memcpy(void* d, const void* s, size_t n) { char* a = d; const char* b = s; while (n--) *a++ = *b++; return d; }
void* memmove(void* d, const void* s, size_t n) {
  char* a = d; const char* b = s;
  if (a < b) while (n--) *a++ = *b++;
  else { a += n; b += n; while (n--) *--a = *--b; }
  return d;
}
void* memset(void* d, int c, size_t n) { char* a = d; while (n--) *a++ = (char)c; return d; }
int memcmp(const void* x, const void* y, size_t n) { const unsigned char *a = x, *b = y; for (; n; n--, a++, b++) if (*a != *b) return *a - *b; return 0; }
/* Hexagon has no integer divide instruction; the DSP build gets these from libgcc.so, this
 * freestanding one needs its own (TopK's threshold rank divides by a runtime n). */
unsigned __hexagon_udivsi3(unsigned a, unsigned b) {
  unsigned q = 0, r = 0;
  for (int i = 31; i >= 0; i--) { r = (r << 1) | ((a >> i) & 1); if (r >= b) { r -= b; q |= 1u << i; } }
  return q;
}
unsigned __hexagon_umodsi3(unsigned a, unsigned b) { return a - __hexagon_udivsi3(a, b) * b; }
int __hexagon_divsi3(int a, int b) {
  unsigned ua = a < 0 ? -(unsigned)a : (unsigned)a, ub = b < 0 ? -(unsigned)b : (unsigned)b, q = __hexagon_udivsi3(ua, ub);
  return (a < 0) != (b < 0) ? -(int)q : (int)q;
}
int __hexagon_modsi3(int a, int b) { return a - __hexagon_divsi3(a, b) * b; }
#include "rpn_kernel.h"
static long sys6(long a, long b, long c, long d, long e, long f, long n) {
  long r; __asm__ volatile("r0=%1;r1=%2;r2=%3;r3=%4;r4=%5;r5=%6;r6=%7;trap0(#1);%0=r0"
    : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(n)
    : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "memory"); return r; }
static unsigned inscount(void) { unsigned r; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r"(r) : : "r0"); return r; }
static void* map(long bytes) { return (void*)sys6(0, (bytes + 4095) & ~4095L, 3, 0x22, -1, 0, 222); }
static long loadf(const char* p, void* b, long cap) {
  int fd = sys6(-100, (long)p, 0, 0, 0, 0, 56); long o = 0;
  if (fd < 0) return -1;
  while (o < cap) { long n = sys6(fd, (long)((char*)b + o), cap - o, 0, 0, 0, 63); if (n <= 0) break; o += n; }
  return o; }
static void puts_(const char* s) { int n = 0; while (s[n]) n++; sys6(1, (long)s, n, 0, 0, 0, 64); }
static void putu(unsigned long v) { char b[24]; int i = 23; b[i] = 0; do { b[--i] = '0' + v % 10; v /= 10; } while (v); puts_(b + i); }
static void cat3(char* d, const char* a, const char* b, const char* c) { while (*a) *d++ = *a++; while (*b) *d++ = *b++; while (*c) *d++ = *c++; *d = 0; }
static void* lvl(const char* dir, int l, const char* name, long cap, long* got) {
  char p[512], s[4] = {'l', (char)('0' + l), '_', 0}, f[64];
  cat3(f, s, name, ".bin"); cat3(p, dir, f, "");
  void* b = map(cap); long n = loadf(p, b, cap); if (got) *got = n;
  if (n < 0) { puts_("missing "); puts_(p); puts_("\n"); sys6(1, 0, 0, 0, 0, 0, 93); }
  return b; }
void __attribute__((noreturn)) main_(long* sp) {
  char** argv = (char**)(sp + 1);
  const char *mdir = argv[1], *idir = argv[2];
  char p[512];
  rpn_model* M = map(sizeof(rpn_model));
  cat3(p, mdir, "model.bin", ""); loadf(p, M, sizeof *M);
  float* an[RPN_NLVL]; float* sc[RPN_NLVL]; float* dl[RPN_NLVL]; uint8_t* nq[RPN_NLVL]; int32_t* rs[RPN_NLVL]; long rn[RPN_NLVL];
  pd_prep* prep = map(sizeof(pd_prep) * RPN_NLVL);
  for (int l = 0; l < RPN_NLVL; l++) {
    const rpn_level_cfg* L = &M->lv[l];
    an[l] = lvl(mdir, l, "anchors", 16L * L->A, 0);
    sc[l] = lvl(idir, l, "scores", 4L * L->A, 0);
    dl[l] = lvl(idir, l, "deltas", 16L * L->A, 0);
    nq[l] = lvl(idir, l, "nchw_q", 12L * L->H * L->W, 0);
    rs[l] = lvl(idir, l, "nms_sel", 4L * RPN_KMAX, &rn[l]); rn[l] /= 4;
    pd_prepare(&L->P, &prep[l]); pd_prepare_anchors(an[l], L->H, L->W, &prep[l]);
  }
  long nprop; cat3(p, idir, "proposals.bin", "");
  float* rprop = map(16L * RPN_MERGE_MAX); nprop = loadf(p, rprop, 16L * RPN_MERGE_MAX) / 16;
  rpn_level_ws* ws = map(sizeof(rpn_level_ws) * RPN_NLVL);
  rpn_level_ws* wp[RPN_NLVL]; for (int l = 0; l < RPN_NLVL; l++) wp[l] = &ws[l];
  rpn_merge_ws* mw = map(sizeof(rpn_merge_ws));
  uint32_t* tks = map(4L * TK_SCRATCH_WORDS(M->lv[0].A));
  float* out = map(16L * RPN_MERGE_MAX);
  int bad = 0;
  for (int v = 0; v < 5; v++) {
    int src = v & 1, hvx = !((v >> 1) & 1), lb = 0, total;
    rpn_exact_filter = v == 4; /* v4: deltas, nms_hvx, filter computed instead of proven */
    unsigned t0 = inscount();
    for (int l = 0; l < RPN_NLVL; l++)
      if (rpn_level(&M->lv[l], &prep[l], an[l], sc[l], src ? 0 : dl[l], src ? nq[l] : 0, &ws[l], tks, rpn_collect_plain, 0, hvx)) lb = 1;
    int kp = rpn_merge(M, wp, mw, out, &total);
    unsigned ins = inscount() - t0;
    for (int l = 0; l < RPN_NLVL; l++) lb |= ws[l].ns != rn[l] || memcmp(ws[l].sel, rs[l], 4L * rn[l]);
    int ob = kp != nprop || memcmp(out, rprop, 16L * kp);
    puts_(src ? "nchw_u8   " : "deltas_f32"); puts_(hvx ? " nms_hvx(portable) " : " nms_scalar        ");
    puts_(v == 4 ? "exact-filter" : "            ");
    puts_(" levels "); puts_(lb ? "DIFF" : "exact"); puts_(" proposals "); puts_(ob ? "DIFF " : "exact ");
    putu(kp); puts_("/"); putu(total); puts_(" insns "); putu(ins); puts_("\n");
    bad |= lb || ob;
  }
  puts_(bad ? "FAIL\n" : "PASS\n");
  sys6(bad, 0, 0, 0, 0, 0, 93); for (;;) {} }
__asm__(".global _start\n_start:\n r0 = r29\n jump main_\n");

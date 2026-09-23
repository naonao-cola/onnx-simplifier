/* Build: clang-19 --target=hexagon -mcpu=hexagonv65 -O2 -ffp-contract=off
 *        -static -nostdlib -ffreestanding -fuse-ld=lld -o nmsq nms_qemu.c
 * Run:   qemu-hexagon-static nmsq DATA_DIR/level  (or DATA_DIR/class)
 * (No -mhvx: qemu 8.2 can't execute the qfloat HVX ops the real kernel uses, so this builds
 * nms_kernel.h's portable nms_chunk -- same greedy loop, same band classification and exact
 * recheck, in scalar IEEE fp32. The qfloat path itself is only checked on the phone.)
 * nms_kernel.h under qemu-hexagon-static (linux-user, freestanding: raw trap0 syscalls, as in
 * ../roialign_fast/roialign_qemu.c). Runs both kernels over one group's calls, checks each call's
 * selected indices exactly against ORT's, prints executed-instruction counts (QEMU control reg 21). */
#include "nms_kernel.h"
static long sys6(long a, long b, long c, long d, long e, long f, long n) {
  long r; __asm__ volatile("r0=%1;r1=%2;r2=%3;r3=%4;r4=%5;r5=%6;r6=%7;trap0(#1);%0=r0"
    : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(n)
    : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "memory"); return r; }
static unsigned inscount(void) { unsigned r; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r"(r) : : "r0"); return r; }
static void* mem(long bytes) { return (void*)sys6(0, (bytes + 4095) & ~4095L, 3, 0x22, -1, 0, 222); }
static long loadf(const char* p, void* b, long cap) {
  int fd = sys6(-100, (long)p, 0, 0, 0, 0, 56); long o = 0;
  while (o < cap) { long n = sys6(fd, (long)((char*)b + o), cap - o, 0, 0, 0, 63); if (n <= 0) break; o += n; }
  return o; }
static void puts_(const char* s) { int n = 0; while (s[n]) n++; sys6(1, (long)s, n, 0, 0, 0, 64); }
static void putu(unsigned long v) { char b[24]; int i = 23; b[i] = 0; do { b[--i] = '0' + v % 10; v /= 10; } while (v); puts_(b + i); }
static void cat(char* d, const char* a, const char* b) { while (*a) *d++ = *a++; while (*b) *d++ = *b++; *d = 0; }
static const char* num(const char* s, long* v) { while (*s == ' ' || *s == '\n') s++; long x = 0; while (*s >= '0' && *s <= '9') x = x * 10 + (*s++ - '0'); *v = x; return s; }
static const char* skip(const char* s) { while (*s == ' ') s++; while (*s && *s != ' ' && *s != '\n') s++; return s; }
void __attribute__((noreturn)) main_(long* sp) {
  char** argv = (char**)(sp + 1); char p[512];
  const long CAP = 1 << 20;
  cat(p, argv[1], "_calls.txt"); char* txt = mem(CAP); txt[loadf(p, txt, CAP - 1)] = 0;
  cat(p, argv[1], "_boxes.bin"); float* boxes = mem(8 * CAP); loadf(p, boxes, 8 * CAP);
  cat(p, argv[1], "_scores.bin"); float* scores = mem(2 * CAP); loadf(p, scores, 2 * CAP);
  cat(p, argv[1], "_ref.bin"); int* ref = mem(2 * CAP); loadf(p, ref, 2 * CAP);
  cat(p, argv[1], "_thr.bin"); float* thrs = mem(CAP); loadf(p, thrs, CAP); /* exact fp32 thresholds */
  int *sel = mem(1 << 16), *ord = mem(1 << 16), *tmp = mem(1 << 16); float* soa = mem(1 << 16);
  unsigned long ins[2] = {0, 0}; int calls = 0, ok[2] = {0, 0};
  long bo = 0, ro = 0; const char* s = txt;
  for (;;) {
    long n, mo, ns; float thr;
    while (*s == ' ' || *s == '\n') s++;
    if (!*s) break;
    s = num(s, &n); s = skip(s); thr = thrs[calls]; s = num(s, &mo); s = num(s, &ns);
    for (int k = 0; k < 2; k++) {
      unsigned t0 = inscount();
      int m = k ? nms_hvx(boxes + 4 * bo, scores + bo, n, thr, mo, sel, ord, tmp, soa)
                : nms_scalar(boxes + 4 * bo, scores + bo, n, thr, mo, sel, ord, tmp);
      ins[k] += inscount() - t0;
      int good = m == ns; for (int j = 0; good && j < m; j++) good = sel[j] == ref[ro + j];
      ok[k] += good;
    }
    bo += n; ro += ns; calls++;
  }
  puts_("calls="); putu(calls); puts_(" exact scalar="); putu(ok[0]); puts_(" hvx-path="); putu(ok[1]);
  puts_(" insns scalar="); putu(ins[0]); puts_(" hvx-path="); putu(ins[1]); puts_("\n");
  sys6(ok[0] == calls && ok[1] == calls ? 0 : 1, 0, 0, 0, 0, 0, 93); for (;;) {} }
__asm__(".global _start\n_start:\n r0 = r29\n jump main_\n");

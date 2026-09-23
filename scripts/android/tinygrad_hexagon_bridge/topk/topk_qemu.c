/* Build: clang-19 --target=hexagon -mcpu=hexagonv73 -mhvx=v73 -mhvx-length=128b -O2 -static -nostdlib
 *        -ffreestanding -fuse-ld=lld -o topkq topk_qemu.c
 * Run:   qemu-hexagon-static topkq callN_x.bin callN_vals.bin callN_idx.bin N K
 * topk_kernel.h under qemu-hexagon-static (linux-user, freestanding: raw trap0 syscalls, the same
 * bare-metal style as ../roialign_fast/roialign_qemu.c). The kernel only uses integer HVX ops
 * (compare/xor/shift), which qemu 8.2 decodes, so the real vector path runs here (unlike RoiAlign's
 * qfloat ops). Checks values+indices byte-exact vs ORT for both the vector and the scalar collect
 * variant and prints QEMU's executed-instruction count (control register 21) for each. */
#include <stddef.h>
void* memcpy(void* d, const void* s, size_t n) { char* a = d; const char* b = s; while (n--) *a++ = *b++; return d; }
void* memset(void* d, int c, size_t n) { char* a = d; while (n--) *a++ = (char)c; return d; }
int memcmp(const void* x, const void* y, size_t n) { const unsigned char *a = x, *b = y; for (; n; n--, a++, b++) if (*a != *b) return *a - *b; return 0; }
#include "topk_kernel.h"
static long sys6(long a, long b, long c, long d, long e, long f, long n) {
  long r; __asm__ volatile("r0=%1;r1=%2;r2=%3;r3=%4;r4=%5;r5=%6;r6=%7;trap0(#1);%0=r0"
    : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(n)
    : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "memory"); return r; }
static unsigned inscount(void) { unsigned r; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r"(r) : : "r0"); return r; }
static void* map(long bytes) { return (void*)sys6(0, (bytes + 4095) & ~4095L, 3, 0x22, -1, 0, 222); }
static void* load(const char* p, long bytes) {
  int fd = sys6(-100, (long)p, 0, 0, 0, 0, 56);
  char* b = map(bytes);
  for (long o = 0; o < bytes;) { long n = sys6(fd, (long)(b + o), bytes - o, 0, 0, 0, 63); if (n <= 0) break; o += n; }
  return b; }
static void puts_(const char* s) { int n = 0; while (s[n]) n++; sys6(1, (long)s, n, 0, 0, 0, 64); }
static void putu(unsigned long v) { char b[24]; int i = 23; b[i] = 0; do { b[--i] = '0' + v % 10; v /= 10; } while (v); puts_(b + i); }
static int atoi_(const char* s) { int v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0'); return v; }
void __attribute__((noreturn)) main_(long* sp) {
  char** argv = (char**)(sp + 1);  /* argv: x vals idx N K */
  int n = atoi_(argv[4]), k = atoi_(argv[5]), bad = 0;
  float* x = load(argv[1], (long)n * 4);
  float* rv = load(argv[2], (long)k * 4);
  int64_t* ri = load(argv[3], (long)k * 8);
  uint32_t* scr = map(4 * TK_SCRATCH_WORDS(n));
  float* ov = map(4L * k + 4);
  int64_t* oi = map(8L * k + 8);
  for (int hvx = 3; hvx >= 0; hvx--) {
    unsigned t0 = inscount();
    int c = topk_desc(x, n, k, ov, oi, scr, hvx);
    unsigned t1 = inscount();
    int ok = memcmp(ov, rv, 4L * k) == 0 && memcmp(oi, ri, 8L * k) == 0;
    bad |= !ok;
    puts_(hvx == 3 ? "vec-reduce_or-mask" : hvx == 2 ? "vec-rot" : hvx == 1 ? "vec-reduce_or" : "scalar"); puts_(" survivors="); putu(c);
    puts_(ok ? " EXACT insns=" : " MISMATCH insns="); putu(t1 - t0); puts_("\n");
  }
  sys6(bad, 0, 0, 0, 0, 0, 93); for (;;) {} }
__asm__(".global _start\n_start:\n r0 = r29\n jump main_\n");

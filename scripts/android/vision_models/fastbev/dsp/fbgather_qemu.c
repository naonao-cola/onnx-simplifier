/* fbgather_kernel.h's HVX body under qemu-hexagon-static (linux-user, freestanding, raw trap0
 * syscalls -- the style of ../../../tinygrad_hexagon_bridge/roialign_fast/roialign_u8_qemu.c).
 * Build: see build_dsp.sh qemu.   Run: qemu-hexagon-static fbgq ROWS t0 t1 t2 t3 l0 l1 l2 l3 ref
 *   t*: (ROWS + 1) * 64 uint8 tables, l*: 160000 int32 LUTs, ref: the expected 41 MB volume
 * Prints exact bytes vs ref for the HVX body and the plain-C body, and the HVX body's executed
 * instruction count; exit 0 iff both are byte-exact. */
#include "fbgather_kernel.h"
#define NR (200 * 200 * 4)
static long sys6(long a, long b, long c, long d, long e, long f, long n) {
  long r; __asm__ volatile("r0=%1;r1=%2;r2=%3;r3=%4;r4=%5;r5=%6;r6=%7;trap0(#1);%0=r0"
    : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(n)
    : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "memory"); return r; }
static unsigned inscount(void) { unsigned r; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r"(r) : : "r0"); return r; }
static void* map_anon(long bytes) { return (void*)sys6(0, (bytes + 4095) & ~4095L, 3, 0x22, -1, 0, 222); }
static void* load(const char* p, long bytes, long extra) {
  int fd = sys6(-100, (long)p, 0, 0, 0, 0, 56);
  char* b = map_anon(bytes + extra);
  for (long o = 0; o < bytes;) { long n = sys6(fd, (long)(b + o), bytes - o, 0, 0, 0, 63); if (n <= 0) break; o += n; }
  sys6(fd, 0, 0, 0, 0, 0, 57);
  return b; }
static void puts_(const char* s) { int n = 0; while (s[n]) n++; sys6(1, (long)s, n, 0, 0, 0, 64); }
static void putu(unsigned long v) { char b[24]; int i = 23; b[i] = 0; do { b[--i] = '0' + v % 10; v /= 10; } while (v); puts_(b + i); }
static int atoi_(const char* s) { int v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0'); return v; }
void* memset(void* d, int c, unsigned long n) { char* p = d; while (n--) *p++ = (char)c; return d; }
void* memcpy(void* d, const void* s, unsigned long n) { char* p = d; const char* q = s; while (n--) *p++ = *q++; return d; }
static long exact(const uint8_t* a, const uint8_t* b, long n) { long e = 0; for (long i = 0; i < n; i++) e += a[i] == b[i]; return e; }
void __attribute__((noreturn)) main_(long* sp) {
  char** a = (char**)(sp + 1) + 1;
  const int rows = atoi_(a[0]);
  const uint8_t* tab[4];
  const int32_t* lut[4];
  for (int t = 0; t < 4; t++) tab[t] = load(a[1 + t], (rows + 1L) * 64, 128);
  for (int t = 0; t < 4; t++) lut[t] = load(a[5 + t], NR * 4L, 0);
  const long total = NR * 256L;
  const uint8_t* ref = load(a[9], total, 0);
  uint8_t* out = map_anon(total);
  unsigned t0 = inscount();
  fbg_run_hvx(tab, lut, rows, 0, NR, out);
  unsigned t1 = inscount();
  long eh = exact(out, ref, total);
  memset(out, 0, total);
  fbg_run_c(tab, lut, rows, 0, NR, out);
  long ec = exact(out, ref, total);
  puts_("bytes="); putu(total); puts_(" hvx_exact="); putu(eh); puts_(" c_exact="); putu(ec);
  puts_(" hvx_insns="); putu(t1 - t0); puts_("\n");
  sys6(eh == total && ec == total ? 0 : 1, 0, 0, 0, 0, 0, 93); for (;;) {} }
__asm__(".global _start\n_start:\n r0 = r29\n jump main_\n");

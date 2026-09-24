/* Build: clang-19 --target=hexagon -mcpu=hexagonv73 -O2 -ffp-contract=off -static -nostdlib
 *        -ffreestanding -fuse-ld=lld -o pdq pd_qemu.c
 * Run:   qemu-hexagon-static pdq lN_params.bin lN_anchors.bin lN_idx.bin lN_deltas.bin lN_nchw_q.bin lN_ref.bin A k H W
 * pd_kernel.h under qemu-hexagon-static (linux-user, freestanding: raw trap0 syscalls, same style as
 * ../roialign_fast/roialign_qemu.c). The kernel is scalar, so unlike RoiAlign it builds for v73
 * directly (no HVX qfloat instructions for qemu 8.2 to choke on). Checks both delta sources, reference
 * and fast paths, against ORT's output and prints QEMU's executed-instruction count for each. */
#include "pd_kernel.h"
static long sys6(long a, long b, long c, long d, long e, long f, long n) {
  long r; __asm__ volatile("r0=%1;r1=%2;r2=%3;r3=%4;r4=%5;r5=%6;r6=%7;trap0(#1);%0=r0"
    : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(n)
    : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "memory"); return r; }
static unsigned inscount(void) { unsigned r; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r"(r) : : "r0"); return r; }
static void* load(const char* p, long bytes) {
  int fd = sys6(-100, (long)p, 0, 0, 0, 0, 56);
  if (fd < 0) { sys6(2, (long)"open failed\n", 12, 0, 0, 0, 64); sys6(2, 0, 0, 0, 0, 0, 93); }
  char* b = (char*)sys6(0, (bytes + 4095) & ~4095L, 3, 0x22, -1, 0, 222);
  for (long o = 0; o < bytes;) { long n = sys6(fd, (long)(b + o), bytes - o, 0, 0, 0, 63); if (n <= 0) break; o += n; }
  return b; }
static void puts_(const char* s) { int n = 0; while (s[n]) n++; sys6(1, (long)s, n, 0, 0, 0, 64); }
static void putu(unsigned long v) { char b[24]; int i = 23; b[i] = 0; do { b[--i] = '0' + v % 10; v /= 10; } while (v); puts_(b + i); }
static int atoi_(const char* s) { int v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0'); return v; }
void __attribute__((noreturn)) main_(long* sp) {
  char** argv = (char**)(sp + 1);
  int A = atoi_(argv[7]), k = atoi_(argv[8]), H = atoi_(argv[9]), W = atoi_(argv[10]);
  pd_params* P = load(argv[1], sizeof(pd_params));
  float* an = load(argv[2], 16L * A); int32_t* idx = load(argv[3], 4L * k);
  float* dl = load(argv[4], 16L * A); uint8_t* nq = load(argv[5], 12L * H * W);
  float* ref = load(argv[6], 16L * k);
  float* out = (float*)sys6(0, (16L * k + 4095) & ~4095L, 3, 0x22, -1, 0, 222);
  int bad = 0;
  pd_prep* Q = (pd_prep*)sys6(0, (sizeof(pd_prep) + 4095) & ~4095L, 3, 0x22, -1, 0, 222);
  pd_prepare(P, Q);
  pd_prepare_anchors(an, H, W, Q);
  for (int v = 0; v < 4; v++) {
    int src = v & 1, fast = v >> 1;
    unsigned t0 = inscount();
    pd_decode(an, idx, k, src ? 0 : dl, src ? nq : 0, H, W, P, fast ? Q : 0, out);
    unsigned t1 = inscount();
    int nd = 0; for (int j = 0; j < 4 * k; j++) nd += out[j] != ref[j];
    puts_(src ? "nchw_u8   " : "deltas_f32"); puts_(fast ? " fast     " : " reference");
    puts_(" diff_elems="); putu(nd); puts_(" insns="); putu(t1 - t0); puts_("\n");
    bad |= nd != 0;
  }
  sys6(bad, 0, 0, 0, 0, 0, 93); for (;;) {} }
__asm__(".global _start\n_start:\n r0 = r29\n jump main_\n");

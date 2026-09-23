/* Build: clang-19 --target=hexagon -mcpu=hexagonv65 -mhvx=v65 -mhvx-length=128b -O2 -static -nostdlib
 *        -ffreestanding -fuse-ld=lld -o roiu8q roialign_u8_qemu.c $HEXAGON_TOOLCHAIN/target/hexagon/lib/v68/G0/libgcc.a
 *        (libgcc.a only for the soft double/int-divide helpers the requant setup and bin index use)
 * Run:   qemu-hexagon-static roiu8q $(python3 roialign_u8_qemu_args.py CAPTURE_DIR box|mask [sort])
 * roialign_u8_kernel.h lowered for Hexagon HVX and run under qemu-hexagon-static (linux-user,
 * freestanding, raw trap0 syscalls -- same style as roialign_qemu.c). The kernel is integer-only, so
 * qemu 8.2's missing qfloat decode doesn't matter. Prints the exact-byte fraction (ppm) and max |diff|
 * vs the capture's QuantizeLinear(ORT fp32 RoiAlign) merged bytes, and QEMU's executed-instruction
 * count for the kernel. Exit status 0 iff max |diff| <= 1 LSB.
 * argv: ref_u8 C OH sr s_out z_out N sort, then per level k=0..3:
 *       map_u8 H W z_in s_in spatial_scale n rois_f32 rows_i32  (n == 0: "-" for the two paths) */
#include "roialign_u8_kernel.h"
static long sys6(long a, long b, long c, long d, long e, long f, long n) {
  long r; __asm__ volatile("r0=%1;r1=%2;r2=%3;r3=%4;r4=%5;r5=%6;r6=%7;trap0(#1);%0=r0"
    : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(n)
    : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "memory"); return r; }
static unsigned inscount(void) { unsigned r; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r"(r) : : "r0"); return r; }
static void* map_anon(long bytes) { return (void*)sys6(0, (bytes + 4095) & ~4095L, 3, 0x22, -1, 0, 222); }
static void* load(const char* p, long bytes) {
  int fd = sys6(-100, (long)p, 0, 0, 0, 0, 56);
  char* b = map_anon(bytes);
  for (long o = 0; o < bytes;) { long n = sys6(fd, (long)(b + o), bytes - o, 0, 0, 0, 63); if (n <= 0) break; o += n; }
  sys6(fd, 0, 0, 0, 0, 0, 57);
  return b; }
static void puts_(const char* s) { int n = 0; while (s[n]) n++; sys6(1, (long)s, n, 0, 0, 0, 64); }
static void putu(unsigned long v) { char b[24]; int i = 23; b[i] = 0; do { b[--i] = '0' + v % 10; v /= 10; } while (v); puts_(b + i); }
static int atoi_(const char* s) { int v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0'); return v; }
/* plain decimal ("0.11570039391517639", python repr of the qparams; no exponent form in meta.txt) */
static float atof_(const char* s) {
  double v = 0, f = 1; while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  if (*s == '.') for (s++; *s >= '0' && *s <= '9'; s++) { f /= 10; v += (*s - '0') * f; }
  return (float)v; }
void* memset(void* d, int c, unsigned long n) { char* p = d; while (n--) *p++ = (char)c; return d; }
void* memcpy(void* d, const void* s, unsigned long n) { char* p = d; const char* q = s; while (n--) *p++ = *q++; return d; }
void __attribute__((noreturn)) main_(long* sp) {
  char** a = (char**)(sp + 1) + 1; /* skip argv[0] */
  const int C = atoi_(a[1]), OH = atoi_(a[2]), sr = atoi_(a[3]), z_out = atoi_(a[5]), N = atoi_(a[6]), sort = atoi_(a[7]);
  const float s_out = atof_(a[4]);
  ru8_level_t lv[4];
  ru8_job_t* jobs = map_anon((long)N * sizeof(ru8_job_t));
  int nj = 0;
  for (int k = 0; k < 4; k++) {
    char** b = a + 8 + 9 * k;
    const int H = atoi_(b[1]), W = atoi_(b[2]), n = atoi_(b[6]);
    lv[k] = (ru8_level_t){load(b[0], (long)H * W * C), H, W, atof_(b[5]), atoi_(b[3]), 0, 0};
    ru8_requant_params(atof_(b[4]), s_out, sr * sr, &lv[k].mult, &lv[k].shift);
    if (!n) continue;
    const float* rois = load(b[7], n * 16L);
    const int* rows = load(b[8], n * 4L);
    for (int i = 0; i < n; i++, nj++) {
      jobs[nj].level = k; jobs[nj].row = rows[i];
      for (int c = 0; c < 4; c++) jobs[nj].box[c] = rois[4 * i + c];
    }
  }
  if (sort) ru8_sort_jobs(jobs, nj);
  const long total = (long)N * OH * OH * C;
  const unsigned char* ref = load(a[0], total);
  unsigned char* out = map_anon(total);
  unsigned t0 = inscount();
  ru8_run_jobs(lv, C, jobs, nj, OH, OH, sr, z_out, 0, out);
  unsigned t1 = inscount();
  long exact = 0; int maxd = 0;
  for (long i = 0; i < total; i++) { int d = (int)out[i] - (int)ref[i]; if (d < 0) d = -d; exact += !d; if (d > maxd) maxd = d; }
  puts_("rois="); putu(nj); puts_(" exact_ppm="); putu((unsigned long)(exact * 1000000.0 / total));
  puts_(" maxdiff="); putu(maxd); puts_(" insns="); putu(t1 - t0); puts_("\n");
  sys6(maxd <= 1 && nj == N ? 0 : 1, 0, 0, 0, 0, 0, 93); for (;;) {} }
__asm__(".global _start\n_start:\n r0 = r29\n jump main_\n");

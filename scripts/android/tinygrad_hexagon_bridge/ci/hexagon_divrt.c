/* 32-bit integer divide helpers for freestanding qemu-hexagon builds (Hexagon has no divide instruction;
 * the DSP builds get these from the toolchain's libgcc). topk_kernel.h's threshold rank divides by a runtime
 * n; ../rpn_fused/rpn_qemu.c defines these itself, ../topk/topk_qemu.c doesn't, so CI links this next to it
 * (clang-19 + ld.lld: "undefined symbol: __hexagon_udivsi3" otherwise). Same code as rpn_qemu.c's copy. */
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

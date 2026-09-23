/* hexagon-sim: map the v69 fp16 HMX tile layouts. Weight halfword q = 1.0 (one-hot), activation
 * halfword p = p+1 (exact in fp16 up to 2048); every nonzero :after.hf output halfword names (q, o, p).
 * argv: span_bytes [bias_mode: 0 none, 1 words 0x3c000000, 2 words 0x00003c00] [qstep] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
static float h2f(uint16_t h){ __fp16 x; memcpy(&x,&h,2); return (float)x; }
static uint16_t f2h(float f){ __fp16 x=(__fp16)f; uint16_t h; memcpy(&h,&x,2); return h; }
int main(int argc, char** argv){
  int span = argc > 1 ? atoi(argv[1]) : 2048, bm = argc > 2 ? atoi(argv[2]) : 0, qstep = argc > 3 ? atoi(argv[3]) : 1;
  unsigned char* v=(unsigned char*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  uint16_t *a=(uint16_t*)v,*w=(uint16_t*)(v+65536),*o=(uint16_t*)(v+131072); unsigned* b=(unsigned*)(v+196608);
  for(int i=0;i<64;i++) b[i]= bm==1 ? 0x3c000000u : bm==2 ? 0x00003c00u : 0;
  if (bm) __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
  int nh = span/2;
  for (int p = 0; p < nh; p++) a[p] = f2h((float)(p % 2048 + 1));
  for (int q = 0; q < nh; q += qstep) {
    memset(w, 0, span); w[q] = 0x3c00; memset(o, 0, 8192);
    __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }"::"r"(a),"r"(span-1),"r"(w),"r"(span-1):"memory");
    __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(o),"r"(0):"memory");
    for (int i = 0; i < 4096; i++) if (o[i]) printf("q %d o %d v %g\n", q, i, h2f(o[i]));
  }
  printf("end\n"); return 0;
}

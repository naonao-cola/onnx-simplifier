/* hexagon-sim: map the v69 int8 HMX tile layouts with one-hot weights and index-encoded activations.
 * For each weight byte q (value 1), two runs encode each activation byte p as (p&127)+1 and (p>>7)+1;
 * every nonzero :sat.uh 2x1 output u16 then names the (q, output index, p) triple that met. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
int main(int argc, char** argv){
  int span = argc > 1 ? atoi(argv[1]) : 2048;      /* bytes per operand (Rt = span-1) */
  int abytes = argc > 2 ? atoi(argv[2]) : span;    /* activation bytes filled */
  int qstep = argc > 3 ? atoi(argv[3]) : 1;
  int amask = argc > 4 ? atoi(argv[4]) : 0; /* 0 all, 1 even bytes only, 2 odd only */
  int st = argc > 5 ? atoi(argv[5]) : 0;    /* 0 sat.uh 2x1, 1 sat.uh 2x2, 2 :deep off */
  unsigned char* v=(unsigned char*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  unsigned char *a=v,*w=v+65536,*b=v+196608; uint16_t* o=(uint16_t*)(v+131072);
  for(int i=0;i<64;i++) ((unsigned*)b)[i]=0x4000;
  __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
  static uint16_t o1[8192];
  for (int q = 0; q < span; q += qstep) {
    memset(w, 0, span + 256); w[q] = 1;
    for (int run = 0; run < 2; run++) {
      memset(a, 0, abytes + 256);
      for (int p = 0; p < abytes; p++) if (!amask || (amask == 1) == !(p & 1)) a[p] = run ? (p >> 7) + 1 : (p & 127) + 1;
      memset(o, 0, 16384);
      if (st == 2) __asm__ volatile("{ activation.ub = mxmem(%0,%1)\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(span-1),"r"(w),"r"(span-1):"memory");
      else __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(span-1),"r"(w),"r"(span-1):"memory");
      if (st == 1) __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x2"::"r"(o),"r"(0):"memory");
      else __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x1"::"r"(o),"r"(0):"memory");
      if (!run) memcpy(o1, o, 16384); else
        for (int i = 0; i < 8192; i++) if (o1[i] || o[i]) printf("q %d o %d p %d\n", q, i, (o[i]-1)*128 + (o1[i]-1));
    }
  }
  printf("end\n"); return 0;
}

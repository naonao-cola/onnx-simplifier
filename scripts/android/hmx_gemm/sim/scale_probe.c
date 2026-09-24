/* hexagon-sim: int8 x int8 (acc = 32*a*w) stored with :after.hf / :after:sat.ub / :sat.uh for a list of
 * per-column table words, to learn the v69 scale/bias word format. argv: a w (activation, weight value) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
static float h2f(uint16_t h){ __fp16 x; memcpy(&x,&h,2); return (float)x; }
int main(int argc,char**argv){
  int av = argc>1?atoi(argv[1]):1, wv = argc>2?atoi(argv[2]):1;
  unsigned char* v=(unsigned char*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  unsigned char *a=v,*w=v+65536,*b=v+196608; uint16_t* o=(uint16_t*)(v+131072);
  memset(a,0,2048); for(int p=1;p<2048;p+=2) a[p]=(unsigned char)av; memset(w,(unsigned char)wv,1024);
  unsigned words[]={0,0x4000,0x2000,0x3c00,0x3c000000,0x40000000,0x00003c00|0x3c000000,0x10000,0x100,0x1,0x8000,0x7fff,0x44004000,0xbc004000};
  for(unsigned i=0;i<sizeof words/4;i++){
    for(int j=0;j<64;j++) ((unsigned*)b)[j]=words[i];
    __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
    for(int st=0;st<3;st++){
      memset(o,0,4096);
      __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
      if(st==0) __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(o),"r"(0):"memory");
      else if(st==1) __asm__ volatile("mxmem(%0,%1):after:sat.ub = acc"::"r"(o),"r"(0):"memory");
      else __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x1"::"r"(o),"r"(0):"memory");
      if(st==0) printf("word %08x hf %g ", words[i], h2f(o[0]));
      else if(st==1) printf("ub %u ", ((unsigned char*)o)[0]);
      else printf("uh %u\n", o[0]);
    }
  }
  return 0;
}

/* hexagon-sim: where does :sat.ub write, and does bias=mxmem2 (2 x 256 B) give int-acc -> hf a scale? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
static float h2f(uint16_t h){ __fp16 x; memcpy(&x,&h,2); return (float)x; }
int main(){
  unsigned char* v=(unsigned char*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  unsigned char *a=v,*w=v+65536,*b=v+196608; unsigned char* o=v+131072;
  memset(a,0,2048); for(int p=1;p<2048;p+=2) a[p]=1; memset(w,1,1024);   /* acc = 32 */
  unsigned words[]={0x4000,0x3c00,0x4400,0x4800,0x5000,0x40004000,0x3c003c00};
  for(unsigned i=0;i<sizeof words/4;i++){
    for(int j=0;j<64;j++) ((unsigned*)b)[j]=words[i];
    __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
    memset(o,0,8192);
    __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
    __asm__ volatile("mxmem(%0,%1):after:sat.ub = acc"::"r"(o),"r"(0):"memory");
    int nz=0,first=-1; for(int k=0;k<8192;k++) if(o[k]){nz++; if(first<0)first=k;}
    printf("ub word %08x: nz %d first %d val %d\n", words[i], nz, first, first>=0?o[first]:0);
  }
  /* mxmem2: 512 B table; try (scale row, bias row) and (bias row, scale row) with fp16 1.0 / 2.0 */
  unsigned pats[][2]={{0x3c003c00,0},{0,0x3c003c00},{0x40004000,0x3c003c00},{0x3c003c00,0x40004000},{0x4000,0},{0x00004000,0x3c000000}};
  for(unsigned i=0;i<sizeof pats/8;i++){
    for(int j=0;j<64;j++){ ((unsigned*)b)[j]=pats[i][0]; ((unsigned*)b)[64+j]=pats[i][1]; }
    __asm__ volatile("bias = mxmem2(%0)"::"r"(b):"memory");
    for(int st=0;st<2;st++){
      memset(o,0,8192);
      __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
      if(st==0) __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(o),"r"(0):"memory");
      else __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x1"::"r"(o),"r"(0):"memory");
      if(st==0) printf("mxmem2 %08x/%08x: hf %g %g ", pats[i][0], pats[i][1], h2f(((uint16_t*)o)[0]), h2f(((uint16_t*)o)[1]));
      else printf("uh %u %u\n", ((uint16_t*)o)[0], ((uint16_t*)o)[1]);
    }
  }
  return 0;
}

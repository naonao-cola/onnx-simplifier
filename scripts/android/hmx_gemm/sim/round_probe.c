/* hexagon-sim: (1) which table word scales output column c; (2) rounding of acc*scale for :sat.uh;
 * (3) fp16 per-column bias word index and fp16 accumulation precision. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
static float h2f(uint16_t h){ __fp16 x; memcpy(&x,&h,2); return (float)x; }
static uint16_t f2h(float f){ __fp16 x=(__fp16)f; uint16_t h; memcpy(&h,&x,2); return h; }
int main(){
  unsigned char* v=(unsigned char*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  unsigned char *a=v,*w=v+65536; unsigned* b=(unsigned*)(v+196608); uint16_t* o=(uint16_t*)(v+131072);
  /* (1) column indexing: word j = 0x4000 only for j==J, others 0 -> which output columns are nonzero */
  memset(a,0,2048); for(int p=1;p<2048;p+=2) a[p]=1; memset(w,1,1024);
  for (int J = 0; J < 64; J += 7) {
    for(int j=0;j<64;j++) b[j] = j==J ? 0x4000 : 0;
    __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
    memset(o,0,4096);
    __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
    __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x1"::"r"(o),"r"(0):"memory");
    printf("colidx word %d ->", J); for(int c=0;c<32;c++) if(o[2*c]) printf(" c%d", c); printf("\n");
  }
  /* (2) rounding: weight row k=0 only, activation row value -> acc = a*w exactly; scale 0x3800 (x0.25) */
  unsigned sc[] = {0x3800, 0x3400, 0x3a00};
  for (unsigned s = 0; s < 3; s++) {
    for(int j=0;j<64;j++) b[j]=sc[s];
    __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
    printf("round scale %04x (x%g):", sc[s], h2f(sc[s])/2);
    for (int accv = 1; accv <= 12; accv++) {
      memset(a,0,2048); a[1] = (unsigned char)accv; memset(w,0,1024); w[0] = 1;  /* A(0,0)=accv, W(0,0)=1 */
      memset(o,0,4096);
      __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
      __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x1"::"r"(o),"r"(0):"memory");
      printf(" %d->%u", accv, o[0]);
    }
    printf("\n");
  }
  /* (3) fp16: bias word J = 0x3c000000 only -> which columns get +1; accumulation: sum of 1024 x 2^-11 terms */
  uint16_t *ah=(uint16_t*)a, *wh=(uint16_t*)w;
  for (int J = 0; J < 64; J += 9) {
    for(int j=0;j<64;j++) b[j] = j==J ? 0x3c000000 : 0;
    __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
    memset(a,0,2048); memset(w,0,2048); memset(o,0,4096);
    __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
    __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(o),"r"(0):"memory");
    printf("hf bias word %d ->", J); for(int c=0;c<32;c++) if(o[2*c]) printf(" c%d", c); printf("\n");
  }
  for(int j=0;j<64;j++) b[j]=0; __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
  /* acc precision: A(0,k)=1.0 for k<32, W(k,0)=1+2^-10 -> exact sum 32*(1+2^-10)=32.03125 (needs 16 bits) */
  for (int i=0;i<1024;i++){ ah[i]=0; wh[i]=0; }
  for (int k=0;k<32;k++){ ah[2*k]=f2h(1.0f); wh[64*(k/2)+k%2]=f2h(1.0f+1.0f/1024); }
  memset(o,0,4096);
  __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
  __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(o),"r"(0):"memory");
  printf("hf acc: sum32 (1+2^-10) = %.6f (fp16 of exact 32.03125 = %.6f)\n", h2f(o[0]), h2f(f2h(32.03125f)));
  /* large+small cancellation: 1000 - 1000 + 0.001*30 */
  for (int i=0;i<1024;i++){ ah[i]=0; wh[i]=0; }
  for (int k=0;k<32;k++){ ah[2*k]=f2h(1.0f); float wv = k==0?1000.0f:(k==1?-1000.0f:0.0009765625f); wh[64*(k/2)+k%2]=f2h(wv); }
  memset(o,0,4096);
  __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
  __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(o),"r"(0):"memory");
  printf("hf acc: 1000-1000+30*2^-10 = %.6f (exact 0.029297)\n", h2f(o[0]));
  return 0;
}

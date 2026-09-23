/* hexagon-sim check of the v69 int8 HMX tile layouts (see README): random 32x32x32 A(ub) x W(b) vs a C reference.
 * Build/run like hmx_sim.c: hexagon-clang -mv69 -mhmx -mhvx -O2 hmx_sim_layout.c; hexagon-sim -mv69 --mhmx 1 a.out */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
int main(){
  unsigned char* v=(unsigned char*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  unsigned char *a=v,*w=v+65536,*b=v+196608; uint16_t* o=(uint16_t*)(v+131072);
  for(int i=0;i<64;i++) ((unsigned*)b)[i]=0x4000;
  __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
  static unsigned char A[32][32]; static signed char W[32][32]; srand(1);
  memset(a,0,2048); memset(w,0,2048);
  for(int i=0;i<32;i++) for(int k=0;k<32;k++){ A[i][k]=rand()%16; a[128*(i/2)+4*k+2*(i%2)+1]=A[i][k]; }
  for(int k=0;k<32;k++) for(int c=0;c<32;c++){ W[k][c]=rand()%16; w[128*(k/4)+4*c+(k%4)]=(unsigned char)W[k][c]; }
  __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(2047),"r"(w),"r"(2047):"memory");
  __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x1"::"r"(o),"r"(0):"memory");
  int bad=0; for(int i=0;i<32;i++) for(int c=0;c<32;c++){ int s=0; for(int k=0;k<32;k++) s+=A[i][k]*W[k][c]; int got=o[64*(i/2)+2*c+(i%2)]; if(got!=s){ if(bad<5) printf("mismatch r%d c%d got %d want %d\n",i,c,got,s); bad++; } }
  printf("random int8 32x32x32: %d/1024 mismatches\n", bad); return 0;
}

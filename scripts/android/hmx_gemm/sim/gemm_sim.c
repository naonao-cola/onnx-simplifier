/* hexagon-sim check of hmx_gemm_f16 vs a double reference: gemm_sim M K N [bias] */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../hmx_gemm.h"
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
static float h2f(uint16_t h){ __fp16 x; memcpy(&x,&h,2); return (float)x; }
static uint16_t f2h(float f){ __fp16 x=(__fp16)f; uint16_t h; memcpy(&h,&x,2); return h; }
static unsigned long long cyc(void){ unsigned long long c; __asm__ volatile("%0 = c15:14":"=r"(c)); return c; }
int main(int argc,char**argv){
  int M=atoi(argv[1]),K=atoi(argv[2]),N=atoi(argv[3]),hb=argc>4?atoi(argv[4]):0;
  uint8_t* v=(uint8_t*)(cfg(0x38)<<16); unsigned vs=cfg(0x3c)*1024;
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  uint16_t *A=malloc((size_t)M*K*2),*W=malloc((size_t)K*N*2),*Wp=malloc((size_t)K*N*2),*C=malloc((size_t)M*N*2),*B=malloc(N*2);
  srand(7);
  for(size_t i=0;i<(size_t)M*K;i++) A[i]=f2h((rand()%2001-1000)/1000.f);
  for(size_t i=0;i<(size_t)K*N;i++) W[i]=f2h((rand()%2001-1000)/4000.f);
  for(int i=0;i<N;i++) B[i]=f2h(hb?(rand()%200-100)/50.f:0);
  hmx_pack_w_f16(W,K,N,Wp);
  unsigned long long t0=cyc();
  int rc=hmx_gemm_f16(A,Wp,hb?B:NULL,C,M,K,N,v,vs);
  unsigned long long t1=cyc();
  double maxe=0,maxr=0; int bad=0;
  for(int i=0;i<M;i++) for(int j=0;j<N;j++){ double s=hb?h2f(B[j]):0; for(int k=0;k<K;k++) s+=(double)h2f(A[(size_t)i*K+k])*h2f(W[(size_t)k*N+j]);
    double got=h2f(C[(size_t)i*N+j]), e=fabs(got-s), tol=fabs(s)*1.0/1024+1e-4; if(e>maxe)maxe=e; if(fabs(s)>maxr)maxr=fabs(s); if(e>tol){ if(bad<4) printf("bad %d,%d got %g want %g\n",i,j,got,s); bad++; } }
  printf("gemm_f16 %dx%dx%d bias %d rc %d vtcm %u: max abs err %.3g (max |ref| %.3g), %d beyond fp16 rounding, %llu cycles\n",M,K,N,hb,rc,vs,maxe,maxr,bad,t1-t0);
  return 0;
}

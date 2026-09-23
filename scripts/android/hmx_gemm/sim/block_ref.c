/* hexagon-sim: bit-exact reference test of hmx_block.h (what a generated kernel is diffed against).
 * Random operands, K = 32 * ktiles (argv[1], default 12); prints the number of mismatching outputs for
 *   fp16:      C = rne_fp16(sum A*W + bias)          (exact accumulation, one rounding)
 *   int8->u16: C = min(65535, floor(acc * s / 2))    (s = 2^-5 here, W >= 0 so acc >= 0)
 *   int8->u8:  C = min(255, floor(acc * s / 512))    (s = 2^3)
 * and "PASS" when all are 0. Run: ./sim/run.sh sim/block_ref.c [ktiles] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../hmx_block.h"
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
static float h2f(uint16_t h){ __fp16 x; memcpy(&x,&h,2); return (float)x; }
static uint16_t f2h(double f){ __fp16 x=(__fp16)f; uint16_t h; memcpy(&h,&x,2); return h; } /* RNE */
int main(int argc,char**argv){
  int kt = argc>1?atoi(argv[1]):12, K = 32*kt;
  uint8_t* v=(uint8_t*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));  /* hmx_lock does this on the phone */
  uint8_t *a=v, *w=v+256*1024, *c=v+512*1024; uint32_t* t=(uint32_t*)(v+768*1024);
  srand(3);
  /* fp16 */
  uint16_t *ah=(uint16_t*)a,*wh=(uint16_t*)w,*ch=(uint16_t*)c;
  static double A[32][1024], W[1024][32], B[32];
  for(int i=0;i<32;i++) for(int k=0;k<K;k++){ A[i][k]=h2f(f2h((rand()%2001-1000)/997.0)); ah[k/32*1024+HMX_IDX(i,k%32)]=f2h(A[i][k]); }
  for(int k=0;k<K;k++) for(int j=0;j<32;j++){ W[k][j]=h2f(f2h((rand()%2001-1000)/3001.0)); wh[k/32*1024+HMX_IDX(k%32,j)]=f2h(W[k][j]); }
  for(int j=0;j<32;j++){ B[j]=h2f(f2h((rand()%200-100)/37.0)); t[j]=(uint32_t)f2h(B[j])<<16; } for(int j=32;j<64;j++) t[j]=0;
  hmx_blk_set_table(t); hmx_blk_mac_f16(a,w,kt); hmx_blk_store_f16(c);
  int bf=0; for(int i=0;i<32;i++) for(int j=0;j<32;j++){ double s=B[j]; for(int k=0;k<K;k++) s+=A[i][k]*W[k][j]; if(ch[HMX_IDX(i,j)]!=f2h(s)) bf++; }
  /* int8 -> u16 and u8 */
  static int Ai[32][1024], Wi[1024][32];
  memset(a,0,(size_t)kt*2048); memset(w,0,(size_t)kt*1024);
  for(int i=0;i<32;i++) for(int k=0;k<K;k++){ Ai[i][k]=rand()%256; a[k/32*2048+2*HMX_IDX(i,k%32)+1]=(uint8_t)Ai[i][k]; }
  for(int k=0;k<K;k++) for(int j=0;j<32;j++){ Wi[k][j]=rand()%128; w[k/32*1024+128*((k%32)/4)+4*j+k%4]=(uint8_t)Wi[k][j]; }
  int bu=0, bb=0;
  for(int mode=0;mode<2;mode++){
    uint16_t s16 = mode==0 ? 0x2c00 : 0x4800;       /* fp16 2^-4 (x 2^-5 after /2), fp16 8 (x 2^-6 after /512) */
    for(int j=0;j<64;j++) t[j]=s16;
    hmx_blk_set_table(t); hmx_blk_mac_u8s8(a,w,kt);
    if(mode==0) hmx_blk_store_u16(c); else hmx_blk_store_u8(c);
    for(int i=0;i<32;i++) for(int j=0;j<32;j++){ long acc=0; for(int k=0;k<K;k++) acc+=(long)Ai[i][k]*Wi[k][j];
      if(mode==0){ long e=acc/32; if(e>65535) e=65535; if(((uint16_t*)c)[HMX_IDX(i,j)]!=e) bu++; }
      else { long e=acc/64; if(e>255) e=255; if(c[2*HMX_IDX(i,j)+1]!=e) bb++; } }
  }
  printf("K=%d: fp16 %d, int8->u16 %d, int8->u8 %d mismatches of 1024 each\n%s\n", K, bf, bu, bb, (bf|bu|bb)?"FAIL":"PASS");
  return 0;
}

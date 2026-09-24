/* Standalone hexagon-sim HMX check (no QuRT): hexagon-sim -mv69 --mhmx 1 hmx_sim.elf -- <variant> <limit> <ssr_bit> [bias_word_hex] [store] [load]
 * variant bit 8: load the bias/scale table (256 B of bias_word) first; ssr_bit 26 enables HMX in standalone mode (QuRT hmx_lock does it on the phone).
 * store: 0 :after:sat.ub, 1 :after.hf, 2 :after:sat.uh 2x1, 3 :after.ub; load: 0 ub x b :deep, 1 hf x hf :deep (1.0 inputs), 2 ub x b. */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
int main(int argc,char**argv){
  unsigned vb = cfg(0x38)<<16, vs = cfg(0x3c);
  printf("cfg vtcm_base=0x%08x tcm_size=%u\n", vb, vs);
  unsigned ssr; __asm__ volatile("%0 = ssr":"=r"(ssr)); printf("ssr=0x%08x\n", ssr);
  unsigned char* v=(unsigned char*)vb;
  unsigned char* a=v, *w=v+65536, *o=v+131072, *b=v+196608;
  int ldf = argc>6? atoi(argv[6]):0; if(ldf==1){ for(int i=0;i<4096;i++){((uint16_t*)a)[i]=0x3c00;((uint16_t*)w)[i]=0x3c00;} } else { memset(a,1,8192); memset(w,1,8192);}  memset(o,0xcd,8192); memset(b,0,256);
  if (argc>4){ unsigned wd=(unsigned)strtoul(argv[4],0,16); for(int i=0;i<256/4;i++) ((unsigned*)b)[i]=wd; }
  int variant = argc>1? atoi(argv[1]):0;
  int sbit = argc>3? atoi(argv[3]):-1;
  if (sbit>=0){ unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<sbit; __asm__ volatile("ssr = %0; isync"::"r"(r)); __asm__ volatile("%0 = ssr":"=r"(r)); printf("ssr now 0x%08x\n",r);}
  int limit = argc>2? atoi(argv[2]):2047;
  printf("variant %d limit %d\n", variant, limit); fflush(stdout);
  if (variant&256) __asm__ volatile("bias = mxmem(%0)"::"r"(b):"memory");
  int stf = argc>5? atoi(argv[5]):0;
  if(ldf==1) __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }"::"r"(a),"r"(limit),"r"(w),"r"(limit):"memory");
  else if(ldf==2) __asm__ volatile("{ activation.ub = mxmem(%0,%1)\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(limit),"r"(w),"r"(limit):"memory");
  else __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(limit),"r"(w),"r"(limit):"memory");
  if(stf==0) __asm__ volatile("mxmem(%0,%1):after:sat.ub = acc"::"r"(o),"r"(0):"memory");
  else if(stf==1) __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(o),"r"(0):"memory");
  else if(stf==2) __asm__ volatile("mxmem(%0,%1):after:sat.uh = acc:2x1"::"r"(o),"r"(0):"memory");
  else __asm__ volatile("mxmem(%0,%1):after.ub = acc"::"r"(o),"r"(0):"memory");
  printf("done; out[0..16]:");
  for(int i=0;i<16;i++) printf(" %02x", o[i]); printf(" | 64..72:"); for(int i=64;i<72;i++) printf(" %02x",o[i]); printf(" | 1024..1032:"); for(int i=1024;i<1032;i++) printf(" %02x",o[i]); printf("\n");
  int nz=0; for(int i=0;i<8192;i++) if(o[i]!=0xcd) nz++; printf("bytes written: %d\n", nz);
  return 0;
}

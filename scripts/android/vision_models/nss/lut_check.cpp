// Host check of nss_lut.h: reads "h_in w_in h_out w_out jy jx taps" lines, prints mod_h mod_w and the LUT.
#include <cstdio>

#include "nss_lut.h"

int main() {
  int hi, wi, ho, wo, taps;
  float jy, jx;
  while (scanf("%d %d %d %d %f %f %d", &hi, &wi, &ho, &wo, &jy, &jx, &taps) == 7) {
    NssLut l = nss_offset_lut(hi, wi, ho, wo, jy, jx, taps);
    printf("%d %d", l.mod_h, l.mod_w);
    for (float v : l.lut) printf(" %.9g", v);
    printf("\n");
  }
  return 0;
}

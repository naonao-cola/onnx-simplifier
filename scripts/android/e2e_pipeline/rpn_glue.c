/* Loads the fused RPN span's model constants (levels.txt, model.txt, lN_anchors.bin in the cwd)
 * with ../tinygrad_hexagon_bridge/rpn_fused/rpn_model_io.h, unchanged, and uploads them to the
 * DSP skel once. Returns the final proposal cap and per level {A, H, W} for the C++ driver. */
#include <stdlib.h>
#include <string.h>
#include "remote.h"
#include "rpcmem.h"
#include "rpn_rpc.h"
#include "rpn_model_io.h"

int rpn_glue_set_model(remote_handle64 h, int* post_cap, int* lv3) {
  static rpn_model M;
  float* an[RPN_NLVL];
  rpn_load_model(".", ".", &M, an);
  long atot = 0;
  for (int l = 0; l < RPN_NLVL; l++) atot += 4L * M.lv[l].A;
  float* anc = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 4 * atot);
  if (!anc) return -1;
  for (long o = 0, l = 0; l < RPN_NLVL; o += 4L * M.lv[l].A, l++) memcpy(anc + o, an[l], 16L * M.lv[l].A);
  int grid = -1;
  int rc = rpn_rpc_set_model(h, anc, atot, (const unsigned char*)&M, sizeof M, &grid);
  rpcmem_free(anc);
  for (int l = 0; l < RPN_NLVL; l++) {
    free(an[l]);
    lv3[3 * l] = M.lv[l].A;
    lv3[3 * l + 1] = M.lv[l].H;
    lv3[3 * l + 2] = M.lv[l].W;
  }
  *post_cap = M.post_cap;
  return rc;
}

/* CPU build of the merged uint8 RoiAlign (roialign_u8_kernel.h, portable vector path) with exactly the
 * argument layout of roialign_u8_rpc_run (roialign_u8_rpc.idl): a host reference / fallback, and what
 * ../../e2e_pipeline/roialign_u8_emulate.py calls through ctypes. Same bytes as the DSP skel.
 *   clang -O2 -shared -fPIC -o libroialign_u8_cpu.so roialign_u8_cpu.c   (clang: vector builtins) */
#include <stdlib.h>
#include "roialign_u8_kernel.h"

int roialign_u8_cpu_run(const uint8_t* const maps[4], const int32_t level_geom[12], const float level_fparams[8],
                        const int32_t level_counts[4], const float* rois, const int32_t* rows, int C, int OH,
                        int OW, int sr, float s_out, int z_out, int flags, uint8_t* out, long out_len) {
  if (C % 128 || C > 128 * RU8_MAX_HALVES || sr < 1 || sr > RU8_MAX_SR || OH * sr > RU8_MAX_AXIS ||
      OW * sr > RU8_MAX_AXIS)
    return -1;
  ru8_level_t lv[4];
  int n = 0;
  for (int k = 0; k < 4; k++) {
    lv[k] = (ru8_level_t){maps[k], level_geom[3 * k], level_geom[3 * k + 1], level_fparams[2 * k + 1],
                          level_geom[3 * k + 2], 0, 0};
    ru8_requant_params(level_fparams[2 * k], s_out, sr * sr, &lv[k].mult, &lv[k].shift);
    n += level_counts[k];
  }
  const long row_bytes = (long)OH * OW * C;
  ru8_job_t* jobs = malloc(sizeof(ru8_job_t) * (n + 1));
  if (!jobs) return -3;
  for (int k = 0, j = 0; k < 4; k++)
    for (int i = 0; i < level_counts[k]; i++, j++) {
      if (rows[j] < 0 || (rows[j] + 1) * row_bytes > out_len) { free(jobs); return -1; }
      jobs[j].level = k; jobs[j].row = rows[j];
      for (int c = 0; c < 4; c++) jobs[j].box[c] = rois[4 * j + c];
    }
  if (flags & 0x200) ru8_sort_jobs(jobs, n);
  ru8_run_jobs(lv, C, jobs, n, OH, OW, sr, z_out, 0, out);
  free(jobs);
  return 0;
}

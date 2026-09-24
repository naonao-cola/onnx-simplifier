/* dfa_core.h's scalar body on the host vs torch (dfa_case.py cases).
 *   cc -O2 -I ../../../msda_hvx -o dfa_host_check dfa_host_check.c -lm && ./dfa_host_check <case dir>... */
#include "dfa_io.h"

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    dfa_args_t a;
    memset(&a, 0, sizeof a);
    dfa_case_load(argv[i], &a);
    uint16_t* stage = NULL;
    const char* mode = getenv("DFA_W") ? getenv("DFA_W") : "";  /* "16": fp16 weights, "2": v2 layout */
    float* wbuf = !strcmp(mode, "16") ? dfa_use_w16(&a) : !strcmp(mode, "2") ? dfa_use_w2(&a, &stage) : NULL;
    if (dfa_check(&a)) { printf("%s: shape rejected\n", argv[i]); return 1; }
    dfa_run(&a, 0, a.Q, wbuf, stage);
    long vis = 0;
    for (long k = 0; k < (long)DFA_CAMS * a.Q; k++) vis += a.vis[k];
    printf("%s: Q %d, visible (cam, anchor) pairs %ld / %d\n", argv[i], a.Q, vis, DFA_CAMS * a.Q);
    dfa_compare(argv[i], "ref_u8.f32", &a);
    dfa_compare(argv[i], "ref_f32.f32", &a);
  }
  return 0;
}

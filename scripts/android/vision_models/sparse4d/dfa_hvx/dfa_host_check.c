/* dfa_core.h's scalar body on the host vs torch (dfa_case.py cases).
 *   cc -O2 -I ../../../msda_hvx -o dfa_host_check dfa_host_check.c -lm && ./dfa_host_check <case dir>... */
#include "dfa_io.h"

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    dfa_args_t a;
    dfa_case_load(argv[i], &a);
    if (dfa_check(&a)) { printf("%s: shape rejected\n", argv[i]); return 1; }
    dfa_run(&a, 0, a.Q);
    long vis = 0;
    for (long k = 0; k < (long)DFA_CAMS * a.Q; k++) vis += a.vis[k];
    printf("%s: Q %d, visible (cam, anchor) pairs %ld / %d\n", argv[i], a.Q, vis, DFA_CAMS * a.Q);
    dfa_compare(argv[i], "ref_u8.f32", &a);
    dfa_compare(argv[i], "ref_f32.f32", &a);
  }
  return 0;
}

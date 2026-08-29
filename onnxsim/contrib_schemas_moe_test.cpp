/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * TEMPORARY diagnostic build: the real test content (committed at 76338d9)
 * is replaced with a minimal main() to determine whether the s390x
 * (big-endian) CI crash -- instant, zero captured output, not even from a
 * flushed breadcrumb printed as the very first statement of main() -- is
 * caused by this test's actual logic (schema registration, function-body
 * building) or by something in how this specific binary is built/linked/
 * executed on that architecture. If this minimal version also fails
 * identically, the cause is not in the test logic. Restore the real
 * content once that's determined.
 */
#include <cstdio>

int main() {
  std::fprintf(stderr, "[diagnostic] minimal main reached\n");
  std::fflush(stderr);
  return 0;
}

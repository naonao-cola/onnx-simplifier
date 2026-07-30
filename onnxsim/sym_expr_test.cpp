// Standalone test: g++ -std=c++17 sym_expr.cpp sym_expr_test.cpp -o t && ./t
// (Builds identically under Emscripten: em++ -std=c++17 ...)
#include "sym_expr.h"

#include <cassert>
#include <iostream>

using onnxsim::SymExpr;
using onnxsim::SymRatio;

int main() {
  const SymExpr batch = SymExpr::Symbol("batch");
  const SymExpr seq = SymExpr::Symbol("seq");

  // --- building a MAC formula, exactly as the counters + _prod would ------
  // 512*batch*seq**2 + 5419008*batch
  const SymExpr macs =
      SymExpr(512) * batch * seq * seq + SymExpr(5419008) * batch;
  assert(macs.is_symbolic());
  assert(macs.str() == "512*batch*seq**2 + 5419008*batch");

  // sympy.factor pulls the integer content (512) as well as the shared batch.
  assert(macs.str_factored() == "512*batch*(seq**2 + 10584)");

  // representative == subs(all symbols -> 1): 512 + 5419008.
  assert(macs.representative() == 512 + 5419008);

  // --- a fully concrete value collapses to a plain integer ----------------
  const SymExpr concrete = SymExpr(6) * SymExpr(7);
  assert(!concrete.is_symbolic());
  assert(concrete.to_int() == 42);
  assert(concrete.str() == "42");

  // --- accumulation with cancellation -------------------------------------
  SymExpr acc = batch;
  acc += SymExpr(-1) * batch;  // batch - batch -> 0
  assert(acc.is_zero());
  assert(acc.str() == "0");

  // --- compute_density: numeric ratio prints a decimal --------------------
  const SymRatio dens_numeric(SymExpr(300), SymExpr(100));
  assert(!dens_numeric.is_symbolic());
  assert(dens_numeric.str() == "3.00");

  // --- compute_density: symbolic ratio cancels the common factor ----------
  // (2*batch*seq) / batch  ->  2*seq
  const SymRatio dens_sym(SymExpr(2) * batch * seq, batch);
  assert(dens_sym.is_symbolic());
  assert(dens_sym.str() == "2*seq");

  // ratio that does not fully cancel keeps a denominator
  // (batch*seq) / (batch + seq)  ->  batch*seq/(batch + seq)
  const SymRatio dens_partial(batch * seq, batch + seq);
  assert(dens_partial.str() == "batch*seq/(batch + seq)");

  std::cout << "all SymExpr tests passed\n";
  std::cout << "  macs           = " << macs.str() << "\n";
  std::cout << "  macs (factored)= " << macs.str_factored() << "\n";
  std::cout << "  density        = " << dens_sym.str() << "\n";
  return 0;
}

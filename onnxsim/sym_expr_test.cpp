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

  // --- subtraction (shape arithmetic: Slice end-start, Sub) ---------------
  assert((SymExpr(5) - SymExpr(3)).to_int() == 2);
  assert((SymExpr(2) * batch - batch).str() == "batch");

  // --- exact division: Reshape's -1, (batch*1024*128) // (1024*128) -> batch
  {
    const SymExpr total = batch * SymExpr(1024) * SymExpr(128);
    const SymExpr non_deferred = SymExpr(1024) * SymExpr(128);
    const auto q = onnxsim::TryExactDivide(total, non_deferred);
    assert(q.has_value() && q->str() == "batch");
  }
  // divide a polynomial by a shared symbol: (512*batch*seq + 8*batch)/batch
  {
    const SymExpr n = SymExpr(512) * batch * seq + SymExpr(8) * batch;
    const auto q = onnxsim::TryExactDivide(n, batch);
    assert(q.has_value() && q->str() == "512*seq + 8");
  }
  // not evenly divisible -> nullopt: (batch + seq) / seq (batch term has no
  // seq)
  assert(!onnxsim::TryExactDivide(batch + seq, seq).has_value());
  // coefficient not divisible -> nullopt: (3*batch) / 2
  assert(!onnxsim::TryExactDivide(SymExpr(3) * batch, SymExpr(2)).has_value());
  // a genuine polynomial (multi-term) divisor is unsupported -> nullopt
  assert(!onnxsim::TryExactDivide(batch * seq, batch + seq).has_value());
  // 0 / anything -> 0
  assert(onnxsim::TryExactDivide(SymExpr(0), batch)->is_zero());

  // --- decidable equality for Equal/Where ---------------------------------
  {
    auto e = onnxsim::TryEqual(SymExpr(512), SymExpr(512));
    assert(e.has_value() && *e);  // equal concretes -> true
  }
  {
    auto e = onnxsim::TryEqual(SymExpr(512), SymExpr(1024));
    assert(e.has_value() && !*e);  // unequal concretes -> false
  }
  {
    auto e = onnxsim::TryEqual(batch, batch);
    assert(e.has_value() && *e);  // structurally identical -> true
  }
  // symbolic vs constant is undecidable -> nullopt (Where leaves it unresolved)
  assert(!onnxsim::TryEqual(batch, SymExpr(512)).has_value());

  std::cout << "all SymExpr tests passed\n";
  std::cout << "  macs           = " << macs.str() << "\n";
  std::cout << "  macs (factored)= " << macs.str_factored() << "\n";
  std::cout << "  density        = " << dens_sym.str() << "\n";
  return 0;
}

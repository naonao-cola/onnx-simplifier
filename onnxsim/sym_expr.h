#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace onnxsim {

// A polynomial in dimension symbols ("batch", "seq", ...) with int64
// coefficients. This is the *entire* symbolic surface model_info needs: MAC
// counts, byte counts and memory traffic are all sums of products of tensor
// dimensions, so they are exactly integer-coefficient polynomials in the
// dim_param names.
//
// Pure standard C++ with no external dependency, so it builds unchanged under
// Emscripten/WASM -- where SymEngine (which needs GMP, or an experimental
// boost-multiprecision build) is not practically available.
//
// Maps onto the sympy calls in onnxsim/model_info.py:
//   sympy.Symbol(name, positive=True, integer=True) -> SymExpr::Symbol(name)
//   a * b, a + b (MAC counters, _prod)              -> operator* / operator+
//   expr.free_symbols / _is_symbolic                -> is_symbolic()
//   expr.subs({s: 1 for s in ...})                  -> representative()
//   int(value)                                      -> to_int()
//   str(expr)                                       -> str()
//   sympy.factor(expr)                              -> str_factored()
class SymExpr {
 public:
  // A monomial is the sorted multiset of symbol names in one product term:
  //   {}                    -> 1                (the constant term)
  //   {"batch"}             -> batch
  //   {"batch","seq","seq"} -> batch*seq**2
  using Monomial = std::vector<std::string>;

  SymExpr() = default;
  // Implicit so `SymExpr(512) * batch` and `expr * 2` read naturally, mirroring
  // Python's int/sympy mixing. A zero coefficient stores no term.
  SymExpr(std::int64_t c) {
    if (c != 0) terms_[Monomial{}] = c;
  }

  static SymExpr Symbol(const std::string& name) {
    SymExpr e;
    e.terms_[Monomial{name}] = 1;
    return e;
  }

  bool is_zero() const { return terms_.empty(); }

  // True iff any term carries a symbol (i.e. the value is a formula, not a
  // plain number) -- the C++ equivalent of model_info._is_symbolic.
  bool is_symbolic() const {
    for (const auto& [mono, coeff] : terms_) {
      (void)coeff;
      if (!mono.empty()) return true;
    }
    return false;
  }

  // subs(every symbol -> 1): collapse to one number for ordering and for the
  // summary table's "which is smaller" highlighting. Never the reported value.
  std::int64_t representative() const {
    std::int64_t sum = 0;
    for (const auto& [mono, coeff] : terms_) {
      (void)mono;
      sum += coeff;
    }
    return sum;
  }

  // Exact value; precondition: !is_symbolic().
  std::int64_t to_int() const {
    auto it = terms_.find(Monomial{});
    return it == terms_.end() ? 0 : it->second;
  }

  SymExpr& operator+=(const SymExpr& o) {
    for (const auto& [mono, coeff] : o.terms_) {
      std::int64_t& c = terms_[mono];
      c += coeff;
      if (c == 0) terms_.erase(mono);
    }
    return *this;
  }

  friend SymExpr operator+(SymExpr a, const SymExpr& b) {
    a += b;
    return a;
  }

  // a - b, needed by shape arithmetic (Slice's end - start, Sub) and by
  // TryEqual below. Defined as a + (-1)*b to reuse the existing operators.
  friend SymExpr operator-(SymExpr a, const SymExpr& b) {
    a += SymExpr(-1) * b;
    return a;
  }

  friend SymExpr operator*(const SymExpr& a, const SymExpr& b) {
    SymExpr r;
    for (const auto& [m1, c1] : a.terms_) {
      for (const auto& [m2, c2] : b.terms_) {
        Monomial m;
        m.reserve(m1.size() + m2.size());
        // Both operands are sorted, so merge keeps the product sorted.
        std::merge(m1.begin(), m1.end(), m2.begin(), m2.end(),
                   std::back_inserter(m));
        std::int64_t& c = r.terms_[m];
        c += c1 * c2;
        if (c == 0) r.terms_.erase(m);
      }
    }
    return r;
  }

  SymExpr& operator*=(const SymExpr& o) {
    *this = *this * o;
    return *this;
  }

  // "512*batch*seq**2 + 5419008*batch" -- sympy-compatible (** for powers).
  std::string str() const;

  // Common monomial and coefficient GCD pulled out, recovering what
  // sympy.factor produced for these polynomials:
  //   512*batch*seq**2 + 5419008*batch  ->  512*batch*(seq**2 + 10584)
  std::string str_factored() const;

  const std::map<Monomial, std::int64_t>& terms() const { return terms_; }

 private:
  // Invariant: no entry has a zero coefficient; every Monomial key is sorted.
  std::map<Monomial, std::int64_t> terms_;
};

// Exact division num / den, for the symbolic *shape* evaluation that a
// data-propagation pass needs (distinct from model_info's MAC formulas). The
// canonical case is Reshape's -1 sentinel, resolved as
//   total // non_deferred_size   e.g.  (batch*1024*128) // (1024*128) -> batch
// and strided/pooled dims. Divisors there are always a single term (a product
// of dimensions times a constant), so only single-term `den` is supported.
//
// Returns the quotient q with q*den == num when den divides num exactly (every
// term's coefficient divisible by den's, and den's monomial a sub-multiset of
// each term's); std::nullopt when it does not divide evenly or den is a genuine
// polynomial sum. Never guesses -- an undecidable division stays unresolved so
// the caller leaves that shape symbolic rather than folding something wrong.
std::optional<SymExpr> TryExactDivide(const SymExpr& num, const SymExpr& den);

// Decidable equality for Equal/Where in shape evaluation. a == b iff a - b ==
// 0:
//   - structurally equal (a - b is 0)             -> true
//   - concrete, non-zero difference (e.g. 512,1024) -> false
//   - symbolic difference (e.g. batch vs 512)       -> std::nullopt (unknown)
// The unknown case is deliberate: a Where whose condition can't be decided
// leaves its output unresolved rather than picking a branch that may be wrong.
std::optional<bool> TryEqual(const SymExpr& a, const SymExpr& b);

// compute_density = flops / mem_access is the one metric that is a *ratio* of
// two polynomials rather than a polynomial, so it gets its own small type. When
// numeric it prints a plain decimal (like model_info.human_readable_density);
// when symbolic it cancels the common monomial and coefficient GCD across
// numerator and denominator for a readable formula.
class SymRatio {
 public:
  SymRatio(SymExpr num, SymExpr den)
      : num_(std::move(num)), den_(std::move(den)) {}

  bool is_symbolic() const { return num_.is_symbolic() || den_.is_symbolic(); }

  // num/den with every symbol set to 1 (for ordering only).
  double representative() const;

  std::string str() const;

 private:
  SymExpr num_;
  SymExpr den_;
};

}  // namespace onnxsim

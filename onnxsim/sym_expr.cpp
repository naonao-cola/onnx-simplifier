#include "sym_expr.h"

#include <sstream>
#include <utility>

namespace onnxsim {
namespace {

using Monomial = SymExpr::Monomial;

std::int64_t gcd_i(std::int64_t a, std::int64_t b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  while (b != 0) {
    std::int64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// Per-symbol powers of a monomial: {"batch","seq","seq"} -> {batch:1, seq:2}.
std::map<std::string, std::size_t> powers_of(const Monomial& mono) {
  std::map<std::string, std::size_t> c;
  for (const std::string& s : mono) ++c[s];
  return c;
}

// "batch*seq**2" for {"batch","seq","seq"}; "" for the empty (constant) mono.
std::string monomial_str(const Monomial& mono) {
  std::string out;
  for (std::size_t i = 0; i < mono.size();) {
    std::size_t j = i;
    while (j < mono.size() && mono[j] == mono[i]) ++j;
    const std::size_t power = j - i;
    if (!out.empty()) out += "*";
    out += mono[i];
    if (power > 1) out += "**" + std::to_string(power);
    i = j;
  }
  return out;
}

// One term rendered without its sign: "512*batch", "batch", "42".
std::string term_str(const Monomial& mono, std::int64_t coeff) {
  const std::string ms = monomial_str(mono);
  if (ms.empty()) return std::to_string(coeff);  // constant term
  if (coeff == 1) return ms;
  return std::to_string(coeff) + "*" + ms;
}

// Readable term order: higher total degree first, then lexicographic. This puts
// the leading term first and the constant last, as sympy's str printer does.
std::vector<std::pair<Monomial, std::int64_t>> ordered(
    const std::map<Monomial, std::int64_t>& terms) {
  std::vector<std::pair<Monomial, std::int64_t>> v(terms.begin(), terms.end());
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
    if (a.first.size() != b.first.size())
      return a.first.size() > b.first.size();
    return a.first < b.first;
  });
  return v;
}

std::string join_terms(const std::map<Monomial, std::int64_t>& terms) {
  const auto v = ordered(terms);
  if (v.empty()) return "0";
  std::string out = term_str(v[0].first, v[0].second);  // keeps its own sign
  for (std::size_t i = 1; i < v.size(); ++i) {
    const std::int64_t c = v[i].second;
    out += (c < 0) ? " - " : " + ";
    out += term_str(v[i].first, c < 0 ? -c : c);
  }
  return out;
}

// The largest factor common to a set of expressions: gcd of every coefficient,
// and the minimum power of each symbol present in *every* term.
struct CommonFactor {
  std::int64_t content = 0;  // gcd of |coeff|; 0 only if there are no terms
  std::map<std::string, std::size_t> monomial;
};

CommonFactor common_factor(std::initializer_list<const SymExpr*> exprs) {
  CommonFactor cf;
  bool first = true;
  for (const SymExpr* e : exprs) {
    for (const auto& [mono, coeff] : e->terms()) {
      cf.content = gcd_i(cf.content, coeff);
      const auto p = powers_of(mono);
      if (first) {
        cf.monomial = p;
        first = false;
      } else {
        for (auto it = cf.monomial.begin(); it != cf.monomial.end();) {
          const auto f = p.find(it->first);
          const std::size_t here = (f == p.end()) ? 0 : f->second;
          it->second = std::min(it->second, here);
          if (it->second == 0)
            it = cf.monomial.erase(it);
          else
            ++it;
        }
      }
    }
  }
  return cf;
}

// e divided by the common factor, built with the public API only. Every term of
// e is divisible by cf (that is what "common" means), so this is exact.
SymExpr divide_by(const SymExpr& e, const CommonFactor& cf) {
  const std::int64_t content = (cf.content == 0) ? 1 : cf.content;
  SymExpr reduced;
  for (const auto& [mono, coeff] : e.terms()) {
    auto p = powers_of(mono);
    for (const auto& [name, power] : cf.monomial) p[name] -= power;
    SymExpr term(coeff / content);
    for (const auto& [name, power] : p)
      for (std::size_t k = 0; k < power; ++k) term *= SymExpr::Symbol(name);
    reduced += term;
  }
  return reduced;
}

std::string factor_str(const CommonFactor& cf) {
  Monomial mono;
  for (const auto& [name, power] : cf.monomial)
    mono.insert(mono.end(), power, name);
  std::sort(mono.begin(), mono.end());
  return term_str(mono, cf.content == 0 ? 1 : cf.content);
}

// Wrap a sum in parentheses so it composes safely into a larger product/ratio.
std::string wrap(const SymExpr& e) {
  const std::string s = e.str();
  return e.terms().size() > 1 ? "(" + s + ")" : s;
}

}  // namespace

std::optional<SymExpr> TryExactDivide(const SymExpr& num, const SymExpr& den) {
  // Only single-term divisors (a product of dimensions times a constant); a
  // genuine polynomial sum is never a shape-arithmetic divisor.
  if (den.terms().size() != 1) return std::nullopt;
  const auto& [den_mono, den_coeff] = *den.terms().begin();
  if (den_coeff == 0) return std::nullopt;  // division by zero
  const auto den_powers = powers_of(den_mono);

  SymExpr quotient;  // starts at 0, so num == 0 divides to 0
  for (const auto& [mono, coeff] : num.terms()) {
    if (coeff % den_coeff != 0)
      return std::nullopt;  // coefficient not divisible
    auto powers = powers_of(mono);
    for (const auto& [name, power] : den_powers) {
      const auto it = powers.find(name);
      if (it == powers.end() || it->second < power) {
        return std::nullopt;  // this term is not divisible by den's monomial
      }
      it->second -= power;
    }
    // Rebuild the quotient term coeff/den_coeff * (remaining symbols), using
    // the public API only (mirrors divide_by above).
    SymExpr term(coeff / den_coeff);
    for (const auto& [name, power] : powers)
      for (std::size_t k = 0; k < power; ++k) term *= SymExpr::Symbol(name);
    quotient += term;
  }
  return quotient;
}

std::optional<bool> TryEqual(const SymExpr& a, const SymExpr& b) {
  const SymExpr diff = a - b;
  if (diff.is_zero()) return true;        // a and b are structurally identical
  if (!diff.is_symbolic()) return false;  // concrete, non-zero difference
  return std::nullopt;                    // symbolic difference: undecidable
}

std::string SymExpr::str() const { return join_terms(terms_); }

std::string SymExpr::str_factored() const {
  if (terms_.size() <= 1) return str();  // a single term is already factored
  const CommonFactor cf = common_factor({this});
  const bool nothing_to_pull =
      (cf.content == 0 || cf.content == 1) && cf.monomial.empty();
  if (nothing_to_pull) return str();
  return factor_str(cf) + "*(" + divide_by(*this, cf).str() + ")";
}

double SymRatio::representative() const {
  const std::int64_t d = den_.representative();
  if (d == 0) return 0.0;
  return static_cast<double>(num_.representative()) / static_cast<double>(d);
}

std::string SymRatio::str() const {
  if (!is_symbolic()) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed << representative();
    return os.str();
  }
  const CommonFactor cf = common_factor({&num_, &den_});
  const SymExpr n = divide_by(num_, cf);
  const SymExpr d = divide_by(den_, cf);
  if (!d.is_symbolic() && d.to_int() == 1) return wrap(n);
  return wrap(n) + "/" + wrap(d);
}

}  // namespace onnxsim

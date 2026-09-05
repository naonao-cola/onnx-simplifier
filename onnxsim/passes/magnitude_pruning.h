// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Magnitude pruning (Han et al., 2015) -- the data-free unstructured
// pruning baseline, C++ port of pruning.py's own apply_magnitude_pruning.
// Zeros the least-magnitude entries of every MatMul/vanilla-Gemm layer's
// constant 2-D FLOAT/FLOAT16/BFLOAT16 weight, every Conv layer's constant
// 4-D FLOAT/FLOAT16/BFLOAT16 weight (ordinary, depthwise, and general
// grouped Conv alike -- see pruning.py's own module docstring for why
// grouping needs no special-casing for unstructured pruning), and every
// ``com.microsoft::Attention``/``DecoderMaskedSelfAttention``/
// ``PackedAttention`` node's constant 2-D FLOAT/FLOAT16/BFLOAT16 merged QKV
// weight, independently per output row/filter so a layer with
// row-dependent weight scale doesn't get some rows pruned to nothing and
// others left untouched.
//
// Full parity with pruning.py's own ``apply_magnitude_pruning``: this port
// also offers N:M (semi-structured) pruning (``MagnitudePruningN()``/
// ``MagnitudePruningM()``, mirroring pruning.py's own ``_nm_mask`` exactly)
// and a ``global_sparsity`` mode (``MagnitudePruningGlobal``, a
// ``FullGraphBasedPass`` mirroring pruning.py's own
// ``_apply_global_unstructured_pruning`` exactly) in addition to the default
// per-layer sparsity-ratio mode.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- and Conv are handled the same way, reshaped to
// [out_channels, flattened_rest] first):
//   Y = MatMul(X, W)         W constant, [K, N], FLOAT/FLOAT16/BFLOAT16
// After:
//   Wp = <W with each output channel's own lowest-|w| fraction zeroed>
//   Y  = MatMul(X, Wp)
//
// The weight's shape, dtype, and every other node attribute are unchanged;
// only element values within the existing tensor are zeroed (via a new
// initializer of the same shape, not an in-place edit -- matching every
// other onnxsim rewrite's "replace, don't mutate" convention for constants).
// A matched weight is read out upcast to float64 for the importance/masking
// math below (mirrors pruning.py's own ``_to_f64``/``_from_f64`` "FP16/
// BFloat16 weight support" convention) and the result cast back down to
// that layer's own original dtype before being written back, so a fp16/bf16
// model's declared dtypes are preserved exactly -- masking never changes a
// surviving entry's own value, only zeros dropped ones. Since the
// replacement is still a plain constant tensor of the same shape/dtype, both
// ``MagnitudePruningMatMul``/``MagnitudePruningConv``/
// ``MagnitudePruningAttention`` predicates below explicitly check whether
// pruning would still change anything (any would-be-dropped entry that
// isn't already exactly zero) -- otherwise, unlike every other pass in this
// directory (whose replacement is no longer constant, or has changed
// kind/shape), the predicate would keep matching its own already-pruned
// output forever, looping OptimizeFixed's fixed point indefinitely.
// ``MagnitudePruningGlobal`` needs no such guard: it is a
// ``FullGraphBasedPass`` with an ``Empty`` analysis type, so
// ``FixedPointPassManager`` runs it exactly once regardless (see
// ``pass_manager.cc``: a pass reporting ``Empty`` analysis is never
// re-invoked in the same fixed-point loop, unlike a ``PredicateBasedPass``'s
// own ``CountBasedPassAnalysis``).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/quantize_conv_common.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Target fraction of each row's entries to zero, ignored when
// MagnitudePruningN()/MagnitudePruningM() are set (N:M mode). Function-local
// statics, the same pattern QuantizeFp16KeepIoTypes() (quantize_fp16.h) uses
// to pass a parameter into a pass that OptimizeFixed's pass-name-list
// interface has no other way to carry -- set by PruneMagnitude
// (pruning_entry.cpp) immediately before calling OptimizeFixed.
inline double& MagnitudePruningSparsity() {
  static double sparsity = 0.5;
  return sparsity;
}

// N:M semi-structured pruning parameters: keep the ``n`` highest-magnitude
// entries per group of ``m`` columns. -1 (the default for both) means
// "N:M mode is off -- use MagnitudePruningSparsity() instead", mirroring
// pruning.py's own ``n``/``m`` being ``None`` together. Set together (both
// > 0, with ``0 < n <= m`` already validated by PruneMagnitude) or left at
// -1 together; never set to conflicting/partial values.
inline int64_t& MagnitudePruningN() {
  static int64_t n = -1;
  return n;
}
inline int64_t& MagnitudePruningM() {
  static int64_t m = -1;
  return m;
}

// --- FLOAT16/BFloat16 support --------------------------------------------
//
// Mirrors pruning.py's own "FP16/BFloat16 weight support" section
// (_is_supported_float_dtype/_to_f64/_from_f64) exactly, but written locally
// against onnxoptimizer's in-memory Tensor/Graph/Node IR rather than
// pruning.py's onnx.TensorProto/numpy, and rather than
// structured_pruning_entry.cpp's own IsSupportedFloatDtype/ReadTensorAsF64/
// WriteF64TensorAs (a genuinely different IR -- raw onnx::TensorProto, not
// this optimizer pass framework's Tensor -- so those helpers' int32_data/
// raw_data plumbing doesn't carry over unchanged; the underlying bit-level
// conversion algorithms below are still the exact same math, just against
// this file's own Tensor accessors).

// True for FLOAT, FLOAT16, and BFLOAT16 -- every element dtype this file's
// matchers accept for a weight initializer, mirroring pruning.py's own
// `_is_supported_float_dtype` exactly.
inline bool IsSupportedFloatDtype(int32_t elem_type) {
  return elem_type == TensorProto_DataType_FLOAT ||
         elem_type == TensorProto_DataType_FLOAT16 ||
         elem_type == TensorProto_DataType_BFLOAT16;
}

// IEEE-754 binary16 -> double, exact (every half value is exactly
// representable in double). Handles zero/subnormal/normal/inf/NaN.
inline double Float16BitsToDouble(uint16_t bits) {
  uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
  uint32_t exp = (bits >> 10) & 0x1Fu;
  uint32_t mant = bits & 0x3FFu;
  uint32_t f32bits;
  if (exp == 0) {
    if (mant == 0) {
      f32bits = sign;
    } else {
      int e = -1;
      do {
        ++e;
        mant <<= 1;
      } while (!(mant & 0x400u));
      mant &= 0x3FFu;
      uint32_t exp32 = (uint32_t)(127 - 15 - e);
      f32bits = sign | (exp32 << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f32bits = sign | 0x7F800000u | (mant << 13);
  } else {
    uint32_t exp32 = exp - 15 + 127;
    f32bits = sign | (exp32 << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &f32bits, sizeof(f));
  return (double)f;
}

// double -> IEEE-754 binary16, round-to-nearest-even. Works directly off the
// double's own 52-bit mantissa/11-bit exponent -- deliberately NOT
// `(float)d` first, which would round TWICE (double -> float32 -> half) and
// can flip a tie-to-even decision right at half's own rounding boundary
// versus rounding straight from double. Only ever actually applied to values
// that came FROM Float16BitsToDouble in the first place (this pass never
// recomputes a kept weight value, only reorders/drops entries -- see this
// file's own top comment), so the exact rounding behavior for a
// genuinely-arbitrary double is academic here: every real call site
// round-trips an already-exact half value, for which any correct
// round-to-nearest implementation reproduces the original bits exactly.
inline uint16_t DoubleToFloat16Bits(double d) {
  uint64_t x;
  std::memcpy(&x, &d, sizeof(x));
  uint32_t sign = (uint32_t)((x >> 48) & 0x8000u);
  int32_t exp11 = (int32_t)((x >> 52) & 0x7FFu);
  uint64_t mant = x & 0xFFFFFFFFFFFFFull;  // 52 bits

  if (exp11 == 0x7FF) {
    if (mant == 0) {
      return (uint16_t)(sign | 0x7C00u);  // inf
    }
    uint16_t m = (uint16_t)(mant >> 42);
    if (m == 0) m = 1;  // stay a NaN, not inf
    return (uint16_t)(sign | 0x7C00u | m);
  }

  int32_t exp = exp11 - 1023 + 15;

  if (exp >= 0x1F) {
    return (uint16_t)(sign | 0x7C00u);  // overflow -> inf
  }

  if (exp <= 0) {
    if (exp < -10) {
      return (uint16_t)sign;  // underflow to zero
    }
    mant |= 0x10000000000000ull;  // implicit leading 1 (bit 52)
    int shift = 43 - exp;         // exp<=0 so shift>=43, mant has 53 bits
    uint64_t half_mant = mant >> shift;
    uint64_t remainder = mant & ((1ull << shift) - 1);
    uint64_t halfway = 1ull << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) {
      half_mant += 1;
    }
    return (uint16_t)(sign | (uint32_t)half_mant);
  }

  uint32_t half_mant = (uint32_t)(mant >> 42);
  uint64_t remainder = mant & ((1ull << 42) - 1);  // low 42 bits
  uint64_t halfway = 1ull << 41;
  if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) {
    half_mant += 1;
    if (half_mant == 0x400u) {
      half_mant = 0;
      exp += 1;
    }
  }
  if (exp >= 0x1F) {
    return (uint16_t)(sign | 0x7C00u);
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
}

// BFloat16 -> double, exact: bfloat16 is literally the top 16 bits of a
// float32, so this is a plain zero-extend into a float32 bit pattern.
inline double BFloat16BitsToDouble(uint16_t bits) {
  uint32_t f32bits = (uint32_t)bits << 16;
  float f;
  std::memcpy(&f, &f32bits, sizeof(f));
  return (double)f;
}

// double -> BFloat16, round-to-nearest-even (via an intermediate float32 --
// unlike DoubleToFloat16Bits, this introduces no double-rounding hazard:
// bfloat16 IS float32's own top 16 bits, so rounding double -> float32 ->
// bfloat16 is the same "round once, truncate the rest" every real bfloat16
// implementation does).
inline uint16_t DoubleToBFloat16Bits(double d) {
  float f = (float)d;
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
    return (uint16_t)((x >> 16) | 0x0040u);  // NaN -- force the quiet bit.
  }
  uint32_t lsb = (x >> 16) & 1u;
  uint32_t rounding_bias = 0x7FFFu + lsb;
  uint32_t rounded = x + rounding_bias;
  return (uint16_t)(rounded >> 16);
}

// Reads a FLOAT/FLOAT16/BFLOAT16 constant `t` (any rank) into a flat
// float64 vector, mirroring pruning.py's own `_to_f64`. Non-raw-data
// storage packs each FLOAT16/BFLOAT16 element's own raw 16 bits into the
// LOW half of one `int32s()` entry -- see ir_pb_converter.cc's own
// `tensorProtoToTensorGeneric`, which imports TensorProto's `int32_data`
// verbatim for these two dtypes.
inline std::vector<double> ReadTensorAsF64Flat(const Tensor& t) {
  int64_t numel = 1;
  for (int64_t s : t.sizes()) {
    numel *= s;
  }
  std::vector<double> out(static_cast<size_t>(numel));
  if (t.elem_type() == TensorProto_DataType_FLOAT) {
    const std::vector<float> f = ReadFloatTensorFlat(t);
    for (size_t i = 0; i < f.size(); ++i) {
      out[i] = static_cast<double>(f[i]);
    }
    return out;
  }
  std::vector<uint16_t> bits(static_cast<size_t>(numel));
  if (t.is_raw_data()) {
    bits = ReadRawDataHostOrder<uint16_t>(
        reinterpret_cast<const uint16_t*>(t.raw().data()), numel);
  } else {
    const std::vector<int32_t>& codes = t.int32s();
    for (int64_t i = 0; i < numel; ++i) {
      bits[static_cast<size_t>(i)] =
          static_cast<uint16_t>(codes[static_cast<size_t>(i)]);
    }
  }
  const bool is_bf16 = t.elem_type() == TensorProto_DataType_BFLOAT16;
  for (size_t i = 0; i < bits.size(); ++i) {
    out[i] =
        is_bf16 ? BFloat16BitsToDouble(bits[i]) : Float16BitsToDouble(bits[i]);
  }
  return out;
}

// Overwrites `out` (a fresh Tensor) with a `elem_type` (FLOAT/FLOAT16/
// BFLOAT16) tensor of `sizes`/`data` -- mirrors pruning.py's own
// `_from_f64`. Always writes `raw_data` for FLOAT16/BFLOAT16 (via
// WriteRawDataLittleEndian, not int32s() -- see endian_read.h's own doc
// comment for why), matching `onnx.numpy_helper.from_array`'s own
// convention for these dtypes.
inline void WriteF64FlatAsTensor(Tensor& out, int32_t elem_type,
                                 const std::vector<int64_t>& sizes,
                                 const std::vector<double>& data) {
  out.elem_type() = elem_type;
  out.sizes() = sizes;
  if (elem_type == TensorProto_DataType_FLOAT) {
    std::vector<float> f(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      f[i] = static_cast<float>(data[i]);
    }
    out.floats() = std::move(f);
    return;
  }
  const bool is_bf16 = elem_type == TensorProto_DataType_BFLOAT16;
  std::vector<uint16_t> bits(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    bits[i] =
        is_bf16 ? DoubleToBFloat16Bits(data[i]) : DoubleToFloat16Bits(data[i]);
  }
  out.set_raw_data(WriteRawDataLittleEndian(bits));
}

// Reads `w_t` (a 2-D FLOAT/FLOAT16/BFLOAT16 constant, [K, N] or, when
// `transposed`, [N, K]) into a flat row-major [N, K] (output channel first)
// float64 buffer -- the float64-widened analogue of quantize_matmul_common.
// h's own `ReadWeightNK`, used here so SparsityMaskRowMajor/NMMaskRowMajor's
// per-row grouping lines up with output channels regardless of the weight's
// own on-disk (transposed or not) layout or dtype.
inline std::vector<double> ReadWeightNKF64(const Tensor& w_t, bool transposed) {
  const int64_t dim0 = w_t.sizes()[0];
  const int64_t dim1 = w_t.sizes()[1];
  const int64_t rows = transposed ? dim0 : dim1;
  const int64_t cols = transposed ? dim1 : dim0;
  const std::vector<double> data = ReadTensorAsF64Flat(w_t);
  std::vector<double> w_nk(static_cast<size_t>(rows * cols));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const double v = data[static_cast<size_t>(i * dim1 + j)];
      if (transposed) {
        w_nk[static_cast<size_t>(i * cols + j)] = v;
      } else {
        w_nk[static_cast<size_t>(j * cols + i)] = v;
      }
    }
  }
  return w_nk;
}

// Inverse of ReadWeightNKF64: converts a (possibly-masked) row-major
// [rows, cols] `w_nk` buffer back to the original on-disk [dim0, dim1]
// layout (transposed or not).
inline std::vector<double> WeightNkToOriginalF64(
    const std::vector<double>& w_nk, int64_t dim0, int64_t dim1,
    bool transposed) {
  const int64_t cols = transposed ? dim1 : dim0;
  std::vector<double> out(static_cast<size_t>(dim0 * dim1));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      out[static_cast<size_t>(i * dim1 + j)] =
          transposed ? w_nk[static_cast<size_t>(i * cols + j)]
                     : w_nk[static_cast<size_t>(j * cols + i)];
    }
  }
  return out;
}

// Row-wise magnitude mask over a flat, row-major [rows, cols] `importance`
// matrix: within each row independently, keeps the
// max(1, round(cols * (1 - sparsity))) highest-importance entries and drops
// the rest -- mirrors pruning.py's own _sparsity_mask exactly (same keep
// count, same "never drop every entry of a row" floor), but via
// std::stable_sort per row instead of a vectorized argsort (tie-breaking on
// exactly-equal importances -- common with exact-zero weights -- may
// therefore differ from the Python port's own choice of which zero-tied
// entry to drop; both are equally valid magnitude-pruning outcomes).
inline std::vector<bool> SparsityMaskRowMajor(
    const std::vector<double>& importance, int64_t rows, int64_t cols,
    double sparsity) {
  std::vector<bool> mask(importance.size(), true);
  // std::nearbyint (current rounding mode is FE_TONEAREST, i.e.
  // round-half-to-even, by default) rather than std::lround
  // (round-half-away-from-zero) to match Python's own round() -- the two
  // differ only on an exact .5 tie, but pruning.py's _sparsity_mask uses
  // Python's round() for its own keep count.
  const int64_t keep = std::max<int64_t>(
      1, static_cast<int64_t>(std::nearbyint(cols * (1.0 - sparsity))));
  if (keep >= cols) {
    return mask;
  }
  const int64_t drop = cols - keep;
  std::vector<int64_t> order(static_cast<size_t>(cols));
  for (int64_t r = 0; r < rows; ++r) {
    const double* row = importance.data() + r * cols;
    for (int64_t c = 0; c < cols; ++c) {
      order[static_cast<size_t>(c)] = c;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) { return row[a] < row[b]; });
    for (int64_t i = 0; i < drop; ++i) {
      mask[static_cast<size_t>(r * cols + order[static_cast<size_t>(i)])] =
          false;
    }
  }
  return mask;
}

// Row-wise N:M mask over a flat, row-major [rows, cols] `importance` matrix:
// within every consecutive group of `m` columns, keeps only the `n`
// highest-importance entries -- mirrors pruning.py's own `_nm_mask` exactly,
// including its trailing-partial-group handling: a final group of fewer
// than `m` columns keeps a proportional share (`min(tail, max(1,
// round(n * tail / m)))`) instead of requiring `cols` to be a multiple of
// `m`. Caller must ensure `0 < n <= m` (validated once by PruneMagnitude at
// the top-level entry point, mirroring pruning.py's own `_validate_pattern`).
inline std::vector<bool> NMMaskRowMajor(const std::vector<double>& importance,
                                        int64_t rows, int64_t cols, int64_t n,
                                        int64_t m) {
  std::vector<bool> mask(importance.size(), true);
  const int64_t full_cols = (cols / m) * m;
  std::vector<int64_t> order(static_cast<size_t>(m));
  for (int64_t r = 0; r < rows; ++r) {
    const double* row = importance.data() + r * cols;
    for (int64_t base = 0; base < full_cols; base += m) {
      for (int64_t i = 0; i < m; ++i) {
        order[static_cast<size_t>(i)] = i;
      }
      std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return row[base + a] < row[base + b];
      });
      const int64_t drop = m - n;
      for (int64_t i = 0; i < drop; ++i) {
        mask[static_cast<size_t>(r * cols + base +
                                 order[static_cast<size_t>(i)])] = false;
      }
    }
    const int64_t tail = cols - full_cols;
    if (tail > 0) {
      const int64_t keep = std::min<int64_t>(
          tail, std::max<int64_t>(1, static_cast<int64_t>(std::nearbyint(
                                         static_cast<double>(n) * tail / m))));
      const int64_t drop = tail - keep;
      std::vector<int64_t> torder(static_cast<size_t>(tail));
      for (int64_t i = 0; i < tail; ++i) {
        torder[static_cast<size_t>(i)] = i;
      }
      std::stable_sort(torder.begin(), torder.end(), [&](int64_t a, int64_t b) {
        return row[full_cols + a] < row[full_cols + b];
      });
      for (int64_t i = 0; i < drop; ++i) {
        mask[static_cast<size_t>(r * cols + full_cols +
                                 torder[static_cast<size_t>(i)])] = false;
      }
    }
  }
  return mask;
}

// Dispatches to NMMaskRowMajor (when MagnitudePruningN()/M() are set) or
// SparsityMaskRowMajor (otherwise) over `data`'s own |.| importance --
// the single mask entry point every per-layer pass (MatMul/Conv/Attention)
// below shares, so N:M mode is "free" once a matcher/writer is expressed in
// terms of this function rather than SparsityMaskRowMajor directly.
inline std::vector<bool> MagnitudePruningMaskRowMajor(
    const std::vector<double>& data, int64_t rows, int64_t cols) {
  std::vector<double> importance(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    importance[i] = std::fabs(data[i]);
  }
  const int64_t n = MagnitudePruningN();
  const int64_t m = MagnitudePruningM();
  if (n > 0 && m > 0) {
    return NMMaskRowMajor(importance, rows, cols, n, m);
  }
  return SparsityMaskRowMajor(importance, rows, cols,
                              MagnitudePruningSparsity());
}

// Whether applying MagnitudePruningMaskRowMajor's own mask to `data`
// (row-major [rows, cols], matching layout) would zero out at least one
// currently nonzero entry -- see this file's own doc comment for why every
// per-layer predicate below must check this before matching.
inline bool MagnitudePruningWouldChange(const std::vector<double>& data,
                                        int64_t rows, int64_t cols) {
  const std::vector<bool> mask = MagnitudePruningMaskRowMajor(data, rows, cols);
  for (size_t i = 0; i < data.size(); ++i) {
    if (!mask[i] && data[i] != 0.0) {
      return true;
    }
  }
  return false;
}

// Applies `mask` (row-major [rows, cols]) to `data` (the same flat row-major
// layout), zeroing every masked-out entry in place.
inline void ApplyMaskRowMajor(std::vector<double>& data,
                              const std::vector<bool>& mask) {
  for (size_t i = 0; i < data.size(); ++i) {
    if (!mask[i]) {
      data[i] = 0.0;
    }
  }
}

// --- com.microsoft Attention-family merged-QKV-weight matching -----------
//
// Mirrors pruning.py's own `_match_attention_weight_only`, which reuses
// `_match_attention_producer`'s own validation UNRESTRICTED by op_type --
// so, to match it exactly, this also accepts `com.microsoft::Attention`'s
// two schema-compatible siblings, `DecoderMaskedSelfAttention` (ONNX
// Runtime's own in-place rewrite target for a step-of-1 decoder `Attention`
// node) and `PackedAttention` (ONNX Runtime's own packing-mode fusion
// target for `Attention`) -- not just plain `Attention` itself. All three
// share `weight`/`bias` at input indices 1/2 and `attention_bias` at index
// 5; `PackedAttention` needs no extra handling beyond that (its own indices
// 3/4 are packing-mode bookkeeping tensors this matcher never inspects,
// exactly like `Attention`'s own optional `mask_index`/`past` at those same
// positions); `DecoderMaskedSelfAttention` needs the three extra checks
// below (`do_rotary`, `qkv_hidden_sizes` absence, a non-constant `past`).

struct AttentionQkvMatch {
  Value* x = nullptr;  // activation (not required to be constant)
  Value* w = nullptr;  // merged QKV weight; constant 2-D FLOAT/FLOAT16/
                       // BFLOAT16 tensor, [K, Nq+Nk+Nv]
};

inline bool IsMicrosoftAttentionOp(Node* n) {
  if (!n->has_domain() || n->domain() != "com.microsoft") {
    return false;
  }
  return CheckKind(n, "Attention") ||
         CheckKind(n, "DecoderMaskedSelfAttention") ||
         CheckKind(n, "PackedAttention");
}

// Classifies a broadcastable per-head `attention_bias` constant's shape
// against the schema-documented rank-4 layout `(batch_size or 1, num_heads
// or 1, q_sequence_length, kv_sequence_length)` -- mirrors pruning.py's own
// `_head_bias_axis`. A rank < 3 tensor is unconditionally head-count-
// independent (true); rank 3 or 4 resolves to a genuine per-head axis at
// position `rank - 3`, safe only when that axis's size is exactly 1
// (broadcast) or `num_heads` (true either way); rank > 4, or a rank-3/4
// tensor whose axis lands on neither 1 nor `num_heads`, is unresolvable
// (false) -- the caller declines the whole match rather than guess, the
// same "don't guess at a malformed node" bar pruning.py's own version
// holds to.
inline bool HeadBiasAxisIsSafe(const std::vector<int64_t>& dims,
                               int64_t num_heads) {
  const int64_t r = static_cast<int64_t>(dims.size());
  if (r > 4) {
    return false;
  }
  if (r < 3) {
    return true;
  }
  const int64_t axis = r - 3;
  const int64_t sz = dims[static_cast<size_t>(axis)];
  return sz == 1 || sz == num_heads;
}

// If `n` is a `com.microsoft` Attention-family node (see
// `IsMicrosoftAttentionOp`) with a constant 2-D FLOAT/FLOAT16/BFLOAT16
// merged QKV weight (`[K, Nq+Nk+Nv]`), fills `info` and returns true.
// Mirrors pruning.py's own `_match_attention_producer` validation
// (`num_heads`/`qkv_hidden_sizes` consistency, and the `attention_bias`
// shape gate for a *constant* one -- a dynamic `attention_bias` never
// blocks the match here, exactly like pruning.py's own
// `_match_attention_weight_only`, which always passes
// `value_info_by_name=None`: unstructured/magnitude pruning never changes
// `num_heads` or any shape at all, so a per-head `attention_bias`'s own
// shape can never go stale under it) even though nothing here reads
// `num_heads` itself for masking -- so a node this module's own structural
// head-pruning family would decline as malformed is declined the same way
// here. `do_rotary`/a constant `past`/`qkv_hidden_sizes` presence are
// DecoderMaskedSelfAttention-only concerns, checked below only for that op.
inline bool MatchAttentionQkvWeightOnly(Node* n, AttentionQkvMatch& info) {
  if (!IsMicrosoftAttentionOp(n) || n->inputs().size() < 2) {
    return false;
  }
  const bool is_dmsa = CheckKind(n, "DecoderMaskedSelfAttention");
  const Tensor* w_t = FetchConstantTensor(n->input(1));
  if (w_t == nullptr || !IsSupportedFloatDtype(w_t->elem_type()) ||
      w_t->sizes().size() != 2) {
    return false;
  }
  const int64_t total_n = w_t->sizes()[1];

  // bias (input 2), optional.
  if (n->inputs().size() > 2 && n->input(2)->node()->kind() != kUndefined) {
    const Tensor* b_t = FetchConstantTensor(n->input(2));
    if (b_t == nullptr || !IsSupportedFloatDtype(b_t->elem_type()) ||
        b_t->sizes().size() != 1 || b_t->sizes()[0] != total_n) {
      return false;
    }
  }

  // DecoderMaskedSelfAttention's own `past` (input 4) is REQUIRED by its
  // schema and holds combined key+value decode state -- a real export
  // always leaves it dynamic (runtime state), but a hand-built graph could
  // bind a constant there; this matcher has no established/tested slicing
  // path for that (mirrors pruning.py's own `_match_attention_producer`
  // exactly, even though this pass never slices anything at all -- the
  // point is matching the same node set, not this matcher's own slicing
  // ability), so a constant `past` conservatively declines the whole match.
  if (is_dmsa && n->inputs().size() > 4 &&
      n->input(4)->node()->kind() != kUndefined) {
    if (FetchConstantTensor(n->input(4)) != nullptr) {
      return false;
    }
  }

  int64_t num_heads = 0;
  if (!GetValueFromAttr(n, "num_heads", num_heads) || num_heads <= 0) {
    return false;
  }

  int64_t do_rotary = 0;
  GetValueFromAttr(n, "do_rotary", do_rotary);  // optional; default 0
  if (is_dmsa && do_rotary) {
    return false;  // fused RoPE -- no confirmed per-head-safe path here
  }

  std::vector<int64_t> qkv_hidden_sizes;
  const bool has_qkv_hidden_sizes =
      GetValueFromAttr(n, "qkv_hidden_sizes", qkv_hidden_sizes);
  if (is_dmsa && has_qkv_hidden_sizes) {
    // Not a real attribute on DecoderMaskedSelfAttention's own schema --
    // decline conservatively rather than guess what a hand-edited graph
    // carrying it anyway might have intended.
    return false;
  }

  int64_t nq, nk, nv;
  if (has_qkv_hidden_sizes) {
    if (qkv_hidden_sizes.size() != 3) {
      return false;
    }
    nq = qkv_hidden_sizes[0];
    nk = qkv_hidden_sizes[1];
    nv = qkv_hidden_sizes[2];
  } else {
    // Schema default: Q/K/V evenly split the merged width.
    if (total_n % 3 != 0) {
      return false;
    }
    nq = nk = nv = total_n / 3;
  }
  if (nq <= 0 || nk <= 0 || nv <= 0 || nq + nk + nv != total_n ||
      nq % num_heads || nk % num_heads || nv % num_heads) {
    return false;
  }

  // attention_bias (input 5), optional.
  if (n->inputs().size() > 5 && n->input(5)->node()->kind() != kUndefined) {
    const Tensor* bias_t = FetchConstantTensor(n->input(5));
    if (bias_t != nullptr) {
      int64_t numel = 1;
      for (int64_t d : bias_t->sizes()) {
        numel *= d;
      }
      if (numel != 0 && !HeadBiasAxisIsSafe(bias_t->sizes(), num_heads)) {
        return false;  // doesn't statically resolve -- decline rather than
                       // guess
      }
    }
    // A dynamic (non-constant) attention_bias is always OK here -- see this
    // function's own doc comment.
  }

  info.x = n->input(0);
  info.w = n->input(1);
  return true;
}

// ReadWeightNKF64 reads a 2-D MatMul/Gemm/Attention weight into a flat
// row-major [N, K] (output channel first) buffer, used here so
// SparsityMaskRowMajor/NMMaskRowMajor's per-row grouping lines up with
// output channels regardless of the weight's own on-disk (transposed or
// not) layout.

struct MagnitudePruningMatMul final : public PredicateBasedPass {
  explicit MagnitudePruningMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "magnitude_pruning_matmul";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || !IsSupportedFloatDtype(w_t->elem_type()) ||
        w_t->sizes().size() != 2) {
      return false;
    }
    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    const int64_t rows = info.weight_transposed ? dim0 : dim1;
    const int64_t cols = info.weight_transposed ? dim1 : dim0;
    const std::vector<double> w_nk =
        ReadWeightNKF64(*w_t, info.weight_transposed);
    return MagnitudePruningWouldChange(w_nk, rows, cols);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || !IsSupportedFloatDtype(w_t->elem_type()) ||
        w_t->sizes().size() != 2) {
      return false;
    }

    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    const int64_t rows = info.weight_transposed ? dim0 : dim1;
    const int64_t cols = info.weight_transposed ? dim1 : dim0;

    std::vector<double> w_nk = ReadWeightNKF64(*w_t, info.weight_transposed);
    const std::vector<bool> mask =
        MagnitudePruningMaskRowMajor(w_nk, rows, cols);
    ApplyMaskRowMajor(w_nk, mask);
    const std::vector<double> out_flat =
        WeightNkToOriginalF64(w_nk, dim0, dim1, info.weight_transposed);

    Tensor w_pruned;
    WriteF64FlatAsTensor(w_pruned, w_t->elem_type(), w_t->sizes(), out_flat);

    Value* w_pruned_v = graph.addInitializerAndCreateValue(w_pruned);
    n->replaceInput(1, w_pruned_v);
    return true;
  }
};

struct MagnitudePruningConv final : public PredicateBasedPass {
  explicit MagnitudePruningConv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "magnitude_pruning_conv"; }

  bool patternMatchPredicate(Node* n) override {
    ConvInfo info;
    if (!MatchConv(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || !IsSupportedFloatDtype(w_t->elem_type()) ||
        w_t->sizes().size() != 4) {
      return false;
    }
    const int64_t rows = w_t->sizes()[0];
    int64_t cols = 1;
    for (size_t i = 1; i < w_t->sizes().size(); ++i) {
      cols *= w_t->sizes()[i];
    }
    const std::vector<double> data = ReadTensorAsF64Flat(*w_t);
    return MagnitudePruningWouldChange(data, rows, cols);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    ConvInfo info;
    if (!MatchConv(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || !IsSupportedFloatDtype(w_t->elem_type()) ||
        w_t->sizes().size() != 4) {
      return false;
    }

    // [out_channels, in_channels/groups, kH, kW] -> [rows=out_channels,
    // cols=flattened rest], the same reshape pruning.py's own _prune_weight
    // uses for Conv.
    const int64_t rows = w_t->sizes()[0];
    int64_t cols = 1;
    for (size_t i = 1; i < w_t->sizes().size(); ++i) {
      cols *= w_t->sizes()[i];
    }

    std::vector<double> data = ReadTensorAsF64Flat(*w_t);
    const std::vector<bool> mask =
        MagnitudePruningMaskRowMajor(data, rows, cols);
    ApplyMaskRowMajor(data, mask);

    Tensor w_pruned;
    WriteF64FlatAsTensor(w_pruned, w_t->elem_type(), w_t->sizes(), data);

    Value* w_pruned_v = graph.addInitializerAndCreateValue(w_pruned);
    n->replaceInput(1, w_pruned_v);
    return true;
  }
};

struct MagnitudePruningAttention final : public PredicateBasedPass {
  explicit MagnitudePruningAttention()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "magnitude_pruning_attention";
  }

  bool patternMatchPredicate(Node* n) override {
    AttentionQkvMatch info;
    if (!MatchAttentionQkvWeightOnly(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr) {
      return false;  // defensive: the matcher above already confirmed this
    }
    // Attention's merged QKV weight has no transpose attribute of its own --
    // it is already [K, N]-shaped by construction (mirrors pruning.py's own
    // `_match_attention_weight_only` doc comment) -- so it is read exactly
    // like a non-transposed MatMul weight.
    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    const std::vector<double> w_nk =
        ReadWeightNKF64(*w_t, /*transposed=*/false);
    return MagnitudePruningWouldChange(w_nk, dim1, dim0);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    AttentionQkvMatch info;
    if (!MatchAttentionQkvWeightOnly(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr) {
      return false;
    }
    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];

    std::vector<double> w_nk = ReadWeightNKF64(*w_t, /*transposed=*/false);
    const std::vector<bool> mask =
        MagnitudePruningMaskRowMajor(w_nk, dim1, dim0);
    ApplyMaskRowMajor(w_nk, mask);
    const std::vector<double> out_flat =
        WeightNkToOriginalF64(w_nk, dim0, dim1, /*transposed=*/false);

    Tensor w_pruned;
    WriteF64FlatAsTensor(w_pruned, w_t->elem_type(), w_t->sizes(), out_flat);

    Value* w_pruned_v = graph.addInitializerAndCreateValue(w_pruned);
    n->replaceInput(1, w_pruned_v);
    return true;
  }
};

// --- global_sparsity mode --------------------------------------------------
//
// Companion to MagnitudePruningMatMul/Conv/Attention's own per-layer
// sparsity-ratio masking (MagnitudePruningMaskRowMajor's own per-row rule,
// applied independently per layer): pools every matched layer's own |W|
// entries across the WHOLE model (every graph -- top-level and every nested
// If/Loop/Scan/BeamSearch-family subgraph body, recursively, exactly like
// pruning.py's own `_iter_subgraphs` -- into ONE combined ranking) into one
// flat array, picks a single keep-count from MagnitudePruningSparsity()'s
// fraction of the total pooled entry count, and zeros exactly the
// lowest-scoring entries wherever they land -- mirrors pruning.py's own
// `_apply_global_unstructured_pruning` exactly (including its "no per-row
// floor" property: a whole row, or even a whole layer's weight, can
// legitimately end up all-zero here). Incompatible with N:M mode -- enforced
// once by PruneMagnitude (pruning_entry.cpp) before this pass ever runs, so
// this pass itself only ever reads MagnitudePruningSparsity().
//
// Implemented as a single FullGraphBasedPass (not three PredicateBasedPass
// instances) because the global threshold genuinely needs every matched
// layer's importance gathered up front before any single layer's mask can
// be decided -- unlike the per-layer passes above, whose predicate/transform
// only ever needs that one node's own weight.
struct MagnitudePruningGlobal final : public FullGraphBasedPass {
  explicit MagnitudePruningGlobal()
      : FullGraphBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "magnitude_pruning_global";
  }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  struct Entry {
    Node* node = nullptr;  // consumer whose weight input (index 1) is replaced
    bool is_conv = false;
    bool transposed = false;  // MatMul/Attention only; Conv is never transposed
    int32_t elem_type = TensorProto_DataType_UNDEFINED;
    std::vector<int64_t> sizes;  // original on-disk tensor sizes
    int64_t dim0 = 0, dim1 = 0;  // MatMul/Attention only (2-D on-disk shape)
    std::vector<double> w_nk;    // row-major [rows, cols], SparsityMaskRowMajor
                                 // layout
    std::vector<double> importance;  // |w_nk|, same shape/layout
  };

  // Every Graph reachable from `g`, including `g` itself, recursively
  // through every node's own GRAPH-/GRAPHS-typed attribute -- mirrors
  // pruning.py's own `_iter_subgraphs` (keyed on attribute *kind*, not a
  // per-op-name allowlist, so a future op with another graph-typed
  // attribute needs no update here either).
  void CollectGraphs(Graph& g, std::vector<Graph*>& out) {
    out.push_back(&g);
    for (Node* n : g.nodes()) {
      DescendOnGraphAttributesUnconstrained(
          n, [&](Graph& sub) { CollectGraphs(sub, out); });
    }
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    std::vector<Graph*> graphs;
    CollectGraphs(graph, graphs);

    std::vector<Entry> entries;
    for (Graph* g : graphs) {
      for (Node* n : g->nodes()) {
        MatMulLikeInfo mm;
        ConvInfo conv;
        AttentionQkvMatch attn;
        if (MatchMatMulLike(n, mm)) {
          const Tensor* w_t = FetchConstantTensor(mm.w);
          if (w_t == nullptr || !IsSupportedFloatDtype(w_t->elem_type()) ||
              w_t->sizes().size() != 2) {
            continue;
          }
          Entry e;
          e.node = n;
          e.is_conv = false;
          e.transposed = mm.weight_transposed;
          e.elem_type = w_t->elem_type();
          e.sizes = w_t->sizes();
          e.dim0 = w_t->sizes()[0];
          e.dim1 = w_t->sizes()[1];
          e.w_nk = ReadWeightNKF64(*w_t, e.transposed);
          e.importance.resize(e.w_nk.size());
          for (size_t i = 0; i < e.w_nk.size(); ++i) {
            e.importance[i] = std::fabs(e.w_nk[i]);
          }
          entries.push_back(std::move(e));
        } else if (MatchConv(n, conv)) {
          const Tensor* w_t = FetchConstantTensor(conv.w);
          if (w_t == nullptr || !IsSupportedFloatDtype(w_t->elem_type()) ||
              w_t->sizes().size() != 4) {
            continue;
          }
          Entry e;
          e.node = n;
          e.is_conv = true;
          e.elem_type = w_t->elem_type();
          e.sizes = w_t->sizes();
          e.w_nk = ReadTensorAsF64Flat(*w_t);
          e.importance.resize(e.w_nk.size());
          for (size_t i = 0; i < e.w_nk.size(); ++i) {
            e.importance[i] = std::fabs(e.w_nk[i]);
          }
          entries.push_back(std::move(e));
        } else if (MatchAttentionQkvWeightOnly(n, attn)) {
          const Tensor* w_t = FetchConstantTensor(attn.w);
          if (w_t == nullptr || !IsSupportedFloatDtype(w_t->elem_type()) ||
              w_t->sizes().size() != 2) {
            continue;
          }
          Entry e;
          e.node = n;
          e.is_conv = false;
          e.transposed = false;
          e.elem_type = w_t->elem_type();
          e.sizes = w_t->sizes();
          e.dim0 = w_t->sizes()[0];
          e.dim1 = w_t->sizes()[1];
          e.w_nk = ReadWeightNKF64(*w_t, /*transposed=*/false);
          e.importance.resize(e.w_nk.size());
          for (size_t i = 0; i < e.w_nk.size(); ++i) {
            e.importance[i] = std::fabs(e.w_nk[i]);
          }
          entries.push_back(std::move(e));
        }
      }
    }

    if (!entries.empty()) {
      int64_t total = 0;
      for (const Entry& e : entries) {
        total += static_cast<int64_t>(e.importance.size());
      }

      const double sparsity = MagnitudePruningSparsity();
      int64_t keep_count =
          static_cast<int64_t>(std::nearbyint(total * (1.0 - sparsity)));
      keep_count = std::max<int64_t>(0, std::min<int64_t>(keep_count, total));
      const int64_t drop_count = total - keep_count;

      std::vector<bool> drop_flat(static_cast<size_t>(total), false);
      if (drop_count > 0) {
        std::vector<double> pooled(static_cast<size_t>(total));
        int64_t off = 0;
        for (const Entry& e : entries) {
          std::copy(e.importance.begin(), e.importance.end(),
                    pooled.begin() + off);
          off += static_cast<int64_t>(e.importance.size());
        }
        std::vector<int64_t> order(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i) {
          order[static_cast<size_t>(i)] = i;
        }
        // Stable sort, ascending -- mirrors pruning.py's own
        // `np.argsort(pooled, kind="stable")`: ties at the cutoff are broken
        // by pooled order (entries order, then each entry's own row-major
        // flatten order), so `keep_count` is always met exactly.
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
          return pooled[static_cast<size_t>(a)] <
                 pooled[static_cast<size_t>(b)];
        });
        for (int64_t i = 0; i < drop_count; ++i) {
          drop_flat[static_cast<size_t>(order[static_cast<size_t>(i)])] = true;
        }
      }

      int64_t off = 0;
      for (Entry& e : entries) {
        const int64_t size = static_cast<int64_t>(e.w_nk.size());
        for (int64_t i = 0; i < size; ++i) {
          if (drop_flat[static_cast<size_t>(off + i)]) {
            e.w_nk[static_cast<size_t>(i)] = 0.0;
          }
        }
        off += size;

        const std::vector<double> out_flat =
            e.is_conv
                ? e.w_nk
                : WeightNkToOriginalF64(e.w_nk, e.dim0, e.dim1, e.transposed);
        Tensor w_pruned;
        WriteF64FlatAsTensor(w_pruned, e.elem_type, e.sizes, out_flat);
        Value* w_pruned_v =
            e.node->owningGraph()->addInitializerAndCreateValue(w_pruned);
        e.node->replaceInput(1, w_pruned_v);
      }
    }

    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE

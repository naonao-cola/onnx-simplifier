// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// A Haar-random orthogonal matrix generator, shared by quarot.h. Serves the
// same purpose as quip_sharp.py's own _random_orthogonal_matrix (see that
// module's docstring for why a uniformly random orthogonal matrix -- rather
// than a Hadamard-structured one, the real QuaRot/QuIP# papers' own
// construction -- suffices for the concentration-of-measure argument this
// rotation relies on, at the cost of an O(K^2) dense MatMul instead of an
// O(K log K) Fast Walsh-Hadamard Transform at deployment), but via a
// DIFFERENT algorithm -- deliberately, not as a gap to close. Read on.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM quarot.py / quip_sharp.py (do not
// "fix" this without re-reading this comment in full):
//
// quip_sharp.py's _random_orthogonal_matrix QR-decomposes a random Gaussian
// matrix via numpy.linalg.qr (LAPACK's Householder-reflector algorithm) and
// then corrects Q's sign using R's own diagonal (``d = sign(diag(r)); q *=
// d``). That correction is necessary there because LAPACK's Householder QR
// picks each reflector's sign for numerical stability (to avoid cancellation
// against the pivot), not at random -- so R's diagonal entries come out
// with an implementation-determined sign, and the resulting Q is NOT
// Haar-uniform without the fix (empirically: Q[0, 0] from an uncorrected
// numpy.linalg.qr on a Gaussian input comes out negative every single time
// for small matrices in a Monte Carlo check -- see the round-16
// investigation this comment summarizes).
//
// Gram-Schmidt -- what THIS file implements -- has no analogous defect and
// needs no analogous correction. Each row's R-equivalent diagonal entry
// here is literally a Euclidean norm (``norm = sqrt(norm_sq)`` above),
// which is always >= 0 by construction, matching exactly the sign
// convention the numpy-side correction has to impose by hand. A Monte
// Carlo comparison (K=5, 300k draws) of this row-wise Gram-Schmidt against
// quip_sharp.py's sign-corrected QR -- both starting from i.i.d. standard
// Gaussian input -- found their first-entry distributions statistically
// indistinguishable (max CDF gap ~0.0016, on par with splitting one
// distribution's own samples in half and comparing the halves, ~0.0022),
// while the SAME comparison against uncorrected QR showed a max CDF gap of
// ~0.5 (obviously biased). In short: Gram-Schmidt of a Gaussian matrix is
// ALREADY Haar-uniform on its own; it is a different, equally-valid
// construction, not "QR without the fix."
//
// Consequently, ApplyQuarot(model, seed=N) and apply_quarot(model,
// seed=N) (quarot.py) produce DIFFERENT rotation matrices for the same
// seed and dimensions, and are not expected to ever alias -- on top of
// which quarot.h's own per-node RNG derivation (a fresh std::mt19937_64
// reseeded per matched node, see QuarotSeed() in quarot.h) is already a
// separate, independently-accepted non-goal of cross-language bit parity
// with quarot.py's single sequentially-advancing numpy.random.Generator.
// Reconciling either difference would mean reimplementing numpy's PCG64
// bit generator and its ziggurat-based standard_normal sampler bit-for-bit
// in C++ (not just swapping in a QR routine) for a rotation whose only
// actual requirement -- per the QuaRot/QuIP# papers themselves -- is being
// SOME uniformly random orthogonal matrix, not any bit-specific one. Not
// worth doing. Both ports remain independently correct, non-interchangeable
// entry points; see quarot.py's apply_quarot docstring and quarot.h's own
// top-of-file comment for the user-facing version of this note.

#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// A Haar-random K x K orthogonal matrix (row-major, float32), built by
// (modified) Gram-Schmidt orthonormalizing the rows of a K x K
// standard-Gaussian random matrix. Orthonormalizing rows rather than
// columns is equivalent (a random Gaussian matrix's transpose is Gaussian
// too) and needs no transpose afterward: the result already satisfies
// U @ U^T == I, the property this rotation needs to be lossless on its own.
inline std::vector<float> RandomOrthogonalMatrix(int64_t k,
                                                 std::mt19937_64& rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> rows(static_cast<size_t>(k * k));
  for (double& v : rows) {
    v = normal(rng);
  }

  for (int64_t i = 0; i < k; ++i) {
    double* ri = rows.data() + i * k;
    for (int64_t j = 0; j < i; ++j) {
      const double* rj = rows.data() + j * k;
      double dot = 0.0;
      for (int64_t c = 0; c < k; ++c) {
        dot += ri[c] * rj[c];
      }
      for (int64_t c = 0; c < k; ++c) {
        ri[c] -= dot * rj[c];
      }
    }
    double norm_sq = 0.0;
    for (int64_t c = 0; c < k; ++c) {
      norm_sq += ri[c] * ri[c];
    }
    double norm = std::sqrt(norm_sq);
    if (norm < 1e-12) {
      norm = 1e-12;  // A degenerate row is vanishingly unlikely with
                     // continuous Gaussian input; floor rather than divide
                     // by zero.
    }
    for (int64_t c = 0; c < k; ++c) {
      ri[c] /= norm;
    }
  }

  std::vector<float> out(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    out[i] = static_cast<float>(rows[i]);
  }
  return out;
}

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE

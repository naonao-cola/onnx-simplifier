// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// A Haar-random orthogonal matrix generator, shared by quarot.h. Ports
// quip_sharp.py's own _random_orthogonal_matrix (QR-decomposing a random
// Gaussian matrix) -- see that module's docstring for why a uniformly
// random orthogonal matrix (rather than a Hadamard-structured one, the real
// QuaRot/QuIP# papers' own construction) suffices for the concentration-of-
// measure argument this rotation relies on, at the cost of an O(K^2) dense
// MatMul instead of an O(K log K) Fast Walsh-Hadamard Transform at
// deployment.

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

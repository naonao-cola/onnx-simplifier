#pragma once

// AffineQuant (Ma et al., 2024, ICLR, "AffineQuant: Affine Transformation
// Quantization for Large Language Models", https://arxiv.org/abs/2403.12544)
// entry point exposed to Python -- C++ port of onnxsim.affinequant's own
// apply_affinequant (see onnxsim/affinequant.py's module docstring for the
// full technique and its own scope note: a block-diagonal, not fully dense,
// generalization of onnxsim.omniquant's own diagonal Learnable Equivalent
// Transformation).
//
// Built on top of omniquant_entry.h's own ApplyOmniquant machinery -- same
// candidate matching (adaround.py's own _find_int4_matmul_candidates,
// transcribed identically), same calibration-activation capture, same LWC
// grid search and diagonal-LET grid search -- extended with a third,
// block-affine LET candidate (a per-block orthogonal rotation, from
// numpy.linalg.eigh's own C++ counterpart: a hand-rolled cyclic Jacobi
// eigendecomposition of each block's own symmetric PSD calibration
// covariance) searched the same alpha grid again in the rotated basis. Same
// self-containment convention as every other *_entry.cpp in this codebase
// (this file's own .cpp duplicates omniquant_entry.cpp's helpers rather
// than including it) -- see gptaq_entry.cpp's own top-of-file comment for
// why.
//
// `num_clip_steps`, `num_alpha_steps`, `min_clip_ratio` mirror
// ApplyOmniquant's own parameters of the same names exactly (and
// apply_affinequant's own, in turn). `affine_block_size` mirrors
// apply_affinequant's own parameter of the same name: the block-affine
// candidate is skipped (falling back to whichever of LWC-only/diagonal-LET
// already won) for a layer whose K isn't evenly divisible by it.
//
// Same `calibration_data`/`executor` shape, same per-layer skip conditions,
// and same FLOAT32-only scope as ApplyOmniquant.
//
// ACCEPTED, PERMANENT DIVERGENCE: this port's own per-block eigendecomposition
// is a hand-rolled cyclic Jacobi kernel (no LAPACK/BLAS linked into this
// codebase, matching spinquant_entry.h's/affinequant.py's own numpy.linalg.eigh
// call), so the eigenvector basis it returns for a block with repeated or
// near-degenerate eigenvalues can differ from numpy's own by more than a
// rotation-ambiguity-free sign/ordering choice -- any two orthonormal bases
// of the same eigenspace reconstruct the exact same rotation matrix up to
// that ambiguity, which this port's own tests verify by orthogonality and
// reconstruction error, not raw eigenvector-matrix byte equality (matching
// spinquant_entry.h's own identical acceptance note for its own
// eigendecomposition-based rotation). Everything else (the grid searches
// themselves) is bounded and deterministic, expected to track the Python
// reference closely the same way ApplyOmniquant's own is.
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see adaround_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

onnx::ModelProto ApplyAffinequant(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_clip_steps = 20, int64_t num_alpha_steps = 20,
    double min_clip_ratio = 0.5, int64_t affine_block_size = 8);

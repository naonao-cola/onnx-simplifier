#pragma once

// BRECQ (Li, Gong, Tan, Yang, Hu, Zhang, Yu, Wang, Gu, 2021, "BRECQ: Pushing
// the Limit of Post-Training Quantization by Block Reconstruction",
// https://arxiv.org/abs/2102.05426, ICLR 2021) entry point exposed to
// Python -- C++ port of onnxsim.brecq's own apply_brecq (see
// onnxsim/brecq.py's module docstring for the full technique and its own
// "what this module simplifies, honestly" scope notes -- block discovery
// only recognizes a linear MatMul/Gemm chain plus an optional trailing
// residual Add, and Fisher-information weighting is approximated by each
// output element's own empirical calibration variance, not a real
// task-loss gradient).
//
// adaround_entry.h's own ApplyAdaround is this port's primary template:
// same rectified-sigmoid relaxation (adaround.py's own _h_and_dhdv,
// transcribed identically -- see adaround_entry.cpp's own inlined h/dh_dv
// computation, duplicated here rather than shared), same hand-rolled Adam
// loop shape, same INT4 [-7, 7] range, same weight/scale layout
// normalization (adaround.py's own _layer_arrays, transcribed identically
// as LayerArrays below). What is new here, mirroring brecq.py's own
// _optimize_block_rounding: every layer inside a caller-delimited block is
// optimized *jointly* -- one shared forward pass runs the whole chain
// (each layer's current weight relaxation feeding the next layer's own
// input), and Adam updates every layer's own per-element relaxation from a
// single backward pass through the *block's own final output*
// reconstruction error (Fisher-diagonal-weighted, post-residual-add if the
// block has one), rather than each layer's own output independently
// (ApplyAdaround's own scope).
//
// Two-model (float_model, its quantize_weight_only_int4-quantized
// counterpart), calibration-driven, protobuf-level shape as
// ApplyAdaround's own. `blocks` mirrors apply_brecq's own parameter of the
// same name exactly: `(block_input_name, block_output_name)` pairs, one per
// residual/linear chain to jointly optimize. A pair whose topology isn't
// recognized (or that matches no quantized layer at all) is silently
// skipped -- mirrors apply_brecq's own tolerance for an unmatched pair
// exactly, the same "no matching candidate" tolerance ApplyAdaround itself
// has.
//
// `num_iterations`, `learning_rate`, `reg_param`, `warm_start`,
// `beta_start`/`beta_end` (the two ends of apply_brecq's own `beta_range`
// tuple, split into separate parameters here since this binding layer has
// no tuple type, matching ApplyAdaround's own `beta_start`/`beta_end`
// split) mirror apply_brecq's own parameters of the same names exactly.
// `fisher_eps` mirrors apply_brecq's own parameter of the same name (the
// numerical floor added before normalizing the empirical per-element
// variance into a Fisher-diagonal loss weight).
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `float_model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `float_model`'s own graph inputs throws `std::invalid_argument`. A block
// whose calibration activations don't pair up (per-batch row-count
// mismatch between its own input and final-output probes) is left
// untouched -- mirrors apply_brecq's own per-batch pairing tolerance
// exactly (see brecq.py's own comment on why batches, not a flat
// concatenation, are paired).
//
// FLOAT32-only throughout, mirroring every other calibration-driven pass in
// this codebase's own FLOAT32-only scope.
//
// ACCEPTED NUMERICAL SCOPE (same class as ApplyAdaround's own): this is a
// `num_iterations`-step Adam optimization, not a single closed-form
// computation -- floating-point summation-order differences between this
// TU's own scalar dense-matmul kernels and numpy's own (possibly
// BLAS-backed) `@` can compound across iterations, and the joint block
// forward/backward pass here compounds that further across every layer in
// the chain (each layer's own gradient depends on every downstream layer's
// current weights). Measured empirically rather than assumed correct --
// see tests/test_brecq_cpp.py for exactly how closely (or not) this tracks
// the pure-Python reference.
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declaration only -- see adaround_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

onnx::ModelProto ApplyBrecq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::pair<std::string, std::string>>& blocks,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations = 300, double learning_rate = 0.1,
    double reg_param = 0.01, double warm_start = 0.2, double beta_start = 20.0,
    double beta_end = 2.0, double fisher_eps = 1e-3);

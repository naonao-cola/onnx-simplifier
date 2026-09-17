#pragma once

// Calibration-driven pruning-recovery fine-tuning entry point exposed to
// Python -- C++ port of onnxsim.finetune's own apply_pruning_finetune
// (see onnxsim/finetune.py's own module docstring for the full
// technique, its "exactness argument" for why a single closed-form pass
// suffices, and its scope boundaries: MatMul/vanilla-Gemm only, a layer
// pruned on both its own input and output channels at once is left
// untouched, no Conv support).
//
// Unlike every other calibration-driven *_entry.h in this codebase
// (GPTQ-family Hessian/Cholesky machinery, AdaRound/BRECQ-family
// iterative Adam optimization), this is an ORDINARY (convex) ridge-
// regression least-squares fit, solved in closed form by one dense
// linear system per layer -- no iterative optimizer, no rounding
// relaxation, no Hessian beyond the fit's own Gram matrix. Closest in
// SHAPE to gptq_entry.h's own ApplyGptq (a TWO-model, protobuf-level,
// calibration-driven pass: `original_model`/`pruned_model` here play
// exactly the role `float_model`/`quantized_model` play there), but the
// numerical method itself is unrelated to any Hessian-compensated
// rounding scheme.
//
// Like every calibration-driven pass in this codebase, this operates
// directly on onnx::GraphProto rather than through onnxoptimizer's
// Node/Value IR (no established path to thread a live ModelExecutor
// through OptimizeFixed's PredicateBasedPass model). Unlike every other
// pass in this family, though, this one never inserts a single new node
// -- pruning already left the graph in its final topological shape; this
// pass only OVERWRITES existing weight/bias initializer VALUES in place
// (same node, same input names, same output names), matching
// apply_pruning_finetune's own final `t.CopyFrom(...)`-only rewrite loop
// exactly.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Fine-tunes every surviving MatMul/vanilla-Gemm layer present (by node
// output name and op type) in both `original_model` and `pruned_model`
// to better reconstruct `original_model`'s own real activations over
// `calibration_data`, run through `executor`. For each matched layer:
//
// 1. That layer's own surviving output/input channel `keep` index sets
//    are reconstructed from the pruned weight's own content alone, by
//    exact (bit-identical, after widening to float64) forward two-
//    pointer subsequence matching against the original weight -- see
//    onnxsim/finetune.py's own "Channel correspondence" docstring
//    section for the full argument (and its own stated boundary: a
//    layer pruned on both its own axes at once is declined outright,
//    never guessed at).
// 2. `X` is `original_model`'s own captured activation restricted to
//    `keep_in`; `Y` is `original_model`'s own analytically recomputed
//    output (`X_full @ W_orig^T [+ bias_orig]`) restricted to
//    `keep_out` -- one probe point per layer, mirroring ApplyAdaround's
//    own single-probe-point-per-layer design (this technique's own
//    "exactness argument": a channel that survives pruning carries
//    EXACTLY `original_model`'s own value at that tensor, so
//    `pruned_model` itself never needs to be re-run).
// 3. The new weight (and bias, if the layer has one) minimizes
//    `||X @ W^T [+ b] - Y||^2 + reg_param * ||[W, b] - [W_pruned,
//    b_pruned]||^2` -- ordinary ridge regression, solved by ONE dense
//    linear system per layer (`reg_param` scaled relative to the fit's
//    own Gram matrix trace, matching apply_pruning_finetune's own
//    `_ridge_fit` exactly).
//
// A layer whose weight isn't found to be a clean row/column subsequence
// of its own `original_model` counterpart (declined per step 1 above,
// including the both-axes-pruned case), or whose captured activation has
// no feature axis at all (rank < 2 after leading-axis flattening), or
// whose feature dim does not match the weight's own reduction dimension
// `K`, is left completely untouched -- mirrors apply_pruning_finetune's
// own per-layer skip conditions exactly.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `original_model`'s own graph inputs) shape as
// every other calibration-driven pass in this codebase -- a
// `calibration_data` batch missing one of `original_model`'s own graph
// inputs throws `std::invalid_argument`. NOT subgraph-aware.
//
// `reg_param` is the ridge-regression regularization strength (relative
// to the fit's own Gram matrix scale) -- mirrors apply_pruning_finetune's
// own parameter of the same name/default exactly.
//
// FLOAT32-only throughout (weights, biases and activations alike),
// mirroring apply_pruning_finetune's own FLOAT-only weight requirement
// and every other calibration-driven pass's own FLOAT32-only scope: a
// FLOAT16/BFLOAT16 activation is never observed (skipped like an
// unobserved one), never converted.
//
// Accepted numerical scope: the dense linear solve at this fit's heart
// (a small, `(K [+1]) x (K [+1])` symmetric system per layer, `K` the
// layer's own surviving reduction dimension) is computed with this TU's
// own scalar double-precision Gaussian-elimination-with-partial-pivoting
// kernel rather than LAPACK's own `numpy.linalg.solve` -- results can
// differ from the reference in the last few ulps on some inputs,
// tracking it closely rather than bit-exactly. See
// tests/test_pruning_finetune_cpp.py for the measured agreement.
onnx::ModelProto ApplyPruningFinetune(
    const onnx::ModelProto& original_model,
    const onnx::ModelProto& pruned_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double reg_param = 1e-2);

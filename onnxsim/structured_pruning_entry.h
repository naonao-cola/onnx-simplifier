#pragma once

// Structured (channel) and attention-head pruning entry points exposed to
// Python, mirroring pruning_entry.h's own "not a Quantize* scheme"
// rationale for its separate file. Unlike every pass in onnxsim/passes/
// (which run through onnxoptimizer's Node/Value IR via OptimizeFixed),
// both operate directly on onnx::GraphProto: the algorithm needs
// whole-graph, multi-hop forward/backward tensor-name-based analysis
// (producer/consumer maps, chain walking) that maps far more directly onto
// the same protobuf-level approach onnxsim/pruning.py's own reference
// implementation takes than onto the optimizer's PredicateBasedPass
// single-node-match model -- see structured_pruning_entry.cpp's own
// top-of-file comment for the details and scope of both ports (attention-
// head pruning lives in the same translation unit specifically to reuse
// its producer/consumer/slicing helpers directly, rather than duplicating
// them). See onnxsim.h for the documentation this mirrors.

#include <onnx/onnx_pb.h>

onnx::ModelProto ApplyStructuredPruning(const onnx::ModelProto& model,
                                        double sparsity);

onnx::ModelProto ApplyAttentionHeadPruning(const onnx::ModelProto& model,
                                           double sparsity);

// Removes intermediate (`inter_size`) channels from every expert of a
// matched `com.microsoft::MoE` node at once -- real structural pruning
// (smaller fc1/fc2 weight tensors, smaller per-expert matmuls on any
// runtime), the C++ port of pruning.py's own
// `apply_moe_expert_channel_pruning`. `num_experts` (whole-expert pruning,
// which needs runtime calibration data this build has no ONNX Runtime to
// provide -- see CLAUDE.md) is out of scope; see
// structured_pruning_entry.cpp's own "MoE (com.microsoft::MoE) expert-
// intermediate-channel pruning" section comment for the full scope and
// safety argument.
onnx::ModelProto ApplyMoeExpertChannelPruning(const onnx::ModelProto& model,
                                              double sparsity);

onnx::ModelProto ApplyQMoEExpertChannelPruning(const onnx::ModelProto& model,
                                               double sparsity);

#pragma once

// Structured (channel) pruning entry point exposed to Python, mirroring
// pruning_entry.h's own "not a Quantize* scheme" rationale for its separate
// file. Unlike every pass in onnxsim/passes/ (which run through
// onnxoptimizer's Node/Value IR via OptimizeFixed), this operates directly
// on onnx::GraphProto: the algorithm needs whole-graph, multi-hop forward/
// backward tensor-name-based analysis (producer/consumer maps, chain
// walking) that maps far more directly onto the same protobuf-level
// approach onnxsim/pruning.py's own reference implementation takes than
// onto the optimizer's PredicateBasedPass single-node-match model -- see
// structured_pruning_entry.cpp's own top-of-file comment for the details
// and scope of this port. See onnxsim.h for the documentation this mirrors.

#include <onnx/onnx_pb.h>

onnx::ModelProto ApplyStructuredPruning(const onnx::ModelProto& model,
                                        double sparsity);

#pragma once

// Single-call pruning entry points exposed to Python, mirroring
// quantize_entry.h's own "one named custom_optimizer_passes.cpp pass,
// standalone via OptimizeFixed" shape -- but for pruning, which is not
// itself a quantization scheme, so it does not belong in quantize_entry.h
// alongside the actual Quantize* schemes (same rationale
// cross_layer_equalization_entry.h gives for its own separate file). See
// onnxsim.h for the per-function documentation these mirror.

#include <onnx/onnx_pb.h>

onnx::ModelProto PruneMagnitude(const onnx::ModelProto& model, double sparsity);

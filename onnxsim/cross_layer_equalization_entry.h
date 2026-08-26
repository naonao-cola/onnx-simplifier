#pragma once

// Single-call preprocessing entry point exposed to Python, mirroring
// quantize_entry.h's own "one named custom_optimizer_passes.cpp pass,
// standalone via OptimizeFixed" shape -- but for cross_layer_equalization,
// which is not itself a quantization scheme (no Quantize/DequantizeLinear
// node is ever introduced), so it does not belong in quantize_entry.h
// alongside the actual Quantize* schemes. See onnxsim.h for the
// documentation this mirrors.

#include <onnx/onnx_pb.h>

onnx::ModelProto CrossLayerEqualize(const onnx::ModelProto& model);

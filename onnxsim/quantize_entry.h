#pragma once

// Single-call quantization entry points exposed to Python: each wraps one (or
// a small group of) onnxsim optimizer passes registered in
// custom_optimizer_passes.cpp, running it standalone via OptimizeFixed rather
// than as part of the full Simplify() fixed point. See onnxsim.h for the
// per-function documentation these mirror.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

onnx::ModelProto QuantizeDynamic(const onnx::ModelProto& model);
onnx::ModelProto QuantizeTernary(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnly(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyInt4(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyMatMulNBits(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyInt16(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyInt8Block(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyMXFP4(const onnx::ModelProto& model);

std::vector<std::string> ListQuantizableActivations(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeStatic(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

onnx::ModelProto QuantizeStaticInt16(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

std::vector<std::string> ListQOperatorQuantizableOutputs(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeQOperator(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

std::vector<std::string> ListQOperatorElementwiseQuantizableTensors(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeQOperatorElementwise(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

std::vector<std::string> ListQOperatorActivationQuantizableTensors(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeQOperatorActivation(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

std::vector<std::string> ListQOperatorConcatQuantizableTensors(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeQOperatorConcat(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

onnx::ModelProto QuantizeFp16(const onnx::ModelProto& model,
                              bool keep_io_types);
onnx::ModelProto QuantizeBf16(const onnx::ModelProto& model,
                              bool keep_io_types);
onnx::ModelProto QuantizeFp8(const onnx::ModelProto& model,
                             const std::string& format, bool keep_io_types);

onnx::ModelProto ApplyDoubleQuantization(const onnx::ModelProto& model);
onnx::ModelProto ApplyAnyPrecisionLlm(const onnx::ModelProto& model,
                                      int64_t bits, int64_t max_bits,
                                      int64_t block_size);
onnx::ModelProto ApplyQuarot(const onnx::ModelProto& model, uint64_t seed,
                             int64_t block_size, float epsilon);
onnx::ModelProto ApplyIQ4NL(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ4_0(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ4_1(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ5_0(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ5_1(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ8_0(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ2K(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ3K(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ4K(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ5K(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufTernaryQuant(const onnx::ModelProto& model);
onnx::ModelProto ApplyFp6Llm(const onnx::ModelProto& model);
onnx::ModelProto ApplyGgufQ6K(const onnx::ModelProto& model);
onnx::ModelProto ApplyLeptoquant(const onnx::ModelProto& model);
onnx::ModelProto ApplyNF4(const onnx::ModelProto& model);
onnx::ModelProto ApplyIF4(const onnx::ModelProto& model);
onnx::ModelProto ApplyNVFP4Quantization(const onnx::ModelProto& model);
onnx::ModelProto ApplyDeepSeekFp8(const onnx::ModelProto& model);
onnx::ModelProto ApplyKMeansQuantization(const onnx::ModelProto& model);
onnx::ModelProto ApplyHQQ(const onnx::ModelProto& model);
onnx::ModelProto ApplyIBertGelu(const onnx::ModelProto& model);
onnx::ModelProto ApplyIBertSoftmax(const onnx::ModelProto& model);
onnx::ModelProto ApplyADPQ(const onnx::ModelProto& model);
onnx::ModelProto ApplyICQuant(const onnx::ModelProto& model);
onnx::ModelProto ApplyOlive(const onnx::ModelProto& model);
onnx::ModelProto ApplyAQLM(const onnx::ModelProto& model);
onnx::ModelProto ApplyDropByDrop(const onnx::ModelProto& model);
onnx::ModelProto ApplyLoBcq(const onnx::ModelProto& model);
onnx::ModelProto ApplyQuipSharp(const onnx::ModelProto& model);
onnx::ModelProto ApplyAttentionQuantization(const onnx::ModelProto& model);
onnx::ModelProto ApplyZeroQuant(const onnx::ModelProto& model,
                                int64_t block_size, float epsilon);
onnx::ModelProto ApplyIntactKv(const onnx::ModelProto& model);
onnx::ModelProto ApplyKbvqMoe(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyLlmFp4(const onnx::ModelProto& model);
onnx::ModelProto ApplyQoq(const onnx::ModelProto& model);
onnx::ModelProto ApplyDsq(const onnx::ModelProto& model);

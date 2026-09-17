#pragma once

// Data-free embedding-output binarization entry point exposed to Python --
// C++ port of onnxsim.embedding_quantization's own quantize_embedding_binary
// (see that module's own docstring for the full rationale: unlike every
// other quantizer in this repo, which rewrites how a model *computes*, this
// one compresses what the model's own graph *emits* -- a retrieval
// encoder's own embedding vector -- for downstream storage in a vector
// index, not for cheaper on-device math).
//
// Unlike every PredicateBasedPass in onnxsim/passes/ (which matches a
// single node kind, e.g. MatMul/Gemm), this port targets a whole GRAPH
// OUTPUT declaration directly -- there is no node to match at all before
// the graph's own declared output is quantized -- so, like daq_entry.h/
// low_rank_compensation_entry.h (see those headers' own top-of-file
// comments for the general rationale), this is a plain top-level function
// operating directly on onnx::GraphProto/NodeProto/TensorProto, not
// onnx-optimizer's Node/Graph IR (unavailable outside the single-graph
// PredicateBasedPass registry mechanism this whole-output-targeting shape
// doesn't fit). This one needs no ModelExecutor/calibration data at all
// either -- every decision is a closed-form threshold-and-pack operation.
//
// `output_name`, when non-empty, must name an existing FLOAT graph output
// (a missing name, or one that isn't FLOAT, declines the whole model
// unchanged) -- the empty string stands in for Python's own
// `output_name=None` sentinel, meaning "resolve automatically, requiring
// the graph to have *exactly one* FLOAT output" (declining, rather than
// guessing, when there is zero or more than one).
//
// Before:
//   Y: graph output, float32 [..., embed_dim]     -- embed_dim % 8 == 0
// After (opset 13+ only -- Python's own _has_min_opset(model, 13) gate;
// this port's own ReduceSum(axes-as-input) usage needs it):
//   Bits    = Cast(Greater(Y, 0.0), INT64)          -- 0/1 per element
//   Shape   = Shape(Y)
//   Prefix  = Slice(Shape, [0], [-1])                -- leading dims only
//   NewShape = Concat(Prefix, [embed_dim/8, 8], axis=0)
//   Grouped = Reshape(Bits, NewShape)                -- [..., embed_dim/8, 8]
//   Weighted = Mul(Grouped, [128,64,32,16,8,4,2,1])  -- MSB-first bit weights
//   PackedI64 = ReduceSum(Weighted, axes=[-1], keepdims=0)
//   Y' = Cast(PackedI64, UINT8)                      -- takes over the
//        ORIGINAL output binding; its own declared shape becomes the same
//        leading dims plus a last dim of embed_dim/8, its own dtype UINT8
//
// This is exactly `numpy.packbits(Y > 0, axis=-1, bitorder="big")` --
// see quantize_embedding_binary's own docstring for that exact
// cross-reference -- expressed with ordinary opset-13+ ops (Greater/Cast/
// Shape/Slice/Concat/Reshape/Mul/ReduceSum/Cast), no contrib op, no custom
// bit-packed tensor type.
//
// A model whose resolved output's own last dimension is not statically
// known, or is not a multiple of 8, or whose opset is older than 13, is
// returned completely unchanged -- mirrors quantize_embedding_binary's own
// per-model skip conditions exactly.
//
// This port hardcodes none of quantize_embedding_binary's own optional
// knobs away (it has only the two parameters mirrored here); no
// calibration-related parameters exist for this technique at all.
//
// ACCEPTED, PERMANENT DIVERGENCE: none -- this is a closed-form,
// deterministic bit-packing scheme with no RNG or fitting step anywhere in
// it, so this port is expected to track quantize_embedding_binary's own
// graph rewrite exactly (up to the ordinary IR-vs-protobuf representational
// differences every other port in this repo already carries, and no
// floating-point summation at all beyond a single Greater-than-zero
// comparison per element).

#include <onnx/onnx_pb.h>

#include <string>

// Binarizes the resolved FLOAT graph output (see this header's own
// top-of-file comment for exactly how `output_name` resolves, and for the
// full before/after graph rewrite): each element thresholded at zero (1 if
// greater than zero, else 0), then 8 consecutive elements along the last
// axis packed MSB-first into one UINT8 byte. Returns `model` unchanged
// when the output can't be resolved, its own last dimension isn't
// statically known or isn't a multiple of 8, or `model`'s opset is older
// than 13.
onnx::ModelProto ApplyEmbeddingQuantizationBinary(
    const onnx::ModelProto& model, const std::string& output_name = "");

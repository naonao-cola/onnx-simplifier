/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generates a single, self-contained C source file that reconstructs an
 * onnx::ModelProto as an XNNPACK Subgraph (https://github.com/google/XNNPACK's
 * Subgraph API) plus a tiny create/run/destroy harness -- for embedding into a
 * target that cannot carry onnxsim/onnx/protobuf at runtime, only libxnnpack
 * itself. This is a genuinely different thing from onnx_to_xnnpack_subgraph.h:
 * that lowering builds a *live* xnn_subgraph_t in-process, for onnxsim's own
 * constant-folding backend; this one emits *text* -- valid C source calling
 * the exact same API -- and is not part of that constant-folding path at all.
 *
 * Because the output is text naming XNNPACK's C API by identifier, not a
 * live call into it, this file has no XNNPACK dependency itself and does not
 * need ONNXSIM_BUILTIN_XNNPACK: it only needs to get every argument, macro
 * name, and struct/function name right, which onnxsim/xnnpack_codegen.cpp's
 * module comment cross-references against the pinned XNNPACK commit
 * (cmake/build_xnnpack.cmake's ONNXSIM_XNNPACK_GIT_TAG) it was written
 * against. The generated file's own #include <xnnpack.h> is resolved when
 * *that* file is compiled, against whatever XNNPACK the target links.
 *
 * Scope (v1) -- see xnnpack_codegen.cpp's module comment for the full
 * rationale, especially the NHWC layout convention every rank-4 tensor is
 * emitted under:
 *   * Ops: Add, Sub, Mul, Div, Relu, Sigmoid, Gemm, MatMul, Reshape (only
 *     where layout-safe -- see below), Conv (regular, grouped, and
 *     depthwise), GlobalAveragePool. Anything else throws
 *     std::runtime_error naming the unsupported op.
 *   * fp32 only, no quantization (unlike onnx_to_xnnpack_subgraph.h, which
 *     has int8/uint8 support -- out of scope here, at least for v1).
 *   * Every rank-4 tensor (a graph input/output or a Conv/pooling-adjacent
 *     activation) is emitted NHWC, not ONNX's own NCHW -- matching XNNPACK's
 *     native convolution layout, and not coincidentally the natural
 *     in-memory layout of a cv::Mat/interleaved image buffer (see
 *     onnxsim/xnnpack_cv_mat.hpp for that glue). Conv/Gemm weights are
 *     permuted once, here, at generation time -- a free, one-time transpose
 *     of constant data, never a runtime cost.
 *   * A Reshape/Flatten of a rank-4 tensor is only supported when its
 *     spatial (H, W) dimensions are both 1 (i.e. immediately after a
 *     GlobalAveragePool, or any op whose output already collapsed them) --
 *     the one case where NCHW- and NHWC-order flattening produce identical
 *     results, so no data reordering is needed. Flattening a genuinely
 *     multi-pixel spatial map is rejected with an explanatory error rather
 *     than silently producing wrong numbers; see the module comment.
 *   * No control-flow (If/Loop/Scan), no dynamic shapes -- every tensor in
 *     the graph must resolve to a concrete shape (the same "not symbolic"
 *     bar onnxsim/memory_planning.h's plan_activation_memory uses), since
 *     generated code bakes shapes in as literals.
 */
#ifndef ONNXSIM_XNNPACK_CODEGEN_H_
#define ONNXSIM_XNNPACK_CODEGEN_H_

#include <onnx/onnx_pb.h>

#include <string>

namespace onnxsim {
namespace xnnpack_backend {

// Translate `model` into a standalone C source file, returned as a string.
// `function_prefix` names every emitted public symbol (`<prefix>_model_t`,
// `<prefix>_create`, `<prefix>_run`, `<prefix>_destroy`) and every static
// data array, so more than one generated file can be linked into the same
// binary without symbol collisions; it must already be a valid C identifier
// (this function does not sanitize it -- unlike tensor names, which it does
// sanitize internally).
//
// Throws std::invalid_argument for a malformed `function_prefix`, and
// std::runtime_error -- naming the offending node/tensor -- for anything
// outside this generator's v1 scope (see the module comment above and in
// xnnpack_codegen.cpp): an unsupported op, a non-fp32 tensor, a shape that
// does not resolve to a concrete size, or a layout-unsafe Reshape.
std::string GenerateXnnpackC(const onnx::ModelProto& model,
                             const std::string& function_prefix);

}  // namespace xnnpack_backend
}  // namespace onnxsim

#endif  // ONNXSIM_XNNPACK_CODEGEN_H_

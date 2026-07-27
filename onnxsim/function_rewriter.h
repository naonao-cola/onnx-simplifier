/*
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ONNXSIM_FUNCTION_REWRITER_H_
#define ONNXSIM_FUNCTION_REWRITER_H_

#include <onnx/onnx_pb.h>

#include <memory>
#include <vector>

#include "onnxsim.h"

namespace onnxsim {

// A single data-driven rewrite rule expressed entirely as ONNX protobuf:
// ``pattern`` describes the subgraph to match and ``replacement`` what to put
// in its place. Both are ``FunctionProto``s, so a rule is *pure data* that can
// cross any binding boundary (Python, the C ABI, Rust) as serialized bytes --
// unlike the ``PyGraphRewriter`` Python callable, which only the Python binding
// can carry.
//
// Semantics (see function_rewriter.cpp for the details and limitations):
//   * ``pattern.input``  -- wildcards; each binds to a host-graph value.
//   * ``pattern.node``   -- the DAG of ops to match.
//   * ``pattern.output`` -- the values whose consumers are rewired to the
//                           replacement's outputs.
// A node attribute carrying a ``ref_attr_name`` (the onnx text ``@name`` form)
// is an attribute wildcard: it binds to the host node's attribute and is
// substituted into the replacement. A concrete attribute must match exactly.
struct FunctionRewriteRule {
  onnx::FunctionProto pattern;
  onnx::FunctionProto replacement;
};

// Build a ``GraphRewriter`` that applies ``rules``. It plugs into ``Simplify``
// like any other rewriter and runs inside onnxsim's simplification fixed point,
// so a fusion it performs can unlock further optimizer/folding passes and vice
// versa.
//
// The concrete rewriter type is kept private to function_rewriter.cpp and only
// a base ``GraphRewriter`` pointer is returned. onnxsim is built as a shared
// library under a hidden default-visibility preset, and its callers (the C ABI
// and the Python extension) live in *other* shared objects; returning the base
// keeps them from referencing the concrete subclass's vtable across the
// library boundary (which the linker cannot resolve). The returned
// ``shared_ptr`` type-erases its deleter, so destruction still dispatches to
// the concrete type. This is a plain free function, exported exactly like the
// rest of onnxsim's public API (e.g. ``Simplify``).
std::shared_ptr<GraphRewriter> MakeFunctionProtoRewriter(
    std::vector<FunctionRewriteRule> rules);

}  // namespace onnxsim

#endif  // ONNXSIM_FUNCTION_REWRITER_H_

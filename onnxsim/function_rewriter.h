/*
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ONNXSIM_FUNCTION_REWRITER_H_
#define ONNXSIM_FUNCTION_REWRITER_H_

#include <onnx/onnx_pb.h>

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

// A ``GraphRewriter`` that applies a fixed set of ``FunctionRewriteRule``s. It
// plugs into ``Simplify`` exactly like any other rewriter and therefore runs
// inside onnxsim's simplification fixed point, so a fusion it performs can
// unlock further optimizer/folding passes and vice versa.
class FunctionProtoRewriter final : public GraphRewriter {
 public:
  explicit FunctionProtoRewriter(std::vector<FunctionRewriteRule> rules);

  // Repeatedly applies the rules to ``model``'s main graph until no rule fires.
  // Returns true if the model was changed and false if it was left untouched;
  // the "unchanged" answer lets onnxsim skip re-copying the model, matching the
  // rest of the ``GraphRewriter`` contract.
  bool _Run(onnx::ModelProto& model) const override;

 private:
  std::vector<FunctionRewriteRule> rules_;
};

}  // namespace onnxsim

#endif  // ONNXSIM_FUNCTION_REWRITER_H_

// SPDX-License-Identifier: Apache-2.0
//
// GENERATED FILE -- do not edit by hand. Produced by
//   python3 scripts/codegen/generate_grad_templates.py
// from the onnxscript function definitions in that script; see its
// module docstring for what this is and why it takes no ONNX-level
// attributes. graph_grad_templates_gen.py is the same text for the
// Python side -- both are produced from the same entries so they
// cannot drift from each other.
#ifndef ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_
#define ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_

// No enclosing namespace -- graph_grad.cpp, this header's only consumer, has
// none either (it mirrors graph_grad.py's flat module directly).
constexpr const char* kGradAddTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradAdd (g) => (da, db)
{
   [n0] da = Identity (g)
   [n1] db = Identity (g)
})GRAD_TPL";

constexpr const char* kGradBatchNormalizationTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradBatchNormalization (g, x, mean_b, var_b, scale_b, eps, channel_axes, one, neg_half) => (dx, dscale, dbias, dmean, dvar)
{
   [n0] xc = Sub (x, mean_b)
   [n1] tmp = Add (var_b, eps)
   [n2] tmp_0 = Sqrt (tmp)
   [n3] inv = Div (one, tmp_0)
   [n4] xhat = Mul (xc, inv)
   [n5] gs = Mul (g, scale_b)
   [n6] dx = Mul (gs, inv)
   [n7] tmp_1 = Mul (g, xhat)
   [n8] dscale = ReduceSum <keepdims: int = 0> (tmp_1, channel_axes)
   [n9] dbias = ReduceSum <keepdims: int = 0> (g, channel_axes)
   [n10] tmp_2 = ReduceSum <keepdims: int = 0> (dx, channel_axes)
   [n11] dmean = Neg (tmp_2)
   [n12] tmp_3 = Mul (dx, xhat)
   [n13] tmp_4 = Mul (tmp_3, inv)
   [n14] tmp_5 = ReduceSum <keepdims: int = 0> (tmp_4, channel_axes)
   [n15] dvar = Mul (tmp_5, neg_half)
})GRAD_TPL";

constexpr const char* kGradNegTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradNeg (g) => (dx)
{
   [n0] dx = Neg (g)
})GRAD_TPL";

constexpr const char* kGradExpTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradExp (g, y) => (dx)
{
   [n0] dx = Mul (g, y)
})GRAD_TPL";

constexpr const char* kGradSqrtTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradSqrt (g, y, half) => (dx)
{
   [n0] tmp = Mul (g, half)
   [n1] dx = Div (tmp, y)
})GRAD_TPL";

constexpr const char* kGradLogTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradLog (g, x) => (dx)
{
   [n0] dx = Div (g, x)
})GRAD_TPL";

constexpr const char* kGradSigmoidTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradSigmoid (g, y, one) => (dx)
{
   [n0] tmp = Sub (one, y)
   [n1] dy = Mul (y, tmp)
   [n2] dx = Mul (g, dy)
})GRAD_TPL";

constexpr const char* kGradTanhTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradTanh (g, y, one) => (dx)
{
   [n0] tmp = Mul (y, y)
   [n1] dy = Sub (one, tmp)
   [n2] dx = Mul (g, dy)
})GRAD_TPL";

constexpr const char* kGradErfTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradErf (g, x, c) => (dx)
{
   [n0] tmp = Mul (x, x)
   [n1] tmp_0 = Neg (tmp)
   [n2] tmp_1 = Exp (tmp_0)
   [n3] dy = Mul (c, tmp_1)
   [n4] dx = Mul (g, dy)
})GRAD_TPL";

constexpr const char* kGradMulTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradMul (g, a, b) => (da, db)
{
   [n0] da = Mul (g, b)
   [n1] db = Mul (g, a)
})GRAD_TPL";

constexpr const char* kGradDivTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradDiv (g, a, b, y) => (da, db)
{
   [n0] da = Div (g, b)
   [n1] tmp = Mul (g, y)
   [n2] tmp_0 = Div (tmp, b)
   [n3] db = Neg (tmp_0)
})GRAD_TPL";

#endif  // ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_

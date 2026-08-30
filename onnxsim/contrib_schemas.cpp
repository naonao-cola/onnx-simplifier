/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "contrib_schemas.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "contrib_schemas_moe_templates.gen.h"
#include "onnx/defs/data_propagators.h"
#include "onnx/defs/function.h"
#include "onnx/defs/math/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxsim {

namespace {

constexpr const char* kMSDomain = "com.microsoft";

using onnx::DataPropagationContext;
using onnx::FunctionBodyBuildContext;
using onnx::FunctionBuilder;
using onnx::FunctionProto;
using onnx::InferenceContext;
using onnx::OpSchema;

// Attach a data-propagation function to `Reshape`. ONNX ships data-propagation
// functions for the shape family (Shape, Gather, Concat, Slice, Squeeze,
// Unsqueeze, Add/Sub/Mul, ...) but not for Reshape, so a shape tensor threaded
// through a Reshape -- e.g. `Shape(x) -> Reshape(., [-1]) -> Gather(...)` --
// loses its propagated value there and downstream shape arithmetic stops
// folding.
//
// A Reshape only rearranges a tensor's dims; it never changes the number of
// elements or their row-major order. Data propagation tracks a shape tensor's
// *value* as a flat, ordered list of its elements, so that list is invariant
// under a Reshape and can be copied straight through -- the same reasoning, and
// the same helper (PropagateShapeDataFromInputToOutput), that ONNX uses for
// Squeeze/Unsqueeze, which likewise only add or remove size-1 axes.
//
// The schema objects are owned by the registry and Schema() returns a pointer
// into that storage, so we const_cast to augment them in place. Data
// propagation only runs when explicitly enabled (onnxsim's partial-shape pass),
// so this never affects ordinary shape inference.
void RegisterReshapeDataPropagation() {
  std::unordered_set<const OpSchema*> augmented;
  for (int ver = 1; ver <= 64; ++ver) {
    const OpSchema* schema =
        onnx::OpSchemaRegistry::Schema("Reshape", ver, onnx::ONNX_DOMAIN);
    if (schema == nullptr || augmented.count(schema)) {
      continue;
    }
    augmented.insert(schema);
    const_cast<OpSchema*>(schema)->PartialDataPropagationFunction(
        [](DataPropagationContext& ctx) {
          onnx::PropagateShapeDataFromInputToOutput(ctx, 0);
        });
  }
}

// Shape/type inference for the element-wise binary quantized ops (QLinearAdd,
// QLinearMul). Inputs are laid out as
//   A, A_scale, A_zero_point, B, B_scale, B_zero_point, C_scale, C_zero_point
// so the two data tensors that determine the output shape are inputs 0 and 3.
void QLinearBinaryShapeInference(InferenceContext& ctx) {
  // The output is quantized to the same element type as the first operand.
  onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (onnx::hasInputShape(ctx, 0) && onnx::hasInputShape(ctx, 3)) {
    onnx::bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(3)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
  }
}

// Shape/type inference for QLinearConcat. Inputs are
//   Y_scale, Y_zero_point, (T, T_scale, T_zero_point)+
// The output element type follows Y_zero_point (input 1) and the shape is the
// concatenation of the data tensors (inputs 2, 5, 8, ...) along `axis`.
void QLinearConcatShapeInference(InferenceContext& ctx) {
  onnx::propagateElemTypeFromInputToOutput(ctx, 1, 0);

  const auto* axis_attr = ctx.getAttribute("axis");
  if (axis_attr == nullptr || !axis_attr->has_i()) {
    return;
  }
  int64_t axis = axis_attr->i();

  std::vector<size_t> data_indices;
  for (size_t i = 2; i < ctx.getNumInputs(); i += 3) {
    data_indices.push_back(i);
  }
  if (data_indices.empty()) {
    return;
  }

  // Every data tensor must have a known rank and the ranks must agree.
  int rank = -1;
  for (size_t idx : data_indices) {
    if (!onnx::hasInputShape(ctx, idx)) {
      return;
    }
    const int cur_rank =
        ctx.getInputType(idx)->tensor_type().shape().dim_size();
    if (rank == -1) {
      rank = cur_rank;
    } else if (rank != cur_rank) {
      // Inconsistent ranks: leave the output shape unset rather than guessing.
      return;
    }
  }
  if (rank <= 0) {
    return;
  }
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return;
  }

  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  output_shape->clear_dim();
  for (int i = 0; i < rank; ++i) {
    output_shape->add_dim();
  }

  bool axis_dim_known = true;
  int64_t axis_dim_sum = 0;
  for (size_t idx : data_indices) {
    const auto& shape = ctx.getInputType(idx)->tensor_type().shape();
    for (int d = 0; d < rank; ++d) {
      const auto& dim = shape.dim(d);
      if (d == axis) {
        if (dim.has_dim_value()) {
          axis_dim_sum += dim.dim_value();
        } else {
          axis_dim_known = false;
        }
        continue;
      }
      // Non-axis dimensions must match across inputs; keep the most specific
      // information we can (a concrete value, otherwise a symbolic name).
      auto* out_dim = output_shape->mutable_dim(d);
      if (!out_dim->has_dim_value() && dim.has_dim_value()) {
        out_dim->set_dim_value(dim.dim_value());
      } else if (!out_dim->has_dim_value() && !out_dim->has_dim_param() &&
                 dim.has_dim_param()) {
        out_dim->set_dim_param(dim.dim_param());
      }
    }
  }
  if (axis_dim_known) {
    output_shape->mutable_dim(axis)->set_dim_value(axis_dim_sum);
  }
}

// Shape/type inference for QLinearGlobalAveragePool. Per ONNX Runtime's own
// doc ("the output tensor has the same rank as the input, with the N and C
// value keep[ing] its value, while the other dimensions are all 1"), the
// output shape is fully determined by the input shape and rank -- unlike
// QLinearAveragePool below, whose true output shape additionally depends on
// kernel_shape/strides/pads/ceil_mode/auto_pad arithmetic this function
// does not replicate.
void QLinearGlobalAveragePoolShapeInference(InferenceContext& ctx) {
  onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!onnx::hasInputShape(ctx, 0)) {
    return;
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const int rank = input_shape.dim_size();
  if (rank < 2) {
    return;
  }
  int64_t channels_last = 0;
  const auto* channels_last_attr = ctx.getAttribute("channels_last");
  if (channels_last_attr != nullptr && channels_last_attr->has_i()) {
    channels_last = channels_last_attr->i();
  }
  const int channel_dim = channels_last != 0 ? rank - 1 : 1;

  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  output_shape->clear_dim();
  for (int i = 0; i < rank; ++i) {
    if (i == 0 || i == channel_dim) {
      *output_shape->add_dim() = input_shape.dim(i);
    } else {
      output_shape->add_dim()->set_dim_value(1);
    }
  }
}

// Shape/type inference for QLinearWhere. Inputs are laid out as
//   condition, X, x_scale, x_zero_point, Y, y_scale, y_zero_point,
//   z_scale, z_zero_point
// so the output's element type follows X (input 1) and its shape is the
// 3-way broadcast of condition (0), X (1), and Y (4) -- exactly mirroring
// ONNX Runtime's own QLinearWhere inference function.
void QLinearWhereShapeInference(InferenceContext& ctx) {
  onnx::propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (!onnx::hasNInputShapes(ctx, 9)) {
    return;
  }
  std::vector<const onnx::TensorShapeProto*> shapes;
  shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
  shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
  shapes.push_back(&ctx.getInputType(4)->tensor_type().shape());
  onnx::multidirectionalBroadcastShapeInference(
      shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

// Shape/type inference for QGemm. The output's element type follows
// `y_zero_point` (input 8) when present -- meaning the output is itself
// quantized -- else it is plain float32 (QGemm's schema allows an
// unquantized float output when `y_scale`/`y_zero_point` are omitted; this
// onnxsim pass never omits them, always producing the quantized-output
// form, but the schema itself supports both). The output shape is the
// standard Gemm (M, N) computed from A's and B's 2-D shapes and the
// transA/transB attributes.
void QGemmShapeInference(InferenceContext& ctx) {
  if (ctx.getNumInputs() == 9 && ctx.getInputType(8) != nullptr) {
    onnx::propagateElemTypeFromInputToOutput(ctx, 8, 0);
  } else {
    onnx::updateOutputElemType(ctx, 0, onnx::TensorProto::FLOAT);
  }
  if (!onnx::hasInputShape(ctx, 0) || !onnx::hasInputShape(ctx, 3)) {
    return;
  }
  const auto& a_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto& b_shape = ctx.getInputType(3)->tensor_type().shape();
  if (a_shape.dim_size() != 2 || b_shape.dim_size() != 2) {
    return;
  }
  int64_t trans_a = 0;
  const auto* trans_a_attr = ctx.getAttribute("transA");
  if (trans_a_attr != nullptr && trans_a_attr->has_i()) {
    trans_a = trans_a_attr->i();
  }
  int64_t trans_b = 0;
  const auto* trans_b_attr = ctx.getAttribute("transB");
  if (trans_b_attr != nullptr && trans_b_attr->has_i()) {
    trans_b = trans_b_attr->i();
  }
  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  output_shape->clear_dim();
  *output_shape->add_dim() = a_shape.dim(trans_a != 0 ? 1 : 0);
  *output_shape->add_dim() = b_shape.dim(trans_b != 0 ? 0 : 1);
}

// Registers `schema` unless an equivalent schema is already known. Duplicate
// registration is turned into a no-op instead of an error so the function stays
// safe to run alongside a build that already provides these schemas.
void RegisterIfAbsent(OpSchema&& schema) {
  const std::string name = schema.Name();
  if (onnx::OpSchemaRegistry::Schema(name, kMSDomain) != nullptr) {
    return;
  }
  onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/1,
                       /*fail_duplicate_schema=*/false,
                       /*fail_with_exception=*/false);
}

OpSchema MakeQLinearBinarySchema(const char* name) {
  return OpSchema()
      .SetName(name)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantized element-wise binary op contributed by ONNX Runtime.")
      .Input(0, "A", "First quantized operand.", "T")
      .Input(1, "A_scale", "Scale of A.", "tensor(float)")
      .Input(2, "A_zero_point", "Zero point of A.", "T", OpSchema::Optional)
      .Input(3, "B", "Second quantized operand.", "T")
      .Input(4, "B_scale", "Scale of B.", "tensor(float)")
      .Input(5, "B_zero_point", "Zero point of B.", "T", OpSchema::Optional)
      .Input(6, "C_scale", "Scale of the output C.", "tensor(float)")
      .Input(7, "C_zero_point", "Zero point of the output C.", "T",
             OpSchema::Optional)
      .Output(0, "C", "Quantized result.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer tensors.")
      .TypeAndShapeInferenceFunction(QLinearBinaryShapeInference);
}

OpSchema MakeQLinearUnarySchema(const char* name, bool has_alpha) {
  OpSchema schema;
  schema.SetName(name)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantized element-wise unary op contributed by ONNX Runtime.")
      .Input(0, "X", "Quantized input.", "T")
      .Input(1, "X_scale", "Scale of X.", "tensor(float)")
      .Input(2, "X_zero_point", "Zero point of X.", "T", OpSchema::Optional)
      .Input(3, "Y_scale", "Scale of the output Y.", "tensor(float)")
      .Input(4, "Y_zero_point", "Zero point of the output Y.", "T",
             OpSchema::Optional)
      .Output(0, "Y", "Quantized output.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput);
  if (has_alpha) {
    schema.Attr("alpha", "Coefficient of leakage.", onnx::AttributeProto::FLOAT,
                0.01f);
  }
  return schema;
}

OpSchema MakeQLinearConcatSchema() {
  return OpSchema()
      .SetName("QLinearConcat")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantized concatenation contributed by ONNX Runtime.")
      .Attr("axis", "Axis to concatenate on.", onnx::AttributeProto::INT,
            /*required=*/true)
      .Input(0, "Y_scale", "Scale of the output Y.", "TF")
      .Input(1, "Y_zero_point", "Zero point of the output Y.", "T8")
      .Input(2, "inputs",
             "Repeated (tensor, scale, zero_point) triples to concatenate.",
             "TV", OpSchema::Variadic, /*is_homogeneous=*/false)
      .Output(0, "Y", "Concatenated quantized result.", "T8")
      .TypeConstraint("T8", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain quantized tensors to 8-bit integers.")
      .TypeConstraint("TF", {"tensor(float)"}, "Constrain scales to float.")
      .TypeConstraint("TV", {"tensor(uint8)", "tensor(int8)", "tensor(float)"},
                      "Constrain the variadic inputs.")
      .TypeAndShapeInferenceFunction(QLinearConcatShapeInference);
}

// Same layout/attribute set ONNX Runtime itself registers for
// "com.microsoft" QLinearSoftmax: `X`/`Y` share a single `T` type constraint
// (8-bit signed or unsigned), the output's shape and element type simply
// follow the input's (Softmax never changes shape), and the `opset`
// attribute is required -- it tells the runtime kernel which of standard
// ONNX's two incompatible `Softmax` axis semantics (pre-13 flattening vs.
// 13+ in-place per-axis reduction) to replicate. Unlike
// MakeQLinearUnarySchema's `Y_zero_point` (optional, since QLinearSigmoid/
// QLinearLeakyRelu allow a runtime-implied default), QLinearSoftmax's
// `y_zero_point` is required.
OpSchema MakeQLinearSoftmaxSchema() {
  return OpSchema()
      .SetName("QLinearSoftmax")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "QLinearSoftmax computes the normalized exponential values for "
          "the given input: Softmax(input, axis) = Exp(input) / "
          "ReduceSum(Exp(input), axis=axis, keepdims=1).")
      .Attr("axis", "Apply softmax to elements for dimensions axis.",
            onnx::AttributeProto::INT, static_cast<int64_t>(-1))
      .Attr("opset",
            "Opset version of the standard-ONNX Softmax whose axis "
            "semantics this node replicates.",
            onnx::AttributeProto::INT)
      .Input(0, "X", "The input tensor.", "T")
      .Input(1, "X_scale", "Scale of quantized input X. Must be a scalar.",
             "tensor(float)")
      .Input(2, "x_zero_point",
             "Zero point of quantized input X. Must be a scalar.", "T",
             OpSchema::Optional)
      .Input(3, "y_scale", "Scale of quantized output Y. Must be a scalar.",
             "tensor(float)")
      .Input(4, "y_zero_point",
             "Zero point of quantized output Y. Must be a scalar.", "T")
      .Output(0, "Y", "Output data tensor.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output types to 8-bit integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput);
}

// Same layout/attribute set ONNX Runtime itself registers for
// "com.microsoft" QLinearAveragePool: every attribute standard ONNX
// AveragePool has (kernel_shape required; auto_pad/ceil_mode/
// count_include_pad/pads/strides optional, matching AveragePool's own
// defaults), plus a `channels_last` attribute AveragePool itself doesn't
// have. Type/shape inference only propagates the element type -- the true
// output shape depends on the same kernel/stride/pad arithmetic standard
// AveragePool's own inference function implements, which this schema does
// not replicate (qoperator_quantize_pool.h's rewrite doesn't need it either:
// it copies the original node's already-known output shape onto the
// trailing DequantizeLinear directly).
OpSchema MakeQLinearAveragePoolSchema() {
  return OpSchema()
      .SetName("QLinearAveragePool")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "QLinearAveragePool consumes an input tensor X and applies "
          "average pooling across the tensor according to kernel sizes, "
          "stride sizes and pad lengths, computing on dequantized values "
          "and requantizing the result.")
      .Attr("auto_pad",
            "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or "
            "VALID (deprecated, kept for parity with standard ONNX "
            "AveragePool).",
            onnx::AttributeProto::STRING, "NOTSET")
      .Attr("ceil_mode",
            "Whether to use ceil or floor (default) to compute the output "
            "shape.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("channels_last", "Works on NHWC layout or not. Default not.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("count_include_pad",
            "Whether to include pad pixels when calculating values for "
            "the edges. Default 0, doesn't count include pad.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("kernel_shape", "The size of the kernel along each axis.",
            onnx::AttributeProto::INTS, /*required=*/true)
      .Attr("pads",
            "Padding for the beginning and ending along each spatial "
            "axis. Defaults to 0 along start and end of each spatial axis "
            "when absent.",
            onnx::AttributeProto::INTS, /*required=*/false)
      .Attr("strides",
            "Stride along each spatial axis. Defaults to 1 along each "
            "spatial axis when absent.",
            onnx::AttributeProto::INTS, /*required=*/false)
      .Input(0, "X", "Input data tensor from the previous operator.", "T")
      .Input(1, "x_scale", "Scale of quantized input X. Must be a scalar.",
             "tensor(float)")
      .Input(2, "x_zero_point",
             "Zero point of quantized input X. Must be a scalar.", "T",
             OpSchema::Optional)
      .Input(3, "y_scale", "Scale of quantized output Y. Must be a scalar.",
             "tensor(float)")
      .Input(4, "y_zero_point",
             "Zero point of quantized output Y. Must be a scalar.", "T",
             OpSchema::Optional)
      .Output(0, "Y", "Output data tensor from average pooling.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
      });
}

// Same layout ONNX Runtime itself registers for "com.microsoft"
// QLinearGlobalAveragePool: unlike QLinearAveragePool, both zero-points are
// required (not optional), there are no kernel_shape/strides/pads/etc.
// attributes (it always pools over every spatial position), and the output
// shape is simple enough (same rank, N/C kept, every other dim collapsed to
// 1) that QLinearGlobalAveragePoolShapeInference computes it exactly.
OpSchema MakeQLinearGlobalAveragePoolSchema() {
  return OpSchema()
      .SetName("QLinearGlobalAveragePool")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "QLinearGlobalAveragePool consumes an input tensor X and applies "
          "average pooling across the values in the same channel. This is "
          "equivalent to AveragePool with kernel size equal to the "
          "spatial dimensions of the input tensor.")
      .Attr("channels_last", "Works on NHWC layout or not. Default not.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Input(0, "X", "Input data tensor from the previous operator.", "T")
      .Input(1, "x_scale", "Scale of quantized input X. Must be a scalar.",
             "tensor(float)")
      .Input(2, "x_zero_point",
             "Zero point of quantized input X. Must be a scalar.", "T")
      .Input(3, "y_scale", "Scale of quantized output Y. Must be a scalar.",
             "tensor(float)")
      .Input(4, "y_zero_point",
             "Zero point of quantized output Y. Must be a scalar.", "T")
      .Output(0, "Y",
              "Output data tensor from pooling across the input "
              "tensor.",
              "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction(QLinearGlobalAveragePoolShapeInference);
}

// Same layout ONNX Runtime itself registers for "com.microsoft"
// QLinearWhere: unlike every other schema in this file, every input is
// required (no OpSchema::Optional anywhere) -- ONNX Runtime's own doc
// strings for a couple of these inputs are copy-paste typos ("X" is
// documented as "Y's zero point.", verbatim, in ORT's own source); this
// registration keeps the exact same names/types/order but writes correct
// descriptions.
OpSchema MakeQLinearWhereSchema() {
  return OpSchema()
      .SetName("QLinearWhere")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Return elements, either from X or Y, depending on condition.")
      .Input(0, "condition", "When True (nonzero), yield X, otherwise yield Y.",
             "B")
      .Input(1, "X", "First quantized operand.", "T")
      .Input(2, "x_scale", "Scale of X.", "TF")
      .Input(3, "x_zero_point", "Zero point of X.", "T")
      .Input(4, "Y", "Second quantized operand.", "T")
      .Input(5, "y_scale", "Scale of Y.", "TF")
      .Input(6, "y_zero_point", "Zero point of Y.", "T")
      .Input(7, "z_scale", "Scale of the output Z.", "TF")
      .Input(8, "z_zero_point", "Zero point of the output Z.", "T")
      .Output(0, "Z",
              "Tensor of shape equal to the broadcasted shape of "
              "condition, X, and Y.",
              "T")
      .TypeConstraint("B", {"tensor(bool)"}, "Constrain condition to bool.")
      .TypeConstraint("TF", {"tensor(float)"}, "Constrain scales to float.")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction(QLinearWhereShapeInference);
}

// Same layout ONNX Runtime itself registers for "com.microsoft" QGemm --
// the fully-general quantized Gemm: unlike QLinearMatMul (used by
// qoperator_quantize_matmul.h for the "vanilla" transA=0/alpha=1 case),
// QGemm keeps `transA`/`transB`/`alpha` as attributes of its own, so it
// needs no forced weight-transpose or activation restriction the way
// QLinearMatMul does. `C` (bias), `y_scale`, and `y_zero_point` are all
// optional -- an omitted `C` means "as if C is a scalar 0", and omitted
// `y_scale`/`y_zero_point` means the output stays float32; onnxsim's own
// qoperator_quantize_gemm.h rewrite always supplies all three (matching
// every other pass in this family's "always fully quantize" convention),
// but the schema itself supports the leaner cases too.
OpSchema MakeQGemmSchema() {
  return OpSchema()
      .SetName("QGemm")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantized Gemm.")
      .Attr("transA", "Whether A should be transposed.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("transB", "Whether B should be transposed.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("alpha", "Scalar multiplier for the product of A and B.",
            onnx::AttributeProto::FLOAT, 1.0f)
      .Input(0, "A",
             "Input tensor A. Shape (M, K) if transA is 0, else (K, M).", "TA")
      .Input(1, "a_scale", "Scale of quantized input A. Must be a scalar.", "T")
      .Input(2, "a_zero_point", "Zero point of quantized input A.", "TA")
      .Input(3, "B",
             "Input tensor B. Shape (K, N) if transB is 0, else (N, K).", "TB")
      .Input(4, "b_scale",
             "Scale of quantized input B. A scalar (per-tensor) or 1-D "
             "tensor of N elements (per-column).",
             "T")
      .Input(5, "b_zero_point",
             "Zero point of quantized input B. Same shape as b_scale.", "TB")
      .Input(6, "C",
             "Optional bias tensor, unidirectionally broadcastable to "
             "(M, N). Its type is int32 and must already be quantized "
             "with zero_point = 0 and scale = alpha * a_scale * b_scale.",
             "TC", OpSchema::Optional)
      .Input(7, "y_scale",
             "Scale of quantized output Y. Must be a scalar. If omitted "
             "(along with y_zero_point), the output is float32.",
             "T", OpSchema::Optional)
      .Input(8, "y_zero_point",
             "Zero point of quantized output Y. Must be a scalar.", "TYZ",
             OpSchema::Optional)
      .Output(0, "Y", "Output tensor of shape (M, N).", "TY")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain scales to float.")
      .TypeConstraint("TA", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain A and its zero point to 8-bit tensors.")
      .TypeConstraint("TB", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain B and its zero point to 8-bit tensors.")
      .TypeConstraint("TC", {"tensor(int32)"},
                      "Constrain C to 32-bit integer tensors.")
      .TypeConstraint("TYZ", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain the output zero point to 8-bit tensors.")
      .TypeConstraint("TY", {"tensor(float)", "tensor(uint8)", "tensor(int8)"},
                      "Constrain the output to float32 or 8-bit tensors.")
      .TypeAndShapeInferenceFunction(QGemmShapeInference);
}

// Same layout ONNX Runtime itself registers for "com.microsoft"
// MatMulIntegerToFloat: unlike MatMulInteger (no dequantization step of its
// own -- callers Cast+Mul the int32 result themselves) or QLinearMatMul
// (fully re-quantizes its float result back to int8), this op dequantizes
// directly to float/float16 and optionally adds a bias in the same op, with
// no re-quantization step at all -- exactly the fused replacement for
// dynamic_quantize_matmul.h's own MatMulInteger+Cast+Mul(+Add) node chain.
OpSchema MakeMatMulIntegerToFloatSchema() {
  return OpSchema()
      .SetName("MatMulIntegerToFloat")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "Matrix product of dequantized 8-bit integer matrices A and B, "
          "producing a float result directly (no further re-quantization), "
          "with an optional bias added in the same op.")
      .Input(0, "A", "N-dimensional matrix A.", "T1")
      .Input(1, "B", "N-dimensional matrix B.", "T2")
      .Input(2, "a_scale",
             "Scale of quantized input A. A scalar (per-tensor) or 1-D "
             "tensor (per-column).",
             "T3")
      .Input(3, "b_scale",
             "Scale of quantized input B. A scalar (per-tensor) or 1-D "
             "tensor (per-column).",
             "T3")
      .Input(4, "a_zero_point", "Zero point of quantized input A.", "T1",
             OpSchema::Optional)
      .Input(5, "b_zero_point", "Zero point of quantized input B.", "T2",
             OpSchema::Optional)
      .Input(6, "bias",
             "Optional 1-D bias, matching B's last dimension, added after "
             "dequantization.",
             "T3", OpSchema::Optional)
      .Output(0, "Y", "Matrix multiply result of dequantized A and B.", "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"},
                      "Constrain A to 8-bit integer tensors.")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"},
                      "Constrain B to 8-bit integer tensors.")
      .TypeConstraint("T3", {"tensor(float)", "tensor(float16)"},
                      "Constrain scales, bias, and output to float tensors.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        onnx::propagateElemTypeFromInputToOutput(ctx, 2, 0);
        onnx::defs::math::utils::MatMulShapeInference(ctx, 0, 1);
      });
}

// Real ONNX Runtime multi-head self-/cross-attention op (docs/
// ContribOperators.md, onnxruntime/core/graph/contrib_ops/contrib_defs.cc).
// Registered here so fuse_attention.h's fused output -- which only ever
// emits inputs 0-2 (input, weights, bias) and attrs num_heads/scale/
// qkv_hidden_sizes -- passes the ONNX checker and shape-inference sweeps the
// rest of onnxsim's pipeline runs after any Fuse pass fires; the remaining
// attrs/inputs (do_rotary, past/present KV cache, attention_bias,
// mask_index, rotary_embedding_dim, past_present_share_buffer) are declared
// for schema completeness/compatibility with externally-authored models,
// not because this pass ever produces them.
OpSchema MakeAttentionSchema() {
  return OpSchema()
      .SetName("Attention")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "Multi-Head Self/Cross Attention. Bias from input projection is "
          "included.")
      .Attr("num_heads", "Number of attention heads.",
            onnx::AttributeProto::INT, /*required=*/true)
      .Attr("scale",
            "Custom scale will be used if specified. Default value is "
            "1/sqrt(head_size).",
            onnx::AttributeProto::FLOAT, /*required=*/false)
      .Attr("unidirectional",
            "Whether every token can only attend to previous tokens.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("qkv_hidden_sizes",
            "Hidden dimension of Q, K, V: hidden_size, hidden_size and "
            "v_hidden_size.",
            onnx::AttributeProto::INTS, /*required=*/false)
      .Attr("mask_filter_value",
            "The value to be filled in the attention mask. Default value "
            "is -10000.0.",
            onnx::AttributeProto::FLOAT, static_cast<float>(-10000.0f))
      .Attr("do_rotary", "Whether to use rotary position embedding.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("rotary_embedding_dim",
            "Dimension of rotary embedding. Limited to 32, 64 or 128.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("past_present_share_buffer",
            "Corresponding past and present are same tensor, its shape is "
            "(2, batch_size, num_heads, max_sequence_length, head_size).",
            onnx::AttributeProto::INT, /*required=*/false)
      .Input(0, "input",
             "Input tensor with shape (batch_size, sequence_length, "
             "input_hidden_size).",
             "T")
      .Input(1, "weights",
             "Merged Q/K/V weights with shape (input_hidden_size, "
             "hidden_size + hidden_size + v_hidden_size).",
             "T")
      .Input(2, "bias",
             "Bias tensor with shape (hidden_size + hidden_size + "
             "v_hidden_size).",
             "T", OpSchema::Optional)
      .Input(3, "mask_index", "Attention mask.", "M", OpSchema::Optional)
      .Input(4, "past",
             "Past state for key and value with shape (2, batch_size, "
             "num_heads, past_sequence_length, head_size).",
             "T", OpSchema::Optional)
      .Input(5, "attention_bias",
             "Additional add to QxK' with shape broadcastable to "
             "(batch_size, num_heads, sequence_length, "
             "total_sequence_length).",
             "T", OpSchema::Optional)
      .Input(6, "past_sequence_length",
             "When past_present_share_buffer is used, it is required to "
             "specify past_sequence_length (could be 0).",
             "M", OpSchema::Optional)
      .Output(0, "output",
              "3D output tensor with shape (batch_size, sequence_length, "
              "v_hidden_size).",
              "T")
      .Output(1, "present",
              "Present state for key and value with shape (2, batch_size, "
              "num_heads, total_sequence_length, head_size).",
              "T", OpSchema::Optional)
      .TypeConstraint(
          "T", {"tensor(float)", "tensor(float16)"},
          "Constrain input and output to float tensors. (bfloat16 omitted "
          "-- onnxsim's own float32-only fuse target never emits it.)")
      .TypeConstraint("M", {"tensor(int32)"},
                      "Constrain mask index to integer types.");
}

// Quantized counterpart of MakeAttentionSchema, above -- ONNX Runtime's
// int8/uint8 attention kernel. Registered here so
// dynamic_quantize_attention.h's fused output -- which only ever emits
// inputs 0-2 (input, weight, bias), 3-4 (input_scale, weight_scale), 6-7
// (input_zero_point, weight_zero_point, with mask_index at 5 skipped via an
// Undefined placeholder) and attrs num_heads/scale -- passes the ONNX
// checker and shape-inference sweeps the rest of onnxsim's pipeline runs;
// the remaining attrs/inputs (unidirectional, mask_filter_value,
// past_present_share_buffer, do_rotary, past) are declared for schema
// completeness/compatibility with externally-authored models, not because
// this pass ever produces them.
OpSchema MakeQAttentionSchema() {
  return OpSchema()
      .SetName("QAttention")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantization of Multi-Head Self Attention.")
      .Attr("num_heads", "Number of attention heads.",
            onnx::AttributeProto::INT, /*required=*/true)
      .Attr("unidirectional",
            "Whether every token can only attend to previous tokens.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("scale",
            "Custom scale will be used if specified. Default value is "
            "1/sqrt(head_size).",
            onnx::AttributeProto::FLOAT, /*required=*/false)
      .Attr("mask_filter_value",
            "The value to be filled in the attention mask. Default value "
            "is -10000.0.",
            onnx::AttributeProto::FLOAT, static_cast<float>(-10000.0f))
      .Attr("do_rotary", "Whether to use rotary position embedding.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("past_present_share_buffer",
            "Corresponding past and present are same tensor, its shape is "
            "(2, batch_size, num_heads, max_sequence_length, head_size).",
            onnx::AttributeProto::INT, /*required=*/false)
      .Input(0, "input",
             "3D input tensor with shape (batch_size, sequence_length, "
             "input_hidden_size).",
             "T1")
      .Input(1, "weight",
             "2D input tensor with shape (input_hidden_size, 3 * "
             "hidden_size), hidden_size = num_heads * head_size.",
             "T2")
      .Input(2, "bias", "1D input tensor with shape (3 * hidden_size).", "T3")
      .Input(3, "input_scale",
             "Scale of quantized input tensor. Scalar (per-tensor "
             "quantization).",
             "T3")
      .Input(4, "weight_scale",
             "Scale of quantized weight tensor. Scalar or 1D "
             "(per-tensor/per-column quantization).",
             "T3")
      .Input(5, "mask_index",
             "Attention mask index with shape "
             "(batch_size).",
             "T4", OpSchema::Optional)
      .Input(6, "input_zero_point",
             "Zero point of quantized input tensor. Scalar (per-tensor "
             "quantization).",
             "T1", OpSchema::Optional)
      .Input(7, "weight_zero_point",
             "Zero point of quantized weight tensor. Scalar or 1D "
             "(per-tensor/per-column quantization).",
             "T2", OpSchema::Optional)
      .Input(8, "past",
             "Past state for key and value with shape (2, batch_size, "
             "num_heads, past_sequence_length, head_size).",
             "T3", OpSchema::Optional)
      .Output(0, "output",
              "3D output tensor with shape (batch_size, sequence_length, "
              "hidden_size).",
              "T3")
      .Output(1, "present",
              "Present state for key and value with shape (2, batch_size, "
              "num_heads, total_sequence_length, head_size).",
              "T3", OpSchema::Optional)
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"},
                      "Constrain input and its zero point to int8/uint8.")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"},
                      "Constrain weight and its zero point to int8/uint8.")
      .TypeConstraint("T3", {"tensor(float)", "tensor(float16)"},
                      "Constrain bias, scales, past, present, and output to "
                      "float tensors.")
      .TypeConstraint("T4", {"tensor(int32)"},
                      "Constrain mask index to integer types.");
}

// ONNX Runtime's block-wise N-bit weight-only quantized MatMul (ORT GenAI's
// own INT4 weight compression path for LLM/ASR deployment -- distinct from
// onnxsim's own opset-21-native weight_only_quantize_int4_matmul.h, which
// targets any conformant runtime instead of ORT specifically). Registered
// here so weight_only_quantize_matmul_nbits.h's fused output -- which only
// ever emits inputs 0-2 (A, B, scales) and attrs K/N/bits/block_size --
// passes the ONNX checker and shape-inference sweeps the rest of onnxsim's
// pipeline runs; the remaining inputs/attrs (zero_points, g_idx, bias,
// weight_prepacked, accuracy_level) are declared for schema
// completeness/compatibility with externally-authored models, not because
// this pass ever produces them.
OpSchema MakeMatMulNBitsSchema() {
  return OpSchema()
      .SetName("MatMulNBits")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "MatMulNBits performs a matrix multiplication where the "
          "right-hand-side matrix (weights) is quantized to N bits.")
      .Attr("K", "Size of the input feature dimension.",
            onnx::AttributeProto::INT, /*required=*/true)
      .Attr("N", "Size of the output feature dimension.",
            onnx::AttributeProto::INT, /*required=*/true)
      .Attr("block_size",
            "Number of weight values quantized together along the K "
            "dimension. Must be a power of 2 and >= 16.",
            onnx::AttributeProto::INT, /*required=*/true)
      .Attr("bits", "Number of bits used to quantize each weight value.",
            onnx::AttributeProto::INT, static_cast<int64_t>(4))
      .Attr("accuracy_level",
            "Optional accuracy level hint for the internal computation.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("weight_prepacked",
            "Whether B is prepacked into a runtime-specific layout.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Input(0, "A", "The input tensor, not quantized.", "T1")
      .Input(1, "B",
             "Packed uint8 tensor of shape (N, k_blocks, blob_size), "
             "k_blocks = ceil(K / block_size), blob_size = block_size * "
             "bits / 8, bit-packed low-nibble-first along K within each "
             "block.",
             "T2")
      .Input(2, "scales",
             "Per-block scaling factors with shape (N, k_blocks), same "
             "type as A.",
             "T1")
      .Input(3, "zero_points",
             "Per-block zero point. Packed (uint8, shape (N, "
             "ceil(k_blocks * bits / 8))) or unpacked (same type as A, "
             "shape (N, k_blocks)). Defaults to 2^(bits-1) when omitted.",
             "T3", OpSchema::Optional)
      .Input(4, "g_idx", "Deprecated group index input.", "T4",
             OpSchema::Optional)
      .Input(5, "bias", "Bias to add to the result, shape [N].", "T1",
             OpSchema::Optional)
      .Output(0, "Y", "The output tensor.", "T1")
      .TypeConstraint("T1",
                      {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                      "Constrain A, scales, bias, and output to float tensors.")
      .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain B to uint8.")
      .TypeConstraint("T3",
                      {"tensor(uint8)", "tensor(float16)", "tensor(float)",
                       "tensor(bfloat16)"},
                      "Constrain zero_points to uint8 (packed) or a float type "
                      "(unpacked).")
      .TypeConstraint("T4", {"tensor(int32)"}, "Constrain g_idx to int32.");
}

// Picks the fixed function body matching (activation, has_fc1_bias,
// has_fc2_bias) out of contrib_schemas_moe_templates.gen.h. That header is
// generated by scripts/codegen/generate_moe_function_templates.py from real
// onnxscript function definitions -- each one type-checked against opset 18,
// round-tripped through onnx's own text-format parser/checker/shape
// inference, and numerically verified against a real onnxruntime session
// when the generator ran -- rather than hand-typed here; see that script's
// module docstring for exactly what's generic inside each fixed body (any
// num_experts via a real ONNX `Loop`; k, normalize_routing_weights, and
// use_sparse_mixer forwarded from the calling node's own attributes and
// dispatched via runtime `If`s) versus what still has to be selected per
// node here, and why. `activation` is assumed to already be one of the four
// BuildMoEFunctionBody below accepts.
const char* SelectMoEFunctionTemplate(const std::string& activation,
                                      bool has_fc1_bias, bool has_fc2_bias) {
  if (activation == "relu") {
    if (has_fc1_bias) {
      return has_fc2_bias ? kMoEFunctionReluFc1BiasFc2Bias
                          : kMoEFunctionReluFc1BiasFc2NoBias;
    }
    return has_fc2_bias ? kMoEFunctionReluFc1NoBiasFc2Bias
                        : kMoEFunctionReluFc1NoBiasFc2NoBias;
  }
  if (activation == "identity") {
    if (has_fc1_bias) {
      return has_fc2_bias ? kMoEFunctionIdentityFc1BiasFc2Bias
                          : kMoEFunctionIdentityFc1BiasFc2NoBias;
    }
    return has_fc2_bias ? kMoEFunctionIdentityFc1NoBiasFc2Bias
                        : kMoEFunctionIdentityFc1NoBiasFc2NoBias;
  }
  if (activation == "silu") {
    if (has_fc1_bias) {
      return has_fc2_bias ? kMoEFunctionSiluFc1BiasFc2Bias
                          : kMoEFunctionSiluFc1BiasFc2NoBias;
    }
    return has_fc2_bias ? kMoEFunctionSiluFc1NoBiasFc2Bias
                        : kMoEFunctionSiluFc1NoBiasFc2NoBias;
  }
  // "gelu".
  if (has_fc1_bias) {
    return has_fc2_bias ? kMoEFunctionGeluFc1BiasFc2Bias
                        : kMoEFunctionGeluFc1BiasFc2NoBias;
  }
  return has_fc2_bias ? kMoEFunctionGeluFc1NoBiasFc2Bias
                      : kMoEFunctionGeluFc1NoBiasFc2NoBias;
}

// Mixtral-style gated MLP -- fc2(silu(fc1(x)) * fc3(x)), a separate
// ungated fc3 Gemm combined with fc1's activated output before fc2. Only
// ever selected for activation_type == "silu": this is ONNX Runtime's own
// "Mixtral case" (onnxruntime/contrib_ops/cuda/moe/moe.cc), which remaps it
// onto the kernel's SwiGLU path by packing weights as [FC3, FC1] (Linear,
// Gate) -- "Kernel supports SwiGLU which is Linear * SiLU(Gate)" -- and is
// gated on that exact activation check, with no analogous defined behavior
// for relu/identity/gelu (see BuildMoEFunctionBody's own comment for why
// those stay declined instead of reusing this). fc3_experts_bias is an
// independent optional input from fc1/fc2's biases (fc3 is its own Gemm),
// hence the extra has_fc3_bias axis here versus SelectMoEFunctionTemplate
// above.
const char* SelectMoESiluFc3FunctionTemplate(bool has_fc1_bias,
                                             bool has_fc2_bias,
                                             bool has_fc3_bias) {
  if (has_fc1_bias) {
    if (has_fc2_bias) {
      return has_fc3_bias ? kMoEFunctionSiluFc1BiasFc2BiasFc3Bias
                          : kMoEFunctionSiluFc1BiasFc2BiasFc3NoBias;
    }
    return has_fc3_bias ? kMoEFunctionSiluFc1BiasFc2NoBiasFc3Bias
                        : kMoEFunctionSiluFc1BiasFc2NoBiasFc3NoBias;
  }
  if (has_fc2_bias) {
    return has_fc3_bias ? kMoEFunctionSiluFc1NoBiasFc2BiasFc3Bias
                        : kMoEFunctionSiluFc1NoBiasFc2BiasFc3NoBias;
  }
  return has_fc3_bias ? kMoEFunctionSiluFc1NoBiasFc2NoBiasFc3Bias
                      : kMoEFunctionSiluFc1NoBiasFc2NoBiasFc3NoBias;
}

// Context-dependent function body for MoE: attaches one of 24 fixed,
// pre-built function bodies (contrib_schemas_moe_templates.gen.h), selected
// purely by which optional inputs the node has (fc1_experts_bias,
// fc2_experts_bias) and which of the four implemented activations it uses --
// no per-node codegen happens here anymore. That selection is the one piece
// of context-dependence ONNX genuinely requires and can't be designed away:
// confirmed empirically (see generate_moe_function_templates.py's module
// docstring) that ONNX's function-inlining requires a function call's actual
// input count to match its formal parameter list, so a function that
// references an optional input directly cannot also serve a call site that
// omits it -- the *node list itself*, not just a runtime value, would need
// to differ. Every one of the 24 bodies is otherwise fully generic: any
// num_experts (a real ONNX `Loop`, not a per-node unrolled copy), and k/
// normalize_routing_weights/use_sparse_mixer forwarded from the calling
// node's own attributes and dispatched via runtime `If`s inside the body,
// rather than read here and compiled into a choice of what to emit.
//
// fc3_experts_weights (a Mixtral-style separate gate/up/down projection, as
// opposed to fc1 alone) is supported for activation_type == "silu" only --
// see SelectMoESiluFc3FunctionTemplate's comment for exactly why it's silu
// specifically, straight from ONNX Runtime's own CUDA kernel. This is a
// real, shipping convention (onnxruntime-genai's Phi-3.5-MoE builder), not
// a synthetic one, but has no CPU-kernel oracle to check against (ORT's own
// CPU MoE kernel rejects fc3 outright, for any activation) -- disclosed as
// the same kind of validation gap as use_sparse_mixer below.
//
// Still declines (returns false, leaving the node opaque but still
// shape-inferable via the TypeAndShapeInferenceFunction above):
//  - swiglu, or an unrecognized activation_type (also swiglu_fusion != 0,
//    which is only meaningful for swiglu) -- not yet implemented; unlike
//    num_experts/k/normalize_routing_weights/use_sparse_mixer above, swiglu
//    needs genuinely different ops (a second per-expert GEMM, a different
//    activation formula, a fused/interleaved input layout), not just a
//    different runtime value, so it can't be folded into the same fixed
//    bodies as a runtime branch the way those were.
//  - fc3_experts_weights with any activation other than silu: real ORT has
//    no defined behavior for that combination either (its CUDA kernel's
//    fc3 remapping is itself gated on activation_type == silu), so this
//    stays declined rather than guessing a formula ORT itself doesn't
//    implement.
bool BuildMoEFunctionBody(const FunctionBodyBuildContext& ctx,
                          const OpSchema& schema,
                          FunctionProto& function_proto) {
  const auto* activation_attr = ctx.getAttribute("activation_type");
  std::string activation =
      activation_attr != nullptr && activation_attr->has_s()
          ? activation_attr->s()
          : "relu";
  if (activation != "relu" && activation != "identity" &&
      activation != "silu" && activation != "gelu") {
    return false;  // swiglu, or an unrecognized value.
  }
  const auto* swiglu_fusion_attr = ctx.getAttribute("swiglu_fusion");
  if (swiglu_fusion_attr != nullptr && swiglu_fusion_attr->has_i() &&
      swiglu_fusion_attr->i() != 0) {
    return false;
  }
  const bool has_fc3 = ctx.hasInput(6);
  if (has_fc3 && activation != "silu") {
    // relu/identity/gelu + fc3 has no ORT-defined behavior at all: real
    // ORT's CUDA kernel only remaps fc3 into its SwiGLU path when
    // activation_type is silu specifically (see
    // SelectMoESiluFc3FunctionTemplate's comment); for any other activation it
    // would still repack fc1/fc3 into a doubled-width buffer while running that
    // activation on it unchanged, silently misinterpreting the weights rather
    // than erroring -- so this stays declined instead of guessing a formula for
    // it.
    return false;
  }

  const bool has_fc1_bias = ctx.hasInput(3);
  const bool has_fc2_bias = ctx.hasInput(5);
  const char* text = has_fc3 ? SelectMoESiluFc3FunctionTemplate(
                                   has_fc1_bias, has_fc2_bias, ctx.hasInput(7))
                             : SelectMoEFunctionTemplate(
                                   activation, has_fc1_bias, has_fc2_bias);

  FunctionBuilder builder(function_proto);
  // Every standard-domain op any of the fixed bodies use (Softmax, TopK,
  // Gemm, Relu, Sigmoid, Tanh, Div, Mul, Add, Sub, Abs, Max, Greater, Or,
  // Equal, Cast, Where, Exp, Range, Reshape, Shape, Concat, Gather, Squeeze,
  // Unsqueeze, Slice, ScatterElements, CastLike, Constant, Identity, Loop,
  // If) has been available, with the signature used here, since opset 18 --
  // schema.BuildFunction() below fills in the function's own (com.microsoft)
  // opset_import entry, but not this one, so it has to be added explicitly
  // or the function is left referencing an undeclared "" domain.
  builder.AddOpset("", 18);
  builder.Add(text);
  schema.BuildFunction(function_proto);
  return true;
}

// Real ONNX Runtime Mixture-of-Experts op (docs/ContribOperators.md,
// onnxruntime/core/graph/contrib_ops/contrib_defs.cc). Despite its name,
// `router_probs` is raw per-expert routing *logits*, not an
// already-normalized distribution (confirmed empirically against ONNX
// Runtime's own CPU kernel -- see generate_moe_function_templates.py's
// routing comments); the op applies Softmax over it (or, if
// use_sparse_mixer is set, a different jitter-masked top-2 routing --
// transcribed from ONNX Runtime's CUDA kernel, since no CPU kernel
// implements it to check against), then does top-k expert selection, the
// per-expert two-layer FFN, and the weighted combine. So, unlike
// Attention/QAttention above, MoE's *reference* semantics are expressible
// as a plain composition of standard ONNX ops: Softmax normalizes the
// routing logits, TopK selects the k highest-probability experts per
// token, ScatterElements turns that into a dense (mostly-zero) per-token
// gate row, and each expert's FFN is then computed for every token --
// via a real ONNX `Loop` over however many experts fc1_experts_weights'
// shape says there are, not a per-node unrolled copy -- and masked by its
// gate value before being summed. This computes E/k times more FLOPs than
// a real sparse-dispatch kernel (it evaluates every expert for every token
// instead of only the ones it was routed to), but is numerically identical
// and needs no data-dependent intermediate shapes.
OpSchema MakeMoESchema() {
  return OpSchema()
      .SetName("MoE")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "Mixture of experts. router_probs holds per-token, per-expert "
          "routing logits (a Softmax is applied internally, despite the "
          "name); this op selects the top-k experts per token by the "
          "resulting probability, runs each selected expert's two-layer "
          "FFN, and combines the results weighted by the (optionally "
          "renormalized) routing probabilities of the selected experts.")
      .Attr("activation_type",
            "Activation function to use. Choose from relu, gelu, silu, "
            "swiglu and identity. Default is relu",
            onnx::AttributeProto::STRING, std::string("relu"))
      .Attr("k", "Number of top experts to select from expert pool.",
            onnx::AttributeProto::INT, /*required=*/true)
      .Attr("normalize_routing_weights",
            "Whether to normalize routing weights.", onnx::AttributeProto::INT,
            static_cast<int64_t>(0))
      .Attr("use_sparse_mixer", "Whether to use sparse mixer.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("swiglu_fusion",
            "0: not fused, 1: fused and interleaved, 2: fused and not "
            "interleaved.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("swiglu_limit",
            "The limit used to clamp in SwiGLU. No clamp when limit is not "
            "provided.",
            onnx::AttributeProto::FLOAT, /*required=*/false)
      .Attr("activation_alpha", "Alpha parameter used in activation function.",
            onnx::AttributeProto::FLOAT, /*required=*/false)
      .Attr("activation_beta", "Beta parameter used in activation function.",
            onnx::AttributeProto::FLOAT, /*required=*/false)
      .Input(0, "input",
             "2D input tensor with shape (num_tokens, hidden_size) or 3D "
             "input tensor with shape (batch_size, sequence_length, "
             "hidden_size).",
             "T")
      .Input(1, "router_probs",
             "2D input tensor with shape (num_tokens, num_experts). Despite "
             "the name, these are raw routing logits -- a Softmax is "
             "applied internally before top-k expert selection.",
             "T")
      .Input(2, "fc1_experts_weights",
             "3D input tensor with shape (num_experts, fusion_size * "
             "inter_size, hidden_size), where fusion_size is 2 for fused "
             "swiglu, and 1 otherwise.",
             "T")
      .Input(3, "fc1_experts_bias",
             "2D optional input tensor with shape (num_experts, "
             "fusion_size * inter_size).",
             "T", OpSchema::Optional)
      .Input(4, "fc2_experts_weights",
             "3D input tensor with shape (num_experts, hidden_size, "
             "inter_size).",
             "T")
      .Input(5, "fc2_experts_bias",
             "2D optional input tensor with shape (num_experts, "
             "hidden_size).",
             "T", OpSchema::Optional)
      .Input(6, "fc3_experts_weights",
             "3D optional input tensor with shape (num_experts, inter_size, "
             "hidden_size).",
             "T", OpSchema::Optional)
      .Input(7, "fc3_experts_bias",
             "2D optional input tensor with shape (num_experts, "
             "inter_size).",
             "T", OpSchema::Optional)
      .Output(0, "output",
              "2D output tensor with shape (num_tokens, hidden_size) or 3D "
              "output tensor with shape (batch_size, sequence_length, "
              "hidden_size).",
              "T")
      .TypeConstraint("T",
                      {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                      "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput)
      .SetContextDependentFunctionBodyBuilder(
          [](const FunctionBodyBuildContext& ctx, const OpSchema& schema,
             FunctionProto& function_proto) {
            return BuildMoEFunctionBody(ctx, schema, function_proto);
          });
}

// ONNX Runtime's quantized counterpart of MoE, above. QMoE's weights are
// packed/quantized (int2/int4/int8, or one of several block-scaled FP4/FP8
// layouts selected by `quant_type`) and its inputs consequently balloon to
// up to 21 tensors (per-fc scales, zero points, global scales, activation
// scales/block-scales for the FP8-activation modes, an optional
// `router_weights` for DeepSeek-style select/aggregate-with-different-
// tensors routing, ...). Registered here for shape/type-inference
// completeness only, matching Attention/QAttention/MatMulNBits above --
// unlike MoE, dequantizing each of those layouts correctly is out of scope
// for a reference decomposition, so no function body is attached; a QMoE
// node stays an opaque (but now shape-inferable) op as far as onnxsim's own
// pipeline is concerned.
OpSchema MakeQMoESchema() {
  OpSchema schema;
  schema.SetName("QMoE")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "Quantized mixture of experts (MoE). Weights are quantized "
          "per-expert; see ONNX Runtime's docs/ContribOperators.md for the "
          "full set of supported quantization layouts.")
      .Attr("activation_type",
            "Activation function to use. Choose from relu, gelu, silu, "
            "swiglu and identity. Default is relu",
            onnx::AttributeProto::STRING, std::string("relu"))
      .Attr("k", "Number of top experts to select from expert pool.",
            onnx::AttributeProto::INT, /*required=*/true)
      .Attr("normalize_routing_weights",
            "Whether to normalize routing weights.", onnx::AttributeProto::INT,
            static_cast<int64_t>(0))
      .Attr("use_sparse_mixer", "Whether to use sparse mixer.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("swiglu_fusion",
            "0: not fused, 1: fused and interleaved, 2: fused and not "
            "interleaved.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("swiglu_limit",
            "The limit used to clamp inputs in SwiGLU. It is infinite when "
            "limit is not provided.",
            onnx::AttributeProto::FLOAT, /*required=*/false)
      .Attr("expert_weight_bits",
            "Number of bits used in quantized weights. Supported values "
            "are 2, 4, and 8. Default is 4 bits.",
            onnx::AttributeProto::INT, static_cast<int64_t>(4))
      .Attr("block_size",
            "Size of each quantization block along the K (input feature) "
            "dimension.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("quant_type",
            "Quantization type: 'int' (default), 'fp4', 'nvfp4', 'fp8', or "
            "'wfp4afp8'.",
            onnx::AttributeProto::STRING, std::string("int"))
      .Attr("weights_prepacked",
            "Tri-state control over the layout of the int4/int8 fc1/fc2 "
            "weight initializers. Defaults to -1.",
            onnx::AttributeProto::INT, static_cast<int64_t>(-1))
      .Input(0, "input",
             "2D tensor with shape (num_tokens, hidden_size), or 3D tensor "
             "with shape (batch_size, sequence_length, hidden_size).",
             "T")
      .Input(1, "router_probs",
             "2D tensor with shape (num_tokens, num_experts).", "T")
      .Input(2, "fc1_experts_weights",
             "3D packed/quantized tensor of FC1 expert weights.", "T1")
      .Input(3, "fc1_scales", "Optional FC1 weight scales.", "T2",
             OpSchema::Optional)
      .Input(4, "fc1_experts_bias", "2D optional FC1 expert bias.", "T",
             OpSchema::Optional)
      .Input(5, "fc2_experts_weights",
             "3D packed/quantized tensor of FC2 expert weights.", "T1")
      .Input(6, "fc2_scales", "Optional FC2 weight scales.", "T2",
             OpSchema::Optional)
      .Input(7, "fc2_experts_bias", "2D optional FC2 expert bias.", "T",
             OpSchema::Optional)
      .Input(8, "fc3_experts_weights",
             "3D optional packed/quantized tensor of FC3 expert weights.", "T1",
             OpSchema::Optional)
      .Input(9, "fc3_scales", "Optional FC3 weight scales.", "T2",
             OpSchema::Optional)
      .Input(10, "fc3_experts_bias", "2D optional FC3 expert bias.", "T",
             OpSchema::Optional)
      .Input(11, "fc1_zero_points", "Optional FC1 quantization zero points.",
             "T1", OpSchema::Optional)
      .Input(12, "fc2_zero_points", "Optional FC2 quantization zero points.",
             "T1", OpSchema::Optional)
      .Input(13, "fc3_zero_points", "Optional FC3 quantization zero points.",
             "T1", OpSchema::Optional)
      .Input(14, "router_weights",
             "2D optional tensor with shape (num_tokens, num_experts), used "
             "for DeepSeek-style select/aggregate-with-different-tensors "
             "routing.",
             "T", OpSchema::Optional)
      .Input(15, "fc1_global_scale", "Optional per-expert FC1 global scale.",
             "T4", OpSchema::Optional)
      .Input(16, "fc2_global_scale", "Optional per-expert FC2 global scale.",
             "T4", OpSchema::Optional)
      .Input(17, "fc1_act_scale", "Optional FC1 FP8 activation scale.", "T4",
             OpSchema::Optional)
      .Input(18, "fc2_act_scale", "Optional FC2 FP8 activation scale.", "T4",
             OpSchema::Optional)
      .Input(19, "fc1_act_block_scale",
             "Optional FC1 MXFP activation block-scale tensor.", "T2",
             OpSchema::Optional)
      .Input(20, "fc2_act_block_scale",
             "Optional FC2 MXFP activation block-scale tensor.", "T2",
             OpSchema::Optional)
      .Output(0, "output", "Output tensor with the same shape as input.", "T")
      .TypeConstraint("T",
                      {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                      "Constrain input and output types to float tensors.")
      .TypeConstraint("T1", {"tensor(uint8)", "tensor(float8e4m3fn)"},
                      "Constrain quantized weight types.")
      .TypeConstraint("T2",
                      {"tensor(float)", "tensor(float16)", "tensor(bfloat16)",
                       "tensor(float8e8m0)", "tensor(float8e4m3fn)"},
                      "Constrain scale types.")
      .TypeConstraint("T4", {"tensor(float)"},
                      "Constrain FP4 global scale type to float32 tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput);
  return schema;
}

void RegisterAll() {
  // The custom domain must be known to the schema registry before any schema
  // in it can be registered.
  auto& domain_range = onnx::OpSchemaRegistry::DomainToVersionRange::Instance();
  if (domain_range.Map().count(kMSDomain) == 0) {
    domain_range.AddDomainToVersion(kMSDomain, /*min_version=*/1,
                                    /*max_version=*/1);
  }

  RegisterIfAbsent(MakeQLinearBinarySchema("QLinearAdd"));
  RegisterIfAbsent(MakeQLinearBinarySchema("QLinearMul"));
  RegisterIfAbsent(
      MakeQLinearUnarySchema("QLinearSigmoid", /*has_alpha=*/false));
  RegisterIfAbsent(
      MakeQLinearUnarySchema("QLinearLeakyRelu", /*has_alpha=*/true));
  RegisterIfAbsent(MakeQLinearConcatSchema());
  RegisterIfAbsent(MakeQLinearSoftmaxSchema());
  RegisterIfAbsent(MakeQLinearAveragePoolSchema());
  RegisterIfAbsent(MakeQLinearGlobalAveragePoolSchema());
  RegisterIfAbsent(MakeQLinearWhereSchema());
  RegisterIfAbsent(MakeQGemmSchema());
  RegisterIfAbsent(MakeMatMulIntegerToFloatSchema());
  RegisterIfAbsent(MakeAttentionSchema());
  RegisterIfAbsent(MakeQAttentionSchema());
  RegisterIfAbsent(MakeMatMulNBitsSchema());
  RegisterIfAbsent(MakeMoESchema());
  RegisterIfAbsent(MakeQMoESchema());

  // Augment the standard Reshape schema with a data-propagation function so
  // shape tensors can flow through a Reshape during partial shape evaluation.
  RegisterReshapeDataPropagation();
}

}  // namespace

void RegisterContribOpSchemas() {
  static std::once_flag once;
  std::call_once(once, RegisterAll);
}

}  // namespace onnxsim

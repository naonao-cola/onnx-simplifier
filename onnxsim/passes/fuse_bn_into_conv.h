// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//	 conv = Conv()
//   bn = BatchNormalization()
//
// After:
//	 bn is deleted
//   new inputs/initializers to conv are added to graph
//   any no longer used inputs/initializers are erased from graph
//
//	 this pass can handle the case satisfy all following conditions:
//	   condition 1: Run in testing mode
//     condition 2: Inputs 1 - 4 of bn are all initializer_size
//     condition 3: Output of initial conv has no other uses
//     condition 3: Currently works for only DOUBLE, FLOAT32 tensor types
//
// Formula for transformation
// $$ X_{bn} = \frac{s(X - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
// $$ X_{conv} = X * W + b_{conv} $$
// thus, substituting $X$ with $X_{conv}$ in the BN equation we get:
// $$X_{bn} = X * \frac{sW}{\sqrt{\sigma + \epsilon}} + \frac{s(b_{conv} -
// m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$ or
// $$ W' = W\frac{s}{\sqrt{\sigma + \epsilon}}$$
// $$ b' = (b_{conv} - m)\frac{s}{\sqrt{\sigma + \epsilon}} + b_{bn}$$

#include <cmath>
#include <numeric>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {
// TODO: Currently broken for complex values and float16
struct FuseBNIntoConv final : public PredicateBasedPass {
  explicit FuseBNIntoConv()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "fuse_bn_into_conv"; }

  // This pass instance is reused across every round of onnxsim's
  // simplification fixed point, but a batch of reserved names is only valid
  // for the graph state it was reserved against -- other passes/rounds can
  // introduce new names in between. Drop any leftover reservation from a
  // prior runPass() call so nextReservedName() always reserves fresh against
  // the current graph.
  bool initializePass(Graph&) override {
    reserved_names_.clear();
    reserved_used_ = 0;
    return false;
  }

  // Write `v` into `t`'s own typed storage. Overloaded (not a template
  // specialization) so ModifyConvNumeric<T>() below can call it unqualified
  // for either T -- ordinary member lookup resolves the right one per
  // instantiation since it is a plain overload set, not a template itself.
  static void SetTensorData(Tensor& t, std::vector<float> v) {
    t.floats() = std::move(v);
  }
  static void SetTensorData(Tensor& t, std::vector<double> v) {
    t.doubles() = std::move(v);
  }

  // Fold the BN parameters into the conv/conv-transpose weight and bias with
  // direct numeric arithmetic -- this file's own documented formula above --
  // instead of modify_conv()'s fallback below, which builds a 10-node
  // computation chain (Cast/Add/Sqrt/Div/Unsqueeze/Mul/Sub/Mul/Add) for a
  // *later* constant-folding pass to evaluate. Every node or initializer
  // created anywhere in this pass pays its own full graph (and subgraph) scan
  // -- Graph::createValue() assigns every new Value's numeric id via
  // Graph::getNextUnique(), unconditionally, regardless of any name-string
  // batching (see nextReservedName()'s own comment) -- so the fallback's ~17
  // new Values per match make this pass effectively O(matches * graph size)
  // on a model with many BN layers. This path creates only the 2 new
  // initializers the fused weight/bias actually need.
  //
  // Only takes over for the two element types the file's own doc comment
  // already claims support for (FLOAT/DOUBLE); the caller (modify_conv())
  // only calls this when the conv's existing bias, if any, already shares
  // that exact type too, so no Cast is ever needed here. Evaluates the same
  // sub-expressions in the same order the graph-based formula's constant
  // folding would (var+eps, then sqrt, then divide), so this is not merely
  // approximately but bit-for-bit the same computation, just performed here
  // instead of by a later pass.
  //
  // `axis` is which axis of `conv_W` holds the C output channels (0 for
  // Conv, 1 for ConvTranspose -- see modify_conv()'s own comment on
  // conv_W_out_channels); the weight is scaled per output channel, broadcast
  // over every other axis, matching the graph-based path's Unsqueeze/Mul.
  template <typename T>
  bool ModifyConvNumeric(Node* conv, Node* bn, Graph& graph,
                         const Tensor& bn_scale, const Tensor& bn_bias,
                         const Tensor& bn_mean, const Tensor& bn_var,
                         const Tensor& conv_W, const Tensor* conv_bias,
                         int64_t C, int axis) {
    const T* scale = bn_scale.data<T>();
    const T* bias = bn_bias.data<T>();
    const T* mean = bn_mean.data<T>();
    const T* var = bn_var.data<T>();
    const T* w_in = conv_W.data<T>();
    const T eps =
        static_cast<T>(GetValueFromAttrWithDefault(bn, kepsilon, 1e-5f));

    const auto& w_sizes = conv_W.sizes();
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) {
      outer *= w_sizes[i];
    }
    int64_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < w_sizes.size(); ++i) {
      inner *= w_sizes[i];
    }

    std::vector<T> s(static_cast<size_t>(C));
    std::vector<T> new_bias(static_cast<size_t>(C));
    for (int64_t c = 0; c < C; ++c) {
      s[static_cast<size_t>(c)] = scale[c] / std::sqrt(var[c] + eps);
      const T orig_bias_c =
          conv_bias == nullptr ? T(0) : conv_bias->data<T>()[c];
      new_bias[static_cast<size_t>(c)] =
          (orig_bias_c - mean[c]) * s[static_cast<size_t>(c)] + bias[c];
    }

    std::vector<T> new_w(static_cast<size_t>(outer * C * inner));
    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t c = 0; c < C; ++c) {
        const T sc = s[static_cast<size_t>(c)];
        const int64_t base = o * C * inner + c * inner;
        for (int64_t j = 0; j < inner; ++j) {
          new_w[static_cast<size_t>(base + j)] = w_in[base + j] * sc;
        }
      }
    }

    Tensor w_t;
    w_t.setName(nextReservedName(graph));
    w_t.elem_type() = conv_W.elem_type();
    w_t.sizes() = conv_W.sizes();
    SetTensorData(w_t, std::move(new_w));
    Value* new_w_value = graph.addInitializerAndCreateValue(w_t);

    Tensor b_t;
    b_t.setName(nextReservedName(graph));
    b_t.elem_type() = bn_bias.elem_type();
    b_t.sizes().push_back(C);
    SetTensorData(b_t, std::move(new_bias));
    Value* new_b_value = graph.addInitializerAndCreateValue(b_t);

    const auto& conv_inputs = conv->inputs();
    Value* old_w_value = conv_inputs[1];
    conv->replaceInput(1, new_w_value);
    if (old_w_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_w_value);
    }

    if (conv_inputs.size() == 3) {
      Value* old_b_value = conv_inputs[2];
      conv->replaceInput(2, new_b_value);
      if (old_b_value->uses().size() == 0) {
        graph.eraseInitializerAndInput(old_b_value);
      }
    } else {
      conv->addInput(new_b_value);
    }
    return true;
  }

  bool modify_conv(Node* conv, Node* bn, Graph& graph, const bool is_conv) {
    const auto& bn_inputs = bn->inputs();
    const auto& conv_inputs = conv->inputs();

    const Tensor* bn_scale_p = FetchConstantTensor(bn_inputs[1]);
    const Tensor* bn_bias_p = FetchConstantTensor(bn_inputs[2]);
    const Tensor* bn_mean_p = FetchConstantTensor(bn_inputs[3]);
    const Tensor* bn_var_p = FetchConstantTensor(bn_inputs[4]);
    const Tensor* conv_W_p = FetchConstantTensor(conv_inputs[1]);

    /// scale bias mean var must be the same shape (C)
    ONNX_ASSERT(bn_scale_p->sizes() == bn_bias_p->sizes());
    ONNX_ASSERT(bn_scale_p->sizes() == bn_mean_p->sizes());
    ONNX_ASSERT(bn_scale_p->sizes() == bn_var_p->sizes());
    ONNX_ASSERT(bn_scale_p->sizes().size() == 1);
    int64_t C = bn_scale_p->sizes()[0];
    ONNX_ASSERT(conv_W_p->sizes().size() > 2);
    // The BatchNormalization channel count C corresponds to the convolution's
    // output channels. Conv weight layout is (out_channels, in_channels/group,
    // kH, kW), so the output channels live on axis 0. ConvTranspose weight
    // layout is (in_channels, out_channels/group, kH, kW), so they live on
    // axis 1 instead. Picking the wrong axis here is what caused the reported
    // `conv_W.sizes()[0] == C` assertion failure for ConvTranspose whenever
    // in_channels != out_channels. For grouped ConvTranspose the axis-1 size is
    // out_channels/group != C, so skip the fusion rather than miscompiling.
    const int axis = is_conv ? 0 : 1;
    const int64_t conv_W_out_channels =
        conv_W_p->sizes()[static_cast<size_t>(axis)];
    if (conv_W_out_channels != C) {
      return false;
    }
    if (bn_scale_p->elem_type() != bn_bias_p->elem_type() ||
        bn_scale_p->elem_type() != bn_mean_p->elem_type() ||
        bn_scale_p->elem_type() != bn_var_p->elem_type() ||
        bn_scale_p->elem_type() != conv_W_p->elem_type()) {
      return false;
    }

    // The conv's existing bias (if any) is fetched once, up front, so both
    // the numeric fast path and the graph-based fallback below can share it.
    const Tensor* conv_bias_p = nullptr;
    if (conv_inputs.size() == 3) {
      if (!IsConstantTensor(conv_inputs[2])) {
        return false;
      }
      conv_bias_p = FetchConstantTensor(conv_inputs[2]);
      ONNX_ASSERT(conv_bias_p->sizes() == bn_scale_p->sizes());
    }

    const bool bias_type_matches =
        conv_bias_p == nullptr ||
        conv_bias_p->elem_type() == bn_scale_p->elem_type();
    if (bias_type_matches) {
      if (bn_scale_p->elem_type() ==
          ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        return ModifyConvNumeric<float>(conv, bn, graph, *bn_scale_p,
                                        *bn_bias_p, *bn_mean_p, *bn_var_p,
                                        *conv_W_p, conv_bias_p, C, axis);
      }
      if (bn_scale_p->elem_type() ==
          ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
        return ModifyConvNumeric<double>(conv, bn, graph, *bn_scale_p,
                                         *bn_bias_p, *bn_mean_p, *bn_var_p,
                                         *conv_W_p, conv_bias_p, C, axis);
      }
    }

    // Fallback: any other element type, or a conv bias whose type differs
    // from the BN parameters' own type (which needs the Cast node below).
    auto bn_scale = *bn_scale_p;
    auto bn_bias = *bn_bias_p;
    auto bn_mean = *bn_mean_p;
    auto bn_var = *bn_var_p;
    auto conv_W = *conv_W_p;
    // Each rename below used to call graph.getNextUniqueName() directly, and
    // every one of those (like every fresh Value -- see the node-creation
    // calls further down) pays a full graph (and subgraph) scan via
    // Graph::isNameUnique(). Drawing them from one batched reservation
    // instead (Graph::reserveUniqueNames()) turns the handful of scans this
    // function's own renames would otherwise pay into a small, amortized
    // number shared across many fusions.
    bn_scale.setName(nextReservedName(graph));
    bn_bias.setName(nextReservedName(graph));
    bn_mean.setName(nextReservedName(graph));
    bn_var.setName(nextReservedName(graph));
    conv_W.setName(nextReservedName(graph));

    Value* conv_bias = nullptr;
    if (conv_inputs.size() == 3) {
      auto bc_t = *conv_bias_p;
      bc_t.setName(nextReservedName(graph));
      conv_bias = graph.addInitializerAndCreateValue(bc_t);
    } else {
      Tensor bc_t;
      bc_t.setName(nextReservedName(graph));
      bc_t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
      bc_t.sizes().push_back(C);
      for (int i = 0; i < C; ++i) {
        bc_t.floats().push_back(float{0});
      }
      conv_bias = graph.addInitializerAndCreateValue(bc_t);
    }

    /// scalar
    Tensor eps_t;
    eps_t.setName(nextReservedName(graph));
    eps_t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    eps_t.floats().push_back(GetValueFromAttrWithDefault(bn, kepsilon, 1e-5f));
    Value* eps = graph.addInitializerAndCreateValue(eps_t);

    Node* cast = graph.create(kCast, 1);
    cast->addInput(eps);
    cast->i_(kto, bn_var.elem_type());
    cast->insertBefore(conv);

    Node* var_add = graph.create(kAdd, 1);
    var_add->insertAfter(cast);
    var_add->addInput(graph.addInitializerAndCreateValue(bn_var));
    var_add->addInput(cast->output());

    Node* sqrt = graph.create(kSqrt, 1);
    sqrt->insertAfter(var_add);
    sqrt->addInput(var_add->output());

    Node* scale = graph.create(kDiv, 1);
    scale->insertAfter(sqrt);
    scale->addInput(graph.addInitializerAndCreateValue(bn_scale));
    scale->addInput(sqrt->output());

    Node* unsqueeze = graph.create(kUnsqueeze, 1);
    unsqueeze->insertAfter(scale);
    unsqueeze->addInput(scale->output());
    // Broadcast the per-output-channel scale so it lines up with the weight's
    // output-channel axis: axis 0 for Conv, axis 1 for ConvTranspose.
    std::vector<int64_t> insert_dims(conv_W.sizes().size());
    std::iota(insert_dims.begin(), insert_dims.end(), 0);
    insert_dims.erase(insert_dims.begin() + (is_conv ? 0 : 1));
    if (getOpsetVersion(graph) >= 13) {
      Tensor shape_s_t;
      shape_s_t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
      shape_s_t.sizes().push_back(insert_dims.size());
      shape_s_t.int64s() = insert_dims;
      unsqueeze->addInput(graph.addInitializerAndCreateValue(shape_s_t));
    } else {
      unsqueeze->is_(kaxes, std::move(insert_dims));
    }

    Node* mul_w = graph.create(kMul, 1);
    mul_w->insertAfter(unsqueeze);
    mul_w->addInput(graph.addInitializerAndCreateValue(conv_W));
    mul_w->addInput(unsqueeze->output());

    Node* cast1 = graph.create(kCast, 1);
    cast1->insertAfter(mul_w);
    cast1->addInput(conv_bias);
    cast1->i_(kto, bn_mean.elem_type());

    Node* sub = graph.create(kSub, 1);
    sub->insertAfter(cast1);
    sub->addInput(cast1->output());
    sub->addInput(graph.addInitializerAndCreateValue(bn_mean));

    Node* mul = graph.create(kMul, 1);
    mul->insertAfter(sub);
    mul->addInput(sub->output());
    mul->addInput(scale->output());

    Node* bias_add = graph.create(kAdd, 1);
    bias_add->insertAfter(mul);
    bias_add->addInput(mul->output());
    bias_add->addInput(graph.addInitializerAndCreateValue(bn_bias));

    Value* old_w_value = conv_inputs[1];
    conv->replaceInput(1, mul_w->output());
    if (old_w_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_w_value);
    }

    if (conv_inputs.size() == 3) {
      Value* old_b_value = conv_inputs[2];
      conv->replaceInput(2, bias_add->output());
      if (old_b_value->uses().size() == 0) {
        graph.eraseInitializerAndInput(old_b_value);
      }
    } else {
      conv->addInput(bias_add->output());
    }
    return true;
  }

  bool patternMatchPredicate(Node* n) override {
    return (CheckKind(n, kBatchNormalization, 0, kConv) ||
            CheckKind(n, kBatchNormalization, 0, kConvTranspose)) &&
           GetValueFromAttrWithDefault(n, "training_mode", (int64_t)0) == 0 &&
           n->input(0)->uses().size() == 1 && n->outputs().size() == 1 &&
           IsConstantTensor(n, 1) && IsConstantTensor(n, 2) &&
           IsConstantTensor(n, 3) && IsConstantTensor(n, 4) &&
           IsConstantTensor(PrevNode(n, 0), 1);
  }
  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    const bool is_conv = CheckKind(n, kBatchNormalization, 0, kConv);

    Node* bn = n;
    Node* conv = PrevNode(n, 0);
    auto origInput = bn->inputs()[0];
    if (!modify_conv(conv, bn, graph, is_conv)) {
      destroy_current = NodeDestroyType::DestroyZero;
      return false;
    }
    // clean
    for (int i = 4; i >= 1; --i) {
      if (bn->inputs()[i]->uses().size() == 1) {
        auto input = bn->inputs()[i];
        bn->removeInput(i);
        graph.eraseInitializerAndInput(input);
      }
    }
    const bool replacing_success =
        tryReplacingAllUsesWith(bn->output(), origInput);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }

 private:
  // See modify_conv()'s comment on the rename call sites above: batch the
  // names this pass draws for renamed/new initializers so their scans are
  // amortized across many fusions instead of paid one full graph scan at a
  // time. Does not (and cannot, from here) avoid the same per-Value scan
  // Graph::createValue() itself pays for every new node the graph-based
  // fallback creates -- that is intrinsic to onnx's IR (every Value's
  // numeric unique id is assigned via Graph::getNextUnique() in its
  // constructor) and outside this pass's reach; ModifyConvNumeric() above
  // sidesteps it instead by not creating those nodes in the first place.
  static constexpr size_t kNameBatchSize = 256;
  std::vector<std::string> reserved_names_;
  size_t reserved_used_ = 0;

  std::string nextReservedName(Graph& graph) {
    if (reserved_used_ >= reserved_names_.size()) {
      reserved_names_ = graph.reserveUniqueNames(kNameBatchSize);
      reserved_used_ = 0;
    }
    return std::move(reserved_names_[reserved_used_++]);
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE

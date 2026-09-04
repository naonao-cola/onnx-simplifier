// SPDX-License-Identifier: Apache-2.0
//
// C++ port of pruning.py's own apply_structured_pruning -- see that
// function's docstring for the full technique description (this is quoted
// here only where it constrains scope). This port covers all seven of
// pruning.py's own chain finders: a MatMul/vanilla-Gemm producer ->
// consumer pair (_find_chains), a Conv producer -> consumer pair, including
// depthwise pass-through hops and general grouped Conv on either side
// (_find_conv_chains), the gated-FFN SwiGLU/GeGLU pattern -- two producers
// combined by Mul (or ONNX opset-28+'s native SwiGLU node) feeding one
// consumer, both pruned to the same channel indices (_find_gated_chains),
// Conv/MatMul residual (skip-connection) chains, and Conv/MatMul
// Concat-merged (U-Net-style) skip-connection chains -- a bounded slice of
// the general dependency-graph-grouping problem: a channel-preserving
// `Add(a, b)` where both operands are non-constant forces whichever real
// producer(s) feed `a`/`b` to be pruned to the same channel-index set. A
// backward walk plus union-find grouping across such eligible merge points
// (mirroring pruning.py's own _walk_conv_producer_backward/
// _find_conv_residual_chains and _walk_matmul_producer_backward/
// _find_matmul_residual_chains) covers not just a single `Add(x, f(x))` but
// a whole chain of such merges transitively sharing one spine channel
// count; a group with any branch that fails to resolve, or whose leaf
// producers disagree on channel count, is declined in its entirety, never
// partially pruned -- the same conservative "no branch-following" boundary
// every other chain finder here already holds. Once a group's shared
// channel-index set is established, though, it can also fan out *forward*
// to more than one independent ordinary consumer (ResolveConvFanoutBranches/
// ResolveMatmulFanoutBranches) -- so a real multi-block ResNet/transformer
// stage's shared "post-block" tensor, read by both the next block's own
// first Conv/MatMul *and*, unchanged, that block's own `Add`, is reached
// rather than declined; a general grouped Conv may take part in this merge
// too, as a producer, the primary consumer, and/or an extra fan-out branch,
// as long as every one of those that is grouped shares the exact same
// `group` count. For MatMul/Gemm specifically, a fused
// com.microsoft::SkipLayerNormalization/
// SkipSimplifiedLayerNormalization node -- what onnxruntime's transformer
// optimizer collapses a bare residual `Add` plus the following LayerNorm
// into, and so what a fully-optimized transformer's residual connections
// typically look like -- is also recognized as an eligible merge point
// (mirroring pruning.py's own _match_matmul_residual_merge), its own
// gamma/beta/bias constants riding along as a per-channel affine hop on the
// resolved chain; a gated (SwiGLU/GeGLU) combine feeding a residual branch
// with no downstream projection in between is resolved the same way a
// gated pair outside a residual chain already is. Conv residual chains only
// ever see a bare `Add` -- there is no Conv analogue of that fused op. A
// fused com.microsoft::BiasGelu/FastGelu node (a bias-add fused into the
// following Gelu-family activation) is recognized as a per-channel hop too
// (MatMul/Gemm chains only), and com.microsoft::QuickGelu is a plain unary
// pass-through hop everywhere a unary activation is already allowed.
//
// A Concat-merged skip connection (the U-Net-style encoder/decoder merge)
// is matched too, for both MatMul/Gemm (last-axis Concat only) and Conv
// (channel-axis Concat) -- see FindMatmulConcatChains/FindConvConcatChains
// and this file's own "Concat-merged" section comment below: unlike Add, a
// Concat's branches are structurally independent (each owns a fixed,
// disjoint offset range of the merged channel range), so each branch is
// ranked and pruned entirely on its own, reusing the exact same backward
// walkers and fan-out resolution the two residual sections above already
// build; only the shared downstream consumer's weight needs new slicing, at
// each branch's own fixed offset.
//
// Implemented directly on onnx::GraphProto (protobuf), not onnxoptimizer's
// Node/Value IR: pruning.py's own algorithm already works this way (name-
// keyed producer/consumer maps, forward hop-by-hop walks with an explicit
// hop budget), and no new node is ever inserted or removed here -- only
// existing initializers' *values* (and, for a depthwise Conv pass-through
// hop, its own `group` attribute) are overwritten in place -- so there is no
// need for onnxoptimizer's fixed-point pass-registration machinery
// (function_rewriter.cpp already establishes this same "operate on
// GraphProto directly" precedent in this codebase for an equally graph-
// global algorithm).

#include "structured_pruning_entry.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dlpack_dtype.h"

namespace {

constexpr int kMaxChainHops = 8;

// Shape-preserving, channel-order-preserving unary ops that may sit between
// a producer and consumer without blocking the chain -- mirrors pruning.py's
// own _UNARY_PASS_THROUGH exactly.
const std::unordered_set<std::string>& UnaryPassThroughOps() {
  static const std::unordered_set<std::string> kOps = {
      "Relu",
      "LeakyRelu",
      "Elu",
      "Selu",
      "Sigmoid",
      "Tanh",
      "Softplus",
      "Softsign",
      "Gelu",
      "HardSigmoid",
      "Mish",
      "Identity",
      "Cast",
      // com.microsoft::QuickGelu(X) = X * Sigmoid(alpha * X) (alpha an
      // attribute, not a second input) -- purely unary/elementwise, so
      // membership here alone extends every walker that already consults
      // this set, mirroring pruning.py's own _UNARY_PASS_THROUGH.
      "QuickGelu",
  };
  return kOps;
}

// com.microsoft's fused bias-add + Gelu-family activation nodes, mirroring
// pruning.py's own _FUSED_BIAS_GELU_OPS/_match_fused_bias_gelu:
// BiasGelu(A, B) = Gelu(A + B) (bias required) and
// FastGelu(X[, bias]) = Gelu_tanh(X [+ bias]) (bias optional) both fuse an
// FFN's bias-add into its following activation. MatMul/Gemm-chain-only, like
// the per-channel Add/Mul hop these sit alongside -- no Conv-side analogue.
constexpr char kComMicrosoftDomain[] = "com.microsoft";

struct FusedBiasGeluMatch {
  std::string data_name;
  std::optional<std::string> bias_name;
};

std::optional<FusedBiasGeluMatch> MatchFusedBiasGelu(
    const onnx::NodeProto& node,
    const std::unordered_map<std::string, const onnx::TensorProto*>& init_map) {
  bool bias_required;
  if (node.op_type() == "BiasGelu") {
    bias_required = true;
  } else if (node.op_type() == "FastGelu") {
    bias_required = false;
  } else {
    return std::nullopt;
  }
  if (node.domain() != kComMicrosoftDomain || node.input_size() == 0 ||
      node.input(0).empty() || node.output_size() != 1) {
    return std::nullopt;
  }
  const std::string data_name = node.input(0);
  const bool has_bias_input = node.input_size() > 1 && !node.input(1).empty();
  if (!has_bias_input) {
    if (bias_required) {
      return std::nullopt;  // BiasGelu's own schema requires a bias operand.
    }
    return FusedBiasGeluMatch{data_name, std::nullopt};
  }
  const std::string bias_name = node.input(1);
  auto it = init_map.find(bias_name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      it->second->dims_size() == 0) {
    return std::nullopt;  // non-constant bias -- can't safely slice/prune it.
  }
  int64_t prod = 1;
  for (int64_t d : it->second->dims()) {
    prod *= d;
  }
  if (prod != it->second->dims(it->second->dims_size() - 1)) {
    return std::nullopt;
  }
  return FusedBiasGeluMatch{data_name, bias_name};
}

// --- Tensor <-> flat float buffer, mirroring onnx.numpy_helper -------------

std::vector<float> ReadFloatTensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<float> out(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    std::memcpy(out.data(), t.raw_data().data(), out.size() * sizeof(float));
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(out.data()),
                                        out.size() * sizeof(float),
                                        sizeof(float));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = t.float_data(static_cast<int>(i));
    }
  }
  return out;
}

// Overwrites `t` in place with a FLOAT tensor of `dims`/`data`, keeping its
// existing name -- the same "replace, don't mutate a live view" convention
// onnx.numpy_helper.from_array's own always-raw_data output gives
// w_init.CopyFrom(...) in the Python reference.
void SetFloatTensorData(onnx::TensorProto* t, const std::vector<int64_t>& dims,
                        const std::vector<float>& data) {
  const std::string name = t->name();
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(data.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), data.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

// The INT64 analogue of ReadFloatTensor, used only by
// FindAttentionChains/FindSeparateQkvChains's own Reshape-target-shape
// reading and rewriting (WalkToAttentionConsumer/SetInt64TensorLastDim).
std::vector<int64_t> ReadInt64Tensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<int64_t> out(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    std::memcpy(out.data(), t.raw_data().data(), out.size() * sizeof(int64_t));
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(out.data()),
                                        out.size() * sizeof(int64_t),
                                        sizeof(int64_t));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = t.int64_data(static_cast<int>(i));
    }
  }
  return out;
}

// Overwrites the last element of an INT64 tensor `t` in place with
// `new_last`, keeping every other dim -- mirrors pruning.py's own
// `dims[-1] = ...; shape_init.CopyFrom(from_array(dims))` pattern for a
// Reshape node's own target-shape constant.
void SetInt64TensorLastDim(onnx::TensorProto* t, int64_t new_last) {
  std::vector<int64_t> data = ReadInt64Tensor(*t);
  if (data.empty()) {
    return;
  }
  data.back() = new_last;
  const std::string name = t->name();
  std::vector<int64_t> dims(t->dims().begin(), t->dims().end());
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::INT64);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(data.size() * sizeof(int64_t), '\0');
  std::memcpy(raw.data(), data.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(int64_t));
  }
  t->set_raw_data(std::move(raw));
}

// The INT64 analogue of SetFloatTensorData -- overwrites `t` in place with a
// fresh INT64 tensor of `dims`/`data`, keeping its existing name. Used only
// by ApplySplitGatedChains's own `split` *input* rewrite (mirrors
// pruning.py's own `size_init.CopyFrom(onnx.numpy_helper.from_array(...))`
// for that same rewrite), which -- unlike SetInt64TensorLastDim's single-
// element Reshape-target-shape rewrite -- replaces every element of a
// 2-element `[keep_count, keep_count]` sizes tensor.
void SetInt64TensorData(onnx::TensorProto* t, const std::vector<int64_t>& dims,
                        const std::vector<int64_t>& data) {
  const std::string name = t->name();
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::INT64);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(data.size() * sizeof(int64_t), '\0');
  std::memcpy(raw.data(), data.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(int64_t));
  }
  t->set_raw_data(std::move(raw));
}

int64_t ConvGroupAttr(const onnx::NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "group") {
      return attr.i();
    }
  }
  return 1;  // ONNX default.
}

void SetOrAddIntAttr(onnx::NodeProto* node, const std::string& name,
                     int64_t value) {
  for (auto& attr : *node->mutable_attribute()) {
    if (attr.name() == name) {
      attr.set_type(onnx::AttributeProto::INT);
      attr.set_i(value);
      return;
    }
  }
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::INT);
  attr->set_i(value);
}

// --- MatMul/vanilla-Gemm matching, mirroring pruning.py's own (imported
// from smoothquant.py) _match_matmul_like -------------------------------

struct MatMulLikeMatch {
  std::string x_name;
  std::string w_name;
  bool weight_transposed;
};

std::optional<MatMulLikeMatch> MatchMatMulLikeRaw(const onnx::NodeProto& node) {
  if (node.op_type() == "MatMul") {
    if (node.input_size() != 2) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), false};
  }
  if (node.op_type() == "Gemm") {
    const int num_inputs = node.input_size();
    if (num_inputs != 2 && num_inputs != 3) {
      return std::nullopt;
    }
    bool has_trans_a = false, has_alpha = false, has_beta = false;
    int64_t trans_a = 0, trans_b = 0;
    double alpha = 1.0, beta = 1.0;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "transA") {
        trans_a = attr.i();
        has_trans_a = true;
      } else if (attr.name() == "alpha") {
        alpha = attr.f();
        has_alpha = true;
      } else if (attr.name() == "beta") {
        beta = attr.f();
        has_beta = true;
      } else if (attr.name() == "transB") {
        trans_b = attr.i();
      }
    }
    if (has_trans_a && trans_a != 0) {
      return std::nullopt;
    }
    if (has_alpha && alpha != 1.0) {
      return std::nullopt;
    }
    if (num_inputs == 3 && has_beta && beta != 1.0) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), trans_b != 0};
  }
  return std::nullopt;
}

using InitMap = std::unordered_map<std::string, const onnx::TensorProto*>;
using ConsumerMap =
    std::unordered_map<std::string, std::vector<onnx::NodeProto*>>;

ConsumerMap ConsumersOf(onnx::GraphProto* graph) {
  ConsumerMap out;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& inp : node->input()) {
      if (!inp.empty()) {
        out[inp].push_back(node);
      }
    }
  }
  return out;
}

size_t ConsumerCount(const ConsumerMap& consumers_of, const std::string& name) {
  auto it = consumers_of.find(name);
  return it == consumers_of.end() ? 0 : it->second.size();
}

// True iff `node` (already confirmed by the caller to be a plain
// (default-domain) `Clip`, `node.input(0)` the tensor being walked through)
// is a pure elementwise clamp with zero channel dependence, so it is safe to
// cross transparently -- mirrors pruning.py's own
// _match_clip_channel_pass_through exactly (see that function's own
// docstring for the full reasoning: this is the `torch.nn.ReLU6` shape
// ubiquitous in MobileNetV2/V3, EfficientNet-Lite, and QAT exports). Unlike
// a Resize/Pad hop, Clip's own `min`/`max` operands are never axis-indexed
// at all -- per Clip's own schema each must already be a scalar (empty or
// single-element shape), broadcasting uniformly over every element
// regardless of axis, so no axis reasoning is needed and the identical check
// works unchanged for a Conv chain's own axis-1 channel convention and a
// MatMul/Gemm chain's own last-axis convention alike -- shared by
// WalkToConvConsumer/WalkConvProducerBackward and
// WalkToConsumer/WalkMatmulProducerBackward below. Declines (false), never
// guesses, whenever a present `min`/`max` (each optional -- a present but
// empty-string input counts as *not* present) is missing from the
// initializer map (a runtime-computed bound) or not single-element shaped.
// Neither bound's own *value* is ever inspected -- clamping is a pure
// elementwise op, so slicing which channels survive first and clamping
// after computes exactly the same result as clamping first and slicing
// after, for any min/max value.
bool MatchClipChannelPassThrough(const onnx::NodeProto& node,
                                 const InitMap& init_map) {
  if (node.op_type() != "Clip" || node.domain() != "") {
    return false;
  }
  if (node.input_size() == 0 || node.input(0).empty()) {
    return false;
  }
  for (int idx : {1, 2}) {  // min, max -- both optional, opset 11+ input-based.
    if (node.input_size() <= idx) {
      continue;
    }
    const std::string& name = node.input(idx);
    if (name.empty()) {
      continue;  // Omitted optional input (empty-string placeholder).
    }
    auto it = init_map.find(name);
    if (it == init_map.end()) {
      return false;  // Non-constant -- declined, never guessed at.
    }
    const auto& dims = it->second->dims();
    const bool is_scalar =
        dims.size() == 0 || (dims.size() == 1 && dims.Get(0) == 1);
    if (!is_scalar) {
      return false;  // Not a scalar -- declined, never guessed at.
    }
  }
  return true;
}

// One PRelu pass-through hop match: `is_per_channel` tells the caller
// whether `slope_name` (present only when `is_per_channel`) needs its own
// axis-0 (Conv chain)/last-axis (MatMul chain) slice, or -- for a scalar/
// single shared parameter slope -- needs no slicing at all, the same
// "nothing of its own to touch" shape a plain unary activation hop already
// gets. Mirrors pruning.py's own `Optional[Tuple[bool, Optional[str]]]`
// return convention for _match_prelu_pass_through and its three siblings
// below.
struct PreluMatch {
  bool is_per_channel;
  std::optional<std::string> slope_name;
};

// The MatMul/Gemm-chain PRelu pass-through matcher used by WalkToConsumer,
// mirroring pruning.py's own _match_prelu_pass_through_matmul: since a
// MatMul/Gemm chain's own channel axis is the tensor's *last* axis (not
// axis 1, as for a Conv chain), `slope`'s per-channel shape here is the same
// flat, last-axis-is-channel vector every other MatMul/Gemm hop's own
// constant operand already is held to (prod(dims) == dims[-1]) -- e.g. a
// bare `[C]`, safe here in a way it is *not* for a Conv chain's own
// `[C, 1, 1]` convention (there is no trailing spatial axis for a rank-1
// `[C]` to spuriously align against instead). Returns
// `(is_per_channel, slope_name_or_none)`: scalar (`prod(dims) == 1`) is left
// completely untouched; per-channel (`dims[-1] == n_channels`) is folded
// into the caller's own chain_ops as an ordinary (node, slope_name) entry --
// no dedicated hop type needed here the way the Conv walk's axis-0 slice
// needs ConvPassThrough. Declines (nullopt) for a missing/non-constant/
// otherwise-malformed slope, the same conservative bar every other hop here
// holds its own constant operand to.
std::optional<PreluMatch> MatchPreluPassThroughMatmul(
    const onnx::NodeProto& node, const InitMap& init_map, int64_t n_channels) {
  if (node.op_type() != "PRelu" || node.domain() != "") {
    return std::nullopt;
  }
  if (node.input_size() != 2 || node.input(0).empty() ||
      node.input(1).empty()) {
    return std::nullopt;
  }
  if (node.output_size() != 1) {
    return std::nullopt;
  }
  const std::string& slope_name = node.input(1);
  auto it = init_map.find(slope_name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT) {
    return std::nullopt;
  }
  const onnx::TensorProto* s = it->second;
  if (s->dims_size() == 0) {
    return std::nullopt;
  }
  int64_t prod = 1;
  for (int64_t d : s->dims()) {
    prod *= d;
  }
  if (prod == 1) {
    return PreluMatch{false, std::nullopt};  // Scalar -- untouched.
  }
  if (prod == s->dims(s->dims_size() - 1) &&
      s->dims(s->dims_size() - 1) == n_channels) {
    return PreluMatch{true, slope_name};
  }
  return std::nullopt;
}

// The backward-walk (WalkMatmulProducerBackward) counterpart of
// MatchPreluPassThroughMatmul, mirroring pruning.py's own
// _match_prelu_pass_through_matmul_self: the backward residual walk doesn't
// know its group's real shared channel count yet at the point it first
// crosses a PRelu hop, so this checks `slope`'s own shape is
// self-consistent by calling that same matcher with `slope`'s own
// `dims[-1]` as the "expected" channel count -- trivially satisfying the
// per-channel case's own `dims[-1] == n_channels` check (never even
// consulted by the scalar case). FindMatmulResidualChains/
// ResolveMatmulResidualGroupForConcat already re-validate every chain_ops
// constant this walk returns against the group's real channel count once
// resolved, so no PRelu-specific re-validation is needed here.
std::optional<PreluMatch> MatchPreluPassThroughMatmulSelf(
    const onnx::NodeProto& node, const InitMap& init_map) {
  if (node.op_type() != "PRelu" || node.domain() != "" ||
      node.input_size() != 2) {
    return std::nullopt;
  }
  const std::string& slope_name = node.input(1);
  if (slope_name.empty()) {
    return std::nullopt;
  }
  auto it = init_map.find(slope_name);
  if (it == init_map.end()) {
    return std::nullopt;
  }
  const int64_t expected = it->second->dims_size() > 0
                               ? it->second->dims(it->second->dims_size() - 1)
                               : 1;
  return MatchPreluPassThroughMatmul(node, init_map, expected);
}

// True for an `Add` node the residual-chain finders below may treat as a
// merge point: exactly two distinct, non-constant operands. Mirrors
// pruning.py's own _is_eligible_add_merge exactly -- not Conv- or
// MatMul-specific, since it only inspects the node's own operands.
bool IsEligibleAddMerge(const onnx::NodeProto& node, const InitMap& init_map) {
  return node.op_type() == "Add" && node.input_size() == 2 &&
         node.output_size() == 1 && node.input(0) != node.input(1) &&
         !init_map.count(node.input(0)) && !init_map.count(node.input(1));
}

// --- Chain data model, mirroring pruning.py's own _Producer/_ConvPassThrough/
// _Chain dataclasses -------------------------------------------------------

struct Producer {
  onnx::NodeProto* node;
  std::string weight;
  bool weight_transposed;
  std::optional<std::string> bias;
  bool is_conv;
  int64_t group;
  // Activation nodes between this producer's raw output and the point it
  // combines with another producer (a gated pair only -- see
  // FindGatedChains; empty for a plain single-producer chain), in forward
  // order (raw output -> ... -> the tensor that feeds the combine op).
  std::vector<onnx::NodeProto*> pre_ops;
};

struct ConvPassThrough {
  onnx::NodeProto* node;
  std::string weight;
  std::optional<std::string> bias;
};

struct ChainOp {
  onnx::NodeProto* node;
  std::optional<std::string> const_name;
};

// One mid-chain `GroupNormalization` node WalkToConvConsumer crossed
// transparently -- the Conv/spatial-path analogue of ConvPassThrough's
// depthwise-Conv hop, for group-normalization statistics rather than a
// channel-mixing-free Conv. Unlike ConvPassThrough, this needs BOTH its own
// `scale` and `bias` sliced (both required by the op's own schema) -- via
// SliceLastAxis, not ConvPassThrough's own axis-0 SliceProducerWeight: a
// GroupNormalization `scale`/`bias` is only ever admitted here when
// FlatChannelConst's `prod(dims) == dims[-1]` bar holds (mirroring
// pruning.py's own `_flat_channel_const`), a looser bar than strictly
// rank-1 that a naive axis-0 slice would get wrong for (e.g. a `[1, 1, C]`
// shape), so this deliberately does NOT reuse ConvPassThrough the way a
// per-channel PRelu `slope` (always exactly `[C, 1, ..., 1]`) safely does --
// see MatchGroupNormPassThrough. Also unlike ConvPassThrough, this hop is
// its own dedicated (at-most-one-per-chain) `Chain::group_norm` field rather
// than living in a vector: its `num_groups` constrains ChainGroup()'s own
// per-block `keep` selection exactly like a general grouped Conv's own
// `group` does (see ChainGroup, MatchGroupNormPassThrough), a whole-chain
// property no other conv_pass_through hop carries. `num_groups` itself is
// never rewritten -- staying valid (the post-prune channel count still
// divides it evenly) without changing it is the entire point of the
// uniform-per-`num_groups`-block scope this hop is held to.
struct GroupNormPassThrough {
  onnx::NodeProto* node;
  std::string scale;
  std::string bias;
  int64_t num_groups;
};

// One extra, independent forward-consumer branch a residual/merge group's
// own fan-out resolves to -- mirroring pruning.py's own _ConsumerBranch --
// fed by the exact same shared `keep` set as a Chain's own primary
// consumer. See ResolveConvFanoutBranches/ResolveMatmulFanoutBranches.
struct ConsumerBranch {
  std::vector<ChainOp> chain_ops;
  onnx::NodeProto* consumer_node = nullptr;
  std::string consumer_weight;
  bool consumer_weight_transposed = false;
  bool consumer_is_conv = false;
  std::vector<ConvPassThrough> conv_pass_through;
  int64_t consumer_group = 1;
};

struct Chain {
  std::vector<Producer> producers;
  std::vector<ChainOp> chain_ops;
  onnx::NodeProto* consumer_node;
  std::string consumer_weight;
  bool consumer_weight_transposed;
  int64_t n_channels;
  bool consumer_is_conv = false;
  std::vector<ConvPassThrough> conv_pass_through;
  int64_t consumer_group = 1;
  // Additional independent consumer branches a residual/merge group's own
  // fan-out resolved -- see pruning.py's own _Chain.extra_consumers. Always
  // empty for every chain kind except a Conv/MatMul residual/merge group.
  std::vector<ConsumerBranch> extra_consumers;
  // A single mid-chain `GroupNormalization` hop the chain walk crossed
  // transparently -- FindConvChains only, for now (see WalkToConvConsumer's
  // own `recognize_group_norm` parameter; always nullopt for every other
  // chain kind -- residual/merge chains, Concat-merged chains, and every
  // MatMul/Gemm chain -- mirroring pruning.py's own `_Chain.group_norm`).
  std::optional<GroupNormPassThrough> group_norm;
};

// --- MatMul/Gemm plain chains, mirroring _match_producer/_walk_to_consumer/
// _find_chains --------------------------------------------------------------

struct ProducerMatch {
  std::string weight;
  bool weight_transposed;
  std::optional<std::string> bias;
  int64_t n_channels;
};

std::optional<ProducerMatch> MatchProducer(const onnx::NodeProto& node,
                                           const InitMap& init_map) {
  auto m = MatchMatMulLikeRaw(node);
  if (!m) {
    return std::nullopt;
  }
  auto it = init_map.find(m->w_name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      it->second->dims_size() != 2) {
    return std::nullopt;
  }
  std::optional<std::string> bias;
  if (node.op_type() == "Gemm" && node.input_size() == 3) {
    bias = node.input(2);
    if (!init_map.count(*bias)) {
      return std::nullopt;  // non-constant bias -- can't safely prune it.
    }
  }
  const onnx::TensorProto* w = it->second;
  const int64_t n_channels = m->weight_transposed ? w->dims(0) : w->dims(1);
  return ProducerMatch{m->w_name, m->weight_transposed, bias, n_channels};
}

struct ConsumerMatch {
  onnx::NodeProto* node;
  std::string weight;
  bool weight_transposed;
};

std::pair<std::optional<ConsumerMatch>, std::vector<ChainOp>> WalkToConsumer(
    const std::string& start, const InitMap& init_map,
    const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int64_t n_channels,
    int max_hops, onnx::NodeProto* forced_first_hop = nullptr) {
  std::vector<ChainOp> chain_ops;
  std::optional<ConsumerMatch> consumer;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    onnx::NodeProto* nxt;
    if (hop == 0 && forced_first_hop != nullptr) {
      // Used only by ResolveMatmulFanoutBranches: `cur` is an
      // already-established residual/merge group's own shared spine tensor,
      // so more than one consumer is expected here -- the caller has
      // already picked this one specific consumer to resolve this branch
      // through. Every hop after the first still enforces the ordinary
      // single-consumer bar unchanged below.
      nxt = forced_first_hop;
    } else {
      auto cit = consumers_of.find(cur);
      if (cit == consumers_of.end() || cit->second.size() != 1) {
        break;
      }
      nxt = cit->second[0];
    }

    auto cm = MatchMatMulLikeRaw(*nxt);
    if (cm && cm->x_name == cur) {
      auto wit = init_map.find(cm->w_name);
      if (wit != init_map.end() &&
          wit->second->data_type() == onnx::TensorProto::FLOAT &&
          wit->second->dims_size() == 2) {
        const int64_t k =
            cm->weight_transposed ? wit->second->dims(1) : wit->second->dims(0);
        if (k == n_channels) {
          consumer = ConsumerMatch{nxt, cm->w_name, cm->weight_transposed};
        }
      }
      break;
    }

    const bool is_unary = UnaryPassThroughOps().count(nxt->op_type()) != 0 &&
                          nxt->input_size() == 1 && nxt->input(0) == cur &&
                          nxt->output_size() == 1;
    std::optional<std::string> const_name;
    if (is_unary) {
      // No constant operand.
    } else if ((nxt->op_type() == "Add" || nxt->op_type() == "Mul") &&
               nxt->input_size() == 2 && nxt->output_size() == 1 &&
               (nxt->input(0) == cur || nxt->input(1) == cur)) {
      const std::string& other =
          (nxt->input(0) == cur) ? nxt->input(1) : nxt->input(0);
      auto oit = init_map.find(other);
      bool valid = false;
      if (oit != init_map.end()) {
        const onnx::TensorProto* c = oit->second;
        int64_t prod = 1;
        for (int64_t d : c->dims()) {
          prod *= d;
        }
        valid = c->data_type() == onnx::TensorProto::FLOAT &&
                c->dims_size() > 0 &&
                c->dims(c->dims_size() - 1) == n_channels && prod == n_channels;
      }
      if (!valid) {
        break;
      }
      const_name = other;
    } else if (nxt->op_type() == "BiasGelu" || nxt->op_type() == "FastGelu") {
      auto fused = MatchFusedBiasGelu(*nxt, init_map);
      if (!fused || fused->data_name != cur) {
        break;
      }
      if (fused->bias_name) {
        const onnx::TensorProto* b = init_map.at(*fused->bias_name);
        if (b->dims(b->dims_size() - 1) != n_channels) {
          break;
        }
      }
      const_name = fused->bias_name;
    } else if (nxt->op_type() == "PRelu" && nxt->domain() == "" &&
               nxt->input_size() > 0 && nxt->input(0) == cur) {
      auto prelu_match =
          MatchPreluPassThroughMatmul(*nxt, init_map, n_channels);
      if (!prelu_match) {
        break;
      }
      const_name =
          prelu_match->is_per_channel ? prelu_match->slope_name : std::nullopt;
    } else if (nxt->op_type() == "Clip" && nxt->domain() == "" &&
               nxt->input_size() > 0 && nxt->input(0) == cur &&
               nxt->output_size() == 1 &&
               MatchClipChannelPassThrough(*nxt, init_map)) {
      // Channel-agnostic -- no const of its own to slice.
    } else {
      break;
    }

    const std::string& out2 = nxt->output(0);
    if (ConsumerCount(consumers_of, out2) != 1 || graph_outputs.count(out2)) {
      break;
    }
    chain_ops.push_back(ChainOp{nxt, const_name});
    cur = out2;
  }
  return {consumer, chain_ops};
}

std::vector<Chain> FindChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };

  std::vector<Chain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = MatchProducer(*node, init_map);
    if (!info) {
      continue;
    }
    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    auto [consumer, chain_ops] =
        WalkToConsumer(out_name, init_map, consumers_of, graph_outputs,
                       info->n_channels, kMaxChainHops);
    if (!consumer) {
      continue;
    }
    Chain chain;
    chain.producers.push_back(Producer{
        node, info->weight, info->weight_transposed, info->bias, false, 1});
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = consumer->weight_transposed;
    chain.n_channels = info->n_channels;
    chains.push_back(std::move(chain));
  }
  return chains;
}

// --- Gated FFN (SwiGLU/GeGLU) chains, mirroring _trace_gate_producer_backward/
// _find_gated_chains ---------------------------------------------------------

struct FullProducerMatch {
  onnx::NodeProto* node;
  std::string weight;
  bool weight_transposed;
  std::optional<std::string> bias;
  int64_t n_channels;
};

// Walks backward from `tensor_name` through unary activation ops until it
// resolves to a matmul-like producer's raw output -- the mirror image of
// WalkToConsumer's forward walk, recognizing a gate branch's own activation
// (e.g. SwiGLU's silu(gate) exported as a separate Sigmoid/Mul-by-a-second-
// operand). Returns the resolved producer plus its pre_ops, in forward
// order (closest to the producer first).
std::optional<std::pair<FullProducerMatch, std::vector<onnx::NodeProto*>>>
TraceGateProducerBackward(
    const std::string& tensor_name,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const std::unordered_map<std::string, FullProducerMatch>& producer_infos,
    const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int max_hops) {
  std::vector<onnx::NodeProto*> pre_ops;  // Backward order; reversed on return.
  std::string cur = tensor_name;
  for (int hop = 0; hop < max_hops; ++hop) {
    if (ConsumerCount(consumers_of, cur) != 1 || graph_outputs.count(cur)) {
      return std::nullopt;
    }
    auto pit = producer_infos.find(cur);
    if (pit != producer_infos.end()) {
      std::reverse(pre_ops.begin(), pre_ops.end());
      return std::make_pair(pit->second, std::move(pre_ops));
    }
    auto nit = node_by_output.find(cur);
    if (nit == node_by_output.end()) {
      return std::nullopt;
    }
    onnx::NodeProto* producer_node = nit->second;
    if (!(UnaryPassThroughOps().count(producer_node->op_type()) != 0 &&
          producer_node->input_size() == 1 &&
          producer_node->output_size() == 1)) {
      return std::nullopt;
    }
    pre_ops.push_back(producer_node);
    cur = producer_node->input(0);
  }
  return std::nullopt;
}

std::vector<Chain> FindGatedChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };

  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  std::unordered_map<std::string, FullProducerMatch> producer_infos;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
    auto info = MatchProducer(*node, init_map);
    if (info) {
      producer_infos[node->output(0)] =
          FullProducerMatch{node, info->weight, info->weight_transposed,
                            info->bias, info->n_channels};
    }
  }

  std::vector<Chain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    std::optional<FullProducerMatch> info_a, info_b;
    std::vector<onnx::NodeProto*> pre_a, pre_b;

    if (node->op_type() == "Mul" && node->input_size() == 2 &&
        node->output_size() == 1) {
      const std::string& a_name = node->input(0);
      const std::string& b_name = node->input(1);
      if (a_name == b_name || init_map.count(a_name) ||
          init_map.count(b_name)) {
        continue;
      }
      auto trace_a =
          TraceGateProducerBackward(a_name, node_by_output, producer_infos,
                                    consumers_of, graph_outputs, kMaxChainHops);
      auto trace_b =
          TraceGateProducerBackward(b_name, node_by_output, producer_infos,
                                    consumers_of, graph_outputs, kMaxChainHops);
      if (!trace_a || !trace_b) {
        continue;
      }
      info_a = trace_a->first;
      pre_a = std::move(trace_a->second);
      info_b = trace_b->first;
      pre_b = std::move(trace_b->second);
    } else if (node->op_type() == "SwiGLU" && node->input_size() == 2 &&
               node->output_size() == 1) {
      const std::string& a_name = node->input(0);
      const std::string& b_name = node->input(1);
      if (init_map.count(a_name) || init_map.count(b_name)) {
        continue;
      }
      if (!(is_internal(a_name) && is_internal(b_name))) {
        continue;
      }
      auto ait = producer_infos.find(a_name);
      auto bit = producer_infos.find(b_name);
      if (ait == producer_infos.end() || bit == producer_infos.end()) {
        continue;
      }
      info_a = ait->second;
      info_b = bit->second;
    } else {
      continue;
    }

    if (info_a->node == info_b->node ||
        info_a->n_channels != info_b->n_channels) {
      continue;
    }

    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    auto [consumer, chain_ops] =
        WalkToConsumer(out_name, init_map, consumers_of, graph_outputs,
                       info_a->n_channels, kMaxChainHops);
    if (!consumer) {
      continue;
    }

    Chain chain;
    chain.producers.push_back(Producer{info_a->node, info_a->weight,
                                       info_a->weight_transposed, info_a->bias,
                                       false, 1, std::move(pre_a)});
    chain.producers.push_back(Producer{info_b->node, info_b->weight,
                                       info_b->weight_transposed, info_b->bias,
                                       false, 1, std::move(pre_b)});
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = consumer->weight_transposed;
    chain.n_channels = info_a->n_channels;
    chains.push_back(std::move(chain));
  }
  return chains;
}

// --- Conv plain chains, mirroring _match_conv_producer/_match_conv_consumer/
// _match_depthwise_conv_pass_through/_walk_to_conv_consumer/_find_conv_chains

struct ConvProducerMatch {
  std::string weight;
  std::optional<std::string> bias;
  int64_t out_channels;
  int64_t group;
};

std::optional<ConvProducerMatch> MatchConvProducer(const onnx::NodeProto& node,
                                                   const InitMap& init_map) {
  if (node.op_type() != "Conv" || node.input_size() < 2) {
    return std::nullopt;
  }
  auto it = init_map.find(node.input(1));
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      it->second->dims_size() != 4) {
    return std::nullopt;
  }
  const onnx::TensorProto* w = it->second;
  const int64_t group = ConvGroupAttr(node);
  if (group < 1) {
    return std::nullopt;
  }
  const int64_t out_channels = w->dims(0);
  const int64_t in_channels = w->dims(1) * group;
  if (group > 1 && (group >= in_channels || group == out_channels ||
                    out_channels % group != 0)) {
    return std::nullopt;
  }
  std::optional<std::string> bias;
  if (node.input_size() == 3 && !node.input(2).empty()) {
    bias = node.input(2);
    if (!init_map.count(*bias)) {
      return std::nullopt;
    }
  }
  return ConvProducerMatch{node.input(1), bias, out_channels, group};
}

struct ConvConsumerMatch {
  std::string weight;
  int64_t in_channels;
  int64_t group;
};

std::optional<ConvConsumerMatch> MatchConvConsumer(const onnx::NodeProto& node,
                                                   const InitMap& init_map) {
  if (node.op_type() != "Conv" || node.input_size() < 2) {
    return std::nullopt;
  }
  auto it = init_map.find(node.input(1));
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      it->second->dims_size() != 4) {
    return std::nullopt;
  }
  const onnx::TensorProto* w = it->second;
  const int64_t group = ConvGroupAttr(node);
  if (group < 1) {
    return std::nullopt;
  }
  const int64_t out_channels = w->dims(0);
  const int64_t in_channels = w->dims(1) * group;
  if (group > 1 && (group >= in_channels || group == out_channels ||
                    out_channels % group != 0)) {
    return std::nullopt;
  }
  return ConvConsumerMatch{node.input(1), in_channels, group};
}

struct DepthwiseMatch {
  std::string weight;
  std::optional<std::string> bias;
};

std::optional<DepthwiseMatch> MatchDepthwiseConvPassThrough(
    const onnx::NodeProto& node, const InitMap& init_map, int64_t n_channels) {
  if (node.op_type() != "Conv" || node.input_size() < 2) {
    return std::nullopt;
  }
  auto it = init_map.find(node.input(1));
  if (it == init_map.end()) {
    return std::nullopt;
  }
  const onnx::TensorProto* w = it->second;
  if (w->data_type() != onnx::TensorProto::FLOAT || w->dims_size() != 4 ||
      w->dims(0) != n_channels || w->dims(1) != 1 ||
      ConvGroupAttr(node) != n_channels) {
    return std::nullopt;
  }
  std::optional<std::string> bias;
  if (node.input_size() == 3 && !node.input(2).empty()) {
    bias = node.input(2);
    auto bit = init_map.find(*bias);
    if (bit == init_map.end() ||
        bit->second->data_type() != onnx::TensorProto::FLOAT) {
      return std::nullopt;
    }
  }
  return DepthwiseMatch{node.input(1), bias};
}

// True if `name` names a constant FLOAT initializer shaped like a flat
// per-channel vector (`prod(dims) == dims[-1]`) -- mirrors pruning.py's own
// `_flat_channel_const` exactly: the self-consistency bar every per-channel
// affine/bias/scale hop in this module checks before ever accepting a
// tensor as a slice target. The real `dims[-1] == n_channels` check, once
// the chain's real channel count is known, is left to the caller
// (MatchGroupNormPassThrough).
bool FlatChannelConst(const std::string& name, const InitMap& init_map) {
  auto it = init_map.find(name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT) {
    return false;
  }
  const auto& dims = it->second->dims();
  if (dims.size() == 0) {
    return false;
  }
  int64_t prod = 1;
  for (int64_t d : dims) {
    prod *= d;
  }
  return prod == dims.Get(dims.size() - 1);
}

struct GroupNormMatch {
  std::string scale;
  std::string bias;
  int64_t num_groups;
};

// The Conv-chain GroupNormalization pass-through matcher used by
// WalkToConvConsumer, mirroring pruning.py's own
// _match_group_norm_pass_through: if `node` is a plain (default-domain)
// `GroupNormalization` node whose own `num_groups` attribute evenly divides
// `n_channels`, with constant, per-channel-shaped (FlatChannelConst,
// `dims[-1] == n_channels` -- this alone already excludes the deprecated
// opset-18 per-*group*-shaped schema whenever `num_groups < n_channels`)
// `scale` (input 1) and `bias` (input 2) -- both required by the op's own
// schema -- returns `{scale_name, bias_name, num_groups}`. Declines
// (nullopt) on a missing/non-constant/wrongly-shaped `scale`/`bias`,
// `num_groups < 1`, `n_channels % num_groups != 0`, or `scale`/`bias`
// naming the same tensor (double-slicing it in ApplyChains's own per-hop
// loop would corrupt it) -- none of these is guessed at. The real "does
// this hop's own `num_groups` agree with a same-chain grouped Conv
// producer's/consumer's own `group`" check is left to the caller
// (FindConvChains), which has visibility into both.
std::optional<GroupNormMatch> MatchGroupNormPassThrough(
    const onnx::NodeProto& node, const InitMap& init_map, int64_t n_channels) {
  if (node.op_type() != "GroupNormalization" || node.domain() != "") {
    return std::nullopt;
  }
  if (node.input_size() != 3 || node.input(1).empty() ||
      node.input(2).empty()) {
    return std::nullopt;
  }
  int64_t num_groups = 0;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "num_groups") {
      num_groups = attr.i();
    }
  }
  if (num_groups < 1 || n_channels % num_groups != 0) {
    return std::nullopt;
  }
  const std::string& scale_name = node.input(1);
  const std::string& bias_name = node.input(2);
  if (scale_name == bias_name) {
    return std::nullopt;  // Tied scale/bias -- double-slicing would corrupt it.
  }
  if (!FlatChannelConst(scale_name, init_map) ||
      !FlatChannelConst(bias_name, init_map)) {
    return std::nullopt;
  }
  const auto& sdims = init_map.at(scale_name)->dims();
  const auto& bdims = init_map.at(bias_name)->dims();
  if (sdims.Get(sdims.size() - 1) != n_channels ||
      bdims.Get(bdims.size() - 1) != n_channels) {
    return std::nullopt;
  }
  return GroupNormMatch{scale_name, bias_name, num_groups};
}

// True iff `node` (already confirmed by the caller to be a plain
// (default-domain) `Resize`, `node.input(0)` the tensor being walked
// through) provably leaves axis 1 -- the NCHW channel axis this module's
// Conv-chain machinery assumes throughout -- unresized, so it is safe to
// cross transparently. Declines (false) rather than guesses whenever it
// cannot statically prove that -- mirrors pruning.py's own
// _match_resize_channel_pass_through exactly (see that function's own
// docstring for the full reasoning, including the empirically-confirmed
// commutativity argument and why a `sizes`-driven Resize is declined
// outright rather than guessed at):
//
// - Needs a length-3-or-4 `node.input` (`X, roi, scales[, sizes]`, the
//   opset 11+ signature) so `scales` lands at a known, fixed position.
// - Only a `scales` (tensor(float))-driven Resize is ever recognized -- a
//   `sizes`-driven one is declined outright, always, never guessed at.
// - `scales` must be a constant FLOAT initializer -- a runtime-computed
//   value means this pass cannot know which axis is affected.
// - The (opset 18+) `axes` attribute, when present, restricts which input
//   axes `scales` actually describes. A negative `axes` entry declines the
//   whole hop (can't resolve without a known rank); axis 1 simply not being
//   named in `axes` at all means it is by definition not resized.
bool MatchResizeChannelPassThrough(const onnx::NodeProto& node,
                                   const InitMap& init_map) {
  if (node.op_type() != "Resize" || node.domain() != "") {
    return false;
  }
  if ((node.input_size() != 3 && node.input_size() != 4) ||
      node.input(0).empty()) {
    return false;
  }
  const std::string& scales_name = node.input(2);
  if (scales_name.empty()) {
    return false;  // Only a `scales`-driven Resize is ever recognized.
  }
  auto it = init_map.find(scales_name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT) {
    return false;
  }
  std::vector<float> values = ReadFloatTensor(*it->second);

  std::optional<std::vector<int64_t>> axes_attr;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "axes") {
      axes_attr = std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
    }
  }

  float channel_value;
  if (!axes_attr) {
    if (values.size() < 2) {
      return false;  // No axis-1 slot in a `scales` this short.
    }
    channel_value = values[1];
  } else {
    for (int64_t a : *axes_attr) {
      if (a < 0) {
        return false;  // Can't resolve a negative axis without a known rank.
      }
    }
    if (values.size() != axes_attr->size()) {
      return false;  // Malformed -- schema requires equal length.
    }
    auto pos = std::find(axes_attr->begin(), axes_attr->end(), int64_t{1});
    if (pos == axes_attr->end()) {
      return true;  // Axis 1 isn't named -- definitely not resized.
    }
    channel_value = values[static_cast<size_t>(pos - axes_attr->begin())];
  }
  return channel_value == 1.0f;
}

// True iff `node` (already confirmed by the caller to be a plain
// (default-domain) `Pad`, `node.input(0)` the tensor being walked through)
// provably pads *nothing* on axis 1 -- the NCHW channel axis -- so it is
// safe to cross transparently, whatever its `mode`. Declines (false) rather
// than guesses whenever it cannot statically prove that -- mirrors
// pruning.py's own _match_pad_channel_pass_through exactly (see that
// function's own docstring for the full reasoning):
//
// - Needs a `pads` *input* at `node.input(1)` (opset 11+'s signature) that
//   is a constant INT64 initializer -- opset < 11's older attribute-based
//   `pads` is declined outright, mirroring MatchResizeChannelPassThrough's
//   own "statically-known-constant-only" bar.
// - The (opset 18+) `axes` *input* (`node.input(3)`), when present, must
//   likewise be a constant INT64 initializer, and restricts which axes
//   `pads` describes (`len(pads) == 2 * len(axes)`). A negative `axes`
//   entry declines the whole hop; axis 1 not named in `axes` at all means
//   it is by definition not padded.
// - When `axes` is omitted, `pads` spans every input axis in order
//   (`len(pads) == 2 * rank`), so `rank = len(pads) / 2` is recovered
//   directly from `pads`'s own length, and axis 1's begin/end pads sit at
//   `pads[1]`/`pads[rank + 1]`.
bool MatchPadChannelPassThrough(const onnx::NodeProto& node,
                                const InitMap& init_map) {
  if (node.op_type() != "Pad" || node.domain() != "") {
    return false;
  }
  if (node.input_size() < 2 || node.input(0).empty() || node.input(1).empty()) {
    return false;
  }
  auto pit = init_map.find(node.input(1));
  if (pit == init_map.end() ||
      pit->second->data_type() != onnx::TensorProto::INT64) {
    return false;
  }
  std::vector<int64_t> pads = ReadInt64Tensor(*pit->second);
  std::string axes_name = node.input_size() > 3 ? node.input(3) : "";

  int64_t begin, end;
  if (!axes_name.empty()) {
    auto ait = init_map.find(axes_name);
    if (ait == init_map.end() ||
        ait->second->data_type() != onnx::TensorProto::INT64) {
      return false;
    }
    std::vector<int64_t> axes = ReadInt64Tensor(*ait->second);
    for (int64_t a : axes) {
      if (a < 0) {
        return false;  // Can't resolve a negative axis without a known rank.
      }
    }
    if (pads.size() != 2 * axes.size()) {
      return false;  // Malformed -- schema requires equal length.
    }
    auto pos = std::find(axes.begin(), axes.end(), int64_t{1});
    if (pos == axes.end()) {
      return true;  // Axis 1 isn't named -- definitely not padded.
    }
    const size_t idx = static_cast<size_t>(pos - axes.begin());
    begin = pads[idx];
    end = pads[axes.size() + idx];
  } else {
    if (pads.size() % 2 != 0) {
      return false;  // Malformed -- schema requires an even length.
    }
    const int64_t rank = static_cast<int64_t>(pads.size()) / 2;
    if (rank < 2) {
      return false;  // No axis-1 slot in a `pads` this short.
    }
    begin = pads[1];
    end = pads[static_cast<size_t>(rank) + 1];
  }
  return begin == 0 && end == 0;
}

// The Conv-chain PRelu pass-through matcher used by WalkToConvConsumer,
// mirroring pruning.py's own _match_prelu_pass_through: if `node` is a plain
// (default-domain) PRelu whose own `slope` (input 1) is a constant float
// initializer cleanly falling into one of two shapes real exporters
// produce, returns `(is_per_channel, slope_name_or_none)`:
//
// - scalar/single shared parameter (every dimension size 1 -- e.g. a bare
//   scalar, `[1]`, or the `[1, 1, 1]` a real `torch.onnx.export` of
//   `nn.PReLU(1)` emits) -- `(false, nullopt)`: the same value multiplies
//   every channel, so pruning some away changes nothing about it -- left
//   completely untouched, the same "no operand of its own to slice" shape a
//   unary activation hop already gets.
// - per-channel (`dims[0] == n_channels`, every other dimension size 1 --
//   e.g. the `[C, 1, 1]` a real `torch.onnx.export` of `nn.PReLU(C)` emits
//   for a 2-D Conv) -- `(true, slope_name)`: one independent value per
//   channel, co-sliced by the chain's own `keep` index set exactly like a
//   depthwise Conv hop's own weight already is -- this reuses ConvPassThrough
//   for exactly that reason (`slope`'s own `[C, 1, ..., 1]` layout needs the
//   identical axis-0, any-trailing-rank slice a depthwise Conv's own weight
//   already gets, and PRelu has no `group` attribute for the caller's own
//   conv-groupedness dispatch to (correctly) leave untouched -- see
//   ApplyChains/ApplyConcatChains's own `op_type() == "Conv"` guard around
//   `SetOrAddIntAttr(..., "group", ...)`).
//
// A bare rank-1 `[C]` slope is deliberately *not* treated as per-channel
// here, unlike a MatMul/Gemm chain's own last-axis-is-channel convention
// (MatchPreluPassThroughMatmul above) -- this module's Conv-chain machinery
// assumes NCHW's axis-1-is-channel convention throughout, and ONNX's
// unidirectional broadcasting aligns a slope's own dimensions against `X`'s
// *trailing* ones: a `[C]` slope padded against a rank-4 NCHW tensor lands
// on axis 3 (W), not axis 1, unless C happens to equal W by coincidence.
// Requiring at least one trailing size-1 dimension (`dims_size() >= 2`) is
// what rules a bare `[C]` out here. Declines (nullopt) whenever `node` isn't
// a plain PRelu, `slope` is missing/non-constant, or its shape doesn't
// cleanly fall into either shape above -- never guessed at.
std::optional<PreluMatch> MatchPreluPassThrough(const onnx::NodeProto& node,
                                                const InitMap& init_map,
                                                int64_t n_channels) {
  if (node.op_type() != "PRelu" || node.domain() != "") {
    return std::nullopt;
  }
  if (node.input_size() != 2 || node.input(0).empty() ||
      node.input(1).empty()) {
    return std::nullopt;
  }
  if (node.output_size() != 1) {
    return std::nullopt;
  }
  const std::string& slope_name = node.input(1);
  auto it = init_map.find(slope_name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT) {
    return std::nullopt;
  }
  const onnx::TensorProto* s = it->second;
  if (s->dims_size() == 0) {
    return std::nullopt;
  }
  int64_t prod = 1;
  for (int64_t d : s->dims()) {
    prod *= d;
  }
  if (prod == 1) {
    return PreluMatch{false, std::nullopt};  // Scalar -- untouched.
  }
  if (s->dims_size() >= 2 && s->dims(0) == n_channels) {
    bool trailing_ones = true;
    for (int i = 1; i < s->dims_size(); ++i) {
      if (s->dims(i) != 1) {
        trailing_ones = false;
        break;
      }
    }
    if (trailing_ones) {
      return PreluMatch{true, slope_name};
    }
  }
  return std::nullopt;
}

struct ConvConsumerResult {
  onnx::NodeProto* node;
  std::string weight;
  int64_t group;
};

std::tuple<std::optional<ConvConsumerResult>, std::vector<ChainOp>,
           std::vector<ConvPassThrough>, std::optional<GroupNormPassThrough>>
WalkToConvConsumer(const std::string& start, const InitMap& init_map,
                   const ConsumerMap& consumers_of,
                   const std::unordered_set<std::string>& graph_outputs,
                   int64_t n_channels, int max_hops,
                   onnx::NodeProto* forced_first_hop = nullptr,
                   bool recognize_group_norm = false) {
  std::vector<ChainOp> chain_ops;
  std::vector<ConvPassThrough> pass_through;
  std::optional<ConvConsumerResult> consumer;
  // At most one mid-chain GroupNormalization hop per chain -- mirrors
  // pruning.py's own `group_norm is None` gate on _walk_to_conv_consumer's
  // own matching `if`. Only ever recognized when `recognize_group_norm`
  // (FindConvChains only, today -- see Chain::group_norm's own comment).
  std::optional<GroupNormPassThrough> group_norm;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    onnx::NodeProto* nxt;
    if (hop == 0 && forced_first_hop != nullptr) {
      // See WalkToConsumer's own matching parameter -- used only by
      // ResolveConvFanoutBranches.
      nxt = forced_first_hop;
    } else {
      auto cit = consumers_of.find(cur);
      if (cit == consumers_of.end() || cit->second.size() != 1) {
        break;
      }
      nxt = cit->second[0];
    }

    if (nxt->op_type() == "Conv" && nxt->input(0) == cur) {
      auto dw = MatchDepthwiseConvPassThrough(*nxt, init_map, n_channels);
      if (dw) {
        const std::string& out2 = nxt->output(0);
        if (ConsumerCount(consumers_of, out2) != 1 ||
            graph_outputs.count(out2)) {
          break;
        }
        pass_through.push_back(ConvPassThrough{nxt, dw->weight, dw->bias});
        cur = out2;
        continue;
      }
      auto match = MatchConvConsumer(*nxt, init_map);
      if (match && match->in_channels == n_channels) {
        consumer = ConvConsumerResult{nxt, match->weight, match->group};
      }
      break;
    }

    if (recognize_group_norm && !group_norm &&
        nxt->op_type() == "GroupNormalization" && nxt->domain() == "" &&
        nxt->input_size() > 0 && nxt->input(0) == cur) {
      auto gn_match = MatchGroupNormPassThrough(*nxt, init_map, n_channels);
      if (!gn_match) {
        break;
      }
      const std::string& out2 = nxt->output(0);
      if (ConsumerCount(consumers_of, out2) != 1 || graph_outputs.count(out2)) {
        break;
      }
      group_norm = GroupNormPassThrough{nxt, gn_match->scale, gn_match->bias,
                                        gn_match->num_groups};
      cur = out2;
      continue;
    }

    if (nxt->op_type() == "Resize" && nxt->domain() == "" &&
        nxt->input_size() > 0 && nxt->input(0) == cur &&
        nxt->output_size() == 1 &&
        MatchResizeChannelPassThrough(*nxt, init_map)) {
      const std::string& out2 = nxt->output(0);
      if (ConsumerCount(consumers_of, out2) != 1 || graph_outputs.count(out2)) {
        break;
      }
      chain_ops.push_back(ChainOp{nxt, std::nullopt});
      cur = out2;
      continue;
    }

    if (nxt->op_type() == "Pad" && nxt->domain() == "" &&
        nxt->input_size() > 0 && nxt->input(0) == cur &&
        nxt->output_size() == 1 && MatchPadChannelPassThrough(*nxt, init_map)) {
      const std::string& out2 = nxt->output(0);
      if (ConsumerCount(consumers_of, out2) != 1 || graph_outputs.count(out2)) {
        break;
      }
      chain_ops.push_back(ChainOp{nxt, std::nullopt});
      cur = out2;
      continue;
    }

    if (nxt->op_type() == "PRelu" && nxt->domain() == "" &&
        nxt->input_size() > 0 && nxt->input(0) == cur &&
        nxt->output_size() == 1) {
      auto prelu_match = MatchPreluPassThrough(*nxt, init_map, n_channels);
      if (!prelu_match) {
        break;
      }
      const std::string& out2 = nxt->output(0);
      if (ConsumerCount(consumers_of, out2) != 1 || graph_outputs.count(out2)) {
        break;
      }
      if (prelu_match->is_per_channel) {
        pass_through.push_back(
            ConvPassThrough{nxt, *prelu_match->slope_name, std::nullopt});
      } else {
        chain_ops.push_back(ChainOp{nxt, std::nullopt});
      }
      cur = out2;
      continue;
    }

    if (nxt->op_type() == "Clip" && nxt->domain() == "" &&
        nxt->input_size() > 0 && nxt->input(0) == cur &&
        nxt->output_size() == 1 &&
        MatchClipChannelPassThrough(*nxt, init_map)) {
      const std::string& out2 = nxt->output(0);
      if (ConsumerCount(consumers_of, out2) != 1 || graph_outputs.count(out2)) {
        break;
      }
      chain_ops.push_back(ChainOp{nxt, std::nullopt});
      cur = out2;
      continue;
    }

    if (!(UnaryPassThroughOps().count(nxt->op_type()) != 0 &&
          nxt->input_size() == 1 && nxt->input(0) == cur &&
          nxt->output_size() == 1)) {
      break;
    }
    const std::string& out2 = nxt->output(0);
    if (ConsumerCount(consumers_of, out2) != 1 || graph_outputs.count(out2)) {
      break;
    }
    chain_ops.push_back(ChainOp{nxt, std::nullopt});
    cur = out2;
  }
  return {consumer, chain_ops, pass_through, group_norm};
}

std::vector<Chain> FindConvChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };

  std::vector<Chain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = MatchConvProducer(*node, init_map);
    if (!info) {
      continue;
    }
    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    auto [consumer, chain_ops, pass_through, group_norm] = WalkToConvConsumer(
        out_name, init_map, consumers_of, graph_outputs, info->out_channels,
        kMaxChainHops, /*forced_first_hop=*/nullptr,
        /*recognize_group_norm=*/true);
    if (!consumer) {
      continue;
    }
    if (info->group > 1 && consumer->group > 1 &&
        info->group != consumer->group) {
      continue;  // Both sides grouped with mismatched group counts: declined.
    }
    if (group_norm &&
        ((info->group > 1 && info->group != group_norm->num_groups) ||
         (consumer->group > 1 && consumer->group != group_norm->num_groups))) {
      // The mid-chain GroupNorm hop's own `num_groups` disagrees with a
      // general grouped Conv producer's or consumer's own `group` -- the
      // two partitions' own block boundaries wouldn't generally align,
      // exactly the same "declined outright" bar the producer/consumer
      // group mismatch above already gets. Mirrors pruning.py's own
      // identical reconciliation check in _find_conv_chains.
      continue;
    }

    Chain chain;
    chain.producers.push_back(
        Producer{node, info->weight, false, info->bias, true, info->group});
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = false;
    chain.n_channels = info->out_channels;
    chain.consumer_is_conv = true;
    chain.conv_pass_through = std::move(pass_through);
    chain.consumer_group = consumer->group;
    chain.group_norm = std::move(group_norm);
    chains.push_back(std::move(chain));
  }
  return chains;
}

// --- Residual (Add-merged) chains, mirroring _is_eligible_add_merge/
// _walk_conv_producer_backward/_find_conv_residual_chains and
// _walk_matmul_producer_backward/_find_matmul_residual_chains. The MatMul
// side also recognizes a fused com.microsoft::SkipLayerNormalization/
// SkipSimplifiedLayerNormalization merge point (see MatchResidualMerge
// below), general grouped Conv producers/consumers on the Conv side, and
// forward "fan-out" to more than one independent ordinary consumer once a
// group's shared channel-index set is established (see
// ResolveConvFanoutBranches/ResolveMatmulFanoutBranches) -- mirroring
// pruning.py's own current _walk_conv_producer_backward/
// _walk_matmul_producer_backward exactly, not just a bare single-consumer
// Add(a, b) pair.

enum class BackwardEdgeKind { kFail, kProducer, kAdd, kGated };

// The backward counterpart of WalkToConvConsumer, used only by
// FindConvResidualChains to resolve one operand of an eligible Add merge
// point back to whatever produces it.
struct ConvBackwardEdge {
  BackwardEdgeKind kind = BackwardEdgeKind::kFail;
  Producer producer;
  int64_t n_channels = 0;
  onnx::NodeProto* add_node = nullptr;
  std::vector<ConvPassThrough> pass_through;  // Forward order.
  std::vector<onnx::NodeProto*> unary_ops;    // Forward order.
  // For every hop that actually advanced `cur`, the pair (new_cur, node)
  // recording that new_cur's own in-group forward consumer is `node` --
  // mirrors pruning.py's own `edges`, in the same (start-to-producer, not
  // reversed) order. Used by ResolveConvFanoutBranches to know which
  // consumer(s) of a backbone tensor are already part of the group's own
  // wiring, so only genuinely extra ones need their own resolution.
  std::vector<std::pair<std::string, onnx::NodeProto*>> edges;
};

// The backward-walk analogue of MatchDepthwiseConvPassThrough: unlike that
// matcher, which validates a hop against an externally supplied
// n_channels, this checks the node's own weight is self-consistently
// depthwise-shaped by calling it with the node's own dims(0) as the
// "expected" count. FindConvResidualChains re-validates every such hop
// against the group's real, established channel count once resolved.
std::optional<DepthwiseMatch> MatchConvPassThroughSelf(
    const onnx::NodeProto& node, const InitMap& init_map) {
  if (node.op_type() != "Conv" || node.input_size() < 2) {
    return std::nullopt;
  }
  auto it = init_map.find(node.input(1));
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      it->second->dims_size() != 4) {
    return std::nullopt;
  }
  return MatchDepthwiseConvPassThrough(node, init_map, it->second->dims(0));
}

// The backward-walk (WalkConvProducerBackward) counterpart of
// MatchPreluPassThrough, mirroring pruning.py's own
// _match_prelu_pass_through_self and MatchConvPassThroughSelf's own
// identical trick: the backward residual walk doesn't know its group's
// shared channel count yet at the point it first crosses a PRelu hop, so
// this checks the node's own `slope` is self-consistently shaped by calling
// that same matcher with `slope`'s own `dims(0)` as the "expected" channel
// count -- trivially satisfying the per-channel case's own
// `dims(0) == n_channels` check and leaving every other one (including the
// scalar case, which never even looks at n_channels) intact.
// FindConvResidualChains/ResolveConvResidualGroupForConcat re-validate every
// per-channel hop this returns against the group's real, established
// channel count once resolved (the same generic `pass_through`
// re-validation a depthwise hop already gets, keyed only on `hop.weight`'s
// own `dims(0)`, needing no PRelu-specific case of its own).
std::optional<PreluMatch> MatchPreluPassThroughSelf(const onnx::NodeProto& node,
                                                    const InitMap& init_map) {
  if (node.op_type() != "PRelu" || node.domain() != "" ||
      node.input_size() != 2) {
    return std::nullopt;
  }
  const std::string& slope_name = node.input(1);
  if (slope_name.empty()) {
    return std::nullopt;
  }
  auto it = init_map.find(slope_name);
  if (it == init_map.end()) {
    return std::nullopt;
  }
  const int64_t expected =
      it->second->dims_size() > 0 ? it->second->dims(0) : 1;
  return MatchPreluPassThrough(node, init_map, expected);
}

// Walks backward from tensor `start` through unary pass-through activations
// and self-consistently-depthwise Conv hops, declining (only) whenever a
// tensor crossed -- `start` itself included -- is a graph output. Unlike the
// version this superseded, *how many* other things also read that same
// tensor is deliberately not checked here (mirroring pruning.py's own
// current _walk_conv_producer_backward): every such extra reader gets its
// own safety check later, in ResolveConvFanoutBranches, once the group's
// real channel count is known. A general grouped Conv producer is also now
// allowed through unconditionally -- the caller (FindConvResidualChains)
// cross-checks group agreement across the whole group.
ConvBackwardEdge WalkConvProducerBackward(
    const std::string& start,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const InitMap& init_map,
    const std::unordered_set<std::string>& graph_outputs, int max_hops) {
  std::vector<ConvPassThrough> pass_through;  // Backward order.
  std::vector<onnx::NodeProto*> unary_ops;    // Backward order.
  std::vector<std::pair<std::string, onnx::NodeProto*>> edges;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    if (graph_outputs.count(cur)) {
      return ConvBackwardEdge{};
    }
    auto nit = node_by_output.find(cur);
    if (nit == node_by_output.end() || nit->second->output_size() != 1 ||
        nit->second->output(0) != cur) {
      return ConvBackwardEdge{};
    }
    onnx::NodeProto* node = nit->second;

    auto prod_info = MatchConvProducer(*node, init_map);
    if (prod_info) {
      ConvBackwardEdge edge;
      edge.kind = BackwardEdgeKind::kProducer;
      edge.producer = Producer{node, prod_info->weight, false, prod_info->bias,
                               true, prod_info->group};
      edge.n_channels = prod_info->out_channels;
      std::reverse(pass_through.begin(), pass_through.end());
      std::reverse(unary_ops.begin(), unary_ops.end());
      edge.pass_through = std::move(pass_through);
      edge.unary_ops = std::move(unary_ops);
      edge.edges = std::move(edges);
      return edge;
    }

    auto dw = MatchConvPassThroughSelf(*node, init_map);
    if (dw) {
      pass_through.push_back(ConvPassThrough{node, dw->weight, dw->bias});
      edges.push_back({node->input(0), node});
      cur = node->input(0);
      continue;
    }

    if (node->op_type() == "Resize" &&
        MatchResizeChannelPassThrough(*node, init_map)) {
      unary_ops.push_back(node);
      edges.push_back({node->input(0), node});
      cur = node->input(0);
      continue;
    }

    if (node->op_type() == "Pad" &&
        MatchPadChannelPassThrough(*node, init_map)) {
      unary_ops.push_back(node);
      edges.push_back({node->input(0), node});
      cur = node->input(0);
      continue;
    }

    auto prelu_self = MatchPreluPassThroughSelf(*node, init_map);
    if (prelu_self) {
      if (prelu_self->is_per_channel) {
        pass_through.push_back(
            ConvPassThrough{node, *prelu_self->slope_name, std::nullopt});
      } else {
        unary_ops.push_back(node);
      }
      edges.push_back({node->input(0), node});
      cur = node->input(0);
      continue;
    }

    if (node->op_type() == "Clip" &&
        MatchClipChannelPassThrough(*node, init_map)) {
      unary_ops.push_back(node);
      edges.push_back({node->input(0), node});
      cur = node->input(0);
      continue;
    }

    if (UnaryPassThroughOps().count(node->op_type()) != 0 &&
        node->input_size() == 1) {
      unary_ops.push_back(node);
      edges.push_back({node->input(0), node});
      cur = node->input(0);
      continue;
    }

    if (IsEligibleAddMerge(*node, init_map)) {
      ConvBackwardEdge edge;
      edge.kind = BackwardEdgeKind::kAdd;
      edge.add_node = node;
      std::reverse(pass_through.begin(), pass_through.end());
      std::reverse(unary_ops.begin(), unary_ops.end());
      edge.pass_through = std::move(pass_through);
      edge.unary_ops = std::move(unary_ops);
      edge.edges = std::move(edges);
      return edge;
    }

    return ConvBackwardEdge{};
  }
  return ConvBackwardEdge{};
}

// For an already-established Conv residual/merge group -- every tensor in
// `backbone_tensors` is one WalkConvProducerBackward's own backward walk
// already proved carries that group's shared channel-index set, `accounted`
// marks, per tensor, which specific consumer node(s) are already part of the
// group's own internal wiring -- finds every *extra* consumer (one not in
// `accounted`) of every backbone tensor and resolves each independently via
// WalkToConvConsumer, seeded at that one specific node (its own
// `forced_first_hop`). Mirrors pruning.py's own
// _resolve_conv_fanout_branches exactly, including its three-way return
// shape: `std::nullopt` -- decline the whole group -- if any backbone tensor
// is itself a graph output, any extra consumer fails to resolve, or two
// branches would end up naming the same consumer weight; otherwise every
// resolved branch (possibly empty, when the group has no extra fan-out at
// all -- the caller treats that exactly like "no consumer found" and
// declines, same as pruning.py's own `if not branches: continue`).
std::optional<std::vector<ConsumerBranch>> ResolveConvFanoutBranches(
    const std::vector<std::string>& backbone_tensors,
    const std::unordered_map<std::string, std::unordered_set<onnx::NodeProto*>>&
        accounted,
    const InitMap& init_map, const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int64_t n_channels) {
  std::vector<ConsumerBranch> branches;
  std::unordered_set<std::string> seen_weights;
  for (const auto& tensor : backbone_tensors) {
    if (graph_outputs.count(tensor)) {
      return std::nullopt;
    }
    auto cit = consumers_of.find(tensor);
    if (cit == consumers_of.end()) {
      continue;
    }
    auto acc_it = accounted.find(tensor);
    std::unordered_set<onnx::NodeProto*> seen_nodes;
    for (onnx::NodeProto* consumer_node : cit->second) {
      if (!seen_nodes.insert(consumer_node).second) {
        continue;
      }
      if (acc_it != accounted.end() && acc_it->second.count(consumer_node)) {
        continue;  // Already part of the group's own established wiring.
      }
      // `recognize_group_norm` stays at its default (false) here -- a
      // fan-out branch's own forward re-walk never recognizes a mid-chain
      // GroupNorm hop, mirroring pruning.py's own _resolve_conv_fanout_branches
      // (which never passes `recognize_group_norm=True` to its own
      // _walk_to_conv_consumer call either).
      auto [resolved, br_chain_ops, br_pass_through, br_group_norm] =
          WalkToConvConsumer(tensor, init_map, consumers_of, graph_outputs,
                             n_channels, kMaxChainHops, consumer_node);
      (void)br_group_norm;  // Always nullopt -- see comment above.
      if (!resolved) {
        return std::nullopt;
      }
      if (seen_weights.count(resolved->weight)) {
        return std::nullopt;  // Two branches naming the same consumer weight.
      }
      seen_weights.insert(resolved->weight);
      ConsumerBranch branch;
      branch.chain_ops = std::move(br_chain_ops);
      branch.consumer_node = resolved->node;
      branch.consumer_weight = resolved->weight;
      branch.consumer_weight_transposed = false;
      branch.consumer_is_conv = true;
      branch.conv_pass_through = std::move(br_pass_through);
      branch.consumer_group = resolved->group;
      branches.push_back(std::move(branch));
    }
  }
  return branches;
}

// Finds Conv residual/skip-connection groups: for every maximal union-find
// group of transitively-connected eligible Add merge points, resolves
// every member's two operands via WalkConvProducerBackward. See this
// section's own comment above and pruning.py's own
// _find_conv_residual_chains for the full algorithm description.
std::vector<Chain> FindConvResidualChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
  }

  std::vector<onnx::NodeProto*> eligible_adds;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    if (IsEligibleAddMerge(*node, init_map)) {
      eligible_adds.push_back(node);
    }
  }
  if (eligible_adds.empty()) {
    return {};
  }
  std::unordered_map<onnx::NodeProto*, int> add_index;
  for (size_t i = 0; i < eligible_adds.size(); ++i) {
    add_index[eligible_adds[i]] = static_cast<int>(i);
  }

  std::vector<int> parent(eligible_adds.size());
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int i) {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  auto unite = [&](int i, int j) {
    const int ri = find(i), rj = find(j);
    if (ri != rj) {
      parent[ri] = rj;
    }
  };

  std::vector<std::vector<ConvBackwardEdge>> edge_results(eligible_adds.size());
  std::unordered_set<int> poisoned;
  for (size_t idx = 0; idx < eligible_adds.size(); ++idx) {
    std::vector<ConvBackwardEdge> results;
    for (const auto& operand : eligible_adds[idx]->input()) {
      ConvBackwardEdge edge = WalkConvProducerBackward(
          operand, node_by_output, init_map, graph_outputs, kMaxChainHops);
      if (edge.kind == BackwardEdgeKind::kFail) {
        poisoned.insert(static_cast<int>(idx));
      } else if (edge.kind == BackwardEdgeKind::kAdd) {
        auto jit = add_index.find(edge.add_node);
        if (jit == add_index.end()) {
          poisoned.insert(
              static_cast<int>(idx));  // Defensive -- shouldn't happen.
        } else {
          unite(static_cast<int>(idx), jit->second);
        }
      }
      results.push_back(std::move(edge));
    }
    edge_results[idx] = std::move(results);
  }

  std::unordered_map<int, std::vector<int>> groups;
  for (size_t idx = 0; idx < eligible_adds.size(); ++idx) {
    groups[find(static_cast<int>(idx))].push_back(static_cast<int>(idx));
  }

  std::vector<Chain> chains;
  for (auto& kv : groups) {
    const std::vector<int>& members = kv.second;
    bool any_poisoned = false;
    for (int i : members) {
      if (poisoned.count(i)) {
        any_poisoned = true;
        break;
      }
    }
    if (any_poisoned) {
      continue;
    }

    std::vector<Producer> leaf_producers;
    std::unordered_set<int64_t> n_channels_set;
    std::vector<ConvPassThrough> pass_through;
    std::vector<onnx::NodeProto*> unary_ops;
    std::unordered_set<int> referenced;
    // Every tensor either walk of every member proved carries this group's
    // own shared channel-index set, and, for each, which specific consumer
    // node is already part of the group's own internal wiring -- fed to
    // ResolveConvFanoutBranches below so only genuinely extra consumers need
    // their own separate resolution. `backbone_tensors` preserves
    // first-seen order, so which resolved branch ends up "primary" is
    // deterministic.
    std::vector<std::string> backbone_tensors;
    std::unordered_map<std::string, std::unordered_set<onnx::NodeProto*>>
        accounted;
    auto mark_backbone = [&](const std::string& tensor, onnx::NodeProto* node) {
      if (!accounted.count(tensor)) {
        backbone_tensors.push_back(tensor);
      }
      accounted[tensor].insert(node);
    };

    for (int idx : members) {
      onnx::NodeProto* add_node = eligible_adds[static_cast<size_t>(idx)];
      const auto& results = edge_results[static_cast<size_t>(idx)];
      for (int oi = 0; oi < add_node->input_size(); ++oi) {
        const std::string& operand = add_node->input(oi);
        const ConvBackwardEdge& edge = results[static_cast<size_t>(oi)];
        mark_backbone(operand, add_node);
        for (const auto& e : edge.edges) {
          mark_backbone(e.first, e.second);
        }
        pass_through.insert(pass_through.end(), edge.pass_through.begin(),
                            edge.pass_through.end());
        unary_ops.insert(unary_ops.end(), edge.unary_ops.begin(),
                         edge.unary_ops.end());
        if (edge.kind == BackwardEdgeKind::kProducer) {
          leaf_producers.push_back(edge.producer);
          n_channels_set.insert(edge.n_channels);
        } else if (edge.kind == BackwardEdgeKind::kAdd) {
          referenced.insert(add_index[edge.add_node]);
        }
      }
    }

    if (n_channels_set.size() != 1) {
      continue;  // Branches disagree on channel count -- decline.
    }
    const int64_t n_channels = *n_channels_set.begin();

    // Every leaf producer's own `group` (1 for an ordinary Conv, > 1 for a
    // general grouped one) must agree with every other non-1 value in the
    // group -- mirrors _find_conv_chains's own "both sides grouped with a
    // different group count" decline.
    std::unordered_set<int64_t> producer_groups;
    for (const auto& p : leaf_producers) {
      if (p.group > 1) {
        producer_groups.insert(p.group);
      }
    }
    if (producer_groups.size() > 1) {
      continue;  // Producers disagree on group count -- decline.
    }

    bool dw_mismatch = false;
    for (const auto& hop : pass_through) {
      if (init_map.at(hop.weight)->dims(0) != n_channels) {
        dw_mismatch = true;
        break;
      }
    }
    if (dw_mismatch) {
      continue;
    }

    std::vector<int> sinks;
    for (int idx : members) {
      if (!referenced.count(idx)) {
        sinks.push_back(idx);
      }
    }
    if (sinks.size() != 1) {
      continue;  // Not a single linear chain of merges -- decline.
    }
    onnx::NodeProto* sink_add = eligible_adds[static_cast<size_t>(sinks[0])];

    std::unordered_set<std::string> seen_weights;
    bool degenerate = false;
    for (const auto& p : leaf_producers) {
      if (!seen_weights.insert(p.weight).second) {
        degenerate = true;
        break;
      }
    }
    if (degenerate) {
      continue;  // The same producer named twice.
    }

    // The sink's own output is never `visited` by any member's own backward
    // walk, so it needs adding explicitly, with no accounted-for consumer of
    // its own yet.
    const std::string& sink_out = sink_add->output(0);
    if (!accounted.count(sink_out)) {
      backbone_tensors.push_back(sink_out);
      accounted[sink_out];
    }

    auto branches_opt =
        ResolveConvFanoutBranches(backbone_tensors, accounted, init_map,
                                  consumers_of, graph_outputs, n_channels);
    if (!branches_opt || branches_opt->empty()) {
      continue;
    }
    std::vector<ConsumerBranch>& branches = *branches_opt;

    // Completes the group-count agreement check started above: every
    // branch's own consumer_group (primary and extra alike) must also agree
    // with `producer_groups`.
    std::unordered_set<int64_t> all_groups = producer_groups;
    for (const auto& b : branches) {
      if (b.consumer_group > 1) {
        all_groups.insert(b.consumer_group);
      }
    }
    if (all_groups.size() > 1) {
      continue;  // Producer(s) and/or branch(es) disagree on group count.
    }

    ConsumerBranch primary = std::move(branches.front());
    std::vector<ConsumerBranch> extra_branches(
        std::make_move_iterator(branches.begin() + 1),
        std::make_move_iterator(branches.end()));

    std::vector<ChainOp> chain_ops;
    for (auto* op : unary_ops) {
      chain_ops.push_back(ChainOp{op, std::nullopt});
    }
    for (int idx : members) {
      chain_ops.push_back(
          ChainOp{eligible_adds[static_cast<size_t>(idx)], std::nullopt});
    }
    for (auto& co : primary.chain_ops) {
      chain_ops.push_back(std::move(co));
    }

    Chain chain;
    chain.producers = std::move(leaf_producers);
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = primary.consumer_node;
    chain.consumer_weight = primary.consumer_weight;
    chain.consumer_weight_transposed = false;
    chain.n_channels = n_channels;
    chain.consumer_is_conv = true;
    chain.extra_consumers = std::move(extra_branches);
    pass_through.insert(pass_through.end(), primary.conv_pass_through.begin(),
                        primary.conv_pass_through.end());
    chain.conv_pass_through = std::move(pass_through);
    chain.consumer_group = primary.consumer_group;
    chains.push_back(std::move(chain));
  }
  return chains;
}

// --- MatMul/Gemm residual (Add-merged) chains, mirroring
// _walk_matmul_producer_backward/_find_matmul_residual_chains ---------------
//
// A com.microsoft::SkipLayerNormalization/SkipSimplifiedLayerNormalization
// node -- what onnxruntime's transformer optimizer fuses a bare residual Add
// plus the following LayerNorm into -- is an eligible merge point too,
// mirroring pruning.py's own _match_matmul_residual_merge: its first two
// inputs (input/skip) play exactly the role Add's two operands do, while its
// constant gamma/beta/bias inputs are a per-channel affine hop riding the
// same node, folded into the resolved chain's own chain_ops as extra
// (node, const_name) entries -- ApplyChains's existing per-hop constant
// slicing picks them up with no changes of its own.

bool IsSkipLayerNormOp(const onnx::NodeProto& node) {
  return node.domain() == kComMicrosoftDomain &&
         (node.op_type() == "SkipLayerNormalization" ||
          node.op_type() == "SkipSimplifiedLayerNormalization");
}

bool IsConstVec(const InitMap& init_map, const std::string& name) {
  auto it = init_map.find(name);
  if (it == init_map.end()) {
    return false;
  }
  const onnx::TensorProto* t = it->second;
  if (t->data_type() != onnx::TensorProto::FLOAT || t->dims_size() == 0) {
    return false;
  }
  int64_t prod = 1;
  for (int64_t d : t->dims()) {
    prod *= d;
  }
  return prod == t->dims(t->dims_size() - 1);
}

struct SkipLayerNormConsts {
  std::string gamma;
  std::optional<std::string> beta;
  std::optional<std::string> bias;
};

// If every constant input a SkipLayerNormalization/
// SkipSimplifiedLayerNormalization `node` needs sliced -- gamma (input 2,
// required), plus beta (input 3, SkipLayerNormalization only) and bias
// (input 4, or input 3 for the simplified/RMSNorm variant) -- is present
// exactly as the node's own input list says and, whenever present, a
// constant float per-channel vector, returns their names. Declines on a
// non-constant gamma, a present-but-non-constant beta/bias, or the same
// tensor named for two of gamma/beta/bias at once (double-slicing it in
// ApplyChains's own per-hop loop would corrupt it).
std::optional<SkipLayerNormConsts> SkipLayerNormConstNames(
    const onnx::NodeProto& node, const InitMap& init_map) {
  const bool simplified = node.op_type() == "SkipSimplifiedLayerNormalization";
  if (node.input_size() < 3 || node.input(2).empty() ||
      !IsConstVec(init_map, node.input(2))) {
    return std::nullopt;  // gamma is required.
  }
  const std::string gamma_name = node.input(2);

  std::optional<std::string> beta_name;
  int bias_idx = 3;
  if (!simplified) {
    bias_idx = 4;
    if (node.input_size() > 3 && !node.input(3).empty()) {
      if (!IsConstVec(init_map, node.input(3))) {
        return std::nullopt;
      }
      beta_name = node.input(3);
    }
  }

  std::optional<std::string> bias_name;
  if (node.input_size() > bias_idx && !node.input(bias_idx).empty()) {
    if (!IsConstVec(init_map, node.input(bias_idx))) {
      return std::nullopt;
    }
    bias_name = node.input(bias_idx);
  }

  std::unordered_set<std::string> seen{gamma_name};
  if (beta_name && !seen.insert(*beta_name).second) {
    return std::nullopt;  // Tied gamma/beta -- double-slicing would corrupt it.
  }
  if (bias_name && !seen.insert(*bias_name).second) {
    return std::nullopt;  // Tied gamma/bias or beta/bias.
  }

  return SkipLayerNormConsts{gamma_name, beta_name, bias_name};
}

struct ResidualMergeMatch {
  std::string input_name;
  std::string skip_name;
  std::vector<ChainOp> extra_ops;
};

// The MatMul/Gemm residual finder's own eligible-merge-point check: `node`
// is either a bare Add (IsEligibleAddMerge, with no extra chain_ops of its
// own) or a SkipLayerNormalization-family node (see this section's own
// comment above). Declines whenever any of the SkipLayerNorm-family node's
// optional secondary outputs (mean/inv_std_var, training-only bookkeeping
// onnxruntime's own CPU kernel never actually writes, or
// input_skip_bias_sum, the raw pre-norm sum whose *shape* -- not
// meaningfulness -- is at risk once input/skip are pruned to a different
// width) are actually consumed by anything else in the graph.
std::optional<ResidualMergeMatch> MatchResidualMerge(
    onnx::NodeProto* node, const InitMap& init_map,
    const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs) {
  if (IsEligibleAddMerge(*node, init_map)) {
    return ResidualMergeMatch{node->input(0), node->input(1), {}};
  }
  if (!IsSkipLayerNormOp(*node) || node->input_size() < 3) {
    return std::nullopt;
  }
  const std::string& input_name = node->input(0);
  const std::string& skip_name = node->input(1);
  if (input_name.empty() || skip_name.empty() || input_name == skip_name ||
      init_map.count(input_name) || init_map.count(skip_name)) {
    return std::nullopt;
  }
  auto consts = SkipLayerNormConstNames(*node, init_map);
  if (!consts) {
    return std::nullopt;
  }
  for (int out_idx : {1, 2, 3}) {  // mean, inv_std_var, input_skip_bias_sum.
    if (node->output_size() > out_idx && !node->output(out_idx).empty()) {
      const std::string& out_name = node->output(out_idx);
      if (ConsumerCount(consumers_of, out_name) != 0 ||
          graph_outputs.count(out_name)) {
        return std::nullopt;
      }
    }
  }
  std::vector<ChainOp> extra_ops;
  extra_ops.push_back(ChainOp{node, consts->gamma});
  if (consts->beta) {
    extra_ops.push_back(ChainOp{node, *consts->beta});
  }
  if (consts->bias) {
    extra_ops.push_back(ChainOp{node, *consts->bias});
  }
  return ResidualMergeMatch{input_name, skip_name, std::move(extra_ops)};
}

// The backward counterpart of WalkToConsumer, used only by
// FindMatmulResidualChains/FindMatmulConcatChains.
struct MatMulBackwardEdge {
  BackwardEdgeKind kind = BackwardEdgeKind::kFail;
  Producer producer;    // kProducer, or the gate/first producer for kGated.
  Producer producer_b;  // The up/second producer, kGated only.
  int64_t n_channels = 0;
  onnx::NodeProto* add_node = nullptr;
  std::vector<ChainOp> chain_ops;  // Forward order.
  // Mirrors ConvBackwardEdge's own `edges` exactly -- see its docstring. A
  // gated Mul/SwiGLU's own two operands are deliberately not added here --
  // see pruning.py's own _walk_matmul_producer_backward docstring for why
  // there's no extra fan-out to track on that shape.
  std::vector<std::pair<std::string, onnx::NodeProto*>> edges;
};

// Walks backward from tensor `start` -- the MatMul/Gemm analogue of
// WalkConvProducerBackward, mirroring pruning.py's own current
// _walk_matmul_producer_backward. `producer_infos`, when given, is
// FindGatedChains's own producer-lookup map (raw producer output -> match
// info), needed to resolve a gated Mul hop via TraceGateProducerBackward and
// a native fused SwiGLU hop via a direct lookup of its own two raw
// operands; left nullptr, neither is ever resolved as a gated pair, and
// both simply fall through to kFail, exactly as before this parameter
// existed.
MatMulBackwardEdge WalkMatmulProducerBackward(
    const std::string& start,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const InitMap& init_map, const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int max_hops,
    const std::unordered_map<std::string, FullProducerMatch>* producer_infos =
        nullptr) {
  std::vector<ChainOp> chain_ops;  // Backward order.
  std::vector<std::pair<std::string, onnx::NodeProto*>> edges;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    if (graph_outputs.count(cur)) {
      return MatMulBackwardEdge{};
    }
    auto nit = node_by_output.find(cur);
    if (nit == node_by_output.end() || nit->second->output_size() == 0 ||
        nit->second->output(0) != cur) {
      return MatMulBackwardEdge{};
    }
    onnx::NodeProto* node = nit->second;

    auto prod_info = MatchProducer(*node, init_map);
    if (prod_info) {
      MatMulBackwardEdge edge;
      edge.kind = BackwardEdgeKind::kProducer;
      edge.producer = Producer{node,
                               prod_info->weight,
                               prod_info->weight_transposed,
                               prod_info->bias,
                               false,
                               1};
      edge.n_channels = prod_info->n_channels;
      std::reverse(chain_ops.begin(), chain_ops.end());
      edge.chain_ops = std::move(chain_ops);
      edge.edges = std::move(edges);
      return edge;
    }

    if (UnaryPassThroughOps().count(node->op_type()) != 0 &&
        node->input_size() == 1) {
      chain_ops.push_back(ChainOp{node, std::nullopt});
      edges.push_back({node->input(0), node});
      cur = node->input(0);
      continue;
    }

    if ((node->op_type() == "Add" || node->op_type() == "Mul") &&
        node->input_size() == 2) {
      const std::string& a_name = node->input(0);
      const std::string& b_name = node->input(1);
      const bool a_const = init_map.count(a_name) != 0;
      const bool b_const = init_map.count(b_name) != 0;
      if (a_const != b_const) {
        const std::string& const_name = a_const ? a_name : b_name;
        const std::string& other = a_const ? b_name : a_name;
        const onnx::TensorProto* c = init_map.at(const_name);
        int64_t prod = 1;
        for (int64_t d : c->dims()) {
          prod *= d;
        }
        const bool valid = c->data_type() == onnx::TensorProto::FLOAT &&
                           c->dims_size() > 0 &&
                           prod == c->dims(c->dims_size() - 1);
        if (valid) {
          chain_ops.push_back(ChainOp{node, const_name});
          edges.push_back({other, node});
          cur = other;
          continue;
        }
        return MatMulBackwardEdge{};
      }
      // Both operands constant (degenerate) or both non-constant: for `Add`
      // the latter is exactly IsEligibleAddMerge's own shape, handled by
      // the merge check below. For `Mul` it's a gated (SwiGLU/GeGLU)
      // combine point -- resolved by walking *both* non-constant operands
      // back to their own real producers, reusing FindGatedChains's own
      // gate-branch tracer unchanged.
      if (producer_infos != nullptr && node->op_type() == "Mul" && !a_const &&
          !b_const && a_name != b_name) {
        auto trace_a =
            TraceGateProducerBackward(a_name, node_by_output, *producer_infos,
                                      consumers_of, graph_outputs, max_hops);
        auto trace_b =
            TraceGateProducerBackward(b_name, node_by_output, *producer_infos,
                                      consumers_of, graph_outputs, max_hops);
        if (trace_a && trace_b) {
          const FullProducerMatch& info_a = trace_a->first;
          const FullProducerMatch& info_b = trace_b->first;
          if (info_a.node != info_b.node &&
              info_a.n_channels == info_b.n_channels) {
            MatMulBackwardEdge edge;
            edge.kind = BackwardEdgeKind::kGated;
            edge.producer = Producer{
                info_a.node,    info_a.weight, info_a.weight_transposed,
                info_a.bias,    false,         1,
                trace_a->second};
            edge.producer_b = Producer{
                info_b.node,    info_b.weight, info_b.weight_transposed,
                info_b.bias,    false,         1,
                trace_b->second};
            edge.n_channels = info_a.n_channels;
            edge.edges = std::move(edges);
            return edge;
          }
        }
      }
      // Not a resolvable gated pair either -- falls through to the merge
      // check (Add only) or SwiGLU/BiasGelu-FastGelu checks below.
    }

    if (producer_infos != nullptr && node->op_type() == "SwiGLU" &&
        node->input_size() == 2 && node->output_size() == 1) {
      const std::string& a_name = node->input(0);
      const std::string& b_name = node->input(1);
      if (!init_map.count(a_name) && !init_map.count(b_name)) {
        auto ait = producer_infos->find(a_name);
        auto bit = producer_infos->find(b_name);
        if (ait != producer_infos->end() && bit != producer_infos->end() &&
            ConsumerCount(consumers_of, a_name) == 1 &&
            !graph_outputs.count(a_name) &&
            ConsumerCount(consumers_of, b_name) == 1 &&
            !graph_outputs.count(b_name)) {
          const FullProducerMatch& info_a = ait->second;
          const FullProducerMatch& info_b = bit->second;
          if (info_a.node != info_b.node &&
              info_a.n_channels == info_b.n_channels) {
            MatMulBackwardEdge edge;
            edge.kind = BackwardEdgeKind::kGated;
            edge.producer =
                Producer{info_a.node, info_a.weight, info_a.weight_transposed,
                         info_a.bias, false,         1};
            edge.producer_b =
                Producer{info_b.node, info_b.weight, info_b.weight_transposed,
                         info_b.bias, false,         1};
            edge.n_channels = info_a.n_channels;
            edge.edges = std::move(edges);
            return edge;
          }
        }
      }
      // Not a resolvable gated pair -- SwiGLU is never an eligible merge
      // node either, so this falls through to kFail below.
    }

    if (node->op_type() == "BiasGelu" || node->op_type() == "FastGelu") {
      auto fused = MatchFusedBiasGelu(*node, init_map);
      if (fused) {
        chain_ops.push_back(ChainOp{node, fused->bias_name});
        edges.push_back({fused->data_name, node});
        cur = fused->data_name;
        continue;
      }
      return MatMulBackwardEdge{};
    }

    if (node->op_type() == "PRelu" && node->domain() == "" &&
        node->input_size() == 2) {
      auto prelu_self = MatchPreluPassThroughMatmulSelf(*node, init_map);
      if (prelu_self) {
        chain_ops.push_back(ChainOp{node, prelu_self->is_per_channel
                                              ? prelu_self->slope_name
                                              : std::nullopt});
        edges.push_back({node->input(0), node});
        cur = node->input(0);
        continue;
      }
      return MatMulBackwardEdge{};
    }

    if (node->op_type() == "Clip" &&
        MatchClipChannelPassThrough(*node, init_map)) {
      chain_ops.push_back(ChainOp{node, std::nullopt});
      edges.push_back({node->input(0), node});
      cur = node->input(0);
      continue;
    }

    if (MatchResidualMerge(node, init_map, consumers_of, graph_outputs)) {
      MatMulBackwardEdge edge;
      edge.kind = BackwardEdgeKind::kAdd;
      edge.add_node = node;
      std::reverse(chain_ops.begin(), chain_ops.end());
      edge.chain_ops = std::move(chain_ops);
      edge.edges = std::move(edges);
      return edge;
    }

    return MatMulBackwardEdge{};
  }
  return MatMulBackwardEdge{};
}

// The MatMul/Gemm analogue of ResolveConvFanoutBranches -- see its own
// docstring for the shared reasoning this mirrors exactly (only the forward
// walker differs: WalkToConsumer instead of WalkToConvConsumer), and there
// is no Conv-style grouped-consumer or depthwise-pass-through concept to
// check or carry for a MatMul/Gemm branch at all.
std::optional<std::vector<ConsumerBranch>> ResolveMatmulFanoutBranches(
    const std::vector<std::string>& backbone_tensors,
    const std::unordered_map<std::string, std::unordered_set<onnx::NodeProto*>>&
        accounted,
    const InitMap& init_map, const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int64_t n_channels) {
  std::vector<ConsumerBranch> branches;
  std::unordered_set<std::string> seen_weights;
  for (const auto& tensor : backbone_tensors) {
    if (graph_outputs.count(tensor)) {
      return std::nullopt;
    }
    auto cit = consumers_of.find(tensor);
    if (cit == consumers_of.end()) {
      continue;
    }
    auto acc_it = accounted.find(tensor);
    std::unordered_set<onnx::NodeProto*> seen_nodes;
    for (onnx::NodeProto* consumer_node : cit->second) {
      if (!seen_nodes.insert(consumer_node).second) {
        continue;
      }
      if (acc_it != accounted.end() && acc_it->second.count(consumer_node)) {
        continue;
      }
      auto [resolved, br_chain_ops] =
          WalkToConsumer(tensor, init_map, consumers_of, graph_outputs,
                         n_channels, kMaxChainHops, consumer_node);
      if (!resolved) {
        return std::nullopt;
      }
      if (seen_weights.count(resolved->weight)) {
        return std::nullopt;
      }
      seen_weights.insert(resolved->weight);
      ConsumerBranch branch;
      branch.chain_ops = std::move(br_chain_ops);
      branch.consumer_node = resolved->node;
      branch.consumer_weight = resolved->weight;
      branch.consumer_weight_transposed = resolved->weight_transposed;
      branch.consumer_is_conv = false;
      branches.push_back(std::move(branch));
    }
  }
  return branches;
}

// Finds MatMul/Gemm residual/skip-connection groups -- the MatMul/Gemm
// analogue of FindConvResidualChains, over WalkMatmulProducerBackward
// instead of WalkConvProducerBackward. Every eligible merge point
// (MatchResidualMerge -- a bare Add or a SkipLayerNormalization-family
// node) contributes its own extra_ops (empty for Add; gamma/beta/bias for
// the normalization-fused case) up front, before any union-find grouping,
// so every member of a resolved group has its own per-channel constants
// folded into the final chain the same way.
std::vector<Chain> FindMatmulResidualChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
  }

  // FindGatedChains's own producer-lookup map, built once here and threaded
  // through every WalkMatmulProducerBackward call below -- needed only to
  // resolve a gated Mul/SwiGLU hop.
  std::unordered_map<std::string, FullProducerMatch> producer_infos;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = MatchProducer(*node, init_map);
    if (info) {
      producer_infos[node->output(0)] =
          FullProducerMatch{node, info->weight, info->weight_transposed,
                            info->bias, info->n_channels};
    }
  }

  struct MergeInfo {
    onnx::NodeProto* node;
    std::string input_name;
    std::string skip_name;
    std::vector<ChainOp> extra_ops;
  };
  std::vector<MergeInfo> merges;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto match =
        MatchResidualMerge(node, init_map, consumers_of, graph_outputs);
    if (match) {
      merges.push_back(MergeInfo{node, std::move(match->input_name),
                                 std::move(match->skip_name),
                                 std::move(match->extra_ops)});
    }
  }
  if (merges.empty()) {
    return {};
  }
  std::unordered_map<onnx::NodeProto*, int> merge_index;
  for (size_t i = 0; i < merges.size(); ++i) {
    merge_index[merges[i].node] = static_cast<int>(i);
  }

  std::vector<int> parent(merges.size());
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int i) {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  auto unite = [&](int i, int j) {
    const int ri = find(i), rj = find(j);
    if (ri != rj) {
      parent[ri] = rj;
    }
  };

  std::vector<std::vector<MatMulBackwardEdge>> edge_results(merges.size());
  std::unordered_set<int> poisoned;
  for (size_t idx = 0; idx < merges.size(); ++idx) {
    std::vector<MatMulBackwardEdge> results;
    for (const auto& operand :
         {merges[idx].input_name, merges[idx].skip_name}) {
      MatMulBackwardEdge edge = WalkMatmulProducerBackward(
          operand, node_by_output, init_map, consumers_of, graph_outputs,
          kMaxChainHops, &producer_infos);
      if (edge.kind == BackwardEdgeKind::kFail) {
        poisoned.insert(static_cast<int>(idx));
      } else if (edge.kind == BackwardEdgeKind::kAdd) {
        auto jit = merge_index.find(edge.add_node);
        if (jit == merge_index.end()) {
          poisoned.insert(
              static_cast<int>(idx));  // Defensive -- shouldn't happen.
        } else {
          unite(static_cast<int>(idx), jit->second);
        }
      }
      results.push_back(std::move(edge));
    }
    edge_results[idx] = std::move(results);
  }

  std::unordered_map<int, std::vector<int>> groups;
  for (size_t idx = 0; idx < merges.size(); ++idx) {
    groups[find(static_cast<int>(idx))].push_back(static_cast<int>(idx));
  }

  std::vector<Chain> chains;
  for (auto& kv : groups) {
    const std::vector<int>& members = kv.second;
    bool any_poisoned = false;
    for (int i : members) {
      if (poisoned.count(i)) {
        any_poisoned = true;
        break;
      }
    }
    if (any_poisoned) {
      continue;
    }

    std::vector<Producer> leaf_producers;
    std::unordered_set<int64_t> n_channels_set;
    std::vector<ChainOp> pre_chain_ops;
    std::unordered_set<int> referenced;
    std::vector<std::string> backbone_tensors;
    std::unordered_map<std::string, std::unordered_set<onnx::NodeProto*>>
        accounted;
    auto mark_backbone = [&](const std::string& tensor, onnx::NodeProto* node) {
      if (!accounted.count(tensor)) {
        backbone_tensors.push_back(tensor);
      }
      accounted[tensor].insert(node);
    };

    for (int idx : members) {
      onnx::NodeProto* merge_node = merges[static_cast<size_t>(idx)].node;
      pre_chain_ops.insert(pre_chain_ops.end(),
                           merges[static_cast<size_t>(idx)].extra_ops.begin(),
                           merges[static_cast<size_t>(idx)].extra_ops.end());
      const std::string operands[2] = {
          merges[static_cast<size_t>(idx)].input_name,
          merges[static_cast<size_t>(idx)].skip_name};
      const auto& results = edge_results[static_cast<size_t>(idx)];
      for (size_t oi = 0; oi < 2; ++oi) {
        const MatMulBackwardEdge& edge = results[oi];
        mark_backbone(operands[oi], merge_node);
        for (const auto& e : edge.edges) {
          mark_backbone(e.first, e.second);
        }
        pre_chain_ops.insert(pre_chain_ops.end(), edge.chain_ops.begin(),
                             edge.chain_ops.end());
        if (edge.kind == BackwardEdgeKind::kProducer) {
          leaf_producers.push_back(edge.producer);
          n_channels_set.insert(edge.n_channels);
        } else if (edge.kind == BackwardEdgeKind::kGated) {
          leaf_producers.push_back(edge.producer);
          leaf_producers.push_back(edge.producer_b);
          n_channels_set.insert(edge.n_channels);
        } else if (edge.kind == BackwardEdgeKind::kAdd) {
          referenced.insert(merge_index[edge.add_node]);
        }
      }
    }

    if (n_channels_set.size() != 1) {
      continue;  // Branches disagree on channel count -- decline.
    }
    const int64_t n_channels = *n_channels_set.begin();

    bool const_mismatch = false;
    for (const auto& co : pre_chain_ops) {
      if (co.const_name &&
          init_map.at(*co.const_name)
                  ->dims(init_map.at(*co.const_name)->dims_size() - 1) !=
              n_channels) {
        const_mismatch = true;
        break;
      }
    }
    if (const_mismatch) {
      continue;
    }

    std::vector<int> sinks;
    for (int idx : members) {
      if (!referenced.count(idx)) {
        sinks.push_back(idx);
      }
    }
    if (sinks.size() != 1) {
      continue;  // Not a single linear chain of merges -- decline.
    }
    onnx::NodeProto* sink_node = merges[static_cast<size_t>(sinks[0])].node;

    std::unordered_set<std::string> seen_weights;
    bool degenerate = false;
    for (const auto& p : leaf_producers) {
      if (!seen_weights.insert(p.weight).second) {
        degenerate = true;
        break;
      }
    }
    if (degenerate) {
      continue;  // The same producer named twice.
    }

    const std::string& sink_out = sink_node->output(0);
    if (!accounted.count(sink_out)) {
      backbone_tensors.push_back(sink_out);
      accounted[sink_out];
    }

    auto branches_opt =
        ResolveMatmulFanoutBranches(backbone_tensors, accounted, init_map,
                                    consumers_of, graph_outputs, n_channels);
    if (!branches_opt || branches_opt->empty()) {
      continue;
    }
    std::vector<ConsumerBranch>& branches = *branches_opt;

    ConsumerBranch primary = std::move(branches.front());
    std::vector<ConsumerBranch> extra_branches(
        std::make_move_iterator(branches.begin() + 1),
        std::make_move_iterator(branches.end()));

    std::vector<ChainOp> chain_ops = std::move(pre_chain_ops);
    for (int idx : members) {
      chain_ops.push_back(
          ChainOp{merges[static_cast<size_t>(idx)].node, std::nullopt});
    }
    for (auto& co : primary.chain_ops) {
      chain_ops.push_back(std::move(co));
    }

    Chain chain;
    chain.producers = std::move(leaf_producers);
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = primary.consumer_node;
    chain.consumer_weight = primary.consumer_weight;
    chain.consumer_weight_transposed = primary.consumer_weight_transposed;
    chain.n_channels = n_channels;
    chain.extra_consumers = std::move(extra_branches);
    chains.push_back(std::move(chain));
  }
  return chains;
}

int64_t ChainGroup(const Chain& chain) {
  int64_t group = chain.consumer_group;
  for (const auto& p : chain.producers) {
    if (p.group > 1) {
      group = p.group;
      break;
    }
  }
  // A mid-chain GroupNormalization hop's own `num_groups` takes priority
  // over the plain producer/consumer `group` fields above -- FindConvChains
  // already declined the chain outright if `num_groups` disagreed with a
  // non-1 producer/consumer `group` (see its own reconciliation check), so
  // whenever both are present they already agree, and returning
  // `num_groups` unconditionally is equivalent to returning either. This is
  // what makes GroupNorm's own per-group statistics stay valid after
  // pruning -- mirrors pruning.py's own _chain_group exactly.
  if (chain.group_norm) {
    group = chain.group_norm->num_groups;
  }
  return group;
}

// --- Slicing, mirroring _slice_producer_weight/_slice_consumer_weight/
// _slice_grouped_consumer_conv_weight/_slice_last_axis ----------------------

// Keeps only rows in `keep` of a [rows, inner] row-major matrix.
std::vector<float> SliceAxis0(const std::vector<float>& data, int64_t /*rows*/,
                              int64_t inner, const std::vector<int64_t>& keep) {
  std::vector<float> out(keep.size() * static_cast<size_t>(inner));
  for (size_t i = 0; i < keep.size(); ++i) {
    std::memcpy(out.data() + i * inner, data.data() + keep[i] * inner,
                static_cast<size_t>(inner) * sizeof(float));
  }
  return out;
}

// Keeps only columns in `keep` of a [dim0, dim1, inner] row-major tensor
// (axis 1 sliced; `inner` is the flattened size of every trailing axis).
std::vector<float> SliceAxis1(const std::vector<float>& data, int64_t dim0,
                              int64_t dim1, int64_t inner,
                              const std::vector<int64_t>& keep) {
  std::vector<float> out(static_cast<size_t>(dim0) * keep.size() *
                         static_cast<size_t>(inner));
  for (int64_t i = 0; i < dim0; ++i) {
    for (size_t j = 0; j < keep.size(); ++j) {
      std::memcpy(
          out.data() + (static_cast<size_t>(i) * keep.size() + j) * inner,
          data.data() + (i * dim1 + keep[j]) * inner,
          static_cast<size_t>(inner) * sizeof(float));
    }
  }
  return out;
}

void SliceProducerWeight(onnx::TensorProto* wt, bool weight_transposed,
                         const std::vector<int64_t>& keep, bool is_conv) {
  std::vector<int64_t> dims(wt->dims().begin(), wt->dims().end());
  std::vector<float> data = ReadFloatTensor(*wt);
  std::vector<float> out;
  std::vector<int64_t> new_dims;
  const int64_t kc = static_cast<int64_t>(keep.size());
  if (is_conv) {
    int64_t inner = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
      inner *= dims[i];
    }
    out = SliceAxis0(data, dims[0], inner, keep);
    new_dims = dims;
    new_dims[0] = kc;
  } else {
    const int64_t dim0 = dims[0], dim1 = dims[1];
    if (weight_transposed) {  // [N, K] -- output channel is axis 0.
      out = SliceAxis0(data, dim0, dim1, keep);
      new_dims = {kc, dim1};
    } else {  // [K, N] -- output channel is axis 1.
      out = SliceAxis1(data, dim0, dim1, 1, keep);
      new_dims = {dim0, kc};
    }
  }
  SetFloatTensorData(wt, new_dims, out);
}

void SliceConsumerWeight(onnx::TensorProto* wt, bool weight_transposed,
                         const std::vector<int64_t>& keep, bool is_conv) {
  std::vector<int64_t> dims(wt->dims().begin(), wt->dims().end());
  std::vector<float> data = ReadFloatTensor(*wt);
  std::vector<float> out;
  std::vector<int64_t> new_dims;
  const int64_t kc = static_cast<int64_t>(keep.size());
  if (is_conv) {
    int64_t inner = 1;
    for (size_t i = 2; i < dims.size(); ++i) {
      inner *= dims[i];
    }
    out = SliceAxis1(data, dims[0], dims[1], inner, keep);
    new_dims = dims;
    new_dims[1] = kc;
  } else {
    const int64_t dim0 = dims[0], dim1 = dims[1];
    if (weight_transposed) {  // [N, K] -- reduction dim is axis 1.
      out = SliceAxis1(data, dim0, dim1, 1, keep);
      new_dims = {dim0, kc};
    } else {  // [K, N] -- reduction dim is axis 0.
      out = SliceAxis0(data, dim0, dim1, keep);
      new_dims = {kc, dim1};
    }
  }
  SetFloatTensorData(wt, new_dims, out);
}

// See pruning.py's own _slice_grouped_consumer_conv_weight for why a global
// `keep` needs per-group-relative local translation here.
void SliceGroupedConsumerConvWeight(onnx::TensorProto* wt,
                                    const std::vector<int64_t>& keep,
                                    int64_t group, int64_t n_channels) {
  std::vector<int64_t> dims(wt->dims().begin(), wt->dims().end());
  std::vector<float> data = ReadFloatTensor(*wt);
  const int64_t out_channels = dims[0];
  const int64_t in_per_group = dims[1];
  int64_t inner = 1;
  for (size_t i = 2; i < dims.size(); ++i) {
    inner *= dims[i];
  }
  const int64_t out_per_group = out_channels / group;
  const int64_t block = n_channels / group;

  std::vector<std::vector<int64_t>> local_keeps(static_cast<size_t>(group));
  for (int64_t k : keep) {
    const int64_t gi = k / block;
    local_keeps[static_cast<size_t>(gi)].push_back(k - gi * block);
  }

  std::vector<float> out;
  out.reserve(data.size());  // upper bound
  for (int64_t gi = 0; gi < group; ++gi) {
    const auto& lk = local_keeps[static_cast<size_t>(gi)];
    const int64_t filt_lo = gi * out_per_group;
    for (int64_t r = filt_lo; r < filt_lo + out_per_group; ++r) {
      for (int64_t local : lk) {
        const float* src = data.data() + (r * in_per_group + local) * inner;
        out.insert(out.end(), src, src + inner);
      }
    }
  }
  std::vector<int64_t> new_dims = dims;
  new_dims[1] = static_cast<int64_t>(keep.size()) / group;
  SetFloatTensorData(wt, new_dims, out);
}

void SliceLastAxis(onnx::TensorProto* t, const std::vector<int64_t>& keep) {
  std::vector<int64_t> dims(t->dims().begin(), t->dims().end());
  std::vector<float> data = ReadFloatTensor(*t);
  std::vector<float> out(keep.size());
  for (size_t i = 0; i < keep.size(); ++i) {
    out[i] = data[static_cast<size_t>(keep[i])];
  }
  std::vector<int64_t> new_dims = dims;
  if (new_dims.empty()) {
    new_dims.push_back(static_cast<int64_t>(keep.size()));
  } else {
    new_dims.back() = static_cast<int64_t>(keep.size());
  }
  SetFloatTensorData(t, new_dims, out);
}

// Selects the `keep_count` highest-`importance` indices, returned sorted
// ascending -- mirrors np.sort(np.argsort(-importance)[:keep_count]) (tie-
// breaking among exactly-equal importances may differ from numpy's own
// argsort, same caveat as magnitude_pruning.h's SparsityMaskRowMajor).
std::vector<int64_t> TopKIndicesAscending(const std::vector<double>& importance,
                                          int64_t keep_count) {
  const int64_t n = static_cast<int64_t>(importance.size());
  std::vector<int64_t> idx(static_cast<size_t>(n));
  std::iota(idx.begin(), idx.end(), int64_t{0});
  std::partial_sort(
      idx.begin(), idx.begin() + keep_count, idx.end(),
      [&](int64_t a, int64_t b) { return importance[a] > importance[b]; });
  idx.resize(static_cast<size_t>(keep_count));
  std::sort(idx.begin(), idx.end());
  return idx;
}

// Transposes a [dim0, dim1] row-major matrix into [dim1, dim0].
std::vector<float> TransposeFlat(const std::vector<float>& data, int64_t dim0,
                                 int64_t dim1) {
  std::vector<float> out(data.size());
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      out[static_cast<size_t>(j * dim0 + i)] =
          data[static_cast<size_t>(i * dim1 + j)];
    }
  }
  return out;
}

// --- Shared apply body, mirroring _apply_chains -----------------------------

// Every touched initializer role and stale value_info name, shared by a
// single ApplyChains call *and* any sibling ApplyConcatChains call over the
// same graph -- mirrors pruning.py's own _TouchedState exactly, so the two
// can never doubly resize the same weight. The caller flushes value_info
// once, from `stale_value_info`, after every such call.
struct TouchedState {
  std::unordered_set<std::string> producer;
  std::unordered_set<std::string> consumer;
  std::unordered_set<std::string> const_names;
  std::unordered_set<std::string> conv_hop;
  std::unordered_set<std::string> stale_value_info;
};

void ApplyChains(onnx::GraphProto* graph, std::vector<Chain>& chains,
                 double sparsity, TouchedState& touched) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }

  // A weight legitimately plays both roles across two different chains --
  // tracked separately per role; bias/scale constants only ever play one
  // role, so a single shared set is enough for those.
  std::unordered_set<std::string>& producer_touched = touched.producer;
  std::unordered_set<std::string>& consumer_touched = touched.consumer;
  std::unordered_set<std::string>& const_touched = touched.const_names;
  std::unordered_set<std::string>& conv_hop_touched = touched.conv_hop;
  std::unordered_set<std::string>& stale_value_info = touched.stale_value_info;

  for (auto& chain : chains) {
    std::unordered_set<std::string> producer_weights;
    bool degenerate = false;
    for (const auto& p : chain.producers) {
      if (!producer_weights.insert(p.weight).second) {
        degenerate = true;
        break;
      }
    }
    if (degenerate) {
      continue;
    }

    // Every consumer branch this chain touches -- just the one primary
    // consumer_* for every chain kind except a residual/merge group with
    // extra fan-out (see Chain::extra_consumers), where there are one or
    // more additional independent branches beyond it. Conflict-checked,
    // touched, and sliced exactly like the single consumer every other
    // chain already has -- each branch is its own axis of its own weight,
    // fed by the exact same shared `keep` this loop computes once, below.
    std::vector<const ConsumerBranch*> extra_ptrs;
    extra_ptrs.reserve(chain.extra_consumers.size());
    for (const auto& b : chain.extra_consumers) {
      extra_ptrs.push_back(&b);
    }

    std::unordered_set<std::string> consumer_weights{chain.consumer_weight};
    size_t n_branches = 1;
    for (const auto* b : extra_ptrs) {
      consumer_weights.insert(b->consumer_weight);
      ++n_branches;
    }
    if (consumer_weights.size() != n_branches) {
      continue;  // Degenerate -- two branches naming the same weight.
    }

    std::unordered_set<std::string> conv_hop_weights;
    for (const auto& h : chain.conv_pass_through) {
      if (!conv_hop_weights.insert(h.weight).second) {
        degenerate = true;
        break;
      }
    }
    size_t n_conv_hops = chain.conv_pass_through.size();
    if (!degenerate) {
      for (const auto* b : extra_ptrs) {
        for (const auto& h : b->conv_pass_through) {
          if (!conv_hop_weights.insert(h.weight).second) {
            degenerate = true;
            break;
          }
        }
        n_conv_hops += b->conv_pass_through.size();
        if (degenerate) {
          break;
        }
      }
    }
    if (degenerate || conv_hop_weights.size() != n_conv_hops) {
      continue;  // Degenerate -- the same depthwise weight named twice.
    }

    std::unordered_set<std::string> consts;
    for (const auto& p : chain.producers) {
      if (p.bias) {
        consts.insert(*p.bias);
      }
    }
    for (const auto& co : chain.chain_ops) {
      if (co.const_name) {
        consts.insert(*co.const_name);
      }
    }
    for (const auto* b : extra_ptrs) {
      for (const auto& co : b->chain_ops) {
        if (co.const_name) {
          consts.insert(*co.const_name);
        }
      }
    }
    // A mid-chain GroupNorm hop's own `scale`/`bias` get exactly the same
    // shared/tied-initializer conflict protection every other chain-op
    // constant already does -- mirrors pruning.py's own
    // `consts.update(_chain_group_norm_consts(chain))`.
    if (chain.group_norm) {
      consts.insert(chain.group_norm->scale);
      consts.insert(chain.group_norm->bias);
    }

    bool conflict = false;
    for (const auto& w : consumer_weights) {
      if (consumer_touched.count(w)) {
        conflict = true;
      }
    }
    for (const auto& w : producer_weights) {
      if (producer_touched.count(w)) {
        conflict = true;
      }
    }
    for (const auto& c : consts) {
      if (const_touched.count(c)) {
        conflict = true;
      }
    }
    for (const auto& w : conv_hop_weights) {
      if (conv_hop_touched.count(w)) {
        conflict = true;
      }
    }
    if (conflict) {
      continue;  // A shared/tied initializer another chain already resized.
    }

    const int64_t n = chain.n_channels;
    const int64_t group = ChainGroup(chain);
    int64_t keep_count, per_group_keep = 0, block = 0;
    if (group > 1) {
      block = n / group;
      per_group_keep = std::max<int64_t>(
          1, std::llround(static_cast<double>(block) * (1.0 - sparsity)));
      keep_count = per_group_keep * group;
    } else {
      keep_count = std::max<int64_t>(
          1, n - std::llround(static_cast<double>(n) * sparsity));
    }
    if (keep_count >= n) {
      continue;  // Rounds down to nothing for this layer -- no-op.
    }

    std::vector<std::vector<float>> w_arrays_nk;
    for (const auto& p : chain.producers) {
      onnx::TensorProto* wt = init_map.at(p.weight);
      std::vector<int64_t> dims(wt->dims().begin(), wt->dims().end());
      std::vector<float> data = ReadFloatTensor(*wt);
      if (p.is_conv) {
        w_arrays_nk.push_back(
            std::move(data));  // Already [Cout, rest] flattened.
      } else if (p.weight_transposed) {
        w_arrays_nk.push_back(std::move(data));  // Already [N, K].
      } else {
        w_arrays_nk.push_back(
            TransposeFlat(data, dims[0], dims[1]));  // [K,N] -> [N,K].
      }
    }

    std::vector<double> importance(static_cast<size_t>(n), 0.0);
    for (const auto& w_nk : w_arrays_nk) {
      const int64_t k = static_cast<int64_t>(w_nk.size()) / n;
      for (int64_t c = 0; c < n; ++c) {
        double sq = 0.0;
        for (int64_t j = 0; j < k; ++j) {
          const double v = w_nk[static_cast<size_t>(c * k + j)];
          sq += v * v;
        }
        importance[static_cast<size_t>(c)] += sq;
      }
    }
    for (double& v : importance) {
      v = std::sqrt(v);
    }

    std::vector<int64_t> keep;
    if (group > 1) {
      keep.reserve(static_cast<size_t>(keep_count));
      for (int64_t gi = 0; gi < group; ++gi) {
        std::vector<double> block_imp(importance.begin() + gi * block,
                                      importance.begin() + (gi + 1) * block);
        for (int64_t li : TopKIndicesAscending(block_imp, per_group_keep)) {
          keep.push_back(li + gi * block);
        }
      }
    } else {
      keep = TopKIndicesAscending(importance, keep_count);
    }

    for (const auto& p : chain.producers) {
      SliceProducerWeight(init_map.at(p.weight), p.weight_transposed, keep,
                          p.is_conv);
      if (p.bias) {
        SliceLastAxis(init_map.at(*p.bias), keep);
      }
    }
    for (const auto& co : chain.chain_ops) {
      if (co.const_name) {
        SliceLastAxis(init_map.at(*co.const_name), keep);
      }
    }
    for (const auto& hop : chain.conv_pass_through) {
      SliceProducerWeight(init_map.at(hop.weight), false, keep, true);
      if (hop.bias) {
        SliceLastAxis(init_map.at(*hop.bias), keep);
      }
      // A PRelu per-channel-slope hop reuses ConvPassThrough for its own
      // slicing (see MatchPreluPassThrough's own comment) but, unlike a
      // depthwise Conv hop, has no `group` attribute of its own to update --
      // mirrors pruning.py's own _apply_conv_pass_through_hop, which only
      // ever touches `group` when the hop node is actually a Conv.
      if (hop.node->op_type() == "Conv") {
        SetOrAddIntAttr(hop.node, "group", static_cast<int64_t>(keep.size()));
      }
    }
    if (chain.group_norm) {
      // Same `keep` index set as the real producer -- `num_groups` itself is
      // left untouched (see GroupNormPassThrough's own comment for why it
      // stays valid without changing it). Sliced via SliceLastAxis, not
      // ConvPassThrough's own axis-0 SliceProducerWeight -- see
      // GroupNormPassThrough's own comment for why.
      SliceLastAxis(init_map.at(chain.group_norm->scale), keep);
      SliceLastAxis(init_map.at(chain.group_norm->bias), keep);
    }
    if (chain.consumer_is_conv && chain.consumer_group > 1) {
      SliceGroupedConsumerConvWeight(init_map.at(chain.consumer_weight), keep,
                                     chain.consumer_group, n);
    } else {
      SliceConsumerWeight(init_map.at(chain.consumer_weight),
                          chain.consumer_weight_transposed, keep,
                          chain.consumer_is_conv);
    }
    // Extra fan-out branches: each is either an ordinary (group == 1)
    // consumer, or, for a Conv residual/merge chain, a general grouped Conv
    // consumer whose own group was already confirmed (in
    // FindConvResidualChains) to agree with `group` above --
    // ResolveMatmulFanoutBranches never resolves a grouped one, so
    // consumer_group stays at its default 1 there. Either way, fed by the
    // exact same `keep` just computed for the group's shared producers
    // above.
    for (const auto* b : extra_ptrs) {
      for (const auto& co : b->chain_ops) {
        if (co.const_name) {
          SliceLastAxis(init_map.at(*co.const_name), keep);
        }
      }
      for (const auto& hop : b->conv_pass_through) {
        SliceProducerWeight(init_map.at(hop.weight), false, keep, true);
        if (hop.bias) {
          SliceLastAxis(init_map.at(*hop.bias), keep);
        }
        if (hop.node->op_type() == "Conv") {
          SetOrAddIntAttr(hop.node, "group", static_cast<int64_t>(keep.size()));
        }
      }
      if (b->consumer_is_conv && b->consumer_group > 1) {
        SliceGroupedConsumerConvWeight(init_map.at(b->consumer_weight), keep,
                                       b->consumer_group, n);
      } else {
        SliceConsumerWeight(init_map.at(b->consumer_weight),
                            b->consumer_weight_transposed, keep,
                            b->consumer_is_conv);
      }
    }

    for (const auto& w : producer_weights) {
      producer_touched.insert(w);
    }
    for (const auto& w : consumer_weights) {
      consumer_touched.insert(w);
    }
    for (const auto& c : consts) {
      const_touched.insert(c);
    }
    for (const auto& w : conv_hop_weights) {
      conv_hop_touched.insert(w);
    }
    for (const auto& p : chain.producers) {
      stale_value_info.insert(p.node->output(0));
      for (const auto* pre_op : p.pre_ops) {
        stale_value_info.insert(pre_op->output(0));
      }
    }
    for (const auto& co : chain.chain_ops) {
      stale_value_info.insert(co.node->output(0));
    }
    for (const auto& hop : chain.conv_pass_through) {
      stale_value_info.insert(hop.node->output(0));
    }
    if (chain.group_norm) {
      stale_value_info.insert(chain.group_norm->node->output(0));
    }
    for (const auto* b : extra_ptrs) {
      for (const auto& co : b->chain_ops) {
        stale_value_info.insert(co.node->output(0));
      }
      for (const auto& hop : b->conv_pass_through) {
        stale_value_info.insert(hop.node->output(0));
      }
    }
  }
}

// --- Attention-head pruning, mirroring pruning.py's own
// _match_attention_producer/_walk_to_attention_consumer/
// _find_attention_chains, _match_gqa_producer/_match_onnx_attention_producer/
// _find_separate_qkv_chains, and _apply_one_plain_attention_chain/
// _apply_one_gqa_chain/_apply_attention_chains. Data-free (magnitude/
// Frobenius-norm) only -- pruning.py's own calibration-driven
// apply_attention_head_wanda_pruning is not ported, matching this codebase's
// established C++-port scope decision (data-free/closed-form techniques
// only). Three fused self-attention ops are matched, each at the
// granularity its own kernel contract allows -- see pruning.py's own
// "Attention-head pruning" section comment for the full rationale (packed-
// QKV vs. separate-Q/K/V weight layout, individual-head vs. whole-KV-group
// pruning unit, and why the plain ai.onnx::Attention op reuses the
// GroupQueryAttention machinery outright rather than a parallel
// implementation) -- this comment only covers what's specific to the port:
//
// - com.microsoft::Attention: a single merged QKV weight/bias, one
//   `num_heads` shared by Q/K/V alike -- every head owns an equally-sized,
//   independent column block, so individual heads drop one at a time
//   (FindAttentionChains/ApplyOnePlainAttentionChain).
// - com.microsoft::GroupQueryAttention and the plain ai.onnx::Attention
//   (opset 24+): separate, un-merged Q/K/V MatMul/vanilla-Gemm producers,
//   `num_heads`/`kv_num_heads` (or `q_num_heads`/`kv_num_heads` for the
//   plain op) attributes -- only a whole KV group (that KV head's own K/V
//   column block, plus every query head mapped to it) is ever removed at
//   once, since every surviving KV head must keep exactly the same number
//   of query heads mapped to it after pruning (FindSeparateQkvChains/
//   ApplyOneGqaChain, shared between the two ops).

// True for a com.microsoft::Attention node with a constant 2-D float32
// merged QKV weight [K, Nq+Nk+Nv] (and, if present, a constant 1-D float32
// merged bias). Mirrors pruning.py's own _match_attention_producer.
struct AttentionProducerMatch {
  std::string weight;
  std::optional<std::string> bias;
  int64_t num_heads;
  int64_t nq, nk, nv;
};

std::optional<AttentionProducerMatch> MatchAttentionProducer(
    const onnx::NodeProto& node, const InitMap& init_map) {
  if (node.domain() != kComMicrosoftDomain || node.op_type() != "Attention") {
    return std::nullopt;
  }
  if (node.input_size() < 2) {
    return std::nullopt;
  }
  const std::string& w_name = node.input(1);
  auto wit = init_map.find(w_name);
  if (wit == init_map.end() ||
      wit->second->data_type() != onnx::TensorProto::FLOAT ||
      wit->second->dims_size() != 2) {
    return std::nullopt;
  }
  const int64_t total_n = wit->second->dims(1);

  std::optional<std::string> bias_name;
  if (node.input_size() >= 3 && !node.input(2).empty()) {
    bias_name = node.input(2);
    auto bit = init_map.find(*bias_name);
    if (bit == init_map.end() ||
        bit->second->data_type() != onnx::TensorProto::FLOAT ||
        bit->second->dims_size() != 1 || bit->second->dims(0) != total_n) {
      return std::nullopt;
    }
  }

  int64_t num_heads = 0;
  bool has_num_heads = false;
  std::optional<std::vector<int64_t>> qkv_hidden_sizes;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "num_heads") {
      num_heads = attr.i();
      has_num_heads = true;
    } else if (attr.name() == "qkv_hidden_sizes") {
      qkv_hidden_sizes =
          std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
    }
  }
  if (!has_num_heads || num_heads <= 0) {
    return std::nullopt;
  }

  int64_t nq, nk, nv;
  if (qkv_hidden_sizes) {
    if (qkv_hidden_sizes->size() != 3) {
      return std::nullopt;
    }
    nq = (*qkv_hidden_sizes)[0];
    nk = (*qkv_hidden_sizes)[1];
    nv = (*qkv_hidden_sizes)[2];
  } else {  // Schema default: Q/K/V evenly split the merged width.
    if (total_n % 3 != 0) {
      return std::nullopt;
    }
    nq = nk = nv = total_n / 3;
  }
  if (nq <= 0 || nk <= 0 || nv <= 0 || nq + nk + nv != total_n ||
      nq % num_heads != 0 || nk % num_heads != 0 || nv % num_heads != 0) {
    return std::nullopt;
  }
  return AttentionProducerMatch{w_name, bias_name, num_heads, nq, nk, nv};
}

// If `node` is a Reshape whose target-shape input is a constant int64
// tensor, returns its last entry (or nullopt for a wildcard/inferred -1/0
// entry, or an unreadable shape).
std::optional<int64_t> ReshapeLastDim(const onnx::NodeProto& node,
                                      const InitMap& init_map) {
  if (node.op_type() != "Reshape" || node.input_size() != 2) {
    return std::nullopt;
  }
  auto it = init_map.find(node.input(1));
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::INT64) {
    return std::nullopt;
  }
  std::vector<int64_t> dims = ReadInt64Tensor(*it->second);
  if (dims.empty()) {
    return std::nullopt;
  }
  const int64_t last = dims.back();
  return last > 0 ? std::optional<int64_t>(last) : std::nullopt;
}

struct AttnChainOp {
  onnx::NodeProto* node;
  std::optional<std::string> shape_name;
};

// From an attention op's raw (V-hidden-size- or Q-hidden-size-wide,
// depending on caller) output tensor `start`, optionally through a single
// Reshape hop whose target shape's last entry is provably still `width`,
// to a MatMul/vanilla-Gemm consumer (the output projection) whose
// reduction dimension matches `width`. Mirrors pruning.py's own
// _walk_to_attention_consumer.
std::pair<std::optional<ConsumerMatch>, std::vector<AttnChainOp>>
WalkToAttentionConsumer(const std::string& start, const InitMap& init_map,
                        const ConsumerMap& consumers_of,
                        const std::unordered_set<std::string>& graph_outputs,
                        int64_t width) {
  auto cit = consumers_of.find(start);
  if (cit == consumers_of.end() || cit->second.size() != 1) {
    return {std::nullopt, {}};
  }
  onnx::NodeProto* node = cit->second[0];
  std::vector<AttnChainOp> chain_ops;
  std::string cur = start;

  if (node->op_type() == "Reshape" && node->input_size() >= 1 &&
      node->input(0) == cur) {
    auto last_dim = ReshapeLastDim(*node, init_map);
    if (!last_dim || *last_dim != width) {
      return {std::nullopt, {}};
    }
    const std::string& shape_name = node->input(1);
    if (ConsumerCount(consumers_of, shape_name) != 1) {
      return {std::nullopt, {}};  // Shared shape constant -- mutating unsafe.
    }
    const std::string& out_name = node->output(0);
    if (ConsumerCount(consumers_of, out_name) != 1 ||
        graph_outputs.count(out_name)) {
      return {std::nullopt, {}};
    }
    chain_ops.push_back(AttnChainOp{node, shape_name});
    cur = out_name;
    node = consumers_of.at(cur)[0];
  }

  auto cm = MatchMatMulLikeRaw(*node);
  if (!cm || cm->x_name != cur) {
    return {std::nullopt, chain_ops};
  }
  auto wit = init_map.find(cm->w_name);
  if (wit == init_map.end() ||
      wit->second->data_type() != onnx::TensorProto::FLOAT ||
      wit->second->dims_size() != 2) {
    return {std::nullopt, chain_ops};
  }
  const int64_t k =
      cm->weight_transposed ? wit->second->dims(1) : wit->second->dims(0);
  if (k != width) {
    return {std::nullopt, chain_ops};
  }
  return {ConsumerMatch{node, cm->w_name, cm->weight_transposed}, chain_ops};
}

enum class AttnChainKind { kPlainAttention, kGqaLike };

// Either kind of matched attention block -- a single tagged struct rather
// than pruning.py's own _AttnLikeChain union of two dataclasses, since
// C++ has no direct analogue of Python's runtime isinstance() dispatch;
// `kind` says which of the two field groups below is populated.
struct AttnChain {
  AttnChainKind kind;
  onnx::NodeProto* node;
  // kPlainAttention fields (com.microsoft::Attention's merged QKV weight):
  std::string weight;
  std::optional<std::string> bias;
  int64_t num_heads = 0;
  int64_t nq = 0, nk = 0, nv = 0;
  // kGqaLike fields (GroupQueryAttention or plain ai.onnx::Attention's
  // separate Q/K/V producers):
  std::string q_weight, k_weight, v_weight;
  bool q_weight_transposed = false;
  bool k_weight_transposed = false;
  bool v_weight_transposed = false;
  std::optional<std::string> q_bias, k_bias, v_bias;
  int64_t kv_num_heads = 0;
  int64_t head_size = 0;
  std::string num_heads_attr = "num_heads";
  // Shared:
  std::vector<AttnChainOp> chain_ops;
  onnx::NodeProto* consumer_node = nullptr;
  std::string consumer_weight;
  bool consumer_weight_transposed = false;
};

std::vector<AttnChain> FindAttentionChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };

  std::vector<AttnChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = MatchAttentionProducer(*node, init_map);
    if (!info) {
      continue;
    }
    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    auto [consumer, chain_ops] = WalkToAttentionConsumer(
        out_name, init_map, consumers_of, graph_outputs, info->nv);
    if (!consumer) {
      continue;
    }

    AttnChain chain;
    chain.kind = AttnChainKind::kPlainAttention;
    chain.node = node;
    chain.weight = info->weight;
    chain.bias = info->bias;
    chain.num_heads = info->num_heads;
    chain.nq = info->nq;
    chain.nk = info->nk;
    chain.nv = info->nv;
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = consumer->weight_transposed;
    chains.push_back(std::move(chain));
  }
  return chains;
}

struct HeadCountsMatch {
  int64_t num_heads;
  int64_t kv_num_heads;
};

// If `node` is a com.microsoft::GroupQueryAttention node this pass can
// safely act on (separate Q/K/V inputs -- rules out the op's packed-QKV
// calling convention; no non-empty constant past_key/past_value), returns
// (num_heads, kv_num_heads). Mirrors pruning.py's own _match_gqa_producer.
std::optional<HeadCountsMatch> MatchGqaProducer(const onnx::NodeProto& node,
                                                const InitMap& init_map) {
  if (node.domain() != kComMicrosoftDomain ||
      node.op_type() != "GroupQueryAttention") {
    return std::nullopt;
  }
  if (node.input_size() < 7 || node.input(0).empty() || node.input(1).empty() ||
      node.input(2).empty()) {
    return std::nullopt;
  }
  int64_t num_heads = 0, kv_num_heads = 0;
  bool has_nh = false, has_kv = false;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "num_heads") {
      num_heads = attr.i();
      has_nh = true;
    } else if (attr.name() == "kv_num_heads") {
      kv_num_heads = attr.i();
      has_kv = true;
    }
  }
  if (!has_nh || !has_kv || num_heads <= 0 || kv_num_heads <= 0) {
    return std::nullopt;
  }
  if (num_heads % kv_num_heads != 0) {
    return std::nullopt;
  }
  for (int idx : {3, 4}) {  // past_key, past_value.
    if (node.input_size() <= idx || node.input(idx).empty()) {
      continue;
    }
    auto it = init_map.find(node.input(idx));
    if (it != init_map.end()) {
      int64_t prod = 1;
      for (int64_t d : it->second->dims()) {
        prod *= d;
      }
      if (prod > 0) {
        return std::nullopt;  // Non-empty KV-cache constant -- needs slicing.
      }
    }
  }
  return HeadCountsMatch{num_heads, kv_num_heads};
}

// If `node` is a plain ai.onnx::Attention node (domain "", opset 24+) this
// pass can safely act on, returns (q_num_heads, kv_num_heads). Mirrors
// pruning.py's own _match_onnx_attention_producer.
std::optional<HeadCountsMatch> MatchOnnxAttentionProducer(
    const onnx::NodeProto& node, const InitMap& init_map) {
  if (!node.domain().empty() || node.op_type() != "Attention") {
    return std::nullopt;
  }
  if (node.input_size() < 3 || node.input(0).empty() || node.input(1).empty() ||
      node.input(2).empty()) {
    return std::nullopt;
  }
  int64_t q_num_heads = 0, kv_num_heads = 0;
  bool has_q = false, has_kv = false;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "q_num_heads") {
      q_num_heads = attr.i();
      has_q = true;
    } else if (attr.name() == "kv_num_heads") {
      kv_num_heads = attr.i();
      has_kv = true;
    }
  }
  if (!has_q || !has_kv || q_num_heads <= 0 || kv_num_heads <= 0) {
    return std::nullopt;
  }
  if (q_num_heads % kv_num_heads != 0) {
    return std::nullopt;
  }
  for (int idx : {3, 4, 5}) {  // attn_mask, past_key, past_value.
    if (node.input_size() <= idx || node.input(idx).empty()) {
      continue;
    }
    auto it = init_map.find(node.input(idx));
    if (it != init_map.end()) {
      int64_t prod = 1;
      for (int64_t d : it->second->dims()) {
        prod *= d;
      }
      if (prod > 0) {
        return std::nullopt;  // Non-empty constant -- would need slicing.
      }
    }
  }
  return HeadCountsMatch{q_num_heads, kv_num_heads};
}

// Shared body for FindGqaChains/FindOnnxAttentionChains: both match a fused
// attention node fed by three separate, un-merged Q/K/V MatMul/vanilla-Gemm
// projections and prune it at whole-KV-group granularity, differing only in
// `match_producer` and which attribute holds the query head count
// (`num_heads_attr`). Mirrors pruning.py's own _find_separate_qkv_chains.
std::vector<AttnChain> FindSeparateQkvChains(
    onnx::GraphProto* graph,
    const std::function<std::optional<HeadCountsMatch>(
        const onnx::NodeProto&, const InitMap&)>& match_producer,
    const std::string& num_heads_attr) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
  }
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };

  std::vector<AttnChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = match_producer(*node, init_map);
    if (!info) {
      continue;
    }

    const std::string& q_name = node->input(0);
    const std::string& k_name = node->input(1);
    const std::string& v_name = node->input(2);
    if (q_name == k_name || q_name == v_name || k_name == v_name) {
      continue;  // Degenerate -- can't independently slice a shared producer.
    }
    if (!is_internal(q_name) || !is_internal(k_name) || !is_internal(v_name)) {
      continue;
    }
    auto qit = node_by_output.find(q_name);
    auto kit = node_by_output.find(k_name);
    auto vit = node_by_output.find(v_name);
    if (qit == node_by_output.end() || kit == node_by_output.end() ||
        vit == node_by_output.end()) {
      continue;
    }
    auto pq = MatchProducer(*qit->second, init_map);
    auto pk = MatchProducer(*kit->second, init_map);
    auto pv = MatchProducer(*vit->second, init_map);
    if (!pq || !pk || !pv) {
      continue;
    }
    if (pq->weight == pk->weight || pq->weight == pv->weight ||
        pk->weight == pv->weight || pq->n_channels % info->num_heads != 0 ||
        pk->n_channels % info->kv_num_heads != 0 ||
        pv->n_channels % info->kv_num_heads != 0) {
      continue;
    }
    const int64_t head_size = pq->n_channels / info->num_heads;
    if (head_size <= 0 || pk->n_channels / info->kv_num_heads != head_size ||
        pv->n_channels / info->kv_num_heads != head_size) {
      // fuse_gqa.h requires equal Q/K/V head_size; the plain ai.onnx op's
      // own more permissive schema allows V its own head_size, but this
      // shared, uniform-head_size body declines that composition rather
      // than mis-slicing it.
      continue;
    }

    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    // Both matched ops' raw output is Nq-wide (num_heads * head_size),
    // unlike plain com.microsoft::Attention's V-hidden-size-wide output.
    auto [consumer, chain_ops] = WalkToAttentionConsumer(
        out_name, init_map, consumers_of, graph_outputs, pq->n_channels);
    if (!consumer) {
      continue;
    }

    AttnChain chain;
    chain.kind = AttnChainKind::kGqaLike;
    chain.node = node;
    chain.q_weight = pq->weight;
    chain.q_bias = pq->bias;
    chain.q_weight_transposed = pq->weight_transposed;
    chain.k_weight = pk->weight;
    chain.k_bias = pk->bias;
    chain.k_weight_transposed = pk->weight_transposed;
    chain.v_weight = pv->weight;
    chain.v_bias = pv->bias;
    chain.v_weight_transposed = pv->weight_transposed;
    chain.num_heads = info->num_heads;
    chain.kv_num_heads = info->kv_num_heads;
    chain.head_size = head_size;
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = consumer->weight_transposed;
    chain.num_heads_attr = num_heads_attr;
    chains.push_back(std::move(chain));
  }
  return chains;
}

std::vector<AttnChain> FindGqaChains(onnx::GraphProto* graph) {
  return FindSeparateQkvChains(graph, MatchGqaProducer, "num_heads");
}

std::vector<AttnChain> FindOnnxAttentionChains(onnx::GraphProto* graph) {
  return FindSeparateQkvChains(graph, MatchOnnxAttentionProducer,
                               "q_num_heads");
}

// Column indices of every kept head's own head_size-wide block, in
// ascending head order -- mirrors pruning.py's own _head_column_indices.
std::vector<int64_t> HeadColumnIndices(const std::vector<int64_t>& keep_heads,
                                       int64_t head_size) {
  std::vector<int64_t> out;
  out.reserve(keep_heads.size() * static_cast<size_t>(head_size));
  for (int64_t h : keep_heads) {
    for (int64_t i = 0; i < head_size; ++i) {
      out.push_back(h * head_size + i);
    }
  }
  return out;
}

struct AppliedAttn {
  std::unordered_set<std::string> producer_weights;
  std::string consumer_weight;
  std::unordered_set<std::string> stale;
};

// Applies whole-head pruning to one matched com.microsoft::Attention block
// in place -- mirrors pruning.py's own _apply_one_plain_attention_chain
// (data-free/magnitude importance only; the Wanda variant's importance
// callback indirection is not ported, see this section's own scope note).
std::optional<AppliedAttn> ApplyOnePlainAttentionChain(
    std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    AttnChain& chain, double sparsity) {
  const int64_t h = chain.num_heads;
  const int64_t keep_count =
      std::max<int64_t>(1, h - std::llround(static_cast<double>(h) * sparsity));
  if (keep_count >= h) {
    return std::nullopt;
  }

  const int64_t dq = chain.nq / h, dk = chain.nk / h, dv = chain.nv / h;
  onnx::TensorProto* w_init = init_map.at(chain.weight);
  const int64_t K = w_init->dims(0);
  const int64_t total_n = w_init->dims(1);
  std::vector<float> w = ReadFloatTensor(*w_init);  // [K, total_n] row-major.

  std::vector<double> importance(static_cast<size_t>(h), 0.0);
  for (int64_t hh = 0; hh < h; ++hh) {
    double sq = 0.0;
    for (int64_t r = 0; r < K; ++r) {
      for (int64_t c = hh * dq; c < (hh + 1) * dq; ++c) {
        const double v = w[static_cast<size_t>(r * total_n + c)];
        sq += v * v;
      }
      for (int64_t c = chain.nq + hh * dk; c < chain.nq + (hh + 1) * dk; ++c) {
        const double v = w[static_cast<size_t>(r * total_n + c)];
        sq += v * v;
      }
      for (int64_t c = chain.nq + chain.nk + hh * dv;
           c < chain.nq + chain.nk + (hh + 1) * dv; ++c) {
        const double v = w[static_cast<size_t>(r * total_n + c)];
        sq += v * v;
      }
    }
    importance[static_cast<size_t>(hh)] = std::sqrt(sq);
  }

  std::vector<int64_t> keep_heads =
      TopKIndicesAscending(importance, keep_count);
  std::vector<int64_t> q_idx = HeadColumnIndices(keep_heads, dq);
  std::vector<int64_t> k_idx = HeadColumnIndices(keep_heads, dk);
  for (auto& x : k_idx) {
    x += chain.nq;
  }
  std::vector<int64_t> v_idx_local = HeadColumnIndices(keep_heads, dv);
  std::vector<int64_t> v_idx = v_idx_local;
  for (auto& x : v_idx) {
    x += chain.nq + chain.nk;
  }
  std::vector<int64_t> all_idx;
  all_idx.reserve(q_idx.size() + k_idx.size() + v_idx.size());
  all_idx.insert(all_idx.end(), q_idx.begin(), q_idx.end());
  all_idx.insert(all_idx.end(), k_idx.begin(), k_idx.end());
  all_idx.insert(all_idx.end(), v_idx.begin(), v_idx.end());

  std::vector<float> sliced_w = SliceAxis1(w, K, total_n, 1, all_idx);
  SetFloatTensorData(w_init, {K, static_cast<int64_t>(all_idx.size())},
                     sliced_w);
  if (chain.bias) {
    SliceLastAxis(init_map.at(*chain.bias), all_idx);
  }

  bool found_qkv = false;
  for (auto& attr : *chain.node->mutable_attribute()) {
    if (attr.name() == "num_heads") {
      attr.set_i(keep_count);
    } else if (attr.name() == "qkv_hidden_sizes") {
      found_qkv = true;
      attr.clear_ints();
      attr.add_ints(keep_count * dq);
      attr.add_ints(keep_count * dk);
      attr.add_ints(keep_count * dv);
    }
  }
  if (!found_qkv) {
    onnx::AttributeProto* attr = chain.node->add_attribute();
    attr->set_name("qkv_hidden_sizes");
    attr->set_type(onnx::AttributeProto::INTS);
    attr->add_ints(keep_count * dq);
    attr->add_ints(keep_count * dk);
    attr->add_ints(keep_count * dv);
  }

  SliceConsumerWeight(init_map.at(chain.consumer_weight),
                      chain.consumer_weight_transposed, v_idx_local, false);

  for (const auto& co : chain.chain_ops) {
    if (co.shape_name) {
      SetInt64TensorLastDim(init_map.at(*co.shape_name), keep_count * dv);
    }
  }

  AppliedAttn out;
  out.producer_weights = {chain.weight};
  out.consumer_weight = chain.consumer_weight;
  out.stale.insert(chain.node->output(0));
  for (const auto& co : chain.chain_ops) {
    out.stale.insert(co.node->output(0));
  }
  return out;
}

// Applies whole-KV-group pruning to one matched GroupQueryAttention or
// plain ai.onnx::Attention block in place -- mirrors pruning.py's own
// _apply_one_gqa_chain (data-free/magnitude importance only).
std::optional<AppliedAttn> ApplyOneGqaChain(
    std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    AttnChain& chain, double sparsity) {
  const int64_t h = chain.kv_num_heads;
  const int64_t keep_count =
      std::max<int64_t>(1, h - std::llround(static_cast<double>(h) * sparsity));
  if (keep_count >= h) {
    return std::nullopt;
  }

  const int64_t d = chain.head_size;
  const int64_t group_size = chain.num_heads / chain.kv_num_heads;

  onnx::TensorProto* wq_init = init_map.at(chain.q_weight);
  onnx::TensorProto* wk_init = init_map.at(chain.k_weight);
  onnx::TensorProto* wv_init = init_map.at(chain.v_weight);
  const std::vector<int64_t> wq_dims(wq_init->dims().begin(),
                                     wq_init->dims().end());
  const std::vector<int64_t> wk_dims(wk_init->dims().begin(),
                                     wk_init->dims().end());
  const std::vector<int64_t> wv_dims(wv_init->dims().begin(),
                                     wv_init->dims().end());
  std::vector<float> wq = ReadFloatTensor(*wq_init);
  std::vector<float> wk = ReadFloatTensor(*wk_init);
  std::vector<float> wv = ReadFloatTensor(*wv_init);

  // Bring each to [K, N] (reduction dim first, head columns last) -- the
  // *opposite* of SliceProducerWeight's "output channel first" convention,
  // matching pruning.py's own comment on this same transpose.
  const int64_t K = wq_dims[chain.q_weight_transposed ? 1 : 0];
  const int64_t Nq = wq_dims[chain.q_weight_transposed ? 0 : 1];
  const int64_t Nk = wk_dims[chain.k_weight_transposed ? 0 : 1];
  const int64_t Nv = wv_dims[chain.v_weight_transposed ? 0 : 1];
  std::vector<float> wq_kn = chain.q_weight_transposed
                                 ? TransposeFlat(wq, wq_dims[0], wq_dims[1])
                                 : wq;
  std::vector<float> wk_kn = chain.k_weight_transposed
                                 ? TransposeFlat(wk, wk_dims[0], wk_dims[1])
                                 : wk;
  std::vector<float> wv_kn = chain.v_weight_transposed
                                 ? TransposeFlat(wv, wv_dims[0], wv_dims[1])
                                 : wv;

  std::vector<double> importance(static_cast<size_t>(chain.kv_num_heads), 0.0);
  for (int64_t kv = 0; kv < chain.kv_num_heads; ++kv) {
    double sq = 0.0;
    for (int64_t r = 0; r < K; ++r) {
      for (int64_t g = kv * group_size; g < (kv + 1) * group_size; ++g) {
        for (int64_t c = g * d; c < (g + 1) * d; ++c) {
          const double v = wq_kn[static_cast<size_t>(r * Nq + c)];
          sq += v * v;
        }
      }
      for (int64_t c = kv * d; c < (kv + 1) * d; ++c) {
        const double v = wk_kn[static_cast<size_t>(r * Nk + c)];
        sq += v * v;
      }
      for (int64_t c = kv * d; c < (kv + 1) * d; ++c) {
        const double v = wv_kn[static_cast<size_t>(r * Nv + c)];
        sq += v * v;
      }
    }
    importance[static_cast<size_t>(kv)] = std::sqrt(sq);
  }

  std::vector<int64_t> keep_groups =
      TopKIndicesAscending(importance, keep_count);
  std::vector<int64_t> keep_q_heads;
  keep_q_heads.reserve(keep_groups.size() * static_cast<size_t>(group_size));
  for (int64_t g : keep_groups) {
    for (int64_t hh = g * group_size; hh < (g + 1) * group_size; ++hh) {
      keep_q_heads.push_back(hh);
    }
  }
  std::vector<int64_t> q_idx = HeadColumnIndices(keep_q_heads, d);
  std::vector<int64_t> kv_idx = HeadColumnIndices(keep_groups, d);

  SliceProducerWeight(wq_init, chain.q_weight_transposed, q_idx, false);
  SliceProducerWeight(wk_init, chain.k_weight_transposed, kv_idx, false);
  SliceProducerWeight(wv_init, chain.v_weight_transposed, kv_idx, false);
  if (chain.q_bias) {
    SliceLastAxis(init_map.at(*chain.q_bias), q_idx);
  }
  if (chain.k_bias) {
    SliceLastAxis(init_map.at(*chain.k_bias), kv_idx);
  }
  if (chain.v_bias) {
    SliceLastAxis(init_map.at(*chain.v_bias), kv_idx);
  }

  const int64_t new_kv_num_heads = keep_count;
  const int64_t new_num_heads = keep_count * group_size;
  for (auto& attr : *chain.node->mutable_attribute()) {
    if (attr.name() == chain.num_heads_attr) {
      attr.set_i(new_num_heads);
    } else if (attr.name() == "kv_num_heads") {
      attr.set_i(new_kv_num_heads);
    }
  }

  SliceConsumerWeight(init_map.at(chain.consumer_weight),
                      chain.consumer_weight_transposed, q_idx, false);

  for (const auto& co : chain.chain_ops) {
    if (co.shape_name) {
      SetInt64TensorLastDim(init_map.at(*co.shape_name), new_num_heads * d);
    }
  }

  AppliedAttn out;
  out.producer_weights = {chain.q_weight, chain.k_weight, chain.v_weight};
  out.consumer_weight = chain.consumer_weight;
  out.stale.insert(chain.node->output(0));
  for (const auto& co : chain.chain_ops) {
    out.stale.insert(co.node->output(0));
  }
  return out;
}

// Shared body dispatching each chain to ApplyOnePlainAttentionChain or
// ApplyOneGqaChain, mirroring pruning.py's own _apply_attention_chains
// (cross-chain touched-role bookkeeping, stale value_info cleanup).
void ApplyAttentionChains(onnx::GraphProto* graph,
                          std::vector<AttnChain>& chains, double sparsity) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }
  std::unordered_set<std::string> producer_touched, consumer_touched,
      stale_value_info;

  for (auto& chain : chains) {
    std::unordered_set<std::string> producer_names =
        chain.kind == AttnChainKind::kGqaLike
            ? std::unordered_set<std::string>{chain.q_weight, chain.k_weight,
                                              chain.v_weight}
            : std::unordered_set<std::string>{chain.weight};

    bool conflict = consumer_touched.count(chain.consumer_weight) != 0;
    for (const auto& w : producer_names) {
      if (producer_touched.count(w)) {
        conflict = true;
      }
    }
    if (conflict) {
      continue;
    }

    std::optional<AppliedAttn> applied =
        chain.kind == AttnChainKind::kGqaLike
            ? ApplyOneGqaChain(init_map, chain, sparsity)
            : ApplyOnePlainAttentionChain(init_map, chain, sparsity);
    if (!applied) {
      continue;
    }

    for (const auto& w : applied->producer_weights) {
      producer_touched.insert(w);
    }
    consumer_touched.insert(applied->consumer_weight);
    for (const auto& s : applied->stale) {
      stale_value_info.insert(s);
    }
  }

  if (!stale_value_info.empty()) {
    google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> kept;
    for (const auto& vi : graph->value_info()) {
      if (!stale_value_info.count(vi.name())) {
        *kept.Add() = vi;
      }
    }
    graph->mutable_value_info()->Swap(&kept);
  }
}

// --- Concat-merged (skip-connection) chains, mirroring pruning.py's own
// section of the same name -- see that section's own comment for the full
// reasoning: unlike Add, whose operands must agree on one shared surviving
// channel-index set, Concat's branches are structurally independent (each
// owns a fixed, disjoint offset range of the merged, pre-pruning tensor),
// so each branch can be ranked and pruned entirely on its own; only the
// shared downstream consumer's weight needs new slicing, at each branch's
// own fixed offset. Reuses the exact same backward walkers
// (WalkMatmulProducerBackward/WalkConvProducerBackward) the two residual
// sections above already built, including their fan-out resolution
// machinery.

std::optional<int64_t> ConcatAxis(const onnx::NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "axis") {
      return attr.i();
    }
  }
  return std::nullopt;  // Required attribute on Concat's own schema.
}

std::unordered_map<std::string, const onnx::ValueInfoProto*> ValueInfoByName(
    const onnx::GraphProto& graph) {
  std::unordered_map<std::string, const onnx::ValueInfoProto*> by_name;
  for (const auto& vi : graph.input()) {
    by_name[vi.name()] = &vi;
  }
  for (const auto& vi : graph.output()) {
    by_name[vi.name()] = &vi;
  }
  for (const auto& vi : graph.value_info()) {
    by_name[vi.name()] = &vi;
  }
  return by_name;
}

std::optional<int64_t> TensorRank(
    const std::string& name,
    const std::unordered_map<std::string, const onnx::ValueInfoProto*>&
        value_info_by_name) {
  auto it = value_info_by_name.find(name);
  if (it == value_info_by_name.end() || !it->second->type().has_tensor_type()) {
    return std::nullopt;
  }
  const auto& tensor_type = it->second->type().tensor_type();
  if (!tensor_type.has_shape()) {
    return std::nullopt;  // ONNX's own "rank not statically known" spelling.
  }
  return static_cast<int64_t>(tensor_type.shape().dim_size());
}

// True if `node`'s own `axis` attribute is confirmed to select the last
// axis of its operands -- `axis == -1` outright, or a positive `axis` only
// when at least one operand's rank is known and every operand with a known
// rank agrees `axis == rank - 1`. Mirrors pruning.py's own
// _concat_axis_is_last -- declined rather than guessed at otherwise.
bool ConcatAxisIsLast(
    const onnx::NodeProto& node,
    const std::unordered_map<std::string, const onnx::ValueInfoProto*>&
        value_info_by_name) {
  auto axis = ConcatAxis(node);
  if (!axis) {
    return false;
  }
  if (*axis < 0) {
    return *axis == -1;
  }
  std::optional<int64_t> known_rank;
  for (const auto& operand : node.input()) {
    auto rank = TensorRank(operand, value_info_by_name);
    if (!rank) {
      continue;
    }
    if (!known_rank) {
      known_rank = rank;
    } else if (*rank != *known_rank) {
      return false;  // Operands disagree -- decline rather than guess.
    }
  }
  if (!known_rank) {
    return false;  // No operand's rank is known -- decline rather than guess.
  }
  return *axis == *known_rank - 1;
}

// One resolved operand of a matched Concat merge group -- mirrors
// pruning.py's own _ConcatBranch. Unlike an Add/SkipLayerNormalization
// residual merge's operands (Chain::producers, all pruned to one *shared*
// keep index set), every ConcatBranch in a ConcatChain is pruned to its own
// *independent* keep set.
struct ConcatBranch {
  // One producer for a plain branch; more than one when this branch
  // instead resolves through a composed residual/merge group or a gated
  // (SwiGLU/GeGLU) combine -- see this section's own comment on the "add"/
  // "gated" outcomes.
  std::vector<Producer> producers;
  // Ops between the producer's own raw output (or, for a composed group,
  // the group's own internal wiring) and this branch's own Concat operand.
  std::vector<ChainOp> pre_ops;
  // Depthwise Conv pass-through hops crossed on this branch (Conv branches
  // only; always empty for a MatMul/Gemm branch).
  std::vector<ConvPassThrough> conv_pass_through;
  int64_t n_channels = 0;
  // This branch's fixed offset into the merged (pre-pruning) channel
  // range, in Concat operand order.
  int64_t offset = 0;
  // The tensor name actually feeding the Concat node at this operand
  // position -- the same probe point apply_structured_wanda_pruning's own
  // Concat-branch activation capture would use, though that calibrated
  // variant is out of this port's scope.
  std::string operand_name;
};

struct ConcatChain {
  std::vector<ConcatBranch> branches;
  onnx::NodeProto* concat_node = nullptr;
  // Ops between the Concat node's own output and the real consumer.
  std::vector<ChainOp> chain_ops;
  onnx::NodeProto* consumer_node = nullptr;
  std::string consumer_weight;
  bool consumer_weight_transposed = false;
  bool consumer_is_conv = false;
  int64_t n_channels = 0;  // Sum of every branch's own n_channels.
  // Depthwise Conv hops crossed between the Concat node and the real
  // consumer (Conv chains only).
  std::vector<ConvPassThrough> conv_pass_through;
};

// True if any tensor a Concat branch's own backward walk crossed -- `start`
// (the branch operand) through the real producer's own output -- has more
// than the one in-group forward consumer the walk itself already accounts
// for. Mirrors pruning.py's own _branch_walk_has_fanout: the backward
// walkers no longer reject a multi-consumer tensor mid-walk themselves
// (that relaxation exists for the residual/fan-out case, resolved
// explicitly afterwards), but a Concat branch has no such resolution -- a
// branch that fans out to another consumer is declined outright.
bool BranchWalkHasFanout(
    const std::string& start,
    const std::vector<std::pair<std::string, onnx::NodeProto*>>& edges,
    const ConsumerMap& consumers_of, onnx::NodeProto* forward_node) {
  onnx::NodeProto* prev_consumer = forward_node;
  std::string cur = start;
  for (const auto& e : edges) {
    auto cit = consumers_of.find(cur);
    if (cit == consumers_of.end() || cit->second.size() != 1 ||
        cit->second[0] != prev_consumer) {
      return true;
    }
    prev_consumer = e.second;
    cur = e.first;
  }
  auto cit = consumers_of.find(cur);
  return cit == consumers_of.end() || cit->second.size() != 1 ||
         cit->second[0] != prev_consumer;
}

struct ResolvedMatmulResidualGroup {
  std::vector<Producer> leaf_producers;
  std::vector<ChainOp> pre_chain_ops;
  int64_t n_channels = 0;
  std::vector<std::string> backbone_tensors;
  std::unordered_map<std::string, std::unordered_set<onnx::NodeProto*>>
      accounted;
};

// Resolves `root` (an Add/SkipLayerNormalization merge a Concat branch's
// own backward walk bottomed out at -- a kAdd outcome from
// WalkMatmulProducerBackward) and its whole transitively-connected
// residual/merge group, mirroring FindMatmulResidualChains's own per-group
// union-find loop exactly (same per-member operand resolution, same "any
// operand fails, the whole group declines" bar, same post-hoc bias/scale-
// constant re-validation) but scoped to just `root`'s own component --
// reached by a plain worklist walk outward from `root` rather than a
// global union-find, since `root` is already known to be the group's own
// sink. Mirrors pruning.py's own
// _resolve_matmul_residual_group_for_concat.
std::optional<ResolvedMatmulResidualGroup> ResolveMatmulResidualGroupForConcat(
    onnx::NodeProto* root,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const InitMap& init_map, const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs) {
  std::vector<onnx::NodeProto*> visited{root};
  std::unordered_set<onnx::NodeProto*> visited_ids{root};
  std::unordered_set<onnx::NodeProto*> referenced;
  std::vector<Producer> leaf_producers;
  std::unordered_set<int64_t> n_channels_set;
  std::vector<ChainOp> pre_chain_ops;
  std::vector<std::string> backbone_tensors;
  std::unordered_map<std::string, std::unordered_set<onnx::NodeProto*>>
      accounted;
  auto mark_backbone = [&](const std::string& tensor, onnx::NodeProto* node) {
    if (!accounted.count(tensor)) {
      backbone_tensors.push_back(tensor);
    }
    accounted[tensor].insert(node);
  };

  for (size_t i = 0; i < visited.size(); ++i) {
    onnx::NodeProto* merge_node = visited[i];
    auto match =
        MatchResidualMerge(merge_node, init_map, consumers_of, graph_outputs);
    if (!match) {
      return std::nullopt;  // Defensive -- every member was matched already.
    }
    pre_chain_ops.insert(pre_chain_ops.end(), match->extra_ops.begin(),
                         match->extra_ops.end());
    const std::string operands[2] = {match->input_name, match->skip_name};
    for (const auto& operand : operands) {
      mark_backbone(operand, merge_node);
      MatMulBackwardEdge edge = WalkMatmulProducerBackward(
          operand, node_by_output, init_map, consumers_of, graph_outputs,
          kMaxChainHops);
      for (const auto& e : edge.edges) {
        mark_backbone(e.first, e.second);
      }
      pre_chain_ops.insert(pre_chain_ops.end(), edge.chain_ops.begin(),
                           edge.chain_ops.end());
      if (edge.kind == BackwardEdgeKind::kProducer) {
        leaf_producers.push_back(edge.producer);
        n_channels_set.insert(edge.n_channels);
      } else if (edge.kind == BackwardEdgeKind::kAdd) {
        referenced.insert(edge.add_node);
        if (!visited_ids.count(edge.add_node)) {
          visited_ids.insert(edge.add_node);
          visited.push_back(edge.add_node);
        }
      } else {
        return std::nullopt;  // kFail (producer_infos not passed, so kGated
                              // is unreachable here) -- decline.
      }
    }
  }

  if (n_channels_set.size() != 1) {
    return std::nullopt;  // Branches disagree on channel count -- decline.
  }
  const int64_t n_channels = *n_channels_set.begin();

  for (const auto& co : pre_chain_ops) {
    if (co.const_name &&
        init_map.at(*co.const_name)
                ->dims(init_map.at(*co.const_name)->dims_size() - 1) !=
            n_channels) {
      return std::nullopt;
    }
  }

  std::vector<onnx::NodeProto*> sinks;
  for (auto* n : visited) {
    if (!referenced.count(n)) {
      sinks.push_back(n);
    }
  }
  if (sinks.size() != 1 || sinks[0] != root) {
    return std::nullopt;  // Not a single linear chain rooted at root.
  }

  std::unordered_set<std::string> seen_weights;
  for (const auto& p : leaf_producers) {
    if (!seen_weights.insert(p.weight).second) {
      return std::nullopt;  // Degenerate -- the same producer named twice.
    }
  }

  return ResolvedMatmulResidualGroup{
      std::move(leaf_producers), std::move(pre_chain_ops), n_channels,
      std::move(backbone_tensors), std::move(accounted)};
}

struct ResolvedConvResidualGroup {
  std::vector<Producer> leaf_producers;
  std::vector<ConvPassThrough> pass_through;
  std::vector<onnx::NodeProto*> unary_ops;
  int64_t n_channels = 0;
  std::vector<std::string> backbone_tensors;
  std::unordered_map<std::string, std::unordered_set<onnx::NodeProto*>>
      accounted;
};

// The Conv analogue of ResolveMatmulResidualGroupForConcat -- see its own
// docstring for the shared reasoning this mirrors exactly (only the
// per-member walker differs: WalkConvProducerBackward instead of
// WalkMatmulProducerBackward, and there is no SkipLayerNormalization
// analogue or per-channel bias/scale hop to re-validate on the Conv side,
// only depthwise pass-through hops). Mirrors pruning.py's own
// _resolve_conv_residual_group_for_concat.
std::optional<ResolvedConvResidualGroup> ResolveConvResidualGroupForConcat(
    onnx::NodeProto* root,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const InitMap& init_map,
    const std::unordered_set<std::string>& graph_outputs) {
  std::vector<onnx::NodeProto*> visited{root};
  std::unordered_set<onnx::NodeProto*> visited_ids{root};
  std::unordered_set<onnx::NodeProto*> referenced;
  std::vector<Producer> leaf_producers;
  std::unordered_set<int64_t> n_channels_set;
  std::vector<ConvPassThrough> pass_through;
  std::vector<onnx::NodeProto*> unary_ops;
  std::vector<std::string> backbone_tensors;
  std::unordered_map<std::string, std::unordered_set<onnx::NodeProto*>>
      accounted;
  auto mark_backbone = [&](const std::string& tensor, onnx::NodeProto* node) {
    if (!accounted.count(tensor)) {
      backbone_tensors.push_back(tensor);
    }
    accounted[tensor].insert(node);
  };

  for (size_t i = 0; i < visited.size(); ++i) {
    onnx::NodeProto* add_node = visited[i];
    if (!IsEligibleAddMerge(*add_node, init_map)) {
      return std::nullopt;  // Defensive -- every member was matched already.
    }
    for (const auto& operand : add_node->input()) {
      mark_backbone(operand, add_node);
      ConvBackwardEdge edge = WalkConvProducerBackward(
          operand, node_by_output, init_map, graph_outputs, kMaxChainHops);
      for (const auto& e : edge.edges) {
        mark_backbone(e.first, e.second);
      }
      pass_through.insert(pass_through.end(), edge.pass_through.begin(),
                          edge.pass_through.end());
      unary_ops.insert(unary_ops.end(), edge.unary_ops.begin(),
                       edge.unary_ops.end());
      if (edge.kind == BackwardEdgeKind::kProducer) {
        leaf_producers.push_back(edge.producer);
        n_channels_set.insert(edge.n_channels);
      } else if (edge.kind == BackwardEdgeKind::kAdd) {
        referenced.insert(edge.add_node);
        if (!visited_ids.count(edge.add_node)) {
          visited_ids.insert(edge.add_node);
          visited.push_back(edge.add_node);
        }
      } else {
        return std::nullopt;  // kFail -- decline the whole group.
      }
    }
  }

  if (n_channels_set.size() != 1) {
    return std::nullopt;
  }
  const int64_t n_channels = *n_channels_set.begin();

  for (const auto& hop : pass_through) {
    if (init_map.at(hop.weight)->dims(0) != n_channels) {
      return std::nullopt;
    }
  }

  std::vector<onnx::NodeProto*> sinks;
  for (auto* n : visited) {
    if (!referenced.count(n)) {
      sinks.push_back(n);
    }
  }
  if (sinks.size() != 1 || sinks[0] != root) {
    return std::nullopt;
  }

  std::unordered_set<std::string> seen_weights;
  for (const auto& p : leaf_producers) {
    if (!seen_weights.insert(p.weight).second) {
      return std::nullopt;
    }
  }

  return ResolvedConvResidualGroup{
      std::move(leaf_producers),   std::move(pass_through),
      std::move(unary_ops),        n_channels,
      std::move(backbone_tensors), std::move(accounted)};
}

// Finds MatMul/Gemm Concat-merged skip connections -- see this section's
// own comment. Every operand of a last-axis Concat is resolved backward via
// WalkMatmulProducerBackward to a real producer (kProducer), an eligible
// residual/SkipLayerNormalization merge's whole group (kAdd, composed via
// ResolveMatmulResidualGroupForConcat), or a gated (SwiGLU/GeGLU-style) Mul
// of two non-constant operands feeding this Concat operand directly
// (kGated, resolved by WalkMatmulProducerBackward itself). If any operand
// fails to resolve, fans out elsewhere, or two operands/producers name the
// same weight, the whole Concat node is declined -- never partially pruned.
// Mirrors pruning.py's own _find_matmul_concat_chains.
std::vector<ConcatChain> FindMatmulConcatChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
  }
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  auto value_info_by_name = ValueInfoByName(*graph);
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };

  std::unordered_map<std::string, FullProducerMatch> producer_infos;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = MatchProducer(*node, init_map);
    if (info) {
      producer_infos[node->output(0)] =
          FullProducerMatch{node, info->weight, info->weight_transposed,
                            info->bias, info->n_channels};
    }
  }

  std::vector<ConcatChain> chains;
  for (int ni = 0; ni < graph->node_size(); ++ni) {
    onnx::NodeProto* node = graph->mutable_node(ni);
    if (node->op_type() != "Concat" || node->input_size() < 2 ||
        node->output_size() != 1) {
      continue;
    }
    if (!ConcatAxisIsLast(*node, value_info_by_name)) {
      continue;
    }
    {
      std::unordered_set<std::string> uniq(node->input().begin(),
                                           node->input().end());
      if (static_cast<int>(uniq.size()) != node->input_size()) {
        continue;  // Degenerate -- the same tensor concatenated with itself.
      }
    }

    std::vector<ConcatBranch> branches;
    std::unordered_set<std::string> seen_weights;
    int64_t offset = 0;
    bool declined = false;
    for (const auto& operand : node->input()) {
      MatMulBackwardEdge edge = WalkMatmulProducerBackward(
          operand, node_by_output, init_map, consumers_of, graph_outputs,
          kMaxChainHops, &producer_infos);
      if (edge.kind == BackwardEdgeKind::kFail) {
        declined = true;
        break;
      }
      if (BranchWalkHasFanout(operand, edge.edges, consumers_of, node)) {
        declined = true;
        break;
      }
      if (edge.kind == BackwardEdgeKind::kGated) {
        if (edge.producer.weight == edge.producer_b.weight ||
            seen_weights.count(edge.producer.weight) ||
            seen_weights.count(edge.producer_b.weight)) {
          declined = true;
          break;
        }
        seen_weights.insert(edge.producer.weight);
        seen_weights.insert(edge.producer_b.weight);
        ConcatBranch branch;
        branch.producers = {edge.producer, edge.producer_b};
        branch.pre_ops = edge.chain_ops;
        branch.n_channels = edge.n_channels;
        branch.offset = offset;
        branch.operand_name = operand;
        offset += edge.n_channels;
        branches.push_back(std::move(branch));
        continue;
      }
      if (edge.kind == BackwardEdgeKind::kAdd) {
        auto resolved = ResolveMatmulResidualGroupForConcat(
            edge.add_node, node_by_output, init_map, consumers_of,
            graph_outputs);
        if (!resolved) {
          declined = true;
          break;
        }
        auto extra = ResolveMatmulFanoutBranches(
            resolved->backbone_tensors, resolved->accounted, init_map,
            consumers_of, graph_outputs, resolved->n_channels);
        // Only an exactly-empty result confirms the group has no consumer
        // anywhere else -- see this section's own comment.
        if (!extra || !extra->empty()) {
          declined = true;
          break;
        }
        bool dup = false;
        for (const auto& p : resolved->leaf_producers) {
          if (seen_weights.count(p.weight)) {
            dup = true;
            break;
          }
        }
        if (dup) {
          declined = true;
          break;
        }
        for (const auto& p : resolved->leaf_producers) {
          seen_weights.insert(p.weight);
        }
        ConcatBranch branch;
        branch.producers = resolved->leaf_producers;
        branch.pre_ops = resolved->pre_chain_ops;
        branch.pre_ops.insert(branch.pre_ops.end(), edge.chain_ops.begin(),
                              edge.chain_ops.end());
        branch.n_channels = resolved->n_channels;
        branch.offset = offset;
        branch.operand_name = operand;
        offset += resolved->n_channels;
        branches.push_back(std::move(branch));
        continue;
      }
      // kProducer.
      if (seen_weights.count(edge.producer.weight)) {
        declined = true;
        break;
      }
      seen_weights.insert(edge.producer.weight);
      ConcatBranch branch;
      branch.producers = {edge.producer};
      branch.pre_ops = edge.chain_ops;
      branch.n_channels = edge.n_channels;
      branch.offset = offset;
      branch.operand_name = operand;
      offset += edge.n_channels;
      branches.push_back(std::move(branch));
    }
    if (declined) {
      continue;
    }

    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    const int64_t total_n = offset;
    auto [consumer, fwd_chain_ops] =
        WalkToConsumer(out_name, init_map, consumers_of, graph_outputs, total_n,
                       kMaxChainHops);
    if (!consumer) {
      continue;
    }

    ConcatChain chain;
    chain.branches = std::move(branches);
    chain.concat_node = node;
    chain.chain_ops = std::move(fwd_chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = consumer->weight_transposed;
    chain.consumer_is_conv = false;
    chain.n_channels = total_n;
    chains.push_back(std::move(chain));
  }
  return chains;
}

// The Conv analogue of FindMatmulConcatChains: every operand of a
// channel-axis Concat (axis in {1, -3}) is resolved backward via
// WalkConvProducerBackward to either a real group=1 Conv producer
// (kProducer) or an eligible Add merge's whole group (kAdd, composed via
// ResolveConvResidualGroupForConcat). The consumer must itself be an
// ordinary (group=1) Conv. Mirrors pruning.py's own
// _find_conv_concat_chains.
std::vector<ConcatChain> FindConvConcatChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
  }
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };

  std::vector<ConcatChain> chains;
  for (int ni = 0; ni < graph->node_size(); ++ni) {
    onnx::NodeProto* node = graph->mutable_node(ni);
    if (node->op_type() != "Concat" || node->input_size() < 2 ||
        node->output_size() != 1) {
      continue;
    }
    auto axis = ConcatAxis(*node);
    if (!axis || (*axis != 1 && *axis != -3)) {
      continue;
    }
    {
      std::unordered_set<std::string> uniq(node->input().begin(),
                                           node->input().end());
      if (static_cast<int>(uniq.size()) != node->input_size()) {
        continue;
      }
    }

    std::vector<ConcatBranch> branches;
    std::unordered_set<std::string> seen_weights;
    int64_t offset = 0;
    bool declined = false;
    for (const auto& operand : node->input()) {
      ConvBackwardEdge edge = WalkConvProducerBackward(
          operand, node_by_output, init_map, graph_outputs, kMaxChainHops);
      if (edge.kind == BackwardEdgeKind::kFail) {
        declined = true;
        break;
      }
      if (BranchWalkHasFanout(operand, edge.edges, consumers_of, node)) {
        declined = true;
        break;
      }
      if (edge.kind == BackwardEdgeKind::kAdd) {
        auto resolved = ResolveConvResidualGroupForConcat(
            edge.add_node, node_by_output, init_map, graph_outputs);
        if (!resolved) {
          declined = true;
          break;
        }
        auto extra = ResolveConvFanoutBranches(
            resolved->backbone_tensors, resolved->accounted, init_map,
            consumers_of, graph_outputs, resolved->n_channels);
        if (!extra || !extra->empty()) {
          declined = true;
          break;
        }
        bool dup = false;
        for (const auto& p : resolved->leaf_producers) {
          if (seen_weights.count(p.weight)) {
            dup = true;
            break;
          }
        }
        if (dup) {
          declined = true;
          break;
        }
        for (const auto& p : resolved->leaf_producers) {
          seen_weights.insert(p.weight);
        }
        ConcatBranch branch;
        branch.producers = resolved->leaf_producers;
        for (auto* op : resolved->unary_ops) {
          branch.pre_ops.push_back(ChainOp{op, std::nullopt});
        }
        for (auto* op : edge.unary_ops) {
          branch.pre_ops.push_back(ChainOp{op, std::nullopt});
        }
        branch.conv_pass_through = resolved->pass_through;
        branch.conv_pass_through.insert(branch.conv_pass_through.end(),
                                        edge.pass_through.begin(),
                                        edge.pass_through.end());
        branch.n_channels = resolved->n_channels;
        branch.offset = offset;
        branch.operand_name = operand;
        offset += resolved->n_channels;
        branches.push_back(std::move(branch));
        continue;
      }
      // kProducer.
      if (seen_weights.count(edge.producer.weight)) {
        declined = true;
        break;
      }
      seen_weights.insert(edge.producer.weight);
      ConcatBranch branch;
      branch.producers = {edge.producer};
      for (auto* op : edge.unary_ops) {
        branch.pre_ops.push_back(ChainOp{op, std::nullopt});
      }
      branch.conv_pass_through = edge.pass_through;
      branch.n_channels = edge.n_channels;
      branch.offset = offset;
      branch.operand_name = operand;
      offset += edge.n_channels;
      branches.push_back(std::move(branch));
    }
    if (declined) {
      continue;
    }

    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    const int64_t total_n = offset;
    // `recognize_group_norm` stays at its default (false) here too -- a
    // Concat-merged chain's own forward consumer walk never recognizes a
    // mid-chain GroupNorm hop either, mirroring pruning.py's own
    // FindConvConcatChains-equivalent walk (GroupNorm pass-through is
    // FindConvChains-only, see Chain::group_norm's own comment).
    auto [consumer, fwd_chain_ops, fwd_pass_through, fwd_group_norm] =
        WalkToConvConsumer(out_name, init_map, consumers_of, graph_outputs,
                           total_n, kMaxChainHops);
    (void)fwd_group_norm;  // Always nullopt -- see comment above.
    if (!consumer) {
      continue;
    }
    if (consumer->group != 1) {
      continue;  // See this section's own comment -- grouped consumer declined.
    }

    ConcatChain chain;
    chain.branches = std::move(branches);
    chain.concat_node = node;
    chain.chain_ops = std::move(fwd_chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = false;
    chain.consumer_is_conv = true;
    chain.n_channels = total_n;
    chain.conv_pass_through = std::move(fwd_pass_through);
    chains.push_back(std::move(chain));
  }
  return chains;
}

// The Concat-merged analogue of ApplyChains: computes one *independent*
// keep index set per branch (a plain branch's own combined-importance L2
// norm, root-sum-square across every producer in a composed/gated
// branch), then slices the shared downstream consumer once, by the
// concatenation of every branch's own keep, each shifted by its own fixed
// offset. `touched` is the same TouchedState a sibling ApplyChains call
// shares. Mirrors pruning.py's own _apply_concat_chains.
void ApplyConcatChains(onnx::GraphProto* graph,
                       std::vector<ConcatChain>& chains, double sparsity,
                       TouchedState& touched) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }

  for (auto& chain : chains) {
    std::unordered_set<std::string> producer_weights;
    size_t n_producers = 0;
    for (const auto& b : chain.branches) {
      for (const auto& p : b.producers) {
        producer_weights.insert(p.weight);
        ++n_producers;
      }
    }
    if (producer_weights.size() != n_producers) {
      continue;  // Degenerate -- two producers naming the same weight.
    }

    std::unordered_set<std::string> conv_hop_weights;
    size_t n_conv_hops = chain.conv_pass_through.size();
    for (const auto& h : chain.conv_pass_through) {
      conv_hop_weights.insert(h.weight);
    }
    for (const auto& b : chain.branches) {
      n_conv_hops += b.conv_pass_through.size();
      for (const auto& h : b.conv_pass_through) {
        conv_hop_weights.insert(h.weight);
      }
    }
    if (conv_hop_weights.size() != n_conv_hops) {
      continue;  // Degenerate -- the same depthwise weight named twice.
    }

    std::unordered_set<std::string> consts;
    for (const auto& b : chain.branches) {
      for (const auto& p : b.producers) {
        if (p.bias) {
          consts.insert(*p.bias);
        }
      }
      for (const auto& co : b.pre_ops) {
        if (co.const_name) {
          consts.insert(*co.const_name);
        }
      }
    }
    for (const auto& co : chain.chain_ops) {
      if (co.const_name) {
        consts.insert(*co.const_name);
      }
    }

    bool conflict = touched.consumer.count(chain.consumer_weight) != 0;
    for (const auto& w : producer_weights) {
      if (touched.producer.count(w)) {
        conflict = true;
      }
    }
    for (const auto& c : consts) {
      if (touched.const_names.count(c)) {
        conflict = true;
      }
    }
    for (const auto& w : conv_hop_weights) {
      if (touched.conv_hop.count(w)) {
        conflict = true;
      }
    }
    if (conflict) {
      continue;  // A shared/tied initializer another chain already resized.
    }

    std::vector<std::vector<int64_t>> branch_keeps;
    branch_keeps.reserve(chain.branches.size());
    bool any_pruned = false;
    for (const auto& b : chain.branches) {
      const int64_t n = b.n_channels;
      const int64_t keep_count = std::max<int64_t>(
          1, n - std::llround(static_cast<double>(n) * sparsity));
      if (keep_count >= n) {
        std::vector<int64_t> full(static_cast<size_t>(n));
        std::iota(full.begin(), full.end(), int64_t{0});
        branch_keeps.push_back(std::move(full));
        continue;
      }
      any_pruned = true;
      std::vector<std::vector<float>> w_arrays_nk;
      for (const auto& p : b.producers) {
        onnx::TensorProto* wt = init_map.at(p.weight);
        std::vector<int64_t> dims(wt->dims().begin(), wt->dims().end());
        std::vector<float> data = ReadFloatTensor(*wt);
        if (p.is_conv || p.weight_transposed) {
          w_arrays_nk.push_back(std::move(data));
        } else {
          w_arrays_nk.push_back(TransposeFlat(data, dims[0], dims[1]));
        }
      }
      std::vector<double> importance(static_cast<size_t>(n), 0.0);
      for (const auto& w_nk : w_arrays_nk) {
        const int64_t k = static_cast<int64_t>(w_nk.size()) / n;
        for (int64_t c = 0; c < n; ++c) {
          double sq = 0.0;
          for (int64_t j = 0; j < k; ++j) {
            const double v = w_nk[static_cast<size_t>(c * k + j)];
            sq += v * v;
          }
          importance[static_cast<size_t>(c)] += sq;
        }
      }
      for (double& v : importance) {
        v = std::sqrt(v);
      }
      branch_keeps.push_back(TopKIndicesAscending(importance, keep_count));
    }

    if (!any_pruned) {
      continue;  // Every branch rounds down to a no-op -- nothing to do.
    }

    for (size_t bi = 0; bi < chain.branches.size(); ++bi) {
      const ConcatBranch& b = chain.branches[bi];
      const std::vector<int64_t>& keep = branch_keeps[bi];
      if (static_cast<int64_t>(keep.size()) == b.n_channels) {
        continue;  // This branch's own sparsity rounded to a no-op.
      }
      for (const auto& p : b.producers) {
        SliceProducerWeight(init_map.at(p.weight), p.weight_transposed, keep,
                            p.is_conv);
        if (p.bias) {
          SliceLastAxis(init_map.at(*p.bias), keep);
        }
      }
      for (const auto& co : b.pre_ops) {
        if (co.const_name) {
          SliceLastAxis(init_map.at(*co.const_name), keep);
        }
      }
      for (const auto& hop : b.conv_pass_through) {
        SliceProducerWeight(init_map.at(hop.weight), false, keep, true);
        if (hop.bias) {
          SliceLastAxis(init_map.at(*hop.bias), keep);
        }
        if (hop.node->op_type() == "Conv") {
          SetOrAddIntAttr(hop.node, "group", static_cast<int64_t>(keep.size()));
        }
      }
    }

    std::vector<int64_t> global_keep;
    for (size_t bi = 0; bi < chain.branches.size(); ++bi) {
      for (int64_t k : branch_keeps[bi]) {
        global_keep.push_back(k + chain.branches[bi].offset);
      }
    }

    for (const auto& co : chain.chain_ops) {
      if (co.const_name) {
        SliceLastAxis(init_map.at(*co.const_name), global_keep);
      }
    }
    for (const auto& hop : chain.conv_pass_through) {
      SliceProducerWeight(init_map.at(hop.weight), false, global_keep, true);
      if (hop.bias) {
        SliceLastAxis(init_map.at(*hop.bias), global_keep);
      }
      if (hop.node->op_type() == "Conv") {
        SetOrAddIntAttr(hop.node, "group",
                        static_cast<int64_t>(global_keep.size()));
      }
    }

    SliceConsumerWeight(init_map.at(chain.consumer_weight),
                        chain.consumer_weight_transposed, global_keep,
                        chain.consumer_is_conv);

    for (const auto& w : producer_weights) {
      touched.producer.insert(w);
    }
    touched.consumer.insert(chain.consumer_weight);
    for (const auto& c : consts) {
      touched.const_names.insert(c);
    }
    for (const auto& w : conv_hop_weights) {
      touched.conv_hop.insert(w);
    }
    touched.stale_value_info.insert(chain.concat_node->output(0));
    for (const auto& b : chain.branches) {
      for (const auto& p : b.producers) {
        touched.stale_value_info.insert(p.node->output(0));
      }
      for (const auto& co : b.pre_ops) {
        touched.stale_value_info.insert(co.node->output(0));
      }
      for (const auto& hop : b.conv_pass_through) {
        touched.stale_value_info.insert(hop.node->output(0));
      }
    }
    for (const auto& co : chain.chain_ops) {
      touched.stale_value_info.insert(co.node->output(0));
    }
    for (const auto& hop : chain.conv_pass_through) {
      touched.stale_value_info.insert(hop.node->output(0));
    }
  }
}

// --- Split-merged (fused gate_up_proj) gated FFN chains, mirroring
// pruning.py's own "Split-merged (fused gate_up_proj) gated FFN chains"
// section -- _SplitGatedChain/_split_axis/_split_axis_is_last/
// _split_explicit_sizes/_trace_split_half_backward/_find_split_gated_chains/
// _apply_split_gated_chains -----------------------------------------------
//
// FindGatedChains above only recognizes the TWO-SEPARATE-PRODUCER shape:
// Mul(gate_proj(x), up_proj(x)) with two distinct MatMul/Gemm weight
// tensors. Real, currently-shipped Phi-3/Phi-3.5 (onnxruntime-genai) exports
// use a different, equally common shape instead: ONE gate_up_proj MatMul/
// Gemm producer (2*H output columns) -> Split(axis=-1, two equal H-wide
// outputs) -> (gate, up) -> act(gate) * up (or native SwiGLU) -> down_proj.
// Unlike the two-producer case, gate and up are two HALVES of the SAME
// physical weight: columns [0, H) are gate, columns [H, 2H) are up (Split's
// own output-order guarantee). "Neuron" i of the intermediate dimension is
// therefore represented by BOTH column i and column H + i of the one
// producer weight -- they must always be kept or dropped TOGETHER, so a
// single `keep` set is chosen once over `h` (not `2h`) candidates and
// applied at both fixed offsets of that one tensor -- see pruning.py's own
// section comment for the full shape derivation, the exact supported/
// declined boundary (MatMul/Gemm only, no quantized producer/consumer, the
// producer's raw output must feed Split directly with no bias-Add hop in
// between, `global_sparsity` mode excludes this family the same way an
// ordinary gated pair already is), and the worked InferenceSession-verified
// correctness argument -- this port covers the identical scope, kept
// deliberately narrower only where the rest of this file already is (no
// recursion into `If` subgraphs, matching every other finder here).
//
// Kept as its own struct (SplitGatedChain) rather than reusing Chain: the
// shape genuinely differs from every other family this file matches --
// exactly one physical producer tensor split by a dedicated Split node, one
// `h`-wide keep set applied at two fixed offsets of that one tensor, plus
// the Split node's own size bookkeeping -- none of which fits Chain's
// "N independent producers, each pruned to the same, un-offset keep set"
// shape. Applied by its own ApplySplitGatedChains, mirroring pruning.py's
// own deliberate `_apply_chains`/`_apply_split_gated_chains` split (called
// from ApplyStructuredPruning alongside, not instead of, ApplyChains) for
// the identical reason: ApplyChains' shared per-chain body (one keep set,
// applied unmodified to every producer/consumer weight it holds) has no
// hook for "the same tensor, sliced at two different offsets" or for a
// Split node's own attribute/input rewrite, and retrofitting one would
// complicate every other chain family's own straight-line path for a
// single caller.

enum class SplitSizesKind { kAuto, kAttr, kInput };

// node.attribute("axis"), Split's own schema default (0) -- unlike Concat's
// *required* attribute, so an un-annotated Split still has a real axis to
// check against (never itself grounds for decline). Mirrors pruning.py's
// own _split_axis.
int64_t SplitAxis(const onnx::NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "axis") {
      return attr.i();
    }
  }
  return 0;
}

// The Split-node analogue of ConcatAxisIsLast, with a single operand
// (Split has exactly one data input) rather than Concat's several --
// axis == -1 outright, or a positive axis only when node.input(0)'s own
// rank is known via value_info and agrees; declined (never guessed at)
// when that rank isn't known at all. Mirrors pruning.py's own
// _split_axis_is_last.
bool SplitAxisIsLast(
    const onnx::NodeProto& node,
    const std::unordered_map<std::string, const onnx::ValueInfoProto*>&
        value_info_by_name) {
  const int64_t axis = SplitAxis(node);
  if (axis < 0) {
    return axis == -1;
  }
  auto rank = TensorRank(node.input(0), value_info_by_name);
  if (!rank) {
    return false;  // Rank unknown -- decline rather than guess.
  }
  return axis == *rank - 1;
}

struct SplitSizesResult {
  // Absent for "auto" (no explicit sizes anywhere -- a fully automatic even
  // split, driven purely by the actual output count).
  std::optional<std::vector<int64_t>> sizes;
  SplitSizesKind kind;
};

// Describes how `node` (assumed already confirmed to be a Split) spells out
// its own two output sizes: opsets before 13 spell explicit sizes as an
// integer-list `split` *attribute*; opset 13+ moved that to an optional
// `split` *input* instead (still accepting no sizes at all, for an even
// split). Returns std::nullopt outright -- decline -- when a `split` input
// IS present but is not a resolvable constant INT64 initializer. Mirrors
// pruning.py's own _split_explicit_sizes exactly.
std::optional<SplitSizesResult> SplitExplicitSizes(const onnx::NodeProto& node,
                                                   const InitMap& init_map) {
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    auto it = init_map.find(node.input(1));
    if (it == init_map.end() ||
        it->second->data_type() != onnx::TensorProto::INT64) {
      return std::nullopt;
    }
    return SplitSizesResult{ReadInt64Tensor(*it->second),
                            SplitSizesKind::kInput};
  }
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "split") {
      return SplitSizesResult{
          std::vector<int64_t>(attr.ints().begin(), attr.ints().end()),
          SplitSizesKind::kAttr};
    }
  }
  return SplitSizesResult{std::nullopt, SplitSizesKind::kAuto};
}

// One of a matched gate_up-style Split node's own two outputs.
struct SplitHalfOf {
  onnx::NodeProto* split_node;
  int half_index;  // 0 or 1, node.output's own index.
};

// The split-half analogue of TraceGateProducerBackward: walks backward from
// `tensor_name` through unary activation ops until it resolves to one
// output of an already-matched gate_up Split node (a key of `split_half_of`)
// instead of a real MatMul/Gemm producer's own raw output. Mirrors
// pruning.py's own _trace_split_half_backward.
std::optional<std::tuple<onnx::NodeProto*, int, std::vector<onnx::NodeProto*>>>
TraceSplitHalfBackward(
    const std::string& tensor_name,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const std::unordered_map<std::string, SplitHalfOf>& split_half_of,
    const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int max_hops) {
  std::vector<onnx::NodeProto*> pre_ops;  // Backward order; reversed on return.
  std::string cur = tensor_name;
  for (int hop = 0; hop < max_hops; ++hop) {
    if (ConsumerCount(consumers_of, cur) != 1 || graph_outputs.count(cur)) {
      return std::nullopt;
    }
    auto sit = split_half_of.find(cur);
    if (sit != split_half_of.end()) {
      std::reverse(pre_ops.begin(), pre_ops.end());
      return std::make_tuple(sit->second.split_node, sit->second.half_index,
                             std::move(pre_ops));
    }
    auto nit = node_by_output.find(cur);
    if (nit == node_by_output.end()) {
      return std::nullopt;
    }
    onnx::NodeProto* producer_node = nit->second;
    if (!(UnaryPassThroughOps().count(producer_node->op_type()) != 0 &&
          producer_node->input_size() == 1 &&
          producer_node->output_size() == 1)) {
      return std::nullopt;
    }
    pre_ops.push_back(producer_node);
    cur = producer_node->input(0);
  }
  return std::nullopt;
}

// One matched fused-gate_up_proj gated FFN block -- see this section's own
// comment. `weight`/`bias` are the ONE physical producer tensor shared by
// both the gate and up halves (columns [0, h) and [h, 2h) respectively).
struct SplitGatedChain {
  onnx::NodeProto* split_node;
  onnx::NodeProto* producer_node;
  std::string weight;
  bool weight_transposed;
  std::optional<std::string> bias;
  int64_t h;  // Width of EACH half; the combined producer output is 2*h wide.
  SplitSizesKind split_sizes_kind;
  // Unary activation hops crossed between split_node.output(0)/(1) and the
  // combine node -- purely for value_info staleness bookkeeping, mirroring
  // Producer::pre_ops's own comment; nothing here ever needs its own
  // slicing, being pure single-input/single-output activations.
  std::vector<onnx::NodeProto*> half0_pre_ops;
  std::vector<onnx::NodeProto*> half1_pre_ops;
  onnx::NodeProto* combine_node;
  std::vector<ChainOp> chain_ops;
  onnx::NodeProto* consumer_node;
  std::string consumer_weight;
  bool consumer_weight_transposed;
};

// Finds fused-gate_up_proj gated FFN blocks -- see this section's own
// comment for the full shape, the co-selection semantics, and exactly
// what's supported/declined and why. Mirrors pruning.py's own
// _find_split_gated_chains.
std::vector<SplitGatedChain> FindSplitGatedChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
  }
  auto value_info_by_name = ValueInfoByName(*graph);
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };

  std::unordered_map<std::string, FullProducerMatch> producer_infos;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = MatchProducer(*node, init_map);
    if (info) {
      producer_infos[node->output(0)] =
          FullProducerMatch{node, info->weight, info->weight_transposed,
                            info->bias, info->n_channels};
    }
  }

  // Every gate_up-style Split matched, up front -- keyed by the node's own
  // pointer identity (mirroring pruning.py's own id(node) key) for the
  // per-chain lookup below, and by each of its own two output tensor names
  // for TraceSplitHalfBackward's own bottom-out check.
  struct SplitMatch {
    onnx::NodeProto* producer_node;
    std::string weight;
    bool weight_transposed;
    std::optional<std::string> bias;
    int64_t h;
    SplitSizesKind kind;
  };
  std::unordered_map<onnx::NodeProto*, SplitMatch> split_matches;
  std::unordered_map<std::string, SplitHalfOf> split_half_of;

  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    if (node->op_type() != "Split" || node->output_size() != 2) {
      continue;
    }
    if (node->input_size() == 0 || node->input(0).empty()) {
      continue;
    }
    if (node->output(0) == node->output(1)) {
      continue;  // Degenerate -- same tensor name twice.
    }
    const std::string& in_name = node->input(0);
    if (!is_internal(in_name)) {
      continue;
    }
    auto pit = producer_infos.find(in_name);
    if (pit == producer_infos.end()) {
      continue;
    }
    const FullProducerMatch& pinfo = pit->second;
    if (pinfo.n_channels % 2 != 0) {
      continue;
    }
    const int64_t h = pinfo.n_channels / 2;
    if (!SplitAxisIsLast(*node, value_info_by_name)) {
      continue;
    }
    auto sizes_result = SplitExplicitSizes(*node, init_map);
    if (!sizes_result) {
      continue;  // A dynamic (non-constant) split-sizes input -- decline.
    }
    if (sizes_result->kind != SplitSizesKind::kAuto) {
      const auto& sizes = *sizes_result->sizes;
      if (sizes.size() != 2 || sizes[0] != h || sizes[1] != h) {
        continue;  // Not an equal two-way split of the producer's own output.
      }
    }
    if (!(is_internal(node->output(0)) && is_internal(node->output(1)))) {
      continue;
    }
    split_matches[node] = SplitMatch{
        pinfo.node, pinfo.weight,      pinfo.weight_transposed, pinfo.bias,
        h,          sizes_result->kind};
    split_half_of[node->output(0)] = SplitHalfOf{node, 0};
    split_half_of[node->output(1)] = SplitHalfOf{node, 1};
  }

  std::vector<SplitGatedChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    onnx::NodeProto* split_a = nullptr;
    onnx::NodeProto* split_b = nullptr;
    int half_a = -1, half_b = -1;
    std::vector<onnx::NodeProto*> pre_a, pre_b;

    if (node->op_type() == "Mul" && node->input_size() == 2 &&
        node->output_size() == 1) {
      const std::string& a_name = node->input(0);
      const std::string& b_name = node->input(1);
      if (a_name == b_name || init_map.count(a_name) ||
          init_map.count(b_name)) {
        continue;
      }
      auto trace_a =
          TraceSplitHalfBackward(a_name, node_by_output, split_half_of,
                                 consumers_of, graph_outputs, kMaxChainHops);
      auto trace_b =
          TraceSplitHalfBackward(b_name, node_by_output, split_half_of,
                                 consumers_of, graph_outputs, kMaxChainHops);
      if (!trace_a || !trace_b) {
        continue;
      }
      split_a = std::get<0>(*trace_a);
      half_a = std::get<1>(*trace_a);
      pre_a = std::move(std::get<2>(*trace_a));
      split_b = std::get<0>(*trace_b);
      half_b = std::get<1>(*trace_b);
      pre_b = std::move(std::get<2>(*trace_b));
    } else if (node->op_type() == "SwiGLU" && node->input_size() == 2 &&
               node->output_size() == 1) {
      const std::string& a_name = node->input(0);
      const std::string& b_name = node->input(1);
      if (init_map.count(a_name) || init_map.count(b_name)) {
        continue;
      }
      if (!(is_internal(a_name) && is_internal(b_name))) {
        continue;
      }
      auto ait = split_half_of.find(a_name);
      auto bit = split_half_of.find(b_name);
      if (ait == split_half_of.end() || bit == split_half_of.end()) {
        continue;
      }
      split_a = ait->second.split_node;
      half_a = ait->second.half_index;
      split_b = bit->second.split_node;
      half_b = bit->second.half_index;
      // pre_a/pre_b stay empty -- SwiGLU's swish lives entirely inside the
      // op, so there's nowhere to hang a pre-op.
    } else {
      continue;
    }

    if (split_a != split_b || half_a == half_b) {
      continue;  // Not both halves of the very same Split.
    }
    const SplitMatch& sm = split_matches.at(split_a);

    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    auto [consumer, chain_ops] = WalkToConsumer(
        out_name, init_map, consumers_of, graph_outputs, sm.h, kMaxChainHops);
    if (!consumer) {
      continue;
    }
    if (consumer->weight == sm.weight) {
      continue;  // Degenerate -- consumer tied to the combined producer weight.
    }

    SplitGatedChain chain;
    chain.split_node = split_a;
    chain.producer_node = sm.producer_node;
    chain.weight = sm.weight;
    chain.weight_transposed = sm.weight_transposed;
    chain.bias = sm.bias;
    chain.h = sm.h;
    chain.split_sizes_kind = sm.kind;
    chain.half0_pre_ops = (half_a == 0) ? std::move(pre_a) : std::move(pre_b);
    chain.half1_pre_ops = (half_a == 0) ? std::move(pre_b) : std::move(pre_a);
    chain.combine_node = node;
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = consumer->weight_transposed;
    chains.push_back(std::move(chain));
  }
  return chains;
}

// Applies every matched SplitGatedChain -- the fused gate_up_proj analogue
// of ApplyChains' own gated-pair handling, deliberately a separate function
// (like ApplyConcatChains) rather than folding into ApplyChains -- see this
// section's own comment for why. `w_arrays_nk`-style combined
// (root-sum-square) importance is computed directly over the two halves of
// the one producer tensor: a channel whose gate-half weight is large but
// whose up-half is negligible (or vice versa) ranks by their *combined*, not
// independently-considered, importance -- mirroring
// _plain_branch_importance's own combining formula, the same one an ordinary
// two-producer gated pair's two producers already get combined by.
//
// A Split node's own explicit output-size spelling (if any), when present,
// is rewritten to the new, still-EVEN [len(keep), len(keep)] once pruning
// finishes -- "still even" is the entire point of co-selection: both halves
// are always pruned by the exact same `keep` set, so they always stay the
// same width as each other post-prune, same as pre-prune. A Split
// `input`-spelled size that happens to be a *shared* constant initializer
// (reused across more than one distinct Split node whose own `h` values
// might disagree) is protected against a silent double-rewrite conflict by
// `touched_split_size_inits` below, local to this one call -- a second
// chain that would need to rewrite an already-rewritten shared initializer
// to a *different* value is declined outright rather than corrupting the
// first chain's own already-applied rewrite. Mirrors pruning.py's own
// _apply_split_gated_chains.
void ApplySplitGatedChains(onnx::GraphProto* graph,
                           std::vector<SplitGatedChain>& chains,
                           double sparsity, TouchedState& touched) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }
  std::unordered_set<std::string> touched_split_size_inits;

  for (auto& chain : chains) {
    if (touched.producer.count(chain.weight) ||
        touched.consumer.count(chain.consumer_weight) ||
        (chain.bias && touched.const_names.count(*chain.bias))) {
      continue;  // A shared/tied initializer another chain already resized.
    }

    std::optional<std::string> size_init_name;
    if (chain.split_sizes_kind == SplitSizesKind::kInput) {
      size_init_name = chain.split_node->input(1);
      if (touched_split_size_inits.count(*size_init_name)) {
        continue;  // A shared split-sizes constant another chain already
                   // rewrote.
      }
    }

    const int64_t h = chain.h;
    const int64_t keep_count = std::max<int64_t>(
        1, h - std::llround(static_cast<double>(h) * sparsity));
    if (keep_count >= h) {
      continue;  // Rounds down to a no-op for this layer.
    }

    onnx::TensorProto* wt = init_map.at(chain.weight);
    std::vector<int64_t> dims(wt->dims().begin(), wt->dims().end());
    std::vector<float> data = ReadFloatTensor(*wt);
    // w_nk: [2h, k] row-major, regardless of the tensor's own on-disk
    // orientation -- mirrors ApplyChains' own w_arrays_nk construction.
    std::vector<float> w_nk;
    int64_t k;
    if (chain.weight_transposed) {  // Already [2h, k].
      w_nk = std::move(data);
      k = dims[1];
    } else {  // [k, 2h] -> [2h, k].
      w_nk = TransposeFlat(data, dims[0], dims[1]);
      k = dims[0];
    }

    std::vector<double> importance(static_cast<size_t>(h), 0.0);
    for (int64_t c = 0; c < h; ++c) {
      double sq_gate = 0.0, sq_up = 0.0;
      for (int64_t j = 0; j < k; ++j) {
        const double vg = w_nk[static_cast<size_t>(c * k + j)];
        sq_gate += vg * vg;
        const double vu = w_nk[static_cast<size_t>((h + c) * k + j)];
        sq_up += vu * vu;
      }
      importance[static_cast<size_t>(c)] = std::sqrt(sq_gate + sq_up);
    }
    const std::vector<int64_t> keep =
        TopKIndicesAscending(importance, keep_count);
    // `keep` (< h) and `keep + h` (>= h) are disjoint ranges, each already
    // ascending -- their concatenation is therefore already ascending
    // overall too, same `keep`-is-ascending invariant every other chain
    // family in this file maintains, so no re-sort is needed here.
    std::vector<int64_t> global_keep;
    global_keep.reserve(keep.size() * 2);
    global_keep.insert(global_keep.end(), keep.begin(), keep.end());
    for (int64_t c : keep) {
      global_keep.push_back(c + h);
    }

    SliceProducerWeight(wt, chain.weight_transposed, global_keep, false);
    if (chain.bias) {
      SliceLastAxis(init_map.at(*chain.bias), global_keep);
    }
    for (const auto& co : chain.chain_ops) {
      if (co.const_name) {
        SliceLastAxis(init_map.at(*co.const_name), keep);
      }
    }
    SliceConsumerWeight(init_map.at(chain.consumer_weight),
                        chain.consumer_weight_transposed, keep, false);

    if (chain.split_sizes_kind == SplitSizesKind::kAttr) {
      for (auto& attr : *chain.split_node->mutable_attribute()) {
        if (attr.name() == "split") {
          attr.clear_ints();
          attr.add_ints(keep_count);
          attr.add_ints(keep_count);
          break;
        }
      }
    } else if (chain.split_sizes_kind == SplitSizesKind::kInput) {
      onnx::TensorProto* size_init = init_map.at(*size_init_name);
      SetInt64TensorData(size_init, {2}, {keep_count, keep_count});
      touched_split_size_inits.insert(*size_init_name);
    }
    // "auto": no explicit sizes anywhere -- the even split stays automatic
    // at the new width, nothing to rewrite.

    touched.producer.insert(chain.weight);
    touched.consumer.insert(chain.consumer_weight);
    if (chain.bias) {
      touched.const_names.insert(*chain.bias);
    }
    for (const auto& co : chain.chain_ops) {
      if (co.const_name) {
        touched.const_names.insert(*co.const_name);
      }
    }

    touched.stale_value_info.insert(chain.producer_node->output(0));
    for (const auto& out : chain.split_node->output()) {
      touched.stale_value_info.insert(out);
    }
    for (auto* op : chain.half0_pre_ops) {
      touched.stale_value_info.insert(op->output(0));
    }
    for (auto* op : chain.half1_pre_ops) {
      touched.stale_value_info.insert(op->output(0));
    }
    touched.stale_value_info.insert(chain.combine_node->output(0));
    for (const auto& co : chain.chain_ops) {
      touched.stale_value_info.insert(co.node->output(0));
    }
  }
}

// --- Subgraph recursion ------------------------------------------------
//
// Mirrors pruning.py's own `_iter_subgraphs` and the "Subgraph recursion"
// section comment just above its definition there -- read that comment
// block first if you're touching this; it's the design rationale for
// everything below, not just for the Python side.
//
// A plain `If` node's `then_branch`/`else_branch`, a `Loop`/`Scan` node's
// `body`, or (per pruning.py's own confirmation against onnxruntime's
// contrib-op schema registry) a `com.microsoft::BeamSearch`/`GreedySearch`/
// `Sampling`/`WhisperBeamSearch` node's own `decoder`/`encoder`/
// `init_decoder` attribute is itself a full GraphProto that can carry
// arbitrary weight-bearing nodes -- for a whole-model-generation export
// (e.g. produced by `onnxruntime.transformers.models.{t5,gpt2,whisper}
// .convert_generation`), essentially 100% of the actual prunable weight
// lives inside one of these, not in the top-level graph at all. Without
// this, the C++ port would silently prune nothing on such a model while
// pruning.py's own `apply_structured_pruning`/`apply_attention_head_
// pruning` (which this file is a behavior-identical port of) would prune
// everything inside the subgraph -- a correctness gap, not just a missed
// optimization.
//
// IterSubgraphs is the one shared primitive both public entry points below
// build on: it walks `graph`'s own `node()` list and recursively every
// nested GraphProto reachable from any node's GRAPH-/GRAPHS-typed
// AttributeProto (genuinely recursive -- a subgraph's own nodes can
// themselves carry further-nested subgraphs, e.g. an `If` inside a `Loop`
// body), matched purely by `AttributeProto::type()` rather than a
// per-op-name allowlist, so -- exactly like the Python original -- it
// needs no update when some future op adds another graph-typed attribute.
//
// Every GraphProto* this returns is handed, completely independently, to
// the existing per-graph Find*/Apply* functions below UNCHANGED -- never
// merged with any sibling or ancestor graph's own state. This is what
// makes the same two correctness properties pruning.py's own comment block
// documents hold here too, with no extra bookkeeping:
//
// - Implicit-capture scoping: ONNX lets a subgraph's own node reference a
//   name defined in an ENCLOSING graph's scope rather than its own
//   node()/initializer() list (e.g. an `If` branch reading a weight that
//   actually lives in the top-level graph's initializer list). Every
//   Find*Chains function below resolves a weight/value strictly via an
//   `init_map`/consumer map built from the one `graph` argument it was
//   given -- never a caller's own enclosing scope -- so a node whose input
//   only resolves in an outer scope simply fails to match and that chain
//   is declined, the same "decline rather than mis-slice" behavior this
//   file already applies to every other unresolvable topology. No
//   subgraph is ever treated as if it could see outward.
// - No double-counting across scopes: since every Apply*Chains function
//   below only ever mutates a tensor found in the CURRENT graph's own
//   mutable_initializer() list (never a parent's), and a parent-scope
//   initializer an inner graph merely *reads* by implicit capture is --
//   by the point above -- never even matched as prunable from inside that
//   inner graph, there is no path by which processing a subgraph could
//   reach into, and corrupt, a tensor that actually belongs to (and is
//   separately, safely processed as part of) an ancestor or sibling
//   graph's own pass. Each subgraph also gets its own fresh TouchedState
//   (see ApplyStructuredPruning below) for the same reason pruning.py
//   resets its own `_TouchedState` per graph: a name is only ever a
//   "shared/tied initializer" conflict against another chain matched
//   *within that same graph*, never across graphs -- ONNX names are
//   scoped per-graph-tree-position and this file never merges two graphs'
//   own initializer/consumer maps together.
//
// Depth-first, `graph.node()` then `node.attribute()` declaration order,
// recursing into a found subgraph's own nested subgraphs before moving on
// to the next node -- byte-for-byte the same traversal order as the
// Python original, so a global_sparsity-style pooled ranking computed
// per-graph (were the C++ port ever extended to that mode) would visit
// graphs in the same order pruning.py does.
std::vector<onnx::GraphProto*> IterSubgraphs(onnx::GraphProto* graph) {
  std::vector<onnx::GraphProto*> graphs;
  graphs.push_back(graph);
  for (onnx::NodeProto& node : *graph->mutable_node()) {
    for (onnx::AttributeProto& attr : *node.mutable_attribute()) {
      if (attr.type() == onnx::AttributeProto::GRAPH) {
        std::vector<onnx::GraphProto*> nested = IterSubgraphs(attr.mutable_g());
        graphs.insert(graphs.end(), nested.begin(), nested.end());
      } else if (attr.type() == onnx::AttributeProto::GRAPHS) {
        for (onnx::GraphProto& g : *attr.mutable_graphs()) {
          std::vector<onnx::GraphProto*> nested = IterSubgraphs(&g);
          graphs.insert(graphs.end(), nested.begin(), nested.end());
        }
      }
    }
  }
  return graphs;
}

}  // namespace

onnx::ModelProto ApplyStructuredPruning(const onnx::ModelProto& model,
                                        double sparsity) {
  if (!(sparsity >= 0.0 && sparsity < 1.0)) {
    throw std::invalid_argument(
        "ApplyStructuredPruning: sparsity must be in [0, 1), got " +
        std::to_string(sparsity));
  }
  onnx::ModelProto out = model;

  // Subgraph-aware (IterSubgraphs, see this file's own "Subgraph
  // recursion" section comment above): every chain family below is
  // matched and pruned inside a nested If/Loop/Scan/BeamSearch-family
  // subgraph, at any nesting depth, exactly as if that subgraph were its
  // own top-level graph -- each returned GraphProto* gets its own fresh
  // TouchedState, so a "shared/tied initializer" conflict is only ever
  // detected against another chain matched *within that same graph*,
  // mirroring pruning.py's own apply_structured_pruning loop over
  // `_iter_subgraphs(out.graph)` exactly (including resetting `touched`,
  // and flushing `stale_value_info` into that same graph's own
  // `value_info`, once per graph rather than once for the whole model).
  for (onnx::GraphProto* graph : IterSubgraphs(out.mutable_graph())) {
    std::vector<Chain> chains = FindChains(graph);
    std::vector<Chain> gated_chains = FindGatedChains(graph);
    std::vector<Chain> conv_chains = FindConvChains(graph);
    std::vector<Chain> conv_residual_chains = FindConvResidualChains(graph);
    std::vector<Chain> matmul_residual_chains = FindMatmulResidualChains(graph);
    chains.insert(chains.end(), std::make_move_iterator(gated_chains.begin()),
                  std::make_move_iterator(gated_chains.end()));
    chains.insert(chains.end(), std::make_move_iterator(conv_chains.begin()),
                  std::make_move_iterator(conv_chains.end()));
    chains.insert(chains.end(),
                  std::make_move_iterator(conv_residual_chains.begin()),
                  std::make_move_iterator(conv_residual_chains.end()));
    chains.insert(chains.end(),
                  std::make_move_iterator(matmul_residual_chains.begin()),
                  std::make_move_iterator(matmul_residual_chains.end()));
    std::vector<ConcatChain> concat_chains = FindMatmulConcatChains(graph);
    std::vector<ConcatChain> conv_concat_chains = FindConvConcatChains(graph);
    concat_chains.insert(concat_chains.end(),
                         std::make_move_iterator(conv_concat_chains.begin()),
                         std::make_move_iterator(conv_concat_chains.end()));
    std::vector<SplitGatedChain> split_gated_chains =
        FindSplitGatedChains(graph);

    TouchedState touched;
    if (!chains.empty()) {
      ApplyChains(graph, chains, sparsity, touched);
    }
    if (!concat_chains.empty()) {
      ApplyConcatChains(graph, concat_chains, sparsity, touched);
    }
    if (!split_gated_chains.empty()) {
      // A genuinely separate application pass, not folded into ApplyChains
      // -- see FindSplitGatedChains/ApplySplitGatedChains's own section
      // comment for why. Shares `touched` with the calls above so a weight
      // this pass resizes can never be double-resized by (or
      // double-resize) an ordinary chain/Concat-chain that happens to
      // touch the same initializer -- still scoped to this one graph only.
      ApplySplitGatedChains(graph, split_gated_chains, sparsity, touched);
    }
    if (!touched.stale_value_info.empty()) {
      google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> kept;
      for (const auto& vi : graph->value_info()) {
        if (!touched.stale_value_info.count(vi.name())) {
          *kept.Add() = vi;
        }
      }
      graph->mutable_value_info()->Swap(&kept);
    }
  }
  return out;
}

onnx::ModelProto ApplyAttentionHeadPruning(const onnx::ModelProto& model,
                                           double sparsity) {
  if (!(sparsity >= 0.0 && sparsity < 1.0)) {
    throw std::invalid_argument(
        "ApplyAttentionHeadPruning: sparsity must be in [0, 1), got " +
        std::to_string(sparsity));
  }
  onnx::ModelProto out = model;

  // Subgraph-aware (IterSubgraphs, see this file's own "Subgraph
  // recursion" section comment above): every chain family below is
  // matched and pruned inside a nested If/Loop/Scan/BeamSearch-family
  // subgraph, at any nesting depth, exactly as if that subgraph were its
  // own top-level graph -- each Find*Chains call below is given one
  // GraphProto* (top-level or nested) at a time, so a chain that would
  // need to reach a producer/consumer only resolvable via an
  // implicitly-captured name from an enclosing scope is declined (never
  // matched) rather than mis-resolved, and ApplyAttentionChains only ever
  // slices an initializer out of the one graph it was actually found in.
  // Mirrors pruning.py's own apply_attention_head_pruning loop over
  // `_iter_subgraphs(out.graph)`.
  for (onnx::GraphProto* graph : IterSubgraphs(out.mutable_graph())) {
    std::vector<AttnChain> chains = FindAttentionChains(graph);
    std::vector<AttnChain> gqa_chains = FindGqaChains(graph);
    std::vector<AttnChain> onnx_attn_chains = FindOnnxAttentionChains(graph);
    chains.insert(chains.end(), std::make_move_iterator(gqa_chains.begin()),
                  std::make_move_iterator(gqa_chains.end()));
    chains.insert(chains.end(),
                  std::make_move_iterator(onnx_attn_chains.begin()),
                  std::make_move_iterator(onnx_attn_chains.end()));
    if (!chains.empty()) {
      ApplyAttentionChains(graph, chains, sparsity);
    }
  }
  return out;
}

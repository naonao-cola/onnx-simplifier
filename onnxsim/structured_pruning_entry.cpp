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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
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
      // ai.onnx (domain "") Erf(X) -> Y, the error function: a single
      // required input, a single output, no attributes at all -- exactly as
      // unary/elementwise/parameter-free as every other entry here.
      // Membership alone extends every walker that already consults this
      // set for free, mirroring pruning.py's own identical addition -- but
      // note it's only *part* of what a decomposed erf-GELU export needs:
      // the rest (the scalar Div/Add operands, and the self-gating Mul
      // against the walk's own running tensor) is a different shape unary
      // membership alone can't reach -- see the "Self-gated activation
      // decomposition" section comment above WalkGateBranch below.
      "Erf",
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

// True if `name` names a constant FLOAT initializer shaped like a flat
// per-channel vector (`prod(dims) == dims[-1]`) -- mirrors pruning.py's own
// `_flat_channel_const` exactly: the self-consistency bar every per-channel
// affine/bias/scale hop in this module checks before ever accepting a
// tensor as a slice target. The real `dims[-1] == n_channels` check, once
// the chain's real channel count is known, is left to the caller
// (MatchGroupNormPassThrough, MatchDecomposedLayerNormPassThrough). Moved
// ahead of MatchClipChannelPassThrough (its original position, still
// unchanged relative to every other caller) so
// MatchDecomposedLayerNormPassThrough below -- itself needed by WalkToConsumer,
// which sits well before this function's own original location -- can call it.
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
  // Set only for a `CausalConvWithState` hop whose own `past_state` input
  // (index 3) is itself a *constant* initializer -- a degenerate,
  // unlikely-in-practice shape no real export produces (every real export
  // feeds it as an ordinary graph input/runtime KV-cache-style buffer,
  // needing no slicing at all -- the caller's own runtime data), but not
  // one MatchCausalConvWithStatePassThrough assumes away: when set,
  // ApplyChains/ApplyConcatChains slice its own channel axis (axis 1) by
  // the same `keep` index set, mirroring GroupQueryAttention's own
  // "slice if constant, leave alone if dynamic" treatment for its
  // analogous `past_key`/`past_value`. Always nullopt for a depthwise Conv,
  // InstanceNormalization, or PRelu hop. Mirrors pruning.py's own
  // `_ConvPassThrough.past_state`.
  std::optional<std::string> past_state;
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

// --- Decomposed (not-yet-fused) LayerNorm pass-through, mirroring
// pruning.py's own _match_decomposed_layer_norm_pass_through -----------------
//
// LayerNormalization's own schema is only opset 17+, so a model exported at
// opset <= 16 -- still extremely common for broad runtime/NPU/TensorRT
// compatibility -- has no fused op to lower nn.LayerNorm to at all, and
// torch.onnx.export emits LayerNorm's own canonical decomposition as plain
// ops instead:
//
//     mean     = ReduceMean(x, axes=[-1])
//     centered = Sub(x, mean)
//     sq       = Pow(centered, 2.0)
//     var      = ReduceMean(sq, axes=[-1])
//     var_eps  = Add(var, eps)
//     std      = Sqrt(var_eps)
//     normed   = Div(centered, std)
//     scaled   = Mul(normed, gamma)
//     out      = Add(scaled, beta)
//
// -- `x` consumed twice (by the first ReduceMean and by Sub), `centered`
// consumed twice (by Pow and by Div), every other intermediate tensor
// consumed exactly once. Recognized by WalkToConsumer as one more
// MatMul/Gemm-chain-only hop (never WalkMatmulProducerBackward -- mirroring
// pruning.py's own scope decision: the far more common shape a residual
// Add's own operand resolves back through is an FFN's down-projection or an
// attention block's own output projection, with no LayerNorm in between at
// all), folding all 9 nodes into chain_ops in one hop. Only the Mul's own
// gamma (input 1) and the final Add's own beta (input 1) have anything
// sliced; every other node's entry carries no const name.
//
// Since this shape's own root tensor (`x`) is read *twice* by design, every
// WalkToConsumer call site that used to gate on a strict single-consumer
// check before ever calling it now uses MatmulWalkRootOk instead -- see its
// own comment below for why dropping that gate down to "not a graph output"
// is safe for every previously-supported shape, not just this one.
//
// Positive-axis ReduceMean `axes` (needing the tensor's own known rank to
// resolve) is deliberately out of scope for this port -- only the
// unambiguous `axes=[-1]` case is recognized (see ReduceMeanAxisIsLast's own
// comment); pruning.py's own value_info-threaded rank resolution is not
// mirrored here, and WalkToConsumer gains no such parameter. This only
// narrows coverage (declines a case pruning.py's own version might still
// accept), never a correctness gap -- the same "declined, not guessed at"
// bar this port always holds itself to.
constexpr int kMaxGateBranchHops = 4;

// True iff `name` names a constant float *scalar* initializer (`dims`
// exactly [] or [1]) -- the same scalar bar MatchClipChannelPassThrough's
// own inline `min`/`max` check already uses, reused here for two different
// callers that each need a plain, channel-agnostic constant operand: the
// decomposed LayerNorm's own `eps` operand
// (DecomposedLayerNormPowExponentIsTwo additionally checks the Pow
// exponent's own *value* is exactly 2.0; `eps`'s own value is never
// inspected, only its shape -- any epsilon is equally safe, it's never
// sliced), and a self-gated activation decomposition's own gate-branch
// "other operand" (sqrt(2), 1.0, erf-GELU's own trailing 0.5 -- see
// WalkGateBranch below). Mirrors pruning.py's own
// _flat_scalar_float_const/_gate_branch_scalar_const, which have identical
// bodies for the identical reason.
bool ScalarFloatConst(const std::string& name, const InitMap& init_map) {
  auto it = init_map.find(name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT) {
    return false;
  }
  const auto& dims = it->second->dims();
  return dims.size() == 0 || (dims.size() == 1 && dims.Get(0) == 1);
}

// True iff `name` names a constant float scalar (ScalarFloatConst) whose
// value is exactly 2.0 -- Pow's own schema (X, Y -> X ** Y, a fixed operand
// order) makes `name` its second input; this confirms the node genuinely
// computes `(x - mean) ** 2`, LayerNorm's own variance term, rather than
// some other exponent this pass has no basis for treating as
// channel-pruning-safe. Mirrors pruning.py's own
// _decomposed_layer_norm_pow_exponent_is_two.
bool DecomposedLayerNormPowExponentIsTwo(const std::string& name,
                                         const InitMap& init_map) {
  if (!ScalarFloatConst(name, init_map)) {
    return false;
  }
  std::vector<float> values = ReadFloatTensor(*init_map.at(name));
  return values.size() == 1 && values[0] == 2.0f;
}

// True iff a plain (default-domain) `ReduceMean` `node` -- one of
// MatchDecomposedLayerNormPassThrough's own two ReduceMean nodes,
// `input_name` its own single input -- is confirmed to reduce *only*
// `input_name`'s last axis, keeping that axis (`keepdims=1`, the shape
// LayerNorm's own mean/variance reduction needs). Only the unambiguous
// `axes == [-1]` case is recognized here -- unlike pruning.py's own
// _reduce_mean_axis_is_last, this deliberately never resolves a positive
// axis against the tensor's own rank (see this section's own comment above
// for why); narrower coverage, declined rather than guessed at, never a
// correctness gap. Declines whenever `axes` is absent (the schema default
// then reduces *every* axis, a different computation) or names more than
// one axis, or `keepdims` is anything but its schema default/an explicit 1
// (`keepdims=0` would drop the reduced axis entirely, breaking the
// following Sub's/Add's own broadcast).
bool ReduceMeanAxisIsLast(const onnx::NodeProto& node,
                          const std::string& input_name) {
  if (node.op_type() != "ReduceMean" || node.domain() != "" ||
      node.input_size() != 1 || node.input(0) != input_name) {
    return false;
  }
  std::optional<std::vector<int64_t>> axes_attr;
  int64_t keepdims = 1;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "axes") {
      axes_attr.emplace(attr.ints().begin(), attr.ints().end());
    } else if (attr.name() == "keepdims") {
      keepdims = attr.i();
    }
  }
  if (keepdims != 1 || !axes_attr || axes_attr->size() != 1) {
    return false;
  }
  return (*axes_attr)[0] == -1;
}

struct DecomposedLayerNormMatch {
  std::string out_name;
  std::vector<ChainOp> chain_ops;  // All 9 nodes, forward order.
};

// If `x_name` is the root of exactly the fixed 9-node decomposed-LayerNorm
// sequence documented in this section's own comment above, returns
// `(out_name, chain_ops)` -- see that comment for the full shape and the
// exact "declined, never partially matched" bar this holds every node/
// tensor along the way to (an extra reader of `x_name`/`centered` beyond the
// two each expects, any other intermediate tensor read more than once or
// itself a graph output, a non-constant-2.0 Pow exponent, a ReduceMean axis
// not resolving to -1, or a non-constant/wrongly-shaped/tied gamma/beta).
// Mirrors pruning.py's own _match_decomposed_layer_norm_pass_through,
// called by WalkToConsumer at two points -- before its own ordinary
// single-consumer dispatch (`cur` itself the sequence's own root), and
// again wherever an ordinary hop's own output would otherwise fail that
// dispatch's single-consumer bar -- both needed because this shape's own
// root tensor is read *twice*, a shape the ordinary single-consumer walk
// can never reach on its own from either point.
std::optional<DecomposedLayerNormMatch> MatchDecomposedLayerNormPassThrough(
    const std::string& x_name, const ConsumerMap& consumers_of,
    const InitMap& init_map,
    const std::unordered_set<std::string>& graph_outputs, int64_t n_channels) {
  auto xit = consumers_of.find(x_name);
  if (xit == consumers_of.end() || xit->second.size() != 2) {
    return std::nullopt;
  }
  onnx::NodeProto* reduce_mean = nullptr;
  onnx::NodeProto* sub = nullptr;
  for (onnx::NodeProto* cand : xit->second) {
    if (cand->op_type() == "ReduceMean" && reduce_mean == nullptr) {
      reduce_mean = cand;
    } else if (cand->op_type() == "Sub" && sub == nullptr) {
      sub = cand;
    }
  }
  if (reduce_mean == nullptr || sub == nullptr || reduce_mean == sub) {
    return std::nullopt;
  }
  if (reduce_mean->output_size() != 1 || reduce_mean->output(0).empty()) {
    return std::nullopt;
  }
  if (!ReduceMeanAxisIsLast(*reduce_mean, x_name)) {
    return std::nullopt;
  }
  const std::string& mean_name = reduce_mean->output(0);
  if (graph_outputs.count(mean_name) ||
      ConsumerCount(consumers_of, mean_name) != 1) {
    return std::nullopt;
  }

  if (sub->domain() != "" || sub->input_size() != 2 ||
      sub->input(0) != x_name || sub->input(1) != mean_name ||
      sub->output_size() != 1 || sub->output(0).empty()) {
    return std::nullopt;
  }
  const std::string& centered_name = sub->output(0);
  if (graph_outputs.count(centered_name)) {
    return std::nullopt;
  }
  auto cit = consumers_of.find(centered_name);
  if (cit == consumers_of.end() || cit->second.size() != 2) {
    return std::nullopt;
  }
  onnx::NodeProto* pow_node = nullptr;
  onnx::NodeProto* div_node = nullptr;
  for (onnx::NodeProto* cand : cit->second) {
    if (cand->op_type() == "Pow" && pow_node == nullptr) {
      pow_node = cand;
    } else if (cand->op_type() == "Div" && div_node == nullptr) {
      div_node = cand;
    }
  }
  if (pow_node == nullptr || div_node == nullptr || pow_node == div_node) {
    return std::nullopt;
  }

  if (pow_node->domain() != "" || pow_node->input_size() != 2 ||
      pow_node->input(0) != centered_name || pow_node->output_size() != 1 ||
      pow_node->output(0).empty() ||
      !DecomposedLayerNormPowExponentIsTwo(pow_node->input(1), init_map)) {
    return std::nullopt;
  }
  const std::string& sq_name = pow_node->output(0);
  if (graph_outputs.count(sq_name) ||
      ConsumerCount(consumers_of, sq_name) != 1) {
    return std::nullopt;
  }

  onnx::NodeProto* var_reduce = consumers_of.at(sq_name)[0];
  if (var_reduce->output_size() != 1 || var_reduce->output(0).empty()) {
    return std::nullopt;
  }
  if (!ReduceMeanAxisIsLast(*var_reduce, sq_name)) {
    return std::nullopt;
  }
  const std::string& var_name = var_reduce->output(0);
  if (graph_outputs.count(var_name) ||
      ConsumerCount(consumers_of, var_name) != 1) {
    return std::nullopt;
  }

  onnx::NodeProto* add_eps = consumers_of.at(var_name)[0];
  if (add_eps->op_type() != "Add" || add_eps->domain() != "" ||
      add_eps->input_size() != 2 ||
      (add_eps->input(0) != var_name && add_eps->input(1) != var_name) ||
      add_eps->output_size() != 1 || add_eps->output(0).empty()) {
    return std::nullopt;
  }
  const std::string& eps_name =
      (add_eps->input(0) == var_name) ? add_eps->input(1) : add_eps->input(0);
  if (eps_name == var_name || !ScalarFloatConst(eps_name, init_map)) {
    return std::nullopt;
  }
  const std::string& var_eps_name = add_eps->output(0);
  if (graph_outputs.count(var_eps_name) ||
      ConsumerCount(consumers_of, var_eps_name) != 1) {
    return std::nullopt;
  }

  onnx::NodeProto* sqrt_node = consumers_of.at(var_eps_name)[0];
  if (sqrt_node->op_type() != "Sqrt" || sqrt_node->domain() != "" ||
      sqrt_node->input_size() != 1 || sqrt_node->input(0) != var_eps_name ||
      sqrt_node->output_size() != 1 || sqrt_node->output(0).empty()) {
    return std::nullopt;
  }
  const std::string& std_name = sqrt_node->output(0);
  if (graph_outputs.count(std_name) ||
      ConsumerCount(consumers_of, std_name) != 1) {
    return std::nullopt;
  }
  if (consumers_of.at(std_name)[0] != div_node) {
    return std::nullopt;  // std must feed the *same* Div node centered forked
                          // to.
  }

  if (div_node->domain() != "" || div_node->input_size() != 2 ||
      div_node->input(0) != centered_name || div_node->input(1) != std_name ||
      div_node->output_size() != 1 || div_node->output(0).empty()) {
    return std::nullopt;
  }
  const std::string& normed_name = div_node->output(0);
  if (graph_outputs.count(normed_name) ||
      ConsumerCount(consumers_of, normed_name) != 1) {
    return std::nullopt;
  }

  onnx::NodeProto* mul_node = consumers_of.at(normed_name)[0];
  if (mul_node->op_type() != "Mul" || mul_node->domain() != "" ||
      mul_node->input_size() != 2 ||
      (mul_node->input(0) != normed_name &&
       mul_node->input(1) != normed_name) ||
      mul_node->output_size() != 1 || mul_node->output(0).empty()) {
    return std::nullopt;
  }
  const std::string& gamma_name = (mul_node->input(0) == normed_name)
                                      ? mul_node->input(1)
                                      : mul_node->input(0);
  if (gamma_name == normed_name || !FlatChannelConst(gamma_name, init_map) ||
      init_map.at(gamma_name)->dims(init_map.at(gamma_name)->dims_size() - 1) !=
          n_channels) {
    return std::nullopt;
  }
  const std::string& scaled_name = mul_node->output(0);
  if (graph_outputs.count(scaled_name) ||
      ConsumerCount(consumers_of, scaled_name) != 1) {
    return std::nullopt;
  }

  onnx::NodeProto* add_beta = consumers_of.at(scaled_name)[0];
  if (add_beta->op_type() != "Add" || add_beta->domain() != "" ||
      add_beta->input_size() != 2 ||
      (add_beta->input(0) != scaled_name &&
       add_beta->input(1) != scaled_name) ||
      add_beta->output_size() != 1 || add_beta->output(0).empty()) {
    return std::nullopt;
  }
  const std::string& beta_name = (add_beta->input(0) == scaled_name)
                                     ? add_beta->input(1)
                                     : add_beta->input(0);
  if (beta_name == scaled_name || beta_name == gamma_name ||
      !FlatChannelConst(beta_name, init_map) ||
      init_map.at(beta_name)->dims(init_map.at(beta_name)->dims_size() - 1) !=
          n_channels) {
    return std::nullopt;
  }

  DecomposedLayerNormMatch result;
  result.out_name = add_beta->output(0);
  result.chain_ops = {
      ChainOp{reduce_mean, std::nullopt}, ChainOp{sub, std::nullopt},
      ChainOp{pow_node, std::nullopt},    ChainOp{var_reduce, std::nullopt},
      ChainOp{add_eps, std::nullopt},     ChainOp{sqrt_node, std::nullopt},
      ChainOp{div_node, std::nullopt},    ChainOp{mul_node, gamma_name},
      ChainOp{add_beta, beta_name},
  };
  return result;
}

// True unless `name` is itself a graph output -- the one condition
// WalkToConsumer can never itself recover from once called (a
// caller-observed tensor's own shape must never silently change out from
// under it). Every *other* "is this safe to walk forward from" question --
// an ordinary single consumer, the decomposed-LayerNorm-root two-consumer
// shape (MatchDecomposedLayerNormPassThrough), or a self-gated activation
// decomposition's own two-consumer origin (MatchSelfGatedActivation, below)
// -- is left entirely to WalkToConsumer's own hop-0 dispatch, which
// declines (returns no consumer) exactly the same way skipping the call
// here already did for every case this module supported before either hop
// existed: a tensor with zero consumers, or two or more that don't happen
// to match either shape, still makes the very first hop's own dispatch fail
// immediately, with no wasted walking. Mirrors pruning.py's own
// _matmul_walk_root_ok. Used, in place of a plain single-consumer gate, by
// every MatMul/Gemm-chain finder (FindChains/FindGatedChains/
// FindSplitGatedChains/FindMatmulConcatChains) that hands WalkToConsumer a
// producer's, a gated combine's, or a Concat's own raw output as `start` --
// never used ahead of a forced_first_hop call (ResolveMatmulFanoutBranches's
// own extra-branch resolution), which always takes hop 0's own dispatch
// directly regardless, so this gate can never be reached there.
bool MatmulWalkRootOk(const std::string& name,
                      const std::unordered_set<std::string>& graph_outputs) {
  return !graph_outputs.count(name);
}

// --- Self-gated activation decomposition (SiLU/erf-GELU, unfused),
// mirroring pruning.py's own "Self-gated activation decomposition" section
// comment, _match_self_gated_activation[_backward], and _walk_gate_branch --
//
// Gelu/com.microsoft::Gelu (already unary, UnaryPassThroughOps()) and the
// fused BiasGelu/FastGelu/QuickGelu hops all require either a late opset
// (native Gelu needs opset 20+, Swish needs opset 24+) or onnxruntime's own
// transformer-optimizer fusion pass. A raw torch.onnx.export, at any
// earlier/default opset and without that fusion pass having run, emits the
// literal decomposition instead -- a "self-gated" shape where the
// producer's own output tensor `cur` feeds *two* consumers at once: the
// gate branch's own first op, and the Mul that combines `cur` with the gate
// branch's own final output. Two concrete shapes, both confirmed live via
// torch.onnx.export in pruning.py's own commit history:
//
// - SiLU/Swish (the default activation in Ultralytics YOLOv5/v8's own Conv
//   blocks, and the gate activation in Llama-family SwiGLU): `s =
//   Sigmoid(cur); out = Mul(cur, s)`.
// - erf-GELU (opset < 20, the overwhelming majority of real exports): `d =
//   Div(cur, sqrt(2)); e = Erf(d); a = Add(e, 1.0); m = Mul(cur, a); out =
//   Mul(m, 0.5)` -- the SiLU shape's own gate Mul plus a longer (Div, Erf,
//   Add) gate branch and one trailing scalar-scale Mul.
//
// On a match, the whole diamond folds into the chain the same "nothing to
// slice" way a pooling/Resize/Pad/Clip hop's own chain_ops entries already
// do -- none of Sigmoid/Erf/the scalar Div/Add/Mul operands need their own
// weight sliced, so every diamond node becomes one more (node, no-const)
// chain_ops entry, and the walk continues from the diamond's own true
// output tensor.
//
// Forward (WalkToConsumer/WalkToConvConsumer), the diamond is recognized
// only as a walk's very *first* hop (`cur == start`, on the ordinary,
// non-forced_first_hop path): every hop *after* the first already only ever
// promotes `out2` to `cur` once `out2`'s own single-consumer bar is
// confirmed, so `cur` can never legitimately have two consumers past hop 0
// in the first place. Reaching hop 0 with a two-consumer `cur` at all needs
// FindChains/FindConvChains's own pre-walk gate to admit exactly two
// consumers there too (see each function's own comment) -- a false
// admission there costs one wasted, safely-declining call, never a wrong
// slice. A forced_first_hop call (ResolveMatmulFanoutBranches/
// ResolveConvFanoutBranches) and a Concat-merge finder's own downstream walk
// from a Concat node's merged output are deliberately left out of that
// admission -- neither is the shape a real export ever produces directly, so
// a diamond immediately following one of those is still declined, the same
// conservative scope boundary pruning.py's own version draws.
//
// Backward (WalkMatmulProducerBackward/WalkConvProducerBackward), no
// equivalent admission problem exists: every hop the backward walk crosses
// is already checked fresh via node_by_output, with no "already vetted as
// single-consumer" precondition the way the forward walk's post-hop-0 `cur`
// has -- so the diamond is recognized the same way as any other backward
// hop, checked ahead of the ordinary Add/Mul bias/scale dispatch (otherwise
// indistinguishable from it by op type alone), at whatever position in the
// chain it's actually crossed. The diamond's own origin tensor legitimately
// has *two* real in-group consumers (the branch's own first node, and the
// gate Mul) -- both recorded as `edges` entries, tolerated natively by the
// `accounted` bookkeeping every residual/merge-group finder already builds
// from them (a plain multiset-like structure, built for exactly this
// "more than one in-group consumer of the same tensor" case). The one
// caller this does *not* compose with safely is the direct Concat-branch
// walk (BranchWalkHasFanout) -- that check re-derives a strict *linear*
// single-consumer chain directly from `edges`, an assumption a
// two-entries-sharing-one-tensor diamond genuinely breaks, so a Concat
// branch that crosses one is simply declined there, the same safe (if
// suboptimal) fallback a residual/merge fan-out already gets for other
// shapes -- no change needed to that check for this feature.
const std::unordered_set<std::string>& GateBranchBinaryOps() {
  static const std::unordered_set<std::string> kOps = {"Div", "Mul", "Add",
                                                       "Sub"};
  return kOps;
}

// Walks forward from `branch_start`, whose own (already known) sole
// consumer is `first_node`, through a strict chain of single-consumer hops
// -- each either a node already in UnaryPassThroughOps() or a binary op in
// GateBranchBinaryOps() against a genuine scalar constant (ScalarFloatConst)
// -- until the chain's own running tensor reaches `target` exactly. Returns
// the matched node tuple, in forward (execution) order, or nullopt if the
// chain runs past kMaxGateBranchHops, crosses a graph output, hits an
// intermediate tensor with anything other than exactly one consumer, or
// never reaches `target` at all -- declined, never guessed at, the same
// conservative bar every other hop in this file already holds a mid-chain
// tensor to. Shared by both MatchSelfGatedActivation (forward) and
// MatchSelfGatedActivationBackward (backward) -- the shared question both
// ask is identical ("does this chain of allowed ops connect `branch_start`
// to `target`"), only the direction differs. Mirrors pruning.py's own
// _walk_gate_branch.
std::optional<std::vector<onnx::NodeProto*>> WalkGateBranch(
    onnx::NodeProto* first_node, const std::string& branch_start,
    const std::string& target, const ConsumerMap& consumers_of,
    const InitMap& init_map,
    const std::unordered_set<std::string>& graph_outputs) {
  std::vector<onnx::NodeProto*> nodes;
  onnx::NodeProto* nxt = first_node;
  std::string cur = branch_start;
  for (int hop = 0; hop < kMaxGateBranchHops; ++hop) {
    if (UnaryPassThroughOps().count(nxt->op_type()) != 0 &&
        nxt->input_size() == 1 && nxt->input(0) == cur &&
        nxt->output_size() == 1) {
      // Ok -- unary, no operand of its own to check.
    } else if (GateBranchBinaryOps().count(nxt->op_type()) != 0 &&
               nxt->domain() == "" && nxt->input_size() == 2 &&
               (nxt->input(0) == cur || nxt->input(1) == cur) &&
               nxt->output_size() == 1) {
      const std::string& other =
          (nxt->input(0) == cur) ? nxt->input(1) : nxt->input(0);
      if (other == cur || !ScalarFloatConst(other, init_map)) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
    nodes.push_back(nxt);
    const std::string& out = nxt->output(0);
    if (graph_outputs.count(out)) {
      return std::nullopt;
    }
    if (out == target) {
      return nodes;
    }
    auto cit = consumers_of.find(out);
    if (cit == consumers_of.end() || cit->second.size() != 1) {
      return std::nullopt;
    }
    nxt = cit->second[0];
    cur = out;
  }
  return std::nullopt;
}

struct SelfGatedMatch {
  std::vector<onnx::NodeProto*> diamond_nodes;  // Forward order.
  std::string final_out;
};

// If `cur` (a tensor with *exactly* two consumers) is the origin of a
// self-gated activation decomposition -- see this section's own comment
// above -- returns `(diamond_nodes, final_out)`: `diamond_nodes` every node
// the diamond spans in forward order (the gate branch's own nodes, then the
// self-gating Mul, then -- only for the erf-GELU shape -- one trailing
// scalar-scale Mul); `final_out` the diamond's own true output tensor the
// caller should continue walking from. Used by WalkToConsumer/
// WalkToConvConsumer as a walk's very first hop only. Declines (nullopt)
// unless `cur` has exactly two consumers, exactly one of which is a plain
// `Mul(cur, branch_out)` with `branch_out` distinct from `cur` and not
// itself a constant, `branch_out` has exactly one consumer and isn't a
// graph output, and the remaining consumer, walked via WalkGateBranch,
// resolves `cur` to `branch_out` exactly. A candidate optional trailing
// scalar Mul is folded in whenever present; its absence (SiLU's own shape)
// is not itself a decline. Mirrors pruning.py's own
// _match_self_gated_activation.
std::optional<SelfGatedMatch> MatchSelfGatedActivation(
    const std::string& cur, const ConsumerMap& consumers_of,
    const InitMap& init_map,
    const std::unordered_set<std::string>& graph_outputs) {
  auto cit = consumers_of.find(cur);
  if (cit == consumers_of.end() || cit->second.size() != 2) {
    return std::nullopt;
  }
  const std::vector<onnx::NodeProto*>& consumers = cit->second;

  onnx::NodeProto* gate_node = nullptr;
  std::string branch_out;
  int gate_matches = 0;
  for (onnx::NodeProto* c : consumers) {
    if (c->op_type() == "Mul" && c->domain() == "" && c->input_size() == 2 &&
        (c->input(0) == cur || c->input(1) == cur) && c->output_size() == 1) {
      const std::string& other =
          (c->input(0) == cur) ? c->input(1) : c->input(0);
      if (other != cur && !init_map.count(other)) {
        gate_node = c;
        branch_out = other;
        ++gate_matches;
      }
    }
  }
  if (gate_matches != 1) {
    return std::nullopt;  // No gate Mul, or an ambiguous second candidate.
  }

  onnx::NodeProto* branch_first_node = nullptr;
  int branch_first_count = 0;
  for (onnx::NodeProto* c : consumers) {
    if (c != gate_node) {
      branch_first_node = c;
      ++branch_first_count;
    }
  }
  if (branch_first_count != 1) {
    return std::nullopt;
  }

  if (graph_outputs.count(branch_out) ||
      ConsumerCount(consumers_of, branch_out) != 1) {
    return std::nullopt;
  }

  auto branch_nodes = WalkGateBranch(branch_first_node, cur, branch_out,
                                     consumers_of, init_map, graph_outputs);
  if (!branch_nodes) {
    return std::nullopt;
  }

  SelfGatedMatch result;
  result.diamond_nodes = std::move(*branch_nodes);
  result.diamond_nodes.push_back(gate_node);
  result.final_out = gate_node->output(0);

  if (!graph_outputs.count(result.final_out) &&
      ConsumerCount(consumers_of, result.final_out) == 1) {
    onnx::NodeProto* trailing = consumers_of.at(result.final_out)[0];
    if (trailing->op_type() == "Mul" && trailing->domain() == "" &&
        trailing->input_size() == 2 &&
        (trailing->input(0) == result.final_out ||
         trailing->input(1) == result.final_out) &&
        trailing->output_size() == 1) {
      const std::string& scale = (trailing->input(0) == result.final_out)
                                     ? trailing->input(1)
                                     : trailing->input(0);
      if (scale != result.final_out && ScalarFloatConst(scale, init_map)) {
        result.diamond_nodes.push_back(trailing);
        result.final_out = trailing->output(0);
      }
    }
  }

  return result;
}

struct SelfGatedBackwardMatch {
  std::vector<onnx::NodeProto*> diamond_nodes;  // Forward order.
  std::string origin;
  onnx::NodeProto* gate_node;
};

// The backward-walk (WalkMatmulProducerBackward/WalkConvProducerBackward)
// counterpart of MatchSelfGatedActivation: `cur` is the diamond's own
// candidate *output* tensor (node_by_output[cur] either the self-gating Mul
// itself, or -- for the erf-GELU shape -- the trailing scalar-scale Mul
// wrapping it), and this resolves back to the diamond's own true *origin*
// tensor. Returns `(diamond_nodes, origin, gate_node)` -- `diamond_nodes` in
// the identical forward order MatchSelfGatedActivation returns them in (so
// a caller building chain_ops in reverse-of-visit order can walk
// `diamond_nodes` in reverse, the same as any other single node this walk
// crosses); `gate_node` is `diamond_nodes`' own self-gating Mul specifically
// (not necessarily its last element -- the erf-GELU shape's own trailing
// scalar Mul sits after it), returned separately so the caller can record
// `origin`'s own *two* real in-group consumers (the gate branch's own first
// node and this Mul) as two `edges` entries without having to re-derive
// which element of `diamond_nodes` is which.
//
// Since a Mul's two operands are unordered, both candidate origin/target
// pairings are tried (WalkGateBranch, forward from the candidate origin --
// the *same* function MatchSelfGatedActivation uses); a match is accepted
// only when *exactly one* ordering resolves, the same "decline on
// ambiguity" bar MatchSelfGatedActivation's own gate-candidate search
// already applies. Mirrors pruning.py's own
// _match_self_gated_activation_backward.
std::optional<SelfGatedBackwardMatch> MatchSelfGatedActivationBackward(
    const std::string& cur,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const ConsumerMap& consumers_of, const InitMap& init_map,
    const std::unordered_set<std::string>& graph_outputs) {
  auto nit = node_by_output.find(cur);
  if (nit == node_by_output.end()) {
    return std::nullopt;
  }
  onnx::NodeProto* node = nit->second;
  if (node->op_type() != "Mul" || node->domain() != "" ||
      node->input_size() != 2 || node->output_size() != 1 ||
      node->output(0) != cur) {
    return std::nullopt;
  }

  onnx::NodeProto* gate_node = node;
  onnx::NodeProto* trailing_node = nullptr;
  std::string a_name = node->input(0);
  std::string b_name = node->input(1);
  bool a_const = init_map.count(a_name) != 0;
  bool b_const = init_map.count(b_name) != 0;
  if (a_const != b_const) {
    // A candidate trailing scalar-scale Mul (erf-GELU's own final `* 0.5`)
    // -- its own non-constant operand must itself be a plain Mul (the real
    // gate Mul) with no other consumer.
    const std::string& const_name = a_const ? a_name : b_name;
    const std::string& gate_out = a_const ? b_name : a_name;
    if (!ScalarFloatConst(const_name, init_map)) {
      return std::nullopt;
    }
    if (ConsumerCount(consumers_of, gate_out) != 1) {
      return std::nullopt;
    }
    auto git = node_by_output.find(gate_out);
    if (git == node_by_output.end()) {
      return std::nullopt;
    }
    onnx::NodeProto* gate_candidate = git->second;
    if (gate_candidate->op_type() != "Mul" || gate_candidate->domain() != "" ||
        gate_candidate->input_size() != 2 ||
        gate_candidate->output_size() != 1 ||
        gate_candidate->output(0) != gate_out) {
      return std::nullopt;
    }
    trailing_node = node;
    gate_node = gate_candidate;
    a_name = gate_node->input(0);
    b_name = gate_node->input(1);
    a_const = init_map.count(a_name) != 0;
    b_const = init_map.count(b_name) != 0;
  }

  if (a_const || b_const || a_name == b_name) {
    return std::nullopt;  // Not a genuine two-non-constant-operand gate Mul.
  }

  std::vector<onnx::NodeProto*> resolved_branch_nodes;
  std::string resolved_origin;
  int resolved_count = 0;
  const std::pair<std::string, std::string> orderings[2] = {{a_name, b_name},
                                                            {b_name, a_name}};
  for (const auto& ordering : orderings) {
    const std::string& origin_cand = ordering.first;
    const std::string& target = ordering.second;
    if (graph_outputs.count(target) ||
        ConsumerCount(consumers_of, target) != 1) {
      continue;
    }
    auto oit = consumers_of.find(origin_cand);
    if (oit == consumers_of.end() || oit->second.size() != 2) {
      continue;
    }
    onnx::NodeProto* first_node = nullptr;
    int first_count = 0;
    bool has_gate = false;
    for (onnx::NodeProto* c : oit->second) {
      if (c == gate_node) {
        has_gate = true;
      } else {
        first_node = c;
        ++first_count;
      }
    }
    if (!has_gate || first_count != 1) {
      continue;
    }
    auto branch_nodes = WalkGateBranch(first_node, origin_cand, target,
                                       consumers_of, init_map, graph_outputs);
    if (branch_nodes) {
      resolved_branch_nodes = std::move(*branch_nodes);
      resolved_origin = origin_cand;
      ++resolved_count;
    }
  }
  if (resolved_count != 1) {
    return std::nullopt;  // No resolvable ordering, or an ambiguous second one.
  }

  SelfGatedBackwardMatch result;
  result.diamond_nodes = std::move(resolved_branch_nodes);
  result.diamond_nodes.push_back(gate_node);
  if (trailing_node != nullptr) {
    result.diamond_nodes.push_back(trailing_node);
  }
  result.origin = resolved_origin;
  result.gate_node = gate_node;
  return result;
}

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
      // single-consumer bar unchanged below. Neither the decomposed-
      // LayerNorm nor the self-gated-activation hop below is ever attempted
      // on this path -- see MatchDecomposedLayerNormPassThrough's/this
      // file's own "Self-gated activation decomposition" section comment
      // for why that composition is out of scope.
      nxt = forced_first_hop;
    } else {
      // Tried before the ordinary "exactly one consumer" dispatch below,
      // since a decomposed LayerNorm's own root tensor is read *twice* (by
      // its first ReduceMean and its Sub) -- a shape that dispatch can
      // never reach on its own. Declines instantly whenever `cur` doesn't
      // have exactly two consumers, so this costs an ordinary hop nothing.
      auto ln_match = MatchDecomposedLayerNormPassThrough(
          cur, consumers_of, init_map, graph_outputs, n_channels);
      if (ln_match) {
        chain_ops.insert(chain_ops.end(), ln_match->chain_ops.begin(),
                         ln_match->chain_ops.end());
        cur = ln_match->out_name;
        continue;
      }

      auto cit = consumers_of.find(cur);
      const size_t num_candidates =
          cit == consumers_of.end() ? 0 : cit->second.size();
      if (num_candidates == 2) {
        // A self-gated activation decomposition's own origin tensor -- see
        // this file's own "Self-gated activation decomposition" section
        // comment above. Recognized only here, at a walk's very first hop
        // (`cur == start` the very first time this branch runs) -- every
        // later hop only ever promotes `out2` to `cur` once its own
        // single-consumer bar already holds, so `cur` can never
        // legitimately have two consumers past hop 0.
        auto diamond = MatchSelfGatedActivation(cur, consumers_of, init_map,
                                                graph_outputs);
        if (diamond) {
          if (ConsumerCount(consumers_of, diamond->final_out) != 1 ||
              graph_outputs.count(diamond->final_out)) {
            break;
          }
          for (onnx::NodeProto* n : diamond->diamond_nodes) {
            chain_ops.push_back(ChainOp{n, std::nullopt});
          }
          cur = diamond->final_out;
          continue;
        }
        break;  // Two consumers but not this shape -- declined, as before.
      }
      if (num_candidates != 1) {
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
    if (graph_outputs.count(out2)) {
      break;
    }
    // Ordinarily `out2` must have exactly one consumer to safely become the
    // walk's new `cur` -- but a decomposed LayerNorm's own root tensor is,
    // by its own fixed shape, read *twice* (see
    // MatchDecomposedLayerNormPassThrough's own comment), so an `out2` this
    // hop produces right where such a sequence begins (e.g. a bias Add
    // feeding both the sequence's own ReduceMean and Sub) would otherwise
    // always fail this check and break the walk before ever reaching the
    // top-of-loop dispatch that tries this hop for `cur` -- that dispatch
    // only ever sees `cur` *after* a hop already committed to advancing past
    // it. Trying the same matcher here first, before the ordinary
    // single-consumer bar, closes that gap: every other multi-consumer
    // `out2` still declines exactly as before.
    std::optional<DecomposedLayerNormMatch> out2_ln;
    if (ConsumerCount(consumers_of, out2) != 1) {
      out2_ln = MatchDecomposedLayerNormPassThrough(
          out2, consumers_of, init_map, graph_outputs, n_channels);
      if (!out2_ln) {
        break;
      }
    }
    chain_ops.push_back(ChainOp{nxt, const_name});
    if (out2_ln) {
      chain_ops.insert(chain_ops.end(), out2_ln->chain_ops.begin(),
                       out2_ln->chain_ops.end());
      cur = out2_ln->out_name;
    } else {
      cur = out2;
    }
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

  std::vector<Chain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = MatchProducer(*node, init_map);
    if (!info) {
      continue;
    }
    const std::string& out_name = node->output(0);
    // MatmulWalkRootOk only rules out a graph output -- every other "is
    // this safe to walk forward from" question (an ordinary single
    // consumer, the decomposed-LayerNorm two-consumer root shape, or a
    // self-gated activation decomposition's own two-consumer origin) is
    // left entirely to WalkToConsumer's own hop-0 dispatch, which declines
    // (same as before) if the actual consumers don't form a recognized
    // shape.
    if (!MatmulWalkRootOk(out_name, graph_outputs)) {
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
    // See FindChains's own identical comment -- the gated combine's own raw
    // output can be a decomposed-LayerNorm/self-gated-activation root too.
    if (!MatmulWalkRootOk(out_name, graph_outputs)) {
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
  // Any spatial rank n >= 1 (weight rank >= 3: one output-channel axis, one
  // input-channel axis, at least one spatial axis) -- mirrors pruning.py's
  // own _match_conv_producer exactly (see that function's own docstring:
  // "1-D, 2-D, or 3-D alike"). Widened from this port's original
  // exactly-rank-4 (2-D-only) bound so a 1-D Conv -- the only shape a real
  // `CausalConvWithState` pipeline's own bracketing producer/consumer Conv
  // can ever be, since that op's own I/O is always rank-3 -- can match here
  // too; every existing rank-4 caller is unaffected, this only admits more.
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      it->second->dims_size() < 3) {
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
  // See MatchConvProducer's own comment: any spatial rank n >= 1, mirroring
  // pruning.py's own _match_conv_consumer.
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      it->second->dims_size() < 3) {
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
  // Set only by MatchCausalConvWithStatePassThrough, for a hop whose own
  // `past_state` input (index 3) is itself a constant initializer of the
  // documented rank-3 `(*, n_channels, *)` shape -- see that matcher's own
  // comment. Always nullopt for a real depthwise Conv match
  // (MatchDepthwiseConvPassThrough/MatchConvPassThroughSelf never set it --
  // a plain Conv has no such fourth input at all).
  std::optional<std::string> past_state;
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
  // Any spatial rank n >= 1 -- see MatchConvProducer's own comment; mirrors
  // pruning.py's own _match_depthwise_conv_pass_through.
  if (w->data_type() != onnx::TensorProto::FLOAT || w->dims_size() < 3 ||
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
  return DepthwiseMatch{node.input(1), bias, std::nullopt};
}

// The Conv-chain `CausalConvWithState` pass-through matcher used by
// WalkToConvConsumer, mirroring pruning.py's own
// _match_causal_conv_with_state_pass_through: if `node` is a plain
// `ai.onnx::CausalConvWithState` node (opset 27+, domain "") whose constant
// `weight` (input 1) is exactly the depthwise-Conv shape
// MatchDepthwiseConvPassThrough already recognizes (`[n_channels, 1, k]`),
// returns a DepthwiseMatch identical in shape to that matcher's own result
// -- this op mixes no channels at all, identically to a depthwise Conv,
// except it carries two extra per-channel-shaped operands a depthwise Conv
// doesn't: an optional `bias` (input 2, identical treatment to a depthwise
// Conv's own) and an optional `past_state` carry tensor (input 3), set on
// the returned DepthwiseMatch only when it is itself a constant of the
// documented rank-3 `(*, n_channels, *)` shape (axis 1 == n_channels) --
// left nullopt (needing no slicing at all) when dynamic/absent, and this
// whole match declined (nullopt) when it's a constant of any *other* shape
// (never guessed at). This op's own second output, `present_state`, is
// never a tensor this function slices (a runtime *output*, not a weight);
// the caller (WalkToConvConsumer) is responsible for treating it as stale
// alongside `output` -- see this file's own `stale_value_info` handling in
// ApplyChains/ApplyConcatChains, which iterates every one of a hop node's
// outputs, not just output(0), for exactly this reason.
std::optional<DepthwiseMatch> MatchCausalConvWithStatePassThrough(
    const onnx::NodeProto& node, const InitMap& init_map, int64_t n_channels) {
  if (node.op_type() != "CausalConvWithState" || node.domain() != "") {
    return std::nullopt;
  }
  if (node.input_size() < 2) {
    return std::nullopt;
  }
  const std::string& w_name = node.input(1);
  auto wit = init_map.find(w_name);
  if (wit == init_map.end() ||
      wit->second->data_type() != onnx::TensorProto::FLOAT ||
      wit->second->dims_size() != 3 || wit->second->dims(0) != n_channels ||
      wit->second->dims(1) != 1) {
    return std::nullopt;
  }
  std::optional<std::string> bias;
  if (node.input_size() > 2 && !node.input(2).empty()) {
    bias = node.input(2);
    auto bit = init_map.find(*bias);
    if (bit == init_map.end() ||
        bit->second->data_type() != onnx::TensorProto::FLOAT) {
      return std::nullopt;  // Non-constant bias -- can't safely prune it.
    }
  }
  std::optional<std::string> past_state;
  if (node.input_size() > 3 && !node.input(3).empty()) {
    auto pit = init_map.find(node.input(3));
    if (pit != init_map.end()) {
      // A constant past_state -- degenerate, but if present it must be the
      // documented rank-3 (batch, channels, k-1) shape with axis 1 ==
      // n_channels to be safely sliceable; anything else declines the whole
      // hop rather than guessing which axis (if any) is really the channel
      // one.
      const onnx::TensorProto* ps = pit->second;
      if (ps->data_type() != onnx::TensorProto::FLOAT || ps->dims_size() != 3 ||
          ps->dims(1) != n_channels) {
        return std::nullopt;
      }
      past_state = node.input(3);
      // else: dynamic (an ordinary graph input/runtime buffer) -- the
      // caller's own runtime data, needs no slicing, not tracked here.
    }
  }
  return DepthwiseMatch{w_name, bias, past_state};
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

struct InstanceNormMatch {
  std::string scale;
  std::string bias;
};

// The Conv-chain InstanceNormalization pass-through matcher used by
// WalkToConvConsumer, mirroring pruning.py's own
// _match_instance_norm_pass_through: if `node` is a plain (default-domain)
// `InstanceNormalization` node (opset 6+) with constant, exactly-1-D
// (`dims == [n_channels]`) `scale` (input 1) and `B` (input 2) -- both
// required by the op's own schema -- returns `{scale_name, bias_name}`.
// Declines (nullopt) on a missing/non-constant/wrongly-shaped `scale`/`B`,
// or `scale`/`B` naming the same tensor (double-slicing it would corrupt
// it, the same tied-name bar MatchGroupNormPassThrough already applies) --
// none of these is guessed at.
//
// Unlike MatchGroupNormPassThrough's own FlatChannelConst bar
// (`prod(dims) == dims[-1]`, deliberately loose enough to admit a
// broadcastable-but-not-strictly-1-D shape like `[1, 1, C]`), this requires
// `scale`/`B` to be *strictly* rank-1: the returned names are carried on a
// plain ConvPassThrough (this hop's own `scale`/`B` playing that struct's
// `weight`/`bias` role -- no dedicated struct needed here the way
// GroupNormPassThrough needed one), whose own slicing (ApplyChains's own
// SliceProducerWeight(..., is_conv=true) call) always slices axis 0 --
// correct only when axis 0 already *is* the one-and-only axis, i.e.
// strictly rank-1. A looser broadcastable shape would silently slice the
// wrong axis instead of being declined, so this bar is deliberately
// narrower than FlatChannelConst's -- not a missed generalization: the ONNX
// schema itself only ever documents `scale`/`B` as rank-1 `[C]` for this
// op (no opset ever gave it GroupNorm's own per-*group* alternate shape),
// so nothing real is excluded by holding to that.
std::optional<InstanceNormMatch> MatchInstanceNormPassThrough(
    const onnx::NodeProto& node, const InitMap& init_map, int64_t n_channels) {
  if (node.op_type() != "InstanceNormalization" || node.domain() != "" ||
      node.input_size() != 3 || node.input(1).empty() ||
      node.input(2).empty() || node.output_size() != 1) {
    return std::nullopt;
  }
  const std::string& scale_name = node.input(1);
  const std::string& bias_name = node.input(2);
  if (scale_name == bias_name) {
    return std::nullopt;  // Tied scale/B -- double-slicing would corrupt it.
  }
  auto sit = init_map.find(scale_name);
  auto bit = init_map.find(bias_name);
  if (sit == init_map.end() || bit == init_map.end() ||
      sit->second->data_type() != onnx::TensorProto::FLOAT ||
      bit->second->data_type() != onnx::TensorProto::FLOAT ||
      sit->second->dims_size() != 1 || bit->second->dims_size() != 1 ||
      sit->second->dims(0) != n_channels ||
      bit->second->dims(0) != n_channels) {
    return std::nullopt;
  }
  return InstanceNormMatch{scale_name, bias_name};
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
      // ResolveConvFanoutBranches. The self-gated-activation hop below is
      // never attempted on this path either -- see this file's own
      // "Self-gated activation decomposition" section comment for why.
      nxt = forced_first_hop;
    } else {
      auto cit = consumers_of.find(cur);
      const size_t num_candidates =
          cit == consumers_of.end() ? 0 : cit->second.size();
      if (num_candidates == 2) {
        // A self-gated activation decomposition's own origin tensor (SiLU/
        // Swish, ubiquitous after Conv in YOLO-style backbones) -- see this
        // file's own "Self-gated activation decomposition" section comment
        // above WalkGateBranch. Recognized only at a walk's very first hop
        // -- see that comment for why every later hop can never see a
        // two-consumer `cur` in the first place.
        auto diamond = MatchSelfGatedActivation(cur, consumers_of, init_map,
                                                graph_outputs);
        if (diamond) {
          if (ConsumerCount(consumers_of, diamond->final_out) != 1 ||
              graph_outputs.count(diamond->final_out)) {
            break;
          }
          for (onnx::NodeProto* n : diamond->diamond_nodes) {
            chain_ops.push_back(ChainOp{n, std::nullopt});
          }
          cur = diamond->final_out;
          continue;
        }
        break;  // Two consumers but not this shape -- declined, as before.
      }
      if (num_candidates != 1) {
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

    if (nxt->op_type() == "CausalConvWithState" && nxt->domain() == "" &&
        nxt->input_size() > 0 && nxt->input(0) == cur) {
      // This op is never a consumer in its own right (unlike `Conv` above,
      // which falls through to MatchConvConsumer on a depthwise-match
      // miss), so a failed match here simply ends the walk -- mirrors
      // pruning.py's own identical short-circuit in _walk_to_conv_consumer.
      // See MatchCausalConvWithStatePassThrough's own comment for the full
      // empirical schema findings, including why this hop is recognized
      // only by this forward walk (WalkConvProducerBackward's own per-hop
      // loop unconditionally declines any node with output_size() != 1, and
      // this op always has two outputs).
      auto cc = MatchCausalConvWithStatePassThrough(*nxt, init_map, n_channels);
      if (cc) {
        const std::string& out2 = nxt->output(0);
        if (ConsumerCount(consumers_of, out2) != 1 ||
            graph_outputs.count(out2)) {
          break;
        }
        pass_through.push_back(
            ConvPassThrough{nxt, cc->weight, cc->bias, cc->past_state});
        cur = out2;
        continue;
      }
      break;
    }

    if (nxt->op_type() == "InstanceNormalization" && nxt->domain() == "" &&
        nxt->input_size() > 0 && nxt->input(0) == cur) {
      // Recognized unconditionally (no `recognize_*` gate, and any number
      // of times per chain, unlike the GroupNorm hop below) --
      // InstanceNorm's per-channel-only statistics carry no
      // group-boundary-drift risk to guard against, so this hop is exactly
      // as unrestricted as a depthwise Conv hop above, reusing the same
      // ConvPassThrough struct (its own `scale`/`B` playing that struct's
      // `weight`/`bias` role) rather than needing a dedicated one. Like
      // CausalConvWithState above, this op is never a consumer in its own
      // right, so a failed match here simply ends the walk.
      auto in_match = MatchInstanceNormPassThrough(*nxt, init_map, n_channels);
      if (in_match) {
        const std::string& out2 = nxt->output(0);
        if (ConsumerCount(consumers_of, out2) != 1 ||
            graph_outputs.count(out2)) {
          break;
        }
        pass_through.push_back(
            ConvPassThrough{nxt, in_match->scale, in_match->bias});
        cur = out2;
        continue;
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

  std::vector<Chain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto info = MatchConvProducer(*node, init_map);
    if (!info) {
      continue;
    }
    const std::string& out_name = node->output(0);
    // Ordinarily exactly one consumer -- but also admit exactly two, the
    // shape a self-gated activation decomposition's own origin tensor
    // takes (see this file's own "Self-gated activation decomposition"
    // section comment above WalkGateBranch): WalkToConvConsumer decides, at
    // its own first hop, whether those two consumers actually form that
    // shape, declining (same as before) if not. Unlike FindChains's own
    // MatmulWalkRootOk gate, this stays narrower (1 or 2 only, never a bare
    // "not a graph output") -- the decomposed-LayerNorm hop is MatMul/Gemm-
    // chain-only, so there's no analogous reason to admit an arbitrary
    // consumer count here.
    if (graph_outputs.count(out_name) ||
        (ConsumerCount(consumers_of, out_name) != 1 &&
         ConsumerCount(consumers_of, out_name) != 2)) {
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
  // Any spatial rank n >= 1 -- see MatchConvProducer's own comment.
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      it->second->dims_size() < 3) {
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

// The backward-walk (WalkConvProducerBackward) counterpart of
// MatchInstanceNormPassThrough, mirroring pruning.py's own
// _match_instance_norm_pass_through_self and MatchConvPassThroughSelf's own
// identical trick: the backward walk doesn't know its group's shared
// channel count yet at the point it first crosses this hop, so this checks
// the node's own `scale` is self-consistently rank-1 by calling
// MatchInstanceNormPassThrough with `scale`'s own `dims(0)` as the
// "expected" n_channels -- trivially satisfying that one check and leaving
// every other one (constant B, matching length, non-tied names) intact.
// FindConvResidualChains/ResolveConvResidualGroupForConcat re-validate
// every such hop against the group's real, established channel count once
// resolved (the same generic pass_through re-validation a depthwise hop
// already gets, keyed only on `hop.weight`'s own `dims(0)` -- `hop.weight`
// here holds this op's own `scale` name, so that same generic check already
// re-validates this hop correctly, with no dedicated InstanceNorm-specific
// re-check needed).
std::optional<InstanceNormMatch> MatchInstanceNormPassThroughSelf(
    const onnx::NodeProto& node, const InitMap& init_map) {
  if (node.op_type() != "InstanceNormalization" || node.input_size() < 2) {
    return std::nullopt;
  }
  auto sit = init_map.find(node.input(1));
  if (sit == init_map.end() ||
      sit->second->data_type() != onnx::TensorProto::FLOAT ||
      sit->second->dims_size() != 1) {
    return std::nullopt;
  }
  return MatchInstanceNormPassThrough(node, init_map, sit->second->dims(0));
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
    const InitMap& init_map, const ConsumerMap& consumers_of,
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

    // Recognized unconditionally here too (no gate), mirroring
    // MatchInstanceNormPassThrough's own unconditional recognition in
    // WalkToConvConsumer -- InstanceNorm's per-channel-only statistics
    // carry no group-boundary-drift risk to guard against, unlike the
    // GroupNorm hop, which this backward walk never recognizes at all.
    auto in_self = MatchInstanceNormPassThroughSelf(*node, init_map);
    if (in_self) {
      pass_through.push_back(
          ConvPassThrough{node, in_self->scale, in_self->bias});
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

    if (node->op_type() == "Mul" && node->domain() == "") {
      // See WalkMatmulProducerBackward's own identical check for the full
      // reasoning (checked here, ahead of nothing else in this Conv-chain
      // walker -- unlike the MatMul/Gemm walk, there is no ordinary
      // bias/scale Mul hop here to shadow). `origin`'s own two real
      // in-group consumers (the gate branch's own first node, and the gate
      // Mul itself) both become `edges` entries, tolerated natively by the
      // `accounted` bookkeeping FindConvResidualChains builds from them.
      // This is the reason this function gained a `consumers_of` parameter
      // at all -- every other hop above reaches `cur`'s own producer purely
      // via `node_by_output`, needing no forward-consumer lookups.
      auto diamond = MatchSelfGatedActivationBackward(
          cur, node_by_output, consumers_of, init_map, graph_outputs);
      if (diamond) {
        for (auto it = diamond->diamond_nodes.rbegin();
             it != diamond->diamond_nodes.rend(); ++it) {
          unary_ops.push_back(*it);
        }
        edges.push_back({diamond->origin, diamond->diamond_nodes.front()});
        edges.push_back({diamond->origin, diamond->gate_node});
        cur = diamond->origin;
        continue;
      }
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
      ConvBackwardEdge edge =
          WalkConvProducerBackward(operand, node_by_output, init_map,
                                   consumers_of, graph_outputs, kMaxChainHops);
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

    if (node->op_type() == "Mul" && node->domain() == "") {
      // Checked ahead of the ordinary Add/Mul dispatch below: a self-gated
      // activation decomposition's own gate Mul (and, for erf-GELU, its
      // trailing scalar-scale Mul) is otherwise indistinguishable from a
      // plain bias/scale Mul hop by op type alone -- see this file's own
      // "Self-gated activation decomposition" section comment above
      // WalkGateBranch. The diamond's own origin tensor legitimately has
      // two in-group consumers (the gate branch's own first node, and this
      // Mul), so *two* `edges` entries share that one tensor -- the
      // `accounted` bookkeeping every caller of this walk already builds
      // tolerates that natively; the direct Concat-branch walk
      // (BranchWalkHasFanout) does not, and simply declines a branch that
      // crosses one, the same safe fallback a residual/merge fan-out
      // already gets there before this feature existed.
      auto diamond = MatchSelfGatedActivationBackward(
          cur, node_by_output, consumers_of, init_map, graph_outputs);
      if (diamond) {
        for (auto it = diamond->diamond_nodes.rbegin();
             it != diamond->diamond_nodes.rend(); ++it) {
          chain_ops.push_back(ChainOp{*it, std::nullopt});
        }
        // `origin` has two real in-group consumers -- the gate branch's
        // own first node and the gate Mul itself -- both recorded so
        // neither is later mistaken for a stray extra consumer needing its
        // own separate resolution.
        edges.push_back({diamond->origin, diamond->diamond_nodes.front()});
        edges.push_back({diamond->origin, diamond->gate_node});
        cur = diamond->origin;
        continue;
      }
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
      if (hop.past_state) {
        // A CausalConvWithState hop's own constant `past_state` -- rank-3
        // `(batch, channels, k-1)`, channel axis 1 -- reuses
        // SliceConsumerWeight's own `is_conv` branch (SliceAxis1 over
        // dims(0)/dims(1) with `inner = prod(dims[2:])`), which slices
        // exactly that axis; mirrors pruning.py's own _slice_axis1 call in
        // _apply_conv_pass_through_hop.
        SliceConsumerWeight(init_map.at(*hop.past_state), false, keep, true);
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
        if (hop.past_state) {
          SliceConsumerWeight(init_map.at(*hop.past_state), false, keep, true);
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
    // Every output, not just [0] -- a hop node can have more than one (e.g.
    // `CausalConvWithState`'s own `present_state`); see ConvPassThrough's
    // own docstring.
    for (const auto& hop : chain.conv_pass_through) {
      for (const auto& out : hop.node->output()) {
        stale_value_info.insert(out);
      }
    }
    if (chain.group_norm) {
      stale_value_info.insert(chain.group_norm->node->output(0));
    }
    for (const auto* b : extra_ptrs) {
      for (const auto& co : b->chain_ops) {
        stale_value_info.insert(co.node->output(0));
      }
      for (const auto& hop : b->conv_pass_through) {
        for (const auto& out : hop.node->output()) {
          stale_value_info.insert(out);
        }
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
    const InitMap& init_map, const ConsumerMap& consumers_of,
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
      ConvBackwardEdge edge =
          WalkConvProducerBackward(operand, node_by_output, init_map,
                                   consumers_of, graph_outputs, kMaxChainHops);
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
    // See FindChains's own identical comment -- the Concat merge's own raw
    // output can be a decomposed-LayerNorm/self-gated-activation root too.
    if (!MatmulWalkRootOk(out_name, graph_outputs)) {
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
      ConvBackwardEdge edge =
          WalkConvProducerBackward(operand, node_by_output, init_map,
                                   consumers_of, graph_outputs, kMaxChainHops);
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
            edge.add_node, node_by_output, init_map, consumers_of,
            graph_outputs);
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
        if (hop.past_state) {
          SliceConsumerWeight(init_map.at(*hop.past_state), false, keep, true);
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
      if (hop.past_state) {
        SliceConsumerWeight(init_map.at(*hop.past_state), false, global_keep,
                            true);
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
      // Every output, not just [0] -- a hop node can have more than one
      // (e.g. `CausalConvWithState`'s own `present_state`); see
      // ConvPassThrough's own docstring.
      for (const auto& hop : b.conv_pass_through) {
        for (const auto& out : hop.node->output()) {
          touched.stale_value_info.insert(out);
        }
      }
    }
    for (const auto& co : chain.chain_ops) {
      touched.stale_value_info.insert(co.node->output(0));
    }
    for (const auto& hop : chain.conv_pass_through) {
      for (const auto& out : hop.node->output()) {
        touched.stale_value_info.insert(out);
      }
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
    // See FindChains's own identical comment -- the combine's own raw
    // output can be a decomposed-LayerNorm/self-gated-activation root too.
    if (!MatmulWalkRootOk(out_name, graph_outputs)) {
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

// --- MatMulNBits (com.microsoft, block-quantized weight) structured -------
// --- pruning -----------------------------------------------------------
//
// C++ port of pruning.py's own "MatMulNBits (block-quantized weight)
// structured pruning" section -- its PLAIN (_find_matmul_nbits_chains),
// GATED (_find_matmul_nbits_gated_chains), and (as of this round) the two
// fused-op variants from pruning.py's own "MatMulNBitsMlp/MatMulNBitsQkv
// (fused block-quantized weight) structured pruning" section
// (_find_matmul_nbits_mlp_chains/_apply_matmul_nbits_mlp_chains,
// _find_matmul_nbits_qkv_chains/_apply_matmul_nbits_qkv_chains) -- i.e.
// everything from this section's own `_matmul_nbits_int_attr` through the
// end of `apply_structured_pruning_matmul_nbits`, MINUS
// `com.microsoft::MatMulBnb4` (a different quantization format --
// bitsandbytes FP4/NF4, not this section's int4/int8 packing -- still out of
// scope, being ported by a separate effort). `MatMulNBitsMlp`/
// `MatMulNBitsQkv` matching/struct/slicing code (MatMulNBitsMlpWeight/
// MatchMatMulNBitsMlp/FindMatMulNBitsMlpChains/ApplyMatMulNBitsMlpChains,
// MatMulNBitsQkvWeight/MatchMatMulNBitsQkv/FindMatMulNBitsQkvChains/
// ApplyMatMulNBitsQkvChains) still lives in THIS section, right after
// ApplyMatMulNBitsChains below, reusing every plain-`MatMulNBits` helper
// above unchanged (MatMulNBitsWeight/NBitsSide/WalkToMatMulNBitsConsumer/
// MatMulNBitsDequantized/MatMulNBitsBlockAlignedKeepBlocks/
// SliceNBitsSideConsumer) -- but each fused chain kind is WIRED IN from a
// different top-level entry point than the plain/gated ones just above,
// mirroring how differently-shaped their own pruning units are (see each
// one's own comment for the reasoning):
//   * `MatMulNBitsMlp` co-slices its own `gate`/`up` branches' shared
//     OUTPUT-channel axis -- structurally just another MatMul-chain
//     producer/consumer pair, no attention-node metadata needed at all --
//     so `FindMatMulNBitsMlpChains`/`ApplyMatMulNBitsMlpChains` are called
//     from `ApplyStructuredPruning` below, sharing that function's own
//     `TouchedState` with the plain/gated `MatMulNBits` chains (and every
//     other chain family `ApplyStructuredPruning` already applies) the exact
//     same way `ApplyMatMulNBitsChains` itself already does.
//   * `MatMulNBitsQkv` prunes whole KV groups, which needs the downstream
//     `GroupQueryAttention`/plain `ai.onnx::Attention` consumer's own
//     `num_heads`/`kv_num_heads` metadata to even determine the pruning
//     unit -- so `FindMatMulNBitsQkvChains`/`ApplyMatMulNBitsQkvChains`
//     reuse this file's own "Attention-head pruning" section machinery
//     directly (MatchGqaProducer/MatchOnnxAttentionProducer/
//     HeadColumnIndices/AttnChainOp) and are called from
//     `ApplyAttentionHeadPruning` instead, with their own dedicated
//     producer/consumer-touched bookkeeping local to that call (not shared
//     with `ApplyAttentionChains`'s own -- see ApplyMatMulNBitsQkvChains's
//     own comment for why that's always safe: a `MatMulNBitsQkv` node's own
//     `q_b`/`k_b`/`v_b` weight names can never collide with a plain/GQA
//     chain's own separately-matched Q/K/V producer weight names).
// This split-entry-point wiring is a real, deliberate divergence from
// pruning.py's own structure -- where `_apply_matmul_nbits_mlp_chains` AND
// `_apply_matmul_nbits_qkv_chains` are both called from ONE Python function,
// `apply_structured_pruning_matmul_nbits`, sharing one `producer_touched`/
// `consumer_touched` pair -- but continues, rather than introduces, this
// port's own established precedent: the plain/gated `MatMulNBits` chain
// families were ALREADY folded into `ApplyStructuredPruning` by an earlier
// round of this port (rather than getting their own
// `ApplyStructuredPruningMatMulNBits` C++ entry point mirroring Python's
// separate function), so composing each new fused-op variant with whichever
// EXISTING C++ top-level entry point actually owns the matching machinery
// it needs is the consistent choice, confirmed by reading both
// `ApplyStructuredPruning`'s own existing `MatMulNBits` handling and
// `ApplyAttentionHeadPruning`'s own existing GQA-chain handling before
// choosing (see this file's own git history/PR description for that
// analysis) rather than guessed at.
//
// `MatMulNBitsQkv`'s own scope is narrower than pruning.py's here in one
// more real, deliberately documented way: pruning.py's own
// `_apply_matmul_nbits_qkv_chains` reuses `_slice_or_gather_head_bias` (this
// module's shared dynamic-`attention_bias`-Gather-insertion fix -- see
// pruning.py's own "Attention-head pruning" section comment for the full
// narrative) for its own `attention_bias`/`head_sink` handling. This port
// has no C++ analogue of that machinery AT ALL yet -- not even for the
// PLAIN GQA chains `ApplyOneGqaChain` above already handles (that function
// never touches `attention_bias`/`head_sink`/`k_scale`/`v_scale` either, a
// pre-existing scope gap this round does not attempt to close). Rather than
// silently reproduce that same gap for the new `MatMulNBitsQkv` chain kind
// too, `FindMatMulNBitsQkvChains` adds its own narrow, explicit safety net
// (MatMulNBitsQkvAttentionBiasSafe) declining the WHOLE chain whenever
// `attention_bias`/`attn_mask` is anything other than absent or a provably
// broadcast (never-per-head) constant, and whenever `head_sink` is anything
// other than absent -- see that function's own comment for the exact
// reasoning. `past_key`/`past_value` (constant, guaranteed-empty per
// MatchGqaProducer's own existing precondition) and GQA's own `k_scale`/
// `v_scale` (constant, per-KV-head-shaped) ARE sliced, mirroring
// pruning.py's own identical handling -- both are cheap, bounded, and safe
// to port exactly, unlike the dynamic-Gather-insertion machinery.
//
// `com.microsoft::MatMulNBits` packs a `[N, K]` weight-only-quantized Linear
// layer's weight as `B` (`uint8[N, k_blocks, block_size * bits / 8]` --
// `bits`-wide codes, `bits == 4` two nibbles per byte LOW NIBBLE FIRST,
// `bits == 8` one full byte per code, no packing at all), `scales`
// (`float[N, k_blocks]`, one scale per `(output channel, K-block)`), and an
// OPTIONAL `zero_points` (`uint8[N, ceil(k_blocks * bits / 8)]` nibble/byte-
// packed along the k_blocks axis the same way `B` is packed along its own
// block_size axis, OR an unpacked `float[N, k_blocks]` -- the schema's own
// two documented encodings), defaulting to `2 ** (bits - 1)` when absent.
// Every one of these empirical facts (packing order, default zero point,
// the schema's own two zero_points encodings) is pruning.py's own section
// comment's finding, confirmed there against a real onnxruntime
// InferenceSession round-trip -- not re-derived here.
//
// *** THE CORE CORRECTNESS INVARIANT: "slice codes directly, never
// dequant-requant" ***. This module -- exactly like pruning.py's own -- only
// ever DEQUANTIZES a matched weight (MatMulNBitsDequantized, below) for
// IMPORTANCE RANKING: deciding which output channels to keep. The actual
// rewrite always operates on the ORIGINAL packed `B`/`scales`/`zero_points`
// bytes -- row/column-selecting (and, for a nibble-packed axis, unpacking
// codes to bytes, re-selecting, and re-packing -- never a raw byte slice at
// the wrong granularity) the EXISTING int4/int8 codes, never re-quantizing a
// sliced float weight from scratch. This matters because re-quantization
// would introduce different rounding at every surviving output channel's
// own scale/zero-point choice -- a pruned model's surviving channels must
// produce BIT-IDENTICAL results to the unpruned model's own (real hardware/
// kernel) quantization, not merely numerically close ones. See
// tests/test_structured_pruning_cpp.py's own MatMulNBits coverage, which
// checks this exactly: the pruned graph's own tensors must equal a
// HAND-SLICE of the ORIGINAL quantized codes, never a re-quantization of the
// sliced float weight.
//
// *** THE BLOCK-ALIGNMENT CONSTRAINT ***. A `MatMulNBits` CONSUMER's own `K`
// (input-channel) axis is quantized in whole `block_size`-sized blocks --
// every K-value within one block shares one `(scale, zero_point)` pair, so
// an individual K-column cannot be dropped on its own without re-quantizing
// its entire block (out of scope: this module never invents new quantized
// values). So whenever the CONSUMER side of a chain is itself `MatMulNBits`,
// the producer's own importance-ranked keep-set is checked
// (MatMulNBitsBlockAlignedKeepBlocks) against that consumer's own block
// boundaries: every `block_size`-sized block must be either wholly kept or
// wholly dropped. When it isn't -- an "unlucky" ranking that keeps some but
// not all of a block's K-positions -- the WHOLE chain (producer AND
// consumer) is left completely untouched, never forced into a partial-block
// re-quantization or a producer/consumer keep-set mismatch. A plain-float
// consumer has no such block structure, so any keep-set applies to it
// directly. (The producer's own `N` axis has no such constraint at all --
// dropping some output channels never touches any OTHER channel's own
// `(scale, zero_point)`, since those are computed independently per row.)
//
// Every chain side (producer or consumer) may independently be EITHER a
// `MatMulNBits` node OR a plain-float (directly-constant weight, never
// QDQ-fed) `MatMul`/`Gemm` -- `NBitsSide`, a `std::variant` mirroring
// pruning.py's own `_MatMulNBitsChainSide = Union[_MatMulNBitsWeight,
// _PlainMatMulNBitsPeer]` -- covering the "quantized transformer block
// feeding an unquantized lm_head" (and the symmetric "unquantized embedding
// feeding the first quantized block") export shapes, with the chain-finders
// below requiring at least one side to actually be `MatMulNBits` (an
// all-plain-float pair is FindChains/FindGatedChains's own job, not
// duplicated here -- mirrors pruning.py's own identical filter).
//
// Two deliberate, narrower-than-pruning.py scope decisions specific to this
// C++ port (both consistent with this file's own established narrower-
// subset-of-pruning.py precedent elsewhere):
//   * `scales`/`zero_points` (when unpacked)/`bias` are admitted only as
//     plain `FLOAT` (float32) tensors here -- this file's own float-tensor
//     helpers (ReadFloatTensor/SetFloatTensorData) have no FLOAT16/BFLOAT16
//     support anywhere yet, unlike pruning.py's own `_is_supported_float_
//     dtype`, which additionally admits both. A FLOAT16/BFLOAT16-quantized
//     export is simply never matched here (declined, not mis-handled).
//   * The plain-float peer side only ever recognizes bare `MatMul`/`Gemm`
//     (via the existing MatchMatMulLikeRaw), not pruning.py's own widened
//     `_match_matmul_like` (which also recognizes `com.microsoft::
//     FusedGemm`/`GemmFastGelu`) -- mirroring MatchProducer's own identical,
//     already-established restriction elsewhere in this file.
//
// The chain-finding shape itself is deliberately NARROWER than the plain
// float FindChains/FindGatedChains above: WalkToMatMulNBitsConsumer only
// ever crosses a shape-preserving unary activation (UnaryPassThroughOps) --
// no per-channel Add/Mul/BiasGelu/PRelu/Clip hop, no decomposed-LayerNorm or
// self-gated-activation recognition, no grouped/depthwise structure, no
// residual/Concat-merge topology -- mirroring pruning.py's own
// `_walk_to_matmul_nbits_consumer` exactly (itself deliberately narrower
// than its own `_walk_to_consumer`/`_walk_to_consumer_qdq`, per that
// function's own docstring).

// The live schema's own supported `block_size` values -- this environment's
// real CPU kernel rejects anything outside this set at run time (see
// pruning.py's own section comment for the empirical confirmation).
const std::unordered_set<int64_t>& MatMulNBitsValidBlockSizes() {
  static const std::unordered_set<int64_t> kSizes = {16, 32, 64, 128, 256};
  return kSizes;
}

std::optional<int64_t> MatMulNBitsIntAttr(const onnx::NodeProto& node,
                                          const std::string& name) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name) {
      return attr.i();
    }
  }
  return std::nullopt;
}

int64_t MatMulNBitsIntAttrOr(const onnx::NodeProto& node,
                             const std::string& name, int64_t default_value) {
  auto v = MatMulNBitsIntAttr(node, name);
  return v.has_value() ? *v : default_value;
}

bool MatMulNBitsDimsEqual(const onnx::TensorProto& t,
                          std::initializer_list<int64_t> expected) {
  if (static_cast<size_t>(t.dims_size()) != expected.size()) {
    return false;
  }
  int i = 0;
  for (int64_t e : expected) {
    if (t.dims(i) != e) {
      return false;
    }
    ++i;
  }
  return true;
}

// A `com.microsoft::MatMulNBits` node's block-quantized weight operands,
// matched by MatchMatMulNBits -- see this section's own top comment for the
// schema facts and packing layout this depends on. Every tensor is stored by
// NAME (not a TensorProto*) -- mirroring ProducerMatch/ConsumerMatch's own
// established convention elsewhere in this file: matching happens against a
// read-only InitMap built from `graph->initializer()`, while the later
// Apply step rebuilds its own MUTABLE name->TensorProto* map from
// `graph->mutable_initializer()` to actually slice, so a match can never
// hold a stale/const pointer across that phase boundary. `node` itself
// (needed later to rewrite its own `N`/`K` attribute) is safe to store as a
// pointer: it comes from `graph->mutable_node(i)`, and no node is ever
// inserted or removed by this whole pruning pass. `zero_points_packed`
// mirrors pruning.py's own `_MatMulNBitsWeight.zero_points_packed`: `true`
// for nibble/byte-packed uint8 (same per-row layout as `b_name`, along the
// `k_blocks` axis), `false` for one unpacked float value per block (same
// dtype as `scales_name`). Meaningless when `zero_points_name` is unset.
struct MatMulNBitsWeight {
  onnx::NodeProto* node = nullptr;
  std::string b_name;
  std::string scales_name;
  std::optional<std::string> zero_points_name;
  bool zero_points_packed = false;
  std::optional<std::string> bias_name;
  int64_t N = 0;
  int64_t K = 0;
  int64_t bits = 0;
  int64_t block_size = 0;
  int64_t k_blocks = 0;
};

// A plain-float `MatMul`/vanilla-`Gemm` node's own weight, matched
// (MatchPlainMatMulNBitsPeer) as the OTHER side of a MIXED `MatMulNBits`/
// plain-float chain -- see this section's own top comment for the real
// export shape this covers. Deliberately supports ONLY a directly-constant
// float weight, never a QDQ one (mixing a QDQ scheme with `MatMulNBits`
// remains out of scope, mirroring pruning.py's own identical restriction).
struct PlainMatMulNBitsPeer {
  onnx::NodeProto* node = nullptr;
  std::string w_name;
  bool weight_transposed = false;
  std::optional<std::string> bias_name;
  int64_t out_channels = 0;
  int64_t in_channels = 0;
};

// Mirrors pruning.py's own `_MatMulNBitsChainSide = Union[_MatMulNBitsWeight,
// _PlainMatMulNBitsPeer]`.
using NBitsSide = std::variant<MatMulNBitsWeight, PlainMatMulNBitsPeer>;

// Mirrors pruning.py's own `_matmul_nbits_chain_side_key`: a name uniquely
// identifying the underlying weight tensor `side` resolves to, used by
// ApplyMatMulNBitsChains to detect a shared/tied weight playing the same
// role in more than one chain (and shared, via TouchedState, with every
// other chain family this file already applies over the same graph).
std::string NBitsSideKey(const NBitsSide& side) {
  if (const auto* w = std::get_if<MatMulNBitsWeight>(&side)) {
    return w->b_name;
  }
  return std::get<PlainMatMulNBitsPeer>(side).w_name;
}

onnx::NodeProto* NBitsSideNode(const NBitsSide& side) {
  if (const auto* w = std::get_if<MatMulNBitsWeight>(&side)) {
    return w->node;
  }
  return std::get<PlainMatMulNBitsPeer>(side).node;
}

// If `node` is a `com.microsoft::MatMulNBits` node matching every scope
// boundary this section's own top comment documents, returns the match --
// mirrors pruning.py's own `_match_matmul_nbits` exactly. `None`/`nullopt`
// whenever anything is ambiguous or out of scope, rather than guessing.
std::optional<MatMulNBitsWeight> MatchMatMulNBits(
    onnx::NodeProto* node, const InitMap& init_map,
    const ConsumerMap& consumers_of) {
  if (node->op_type() != "MatMulNBits" ||
      node->domain() != kComMicrosoftDomain) {
    return std::nullopt;
  }
  if (node->input_size() < 3 || node->output_size() != 1) {
    return std::nullopt;
  }
  const std::string& a_name = node->input(0);
  const std::string& b_name = node->input(1);
  const std::string& scales_name = node->input(2);
  if (a_name.empty() || b_name.empty() || scales_name.empty()) {
    return std::nullopt;
  }
  const std::string zp_name =
      (node->input_size() > 3) ? node->input(3) : std::string();
  const std::string g_idx_name =
      (node->input_size() > 4) ? node->input(4) : std::string();
  const std::string bias_name =
      (node->input_size() > 5) ? node->input(5) : std::string();
  if (!g_idx_name.empty()) {
    return std::nullopt;  // GPTQ-style permutation -- declined, out of scope.
  }

  const auto block_size_opt = MatMulNBitsIntAttr(*node, "block_size");
  const auto n_opt = MatMulNBitsIntAttr(*node, "N");
  const auto k_opt = MatMulNBitsIntAttr(*node, "K");
  if (!block_size_opt || !n_opt || !k_opt || *n_opt <= 0 || *k_opt <= 0) {
    return std::nullopt;
  }
  const int64_t block_size = *block_size_opt;
  const int64_t N = *n_opt;
  const int64_t K = *k_opt;
  if (!MatMulNBitsValidBlockSizes().count(block_size)) {
    return std::nullopt;
  }
  if (K % block_size != 0) {
    return std::nullopt;  // Padded/partial final block -- declined.
  }
  const int64_t bits =
      MatMulNBitsIntAttrOr(*node, "bits", 4);  // schema default.
  if (bits != 4 && bits != 8) {
    return std::nullopt;  // Only 4-bit/8-bit packing empirically verified.
  }
  if (MatMulNBitsIntAttrOr(*node, "weight_prepacked", 0) != 0) {
    return std::nullopt;  // EP-specific opaque prepacked layout -- declined.
  }

  const int64_t k_blocks = K / block_size;
  const int64_t blob_size = block_size * bits / 8;

  auto b_it = init_map.find(b_name);
  auto s_it = init_map.find(scales_name);
  if (b_it == init_map.end() || s_it == init_map.end()) {
    return std::nullopt;  // Non-constant B/scales -- can't safely slice them.
  }
  const onnx::TensorProto* b_init = b_it->second;
  const onnx::TensorProto* scales_init = s_it->second;
  if (b_init->data_type() != onnx::TensorProto::UINT8) {
    return std::nullopt;
  }
  if (!MatMulNBitsDimsEqual(*b_init, {N, k_blocks, blob_size})) {
    return std::nullopt;
  }
  // Scope: FLOAT32 only -- see this section's own top comment.
  if (scales_init->data_type() != onnx::TensorProto::FLOAT) {
    return std::nullopt;
  }
  if (!MatMulNBitsDimsEqual(*scales_init, {N, k_blocks})) {
    return std::nullopt;
  }

  const bool has_zp = !zp_name.empty();
  bool zp_packed = false;
  if (has_zp) {
    auto zp_it = init_map.find(zp_name);
    if (zp_it == init_map.end()) {
      return std::nullopt;  // Non-constant zero_points -- can't safely slice
                            // it.
    }
    const onnx::TensorProto* zp_init = zp_it->second;
    const int64_t zp_bytes = (k_blocks * bits + 7) / 8;
    if (zp_init->data_type() == onnx::TensorProto::UINT8) {
      if (!MatMulNBitsDimsEqual(*zp_init, {N, zp_bytes})) {
        return std::nullopt;
      }
      zp_packed = true;
    } else if (zp_init->data_type() == scales_init->data_type()) {
      if (!MatMulNBitsDimsEqual(*zp_init, {N, k_blocks})) {
        return std::nullopt;
      }
      zp_packed = false;
    } else {
      return std::nullopt;
    }
  }

  const bool has_bias = !bias_name.empty();
  if (has_bias) {
    auto bias_it = init_map.find(bias_name);
    if (bias_it == init_map.end() ||
        bias_it->second->data_type() != scales_init->data_type()) {
      return std::nullopt;  // Non-constant, or dtype-mismatched, bias.
    }
    if (!MatMulNBitsDimsEqual(*bias_it->second, {N})) {
      return std::nullopt;
    }
  }

  std::vector<std::string> shared_names = {b_name, scales_name};
  if (has_zp) {
    shared_names.push_back(zp_name);
  }
  if (has_bias) {
    shared_names.push_back(bias_name);
  }
  for (const auto& nm : shared_names) {
    if (ConsumerCount(consumers_of, nm) != 1) {
      return std::nullopt;  // Shared/tied tensor -- another node reads it too.
    }
  }

  MatMulNBitsWeight w;
  w.node = node;
  w.b_name = b_name;
  w.scales_name = scales_name;
  w.zero_points_name =
      has_zp ? std::optional<std::string>(zp_name) : std::nullopt;
  w.zero_points_packed = zp_packed;
  w.bias_name = has_bias ? std::optional<std::string>(bias_name) : std::nullopt;
  w.N = N;
  w.K = K;
  w.bits = bits;
  w.block_size = block_size;
  w.k_blocks = k_blocks;
  return w;
}

// If `node` is a MatMul/vanilla-Gemm (MatchMatMulLikeRaw) whose weight is a
// directly-constant FLOAT rank-2 initializer (never a QDQ-fed one), returns
// the match -- mirrors pruning.py's own `_match_plain_matmul_nbits_peer`
// (narrowed to FLOAT32/MatMul-Gemm-only per this section's own top comment).
std::optional<PlainMatMulNBitsPeer> MatchPlainMatMulNBitsPeer(
    onnx::NodeProto* node, const InitMap& init_map,
    const ConsumerMap& consumers_of) {
  auto m = MatchMatMulLikeRaw(*node);
  if (!m) {
    return std::nullopt;
  }
  auto w_it = init_map.find(m->w_name);
  if (w_it == init_map.end() ||
      w_it->second->data_type() != onnx::TensorProto::FLOAT) {
    return std::nullopt;  // Absent, non-constant, wrong dtype, or QDQ-fed.
  }
  if (w_it->second->dims_size() != 2) {
    return std::nullopt;
  }
  const int axis = m->weight_transposed ? 0 : 1;
  const int64_t out_channels = w_it->second->dims(axis);
  const int64_t in_channels = w_it->second->dims(1 - axis);

  std::optional<std::string> bias_name;
  if (node->op_type() == "Gemm" && node->input_size() == 3 &&
      !node->input(2).empty()) {
    bias_name = node->input(2);
    auto bias_it = init_map.find(*bias_name);
    if (bias_it == init_map.end() ||
        bias_it->second->data_type() != onnx::TensorProto::FLOAT) {
      return std::nullopt;  // Non-constant bias -- can't safely slice it.
    }
    if (!MatMulNBitsDimsEqual(*bias_it->second, {out_channels})) {
      return std::nullopt;
    }
  }

  std::vector<std::string> shared_names = {m->w_name};
  if (bias_name) {
    shared_names.push_back(*bias_name);
  }
  for (const auto& nm : shared_names) {
    if (ConsumerCount(consumers_of, nm) != 1) {
      return std::nullopt;  // Shared/tied tensor -- another node reads it too.
    }
  }

  PlainMatMulNBitsPeer p;
  p.node = node;
  p.w_name = m->w_name;
  p.weight_transposed = m->weight_transposed;
  p.bias_name = bias_name;
  p.out_channels = out_channels;
  p.in_channels = in_channels;
  return p;
}

// --- Bit-packing, mirroring _unpack_nbits_nibbles/_pack_nbits_nibbles (4-bit
// specifically) and _unpack_nbits_codes/_pack_nbits_codes (bits-aware
// generalization: bits==8 delegates to a plain truncating slice/identity
// copy -- one full byte per code, no packing at all -- bits==4 to the
// nibble pack/unpack). Every function here operates on a flat
// `[outer, packed_width]` uint8 buffer (row-major, `outer` independent
// packed rows sharing one packing granularity) -- the shared shape both a
// producer-role N-axis slice (whole rows are independent quantization
// blocks: `outer = N`) and a consumer-role k_blocks-axis slice (`outer = N`
// too, but now COLUMN-selecting `count` positions out of each of those N
// packed rows) actually need, so one implementation covers both `B`'s own
// zero_points and every fused-op zero_points site this section might one
// day extend to.

// Unpacks the last axis of `packed` (`[outer, nbytes]`) into `[outer,
// count]` codes in [0, 15], 2 per byte, LOW NIBBLE FIRST -- the schema's own
// documented layout, empirically confirmed (see this section's own top
// comment). Drops the last, padding, half-byte when `count` is odd.
std::vector<uint8_t> UnpackNibblesLastAxis(const std::vector<uint8_t>& packed,
                                           int64_t outer, int64_t nbytes,
                                           int64_t count) {
  std::vector<uint8_t> out(static_cast<size_t>(outer * count), 0);
  for (int64_t r = 0; r < outer; ++r) {
    const uint8_t* prow = packed.data() + r * nbytes;
    uint8_t* orow = out.data() + r * count;
    for (int64_t j = 0; j < count; ++j) {
      const uint8_t byte = prow[j / 2];
      orow[j] = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    }
  }
  return out;
}

// Inverse of UnpackNibblesLastAxis -- this, rather than a raw byte-slice of
// the original packed tensor, is exactly why this section re-packs instead
// of slicing bytes directly: dropping an ODD number of rows/blocks from the
// MIDDLE of a nibble-packed axis shifts every subsequent kept value's
// nibble parity, silently corrupting the result if not accounted for. Pads
// an odd trailing `count` with a zero high nibble.
std::vector<uint8_t> PackNibblesLastAxis(const std::vector<uint8_t>& vals,
                                         int64_t outer, int64_t count) {
  const int64_t nbytes = (count + 1) / 2;
  std::vector<uint8_t> out(static_cast<size_t>(outer * nbytes), 0);
  for (int64_t r = 0; r < outer; ++r) {
    const uint8_t* vrow = vals.data() + r * count;
    uint8_t* prow = out.data() + r * nbytes;
    for (int64_t j = 0; j < nbytes; ++j) {
      const uint8_t lo = vrow[2 * j] & 0x0F;
      const uint8_t hi = (2 * j + 1 < count) ? (vrow[2 * j + 1] & 0x0F) : 0;
      prow[j] = lo | (hi << 4);
    }
  }
  return out;
}

// `bits`-aware generalization of UnpackNibblesLastAxis -- MatchMatMulNBits
// only ever admits bits in {4, 8}.
std::vector<uint8_t> UnpackCodesLastAxis(const std::vector<uint8_t>& packed,
                                         int64_t outer, int64_t packed_width,
                                         int64_t count, int64_t bits) {
  if (bits == 8) {
    std::vector<uint8_t> out(static_cast<size_t>(outer * count));
    for (int64_t r = 0; r < outer; ++r) {
      std::memcpy(out.data() + r * count, packed.data() + r * packed_width,
                  static_cast<size_t>(count));
    }
    return out;
  }
  return UnpackNibblesLastAxis(packed, outer, packed_width, count);
}

// `bits`-aware generalization of PackNibblesLastAxis.
std::vector<uint8_t> PackCodesLastAxis(const std::vector<uint8_t>& vals,
                                       int64_t outer, int64_t count,
                                       int64_t bits) {
  if (bits == 8) {
    return vals;  // Already [outer, count] -- one byte per code, no packing.
  }
  return PackNibblesLastAxis(vals, outer, count);
}

// Column-select (axis 1) of a fully unpacked-then-repacked `[rows, count]`
// code matrix stored packed along its own LAST axis -- the genuine
// nibble-repack hazard this section's own top comment documents: dropping
// blocks from the MIDDLE of the packed axis shifts every subsequent kept
// block's own nibble parity unless unpacked/resliced/repacked, which is
// exactly what this does. Used by SliceMatMulNBitsConsumerBlocks for a
// packed `zero_points`' own k_blocks axis.
std::vector<uint8_t> RepackColumnSelect(const std::vector<uint8_t>& packed,
                                        int64_t rows, int64_t packed_width,
                                        int64_t count, int64_t bits,
                                        const std::vector<int64_t>& keep_cols) {
  const std::vector<uint8_t> unpacked =
      UnpackCodesLastAxis(packed, rows, packed_width, count, bits);
  const int64_t new_count = static_cast<int64_t>(keep_cols.size());
  std::vector<uint8_t> selected(static_cast<size_t>(rows * new_count));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t j = 0; j < new_count; ++j) {
      selected[static_cast<size_t>(r * new_count + j)] =
          unpacked[static_cast<size_t>(r * count + keep_cols[j])];
    }
  }
  return PackCodesLastAxis(selected, rows, new_count, bits);
}

// --- Tensor <-> flat uint8 buffer, the UINT8 analogue of ReadFloatTensor/
// SetFloatTensorData above (no endianness concerns -- a single-byte dtype).

std::vector<uint8_t> ReadUint8Tensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<uint8_t> out(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    std::memcpy(out.data(), t.raw_data().data(), out.size());
  } else {
    // Per the ONNX spec, UINT8 (like INT8/INT16/UINT16/BOOL) is stored in
    // `int32_data`, not a dedicated field, when raw_data isn't used.
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] =
          static_cast<uint8_t>(t.int32_data(static_cast<int>(i)));
    }
  }
  return out;
}

void SetUint8TensorData(onnx::TensorProto* t, const std::vector<int64_t>& dims,
                        const std::vector<uint8_t>& data) {
  const std::string name = t->name();
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::UINT8);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  t->set_raw_data(
      std::string(reinterpret_cast<const char*>(data.data()), data.size()));
}

// The full float64 `[N, K]` dequantized weight matrix `w` refers to, for
// IMPORTANCE RANKING ONLY -- never written back to the graph (this
// section's own top comment: the "slice codes directly, never dequant-
// requant" invariant). `dequant[n, k] = (code[n, k] - zero_point[n, k /
// block_size]) * scale[n, k / block_size]` -- mirrors pruning.py's own
// `_matmul_nbits_dequantized` exactly. `init_map` here is the Apply phase's
// own MUTABLE name->TensorProto* map (this function is only ever called
// from ApplyMatMulNBitsChains, never during matching).
std::vector<double> MatMulNBitsDequantized(
    const MatMulNBitsWeight& w,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map) {
  const onnx::TensorProto* b_init = init_map.at(w.b_name);
  const onnx::TensorProto* scales_init = init_map.at(w.scales_name);
  const int64_t blob_size = w.block_size * w.bits / 8;

  const std::vector<uint8_t> b_raw =
      ReadUint8Tensor(*b_init);  // [N, k_blocks, blob_size]
  // Unpacking the whole [N * k_blocks, blob_size] buffer at once yields
  // exactly the [N, k_blocks, block_size] == [N, K] code matrix, row-major.
  const std::vector<uint8_t> codes = UnpackCodesLastAxis(
      b_raw, w.N * w.k_blocks, blob_size, w.block_size, w.bits);
  const std::vector<float> scales =
      ReadFloatTensor(*scales_init);  // [N, k_blocks]

  std::vector<double> zp(static_cast<size_t>(w.N * w.k_blocks));
  if (w.zero_points_name) {
    const onnx::TensorProto* zp_init = init_map.at(*w.zero_points_name);
    if (w.zero_points_packed) {
      const std::vector<uint8_t> zp_raw =
          ReadUint8Tensor(*zp_init);  // [N, zp_bytes]
      const int64_t zp_bytes = (w.k_blocks * w.bits + 7) / 8;
      const std::vector<uint8_t> zp_codes =
          UnpackCodesLastAxis(zp_raw, w.N, zp_bytes, w.k_blocks, w.bits);
      for (size_t i = 0; i < zp.size(); ++i) {
        zp[i] = static_cast<double>(zp_codes[i]);
      }
    } else {
      const std::vector<float> zp_f =
          ReadFloatTensor(*zp_init);  // [N, k_blocks]
      for (size_t i = 0; i < zp.size(); ++i) {
        zp[i] = static_cast<double>(zp_f[i]);
      }
    }
  } else {
    // Schema's own documented default: 2 ** (bits - 1).
    std::fill(zp.begin(), zp.end(),
              static_cast<double>(int64_t{1} << (w.bits - 1)));
  }

  std::vector<double> out(static_cast<size_t>(w.N * w.K));
  for (int64_t n = 0; n < w.N; ++n) {
    for (int64_t kb = 0; kb < w.k_blocks; ++kb) {
      const double z = zp[static_cast<size_t>(n * w.k_blocks + kb)];
      const double s =
          static_cast<double>(scales[static_cast<size_t>(n * w.k_blocks + kb)]);
      for (int64_t j = 0; j < w.block_size; ++j) {
        const int64_t idx = n * w.K + kb * w.block_size + j;
        const double code =
            static_cast<double>(codes[static_cast<size_t>(idx)]);
        out[static_cast<size_t>(idx)] = (code - z) * s;
      }
    }
  }
  return out;
}

// `[N, K]` float64 view of one chain PRODUCER's own weight, for IMPORTANCE
// RANKING ONLY -- never written back. A `MatMulNBits` side dequantizes via
// MatMulNBitsDequantized; a plain-float peer is read directly, transposed
// only when NOT already stored `[N, K]` -- mirrors pruning.py's own
// `_matmul_nbits_chain_producer_weight_nk`.
std::vector<double> NBitsSideProducerWeightNK(
    const NBitsSide& side,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map) {
  if (const auto* w = std::get_if<MatMulNBitsWeight>(&side)) {
    return MatMulNBitsDequantized(*w, init_map);
  }
  const auto& p = std::get<PlainMatMulNBitsPeer>(side);
  const onnx::TensorProto* wt = init_map.at(p.w_name);
  const std::vector<float> data = ReadFloatTensor(*wt);
  std::vector<float> nk = p.weight_transposed
                              ? data
                              : TransposeFlat(data, wt->dims(0), wt->dims(1));
  return std::vector<double>(nk.begin(), nk.end());
}

// --- Slicing, mirroring _slice_matmul_nbits_producer_rows/
// _slice_matmul_nbits_consumer_blocks/_slice_matmul_nbits_chain_producer/
// _slice_matmul_nbits_chain_consumer -----------------------------------

// Slices `w`'s own N (output-channel) axis to `keep` (ascending indices) --
// the producer role. `B`/`scales`/`bias` (if present) are all row-sliced
// directly (their leading dim IS `N`); `zero_points` (if present) is
// row-sliced directly when unpacked, or unpacked/row-sliced/re-packed when
// packed (whole-row-safe, no nibble-parity hazard for THIS axis -- see this
// section's own top comment -- but still always re-derived through the
// same unpack/repack path as the genuinely hazardous consumer-block axis
// below, rather than special-cased, mirroring pruning.py's own identical
// choice not to special-case it either). Updates the node's own `N`
// attribute to `len(keep)`.
void SliceMatMulNBitsProducerRows(
    const MatMulNBitsWeight& w,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    const std::vector<int64_t>& keep) {
  const int64_t blob_size = w.block_size * w.bits / 8;
  const int64_t row_width = w.k_blocks * blob_size;
  const int64_t kc = static_cast<int64_t>(keep.size());

  onnx::TensorProto* b = init_map.at(w.b_name);
  const std::vector<uint8_t> b_data = ReadUint8Tensor(*b);
  std::vector<uint8_t> b_out(static_cast<size_t>(kc * row_width));
  for (int64_t i = 0; i < kc; ++i) {
    std::memcpy(b_out.data() + i * row_width,
                b_data.data() + keep[i] * row_width,
                static_cast<size_t>(row_width));
  }
  SetUint8TensorData(b, {kc, w.k_blocks, blob_size}, b_out);

  onnx::TensorProto* scales = init_map.at(w.scales_name);
  const std::vector<float> s_data = ReadFloatTensor(*scales);
  std::vector<float> s_out(static_cast<size_t>(kc * w.k_blocks));
  for (int64_t i = 0; i < kc; ++i) {
    std::memcpy(s_out.data() + i * w.k_blocks,
                s_data.data() + keep[i] * w.k_blocks,
                static_cast<size_t>(w.k_blocks) * sizeof(float));
  }
  SetFloatTensorData(scales, {kc, w.k_blocks}, s_out);

  if (w.zero_points_name) {
    onnx::TensorProto* zp = init_map.at(*w.zero_points_name);
    if (w.zero_points_packed) {
      const int64_t zp_bytes = (w.k_blocks * w.bits + 7) / 8;
      const std::vector<uint8_t> zp_data = ReadUint8Tensor(*zp);
      const std::vector<uint8_t> unpacked =
          UnpackCodesLastAxis(zp_data, w.N, zp_bytes, w.k_blocks, w.bits);
      std::vector<uint8_t> kept(static_cast<size_t>(kc * w.k_blocks));
      for (int64_t i = 0; i < kc; ++i) {
        std::memcpy(kept.data() + i * w.k_blocks,
                    unpacked.data() + keep[i] * w.k_blocks,
                    static_cast<size_t>(w.k_blocks));
      }
      const std::vector<uint8_t> repacked =
          PackCodesLastAxis(kept, kc, w.k_blocks, w.bits);
      SetUint8TensorData(zp, {kc, zp_bytes}, repacked);
    } else {
      const std::vector<float> zp_data = ReadFloatTensor(*zp);
      std::vector<float> zp_out(static_cast<size_t>(kc * w.k_blocks));
      for (int64_t i = 0; i < kc; ++i) {
        std::memcpy(zp_out.data() + i * w.k_blocks,
                    zp_data.data() + keep[i] * w.k_blocks,
                    static_cast<size_t>(w.k_blocks) * sizeof(float));
      }
      SetFloatTensorData(zp, {kc, w.k_blocks}, zp_out);
    }
  }

  if (w.bias_name) {
    SliceLastAxis(init_map.at(*w.bias_name), keep);
  }

  SetOrAddIntAttr(w.node, "N", kc);
}

// Returns the ascending block indices (into the consumer's own
// `k_blocks`-sized block axis) that `keep` (ascending element indices into
// its `K`-length input axis) corresponds to when every `block_size`-sized
// block is either wholly present in `keep` or wholly absent -- or
// `nullopt` when some block is only partially kept, meaning this consumer
// cannot be safely pruned to this exact `keep` set at all -- mirrors
// pruning.py's own `_matmul_nbits_block_aligned_keep_blocks`. Since
// MatchMatMulNBits already declines any `K` not an exact multiple of
// `block_size`, every block here is full-width -- there is no partial
// FINAL block to special-case.
std::optional<std::vector<int64_t>> MatMulNBitsBlockAlignedKeepBlocks(
    const std::vector<int64_t>& keep, int64_t k_blocks, int64_t block_size) {
  std::unordered_set<int64_t> keep_set(keep.begin(), keep.end());
  std::vector<int64_t> keep_blocks;
  for (int64_t kb = 0; kb < k_blocks; ++kb) {
    const int64_t k0 = kb * block_size;
    int64_t overlap = 0;
    for (int64_t k = k0; k < k0 + block_size; ++k) {
      if (keep_set.count(k)) {
        ++overlap;
      }
    }
    if (overlap == block_size) {
      keep_blocks.push_back(kb);
    } else if (overlap > 0) {
      return std::nullopt;  // Partial block -- not block-aligned, decline.
    }
  }
  return keep_blocks;
}

// Drops entire `k_blocks`-axis blocks NOT in `keep_blocks` (ascending block
// indices) from `w`'s own `B`/`scales`/`zero_points` -- the consumer role.
// Never invoked with a non-block-aligned `keep_blocks` (see
// MatMulNBitsBlockAlignedKeepBlocks, which computes and validates it before
// this is ever called). `B` needs only a whole-block byte copy per
// `(n, kept k_block)` pair (each `blob_size`-byte group is already a
// self-contained packed block, safe to reorder/drop as a unit); a packed
// `zero_points` genuinely must be unpacked/resliced/repacked (RepackColumn
// Select) -- the block axis IS ITS OWN packed axis (unlike `B`, whose
// packing is along `block_size`, an entirely different axis untouched by
// this slice). Updates the node's own `K` attribute to `len(keep_blocks) *
// block_size`. Mirrors pruning.py's own `_slice_matmul_nbits_consumer_
// blocks`.
void SliceMatMulNBitsConsumerBlocks(
    const MatMulNBitsWeight& w,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    const std::vector<int64_t>& keep_blocks) {
  const int64_t blob_size = w.block_size * w.bits / 8;
  const int64_t new_kb = static_cast<int64_t>(keep_blocks.size());

  onnx::TensorProto* b = init_map.at(w.b_name);
  const std::vector<uint8_t> b_data =
      ReadUint8Tensor(*b);  // [N, k_blocks, blob_size]
  std::vector<uint8_t> b_out(static_cast<size_t>(w.N * new_kb * blob_size));
  for (int64_t n = 0; n < w.N; ++n) {
    for (int64_t j = 0; j < new_kb; ++j) {
      const uint8_t* src =
          b_data.data() + (n * w.k_blocks + keep_blocks[j]) * blob_size;
      uint8_t* dst = b_out.data() + (n * new_kb + j) * blob_size;
      std::memcpy(dst, src, static_cast<size_t>(blob_size));
    }
  }
  SetUint8TensorData(b, {w.N, new_kb, blob_size}, b_out);

  onnx::TensorProto* scales = init_map.at(w.scales_name);
  const std::vector<float> s_data = ReadFloatTensor(*scales);  // [N, k_blocks]
  std::vector<float> s_out(static_cast<size_t>(w.N * new_kb));
  for (int64_t n = 0; n < w.N; ++n) {
    for (int64_t j = 0; j < new_kb; ++j) {
      s_out[static_cast<size_t>(n * new_kb + j)] =
          s_data[static_cast<size_t>(n * w.k_blocks + keep_blocks[j])];
    }
  }
  SetFloatTensorData(scales, {w.N, new_kb}, s_out);

  if (w.zero_points_name) {
    onnx::TensorProto* zp = init_map.at(*w.zero_points_name);
    if (w.zero_points_packed) {
      const int64_t zp_bytes = (w.k_blocks * w.bits + 7) / 8;
      const std::vector<uint8_t> zp_data = ReadUint8Tensor(*zp);
      const std::vector<uint8_t> repacked = RepackColumnSelect(
          zp_data, w.N, zp_bytes, w.k_blocks, w.bits, keep_blocks);
      const int64_t new_zp_bytes = (new_kb * w.bits + 7) / 8;
      SetUint8TensorData(zp, {w.N, new_zp_bytes}, repacked);
    } else {
      const std::vector<float> zp_data = ReadFloatTensor(*zp);  // [N, k_blocks]
      std::vector<float> zp_out(static_cast<size_t>(w.N * new_kb));
      for (int64_t n = 0; n < w.N; ++n) {
        for (int64_t j = 0; j < new_kb; ++j) {
          zp_out[static_cast<size_t>(n * new_kb + j)] =
              zp_data[static_cast<size_t>(n * w.k_blocks + keep_blocks[j])];
        }
      }
      SetFloatTensorData(zp, {w.N, new_kb}, zp_out);
    }
  }

  SetOrAddIntAttr(w.node, "K", new_kb * w.block_size);
}

// Slices one chain PRODUCER's own output channels to `keep` -- dispatches
// to SliceMatMulNBitsProducerRows for a `MatMulNBits` side, or a direct
// SliceProducerWeight (plus its own bias, if present) for a plain-float
// peer. Mirrors pruning.py's own `_slice_matmul_nbits_chain_producer`.
void SliceNBitsSideProducer(
    const NBitsSide& side,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    const std::vector<int64_t>& keep) {
  if (const auto* w = std::get_if<MatMulNBitsWeight>(&side)) {
    SliceMatMulNBitsProducerRows(*w, init_map, keep);
    return;
  }
  const auto& p = std::get<PlainMatMulNBitsPeer>(side);
  SliceProducerWeight(init_map.at(p.w_name), p.weight_transposed, keep, false);
  if (p.bias_name) {
    SliceLastAxis(init_map.at(*p.bias_name), keep);
  }
}

// Slices one chain CONSUMER's own input channels to `keep` -- a
// `MatMulNBits` side requires `keep` to already be whole, block-aligned
// BLOCK indices (checked by the caller via MatMulNBitsBlockAlignedKeepBlocks
// before this is ever invoked for that side) and dispatches to
// SliceMatMulNBitsConsumerBlocks; a plain-float peer has no block structure
// at all, so `keep` there is ordinary element indices, dispatched straight
// to SliceConsumerWeight. Mirrors pruning.py's own `_slice_matmul_nbits_
// chain_consumer`.
void SliceNBitsSideConsumer(
    const NBitsSide& side,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    const std::vector<int64_t>& keep) {
  if (const auto* w = std::get_if<MatMulNBitsWeight>(&side)) {
    SliceMatMulNBitsConsumerBlocks(*w, init_map, keep);
    return;
  }
  const auto& p = std::get<PlainMatMulNBitsPeer>(side);
  SliceConsumerWeight(init_map.at(p.w_name), p.weight_transposed, keep, false);
}

// --- MatMul/Gemm-chain plain and gated chain finding, mirroring
// _walk_to_matmul_nbits_consumer/_find_matmul_nbits_chains/
// _trace_gate_producer_backward_matmul_nbits/_find_matmul_nbits_gated_chains

// From tensor `start` (a `MatMulNBits` OR plain-float MatMul/Gemm
// producer's own output), walks forward through shape-preserving unary
// activations (UnaryPassThroughOps) with no other consumer anywhere along
// the way, until EITHER a `MatMulNBits` consumer OR a plain-float MatMul/
// vanilla-Gemm consumer (MatchPlainMatMulNBitsPeer) is found whose
// input-channel count matches `n_channels`. No per-channel Add/Mul/
// BiasGelu/PRelu/Clip hop, no decomposed-LayerNorm/self-gated-activation
// recognition, no branch -- narrower than WalkToConsumer above, mirroring
// pruning.py's own `_walk_to_matmul_nbits_consumer` exactly (see this
// section's own top comment). Returns `nullopt` if the walk runs out of
// hops, hits a branch, or never reaches such a consumer.
struct MatMulNBitsWalkResult {
  NBitsSide consumer;
  std::vector<onnx::NodeProto*> chain_ops;
};

std::optional<MatMulNBitsWalkResult> WalkToMatMulNBitsConsumer(
    const std::string& start, const InitMap& init_map,
    const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int64_t n_channels,
    int max_hops) {
  std::vector<onnx::NodeProto*> chain_ops;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    auto cit = consumers_of.find(cur);
    if (cit == consumers_of.end() || cit->second.size() != 1) {
      return std::nullopt;
    }
    onnx::NodeProto* nxt = cit->second[0];

    if (nxt->op_type() == "MatMulNBits" && nxt->input_size() > 0 &&
        nxt->input(0) == cur) {
      auto w = MatchMatMulNBits(nxt, init_map, consumers_of);
      if (!w || w->K != n_channels) {
        return std::nullopt;
      }
      return MatMulNBitsWalkResult{NBitsSide(*w), std::move(chain_ops)};
    }

    auto mm = MatchMatMulLikeRaw(*nxt);
    if (mm && mm->x_name == cur) {
      auto peer = MatchPlainMatMulNBitsPeer(nxt, init_map, consumers_of);
      if (!peer || peer->in_channels != n_channels) {
        return std::nullopt;
      }
      return MatMulNBitsWalkResult{NBitsSide(*peer), std::move(chain_ops)};
    }

    if (!(UnaryPassThroughOps().count(nxt->op_type()) != 0 &&
          nxt->input_size() == 1 && nxt->input(0) == cur &&
          nxt->output_size() == 1)) {
      return std::nullopt;
    }
    const std::string& out2 = nxt->output(0);
    auto oc = consumers_of.find(out2);
    if (oc == consumers_of.end() || oc->second.size() != 1 ||
        graph_outputs.count(out2)) {
      return std::nullopt;
    }
    chain_ops.push_back(nxt);
    cur = out2;
  }
  return std::nullopt;
}

struct MatMulNBitsChain {
  NBitsSide producer;
  std::vector<onnx::NodeProto*> chain_ops;
  NBitsSide consumer;
  int64_t n_channels;
};

// The `MatMulNBits` analogue of FindChains: every producer/consumer pair
// connected by WalkToMatMulNBitsConsumer where AT LEAST ONE side is a
// `MatMulNBits` node (a plain-float-to-plain-float pair is FindChains's own
// job, not duplicated here). Mirrors pruning.py's own `_find_matmul_nbits_
// chains`.
std::vector<MatMulNBitsChain> FindMatMulNBitsChains(onnx::GraphProto* graph) {
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

  std::vector<MatMulNBitsChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    std::optional<NBitsSide> producer;
    int64_t n_channels = 0;
    if (node->op_type() == "MatMulNBits") {
      auto w = MatchMatMulNBits(node, init_map, consumers_of);
      if (!w) {
        continue;
      }
      n_channels = w->N;
      producer = NBitsSide(*w);
    } else {
      auto peer = MatchPlainMatMulNBitsPeer(node, init_map, consumers_of);
      if (!peer) {
        continue;
      }
      n_channels = peer->out_channels;
      producer = NBitsSide(*peer);
    }

    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    auto found =
        WalkToMatMulNBitsConsumer(out_name, init_map, consumers_of,
                                  graph_outputs, n_channels, kMaxChainHops);
    if (!found) {
      continue;
    }
    if (std::holds_alternative<PlainMatMulNBitsPeer>(*producer) &&
        std::holds_alternative<PlainMatMulNBitsPeer>(found->consumer)) {
      continue;  // Both plain float -- FindChains's own job.
    }
    chains.push_back(MatMulNBitsChain{std::move(*producer),
                                      std::move(found->chain_ops),
                                      std::move(found->consumer), n_channels});
  }
  return chains;
}

struct NBitsProducerInfo {
  onnx::NodeProto* node;
  NBitsSide side;
  int64_t n_channels;
};

// The `MatMulNBits` analogue of TraceGateProducerBackward: walks backward
// from `tensor_name` through unary activation ops until it resolves to a
// `MatMulNBits`-or-plain-float MatMul/vanilla-Gemm producer's raw output
// (`producer_infos`, built from MatchMatMulNBits/MatchPlainMatMulNBitsPeer
// -- see FindMatMulNBitsGatedChains). Duplicated rather than shared with the
// plain-float walker above for the identical reason FindGatedChains's own
// duplication exists (structurally different `producer_infos` value type).
std::optional<std::pair<NBitsProducerInfo, std::vector<onnx::NodeProto*>>>
TraceGateProducerBackwardNBits(
    const std::string& tensor_name,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const std::unordered_map<std::string, NBitsProducerInfo>& producer_infos,
    const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int max_hops) {
  std::vector<onnx::NodeProto*> pre_ops;
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

struct MatMulNBitsGatedChain {
  NBitsSide producer_a;
  std::vector<onnx::NodeProto*> producer_a_pre_ops;
  NBitsSide producer_b;
  std::vector<onnx::NodeProto*> producer_b_pre_ops;
  std::vector<onnx::NodeProto*> chain_ops;
  NBitsSide consumer;
  int64_t n_channels;
};

// The `MatMulNBits` analogue of FindGatedChains: recognizes the same gated
// (SwiGLU/GeGLU) MatMul/vanilla-Gemm pair -- gate_proj/up_proj sharing one
// input, combined via a plain elementwise `Mul` of two non-constant
// operands (each optionally through its own UnaryPassThroughOps activation)
// or the native `SwiGLU` node -- feeding into exactly one downstream
// consumer whose input-channel count matches (WalkToMatMulNBitsConsumer),
// except gate_proj/up_proj/down_proj may now each independently be EITHER a
// `MatMulNBits` node OR a plain-float MatMul/vanilla-Gemm peer, requiring at
// least one of the three to actually be `MatMulNBits` (an all-plain-float
// triple is FindGatedChains's own job, not duplicated here). Mirrors
// pruning.py's own `_find_matmul_nbits_gated_chains`.
std::vector<MatMulNBitsGatedChain> FindMatMulNBitsGatedChains(
    onnx::GraphProto* graph) {
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
  std::unordered_map<std::string, NBitsProducerInfo> producer_infos;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
    if (node->op_type() == "MatMulNBits") {
      auto w = MatchMatMulNBits(node, init_map, consumers_of);
      if (w) {
        producer_infos[node->output(0)] =
            NBitsProducerInfo{node, NBitsSide(*w), w->N};
      }
    } else {
      auto peer = MatchPlainMatMulNBitsPeer(node, init_map, consumers_of);
      if (peer) {
        producer_infos[node->output(0)] =
            NBitsProducerInfo{node, NBitsSide(*peer), peer->out_channels};
      }
    }
  }

  std::vector<MatMulNBitsGatedChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    std::optional<NBitsProducerInfo> info_a, info_b;
    std::vector<onnx::NodeProto*> pre_a, pre_b;

    if (node->op_type() == "Mul" && node->input_size() == 2 &&
        node->output_size() == 1) {
      const std::string& a_name = node->input(0);
      const std::string& b_name = node->input(1);
      if (a_name == b_name || init_map.count(a_name) ||
          init_map.count(b_name)) {
        continue;
      }
      auto trace_a = TraceGateProducerBackwardNBits(
          a_name, node_by_output, producer_infos, consumers_of, graph_outputs,
          kMaxChainHops);
      auto trace_b = TraceGateProducerBackwardNBits(
          b_name, node_by_output, producer_infos, consumers_of, graph_outputs,
          kMaxChainHops);
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
    auto found = WalkToMatMulNBitsConsumer(out_name, init_map, consumers_of,
                                           graph_outputs, info_a->n_channels,
                                           kMaxChainHops);
    if (!found) {
      continue;
    }

    if (std::holds_alternative<PlainMatMulNBitsPeer>(info_a->side) &&
        std::holds_alternative<PlainMatMulNBitsPeer>(info_b->side) &&
        std::holds_alternative<PlainMatMulNBitsPeer>(found->consumer)) {
      continue;  // All plain float -- FindGatedChains's own job.
    }

    chains.push_back(
        MatMulNBitsGatedChain{info_a->side, std::move(pre_a), info_b->side,
                              std::move(pre_b), std::move(found->chain_ops),
                              std::move(found->consumer), info_a->n_channels});
  }
  return chains;
}

// --- Apply, mirroring apply_structured_pruning_matmul_nbits's own plain and
// gated per-chain loops (the fused MLP/QKV loops it also drives are out of
// scope here, per this section's own top comment) ------------------------

// Applies both plain (`chains`) and gated (`gated_chains`) MatMulNBits
// chains -- ranks the producer's output channels by L2 norm of their own
// (dequantized, for a `MatMulNBits` producer) weight row, drops the
// lowest-`sparsity`-fraction, and -- only when the CONSUMER side is itself
// `MatMulNBits` -- requires that keep-set to land on whole `block_size`
// blocks (MatMulNBitsBlockAlignedKeepBlocks), declining the WHOLE chain
// otherwise (see this section's own top comment for why). Shares `touched`
// with every other chain family ApplyStructuredPruning already applies over
// this same graph, so a weight already resized by an ordinary MatMul/Conv
// chain (or vice versa) can never be double-resized here.
void ApplyMatMulNBitsChains(onnx::GraphProto* graph,
                            std::vector<MatMulNBitsChain>& chains,
                            std::vector<MatMulNBitsGatedChain>& gated_chains,
                            double sparsity, TouchedState& touched) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }

  auto row_norms = [](const std::vector<double>& w_nk, int64_t n) {
    const int64_t k = static_cast<int64_t>(w_nk.size()) / n;
    std::vector<double> importance(static_cast<size_t>(n), 0.0);
    for (int64_t c = 0; c < n; ++c) {
      double sq = 0.0;
      for (int64_t j = 0; j < k; ++j) {
        const double v = w_nk[static_cast<size_t>(c * k + j)];
        sq += v * v;
      }
      importance[static_cast<size_t>(c)] = sq;
    }
    return importance;
  };

  for (auto& chain : chains) {
    const std::string p_key = NBitsSideKey(chain.producer);
    const std::string c_key = NBitsSideKey(chain.consumer);
    if (p_key == c_key) {
      continue;  // Degenerate (the same weight in both roles).
    }
    if (touched.producer.count(p_key) || touched.consumer.count(c_key)) {
      continue;  // A shared/tied weight another chain already resized.
    }

    const int64_t n = chain.n_channels;
    const int64_t keep_count = std::max<int64_t>(
        1, n - std::llround(static_cast<double>(n) * sparsity));
    if (keep_count >= n) {
      continue;  // Rounds down to nothing for this layer -- no-op.
    }

    const std::vector<double> w_nk =
        NBitsSideProducerWeightNK(chain.producer, init_map);
    std::vector<double> importance = row_norms(w_nk, n);
    for (double& v : importance) {
      v = std::sqrt(v);
    }
    const std::vector<int64_t> keep =
        TopKIndicesAscending(importance, keep_count);

    std::vector<int64_t> consumer_keep;
    if (const auto* cw = std::get_if<MatMulNBitsWeight>(&chain.consumer)) {
      auto keep_blocks =
          MatMulNBitsBlockAlignedKeepBlocks(keep, cw->k_blocks, cw->block_size);
      if (!keep_blocks) {
        continue;  // Non-block-aligned request -- decline, see top comment.
      }
      consumer_keep = std::move(*keep_blocks);
    } else {
      consumer_keep = keep;  // Plain-float consumer -- no block structure.
    }

    SliceNBitsSideProducer(chain.producer, init_map, keep);
    SliceNBitsSideConsumer(chain.consumer, init_map, consumer_keep);

    touched.producer.insert(p_key);
    touched.consumer.insert(c_key);
    touched.stale_value_info.insert(NBitsSideNode(chain.producer)->output(0));
    for (auto* op : chain.chain_ops) {
      touched.stale_value_info.insert(op->output(0));
    }
  }

  for (auto& gchain : gated_chains) {
    const std::string pa_key = NBitsSideKey(gchain.producer_a);
    const std::string pb_key = NBitsSideKey(gchain.producer_b);
    const std::string c_key = NBitsSideKey(gchain.consumer);
    if (pa_key == pb_key || pa_key == c_key || pb_key == c_key) {
      continue;  // Degenerate (a weight tied across two roles).
    }
    if (touched.producer.count(pa_key) || touched.producer.count(pb_key) ||
        touched.consumer.count(c_key)) {
      continue;  // A shared/tied weight another chain already resized.
    }

    const int64_t n = gchain.n_channels;
    const int64_t keep_count = std::max<int64_t>(
        1, n - std::llround(static_cast<double>(n) * sparsity));
    if (keep_count >= n) {
      continue;  // Rounds down to nothing for this layer -- no-op.
    }

    const std::vector<double> wa_nk =
        NBitsSideProducerWeightNK(gchain.producer_a, init_map);
    const std::vector<double> wb_nk =
        NBitsSideProducerWeightNK(gchain.producer_b, init_map);
    // Combined (root-sum-square) importance across both producers -- mirrors
    // pruning.py's own `_matmul_nbits_gated_channel_importance` (L2 branch
    // only; this C++ port has no `importance_norm` choice, always L2,
    // matching this file's own established scope elsewhere).
    std::vector<double> sq_a = row_norms(wa_nk, n);
    const std::vector<double> sq_b = row_norms(wb_nk, n);
    std::vector<double> importance(static_cast<size_t>(n));
    for (int64_t c = 0; c < n; ++c) {
      importance[static_cast<size_t>(c)] = std::sqrt(
          sq_a[static_cast<size_t>(c)] + sq_b[static_cast<size_t>(c)]);
    }
    const std::vector<int64_t> keep =
        TopKIndicesAscending(importance, keep_count);

    std::vector<int64_t> consumer_keep;
    if (const auto* cw = std::get_if<MatMulNBitsWeight>(&gchain.consumer)) {
      auto keep_blocks =
          MatMulNBitsBlockAlignedKeepBlocks(keep, cw->k_blocks, cw->block_size);
      if (!keep_blocks) {
        continue;  // Non-block-aligned -- decline the whole gated group.
      }
      consumer_keep = std::move(*keep_blocks);
    } else {
      consumer_keep = keep;  // Plain-float consumer -- no block structure.
    }

    SliceNBitsSideProducer(gchain.producer_a, init_map, keep);
    SliceNBitsSideProducer(gchain.producer_b, init_map, keep);
    SliceNBitsSideConsumer(gchain.consumer, init_map, consumer_keep);

    touched.producer.insert(pa_key);
    touched.producer.insert(pb_key);
    touched.consumer.insert(c_key);
    touched.stale_value_info.insert(
        NBitsSideNode(gchain.producer_a)->output(0));
    for (auto* op : gchain.producer_a_pre_ops) {
      touched.stale_value_info.insert(op->output(0));
    }
    touched.stale_value_info.insert(
        NBitsSideNode(gchain.producer_b)->output(0));
    for (auto* op : gchain.producer_b_pre_ops) {
      touched.stale_value_info.insert(op->output(0));
    }
    for (auto* op : gchain.chain_ops) {
      touched.stale_value_info.insert(op->output(0));
    }
  }
}

// --- MatMulNBitsMlp/MatMulNBitsQkv (fused block-quantized weight)
// structured pruning, mirroring pruning.py's own section of (almost) the
// same name -- see this section's own top comment (above
// ApplyMatMulNBitsChains) for the empirical schema facts both fused ops
// share, and for why each chain kind below is wired into a DIFFERENT
// top-level entry point (ApplyStructuredPruning for MatMulNBitsMlp,
// ApplyAttentionHeadPruning for MatMulNBitsQkv) rather than one shared
// with plain/gated MatMulNBits chains here. ---------------------------

// Row-slices (N/Nq/Nkv axis) `b_name`/`scales_name`/`bias_name` (if present)
// in place, given directly by NAME rather than through a MatMulNBitsWeight
// -- the zero_points-FREE analogue of SliceMatMulNBitsProducerRows, since
// neither `MatMulNBitsMlp` nor `MatMulNBitsQkv` has a `zero_points` input at
// all (see this section's own top comment). Never touches any node
// attribute itself -- `MatMulNBitsMlp`'s single `N` is shared by two
// branches (harmlessly redundant to set twice) and `MatMulNBitsQkv`'s own
// `Nq`/`Nkv` aren't literally named `"N"` at all -- so the caller sets
// whichever attribute is right, once, itself. Mirrors pruning.py's own
// `_slice_matmul_nbits_rows_no_zp`.
void SliceMatMulNBitsRowsNoZp(
    const std::string& b_name, const std::string& scales_name,
    const std::optional<std::string>& bias_name, int64_t block_size,
    int64_t bits, int64_t k_blocks,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    const std::vector<int64_t>& keep) {
  const int64_t blob_size = block_size * bits / 8;
  const int64_t row_width = k_blocks * blob_size;
  const int64_t kc = static_cast<int64_t>(keep.size());

  onnx::TensorProto* b = init_map.at(b_name);
  const std::vector<uint8_t> b_data = ReadUint8Tensor(*b);
  std::vector<uint8_t> b_out(static_cast<size_t>(kc * row_width));
  for (int64_t i = 0; i < kc; ++i) {
    std::memcpy(b_out.data() + i * row_width,
                b_data.data() + keep[i] * row_width,
                static_cast<size_t>(row_width));
  }
  SetUint8TensorData(b, {kc, k_blocks, blob_size}, b_out);

  onnx::TensorProto* scales = init_map.at(scales_name);
  const std::vector<float> s_data = ReadFloatTensor(*scales);
  std::vector<float> s_out(static_cast<size_t>(kc * k_blocks));
  for (int64_t i = 0; i < kc; ++i) {
    std::memcpy(s_out.data() + i * k_blocks, s_data.data() + keep[i] * k_blocks,
                static_cast<size_t>(k_blocks) * sizeof(float));
  }
  SetFloatTensorData(scales, {kc, k_blocks}, s_out);

  if (bias_name) {
    SliceLastAxis(init_map.at(*bias_name), keep);
  }
}

// Slices axis 1 (the second dimension) of a rank>=2 FLOAT tensor `t` in
// place to `keep`, leaving every other axis untouched -- mirrors
// pruning.py's own `_slice_axis1` (used there for GQA's own `past_key`/
// `past_value` BNSH cache and `k_scale`/`v_scale`), generalized here to any
// rank via the existing flat-buffer SliceAxis1 helper (`inner` = the
// flattened size of every trailing axis past axis 1).
void SliceFloatTensorAxis1(onnx::TensorProto* t,
                           const std::vector<int64_t>& keep) {
  std::vector<int64_t> dims(t->dims().begin(), t->dims().end());
  const std::vector<float> data = ReadFloatTensor(*t);
  int64_t inner = 1;
  for (size_t i = 2; i < dims.size(); ++i) {
    inner *= dims[i];
  }
  const std::vector<float> out =
      SliceAxis1(data, dims[0], dims[1], inner, keep);
  std::vector<int64_t> new_dims = dims;
  new_dims[1] = static_cast<int64_t>(keep.size());
  SetFloatTensorData(t, new_dims, out);
}

// A matched `com.microsoft::MatMulNBitsMlp` node's two block-quantized
// weight operands (`gate`/`up`) -- mirrors pruning.py's own
// `_MatMulNBitsMlpWeight`. No `zero_points` field at all: confirmed via live
// schema introspection (pruning.py's own section comment) that neither this
// op nor `MatMulNBitsQkv` has a `zero_points` input on any weight slot --
// every one implicitly uses the schema's own default zero point,
// `2 ** (bits - 1)`.
struct MatMulNBitsMlpWeight {
  onnx::NodeProto* node = nullptr;
  std::string gate_b_name, gate_scales_name;
  std::optional<std::string> gate_bias_name;
  std::string up_b_name, up_scales_name;
  std::optional<std::string> up_bias_name;
  int64_t N = 0;
  int64_t K = 0;
  int64_t bits = 0;
  int64_t block_size = 0;
  int64_t k_blocks = 0;
};

// If `node` is a `com.microsoft::MatMulNBitsMlp` node matching every scope
// boundary this section's own top comment documents, returns the match --
// mirrors pruning.py's own `_match_matmul_nbits_mlp`. `skip`/`norm_scale`
// (inputs 1/2), when present, are never inspected at all (not even read) --
// this matcher, like the plain `MatMulNBits` one above, only ever prunes
// `gate`'s/`up`'s own shared OUTPUT (`N`) axis, never their shared INPUT
// (`K`) axis those two operands would need re-slicing for. `scales`/`bias`
// restricted to FLOAT32 -- the same narrower-than-pruning.py scope
// MatchMatMulNBits above already establishes for plain `MatMulNBits`
// (pruning.py's own schema allows FLOAT16/BFLOAT16 there too).
std::optional<MatMulNBitsMlpWeight> MatchMatMulNBitsMlp(
    onnx::NodeProto* node, const InitMap& init_map,
    const ConsumerMap& consumers_of) {
  if (node->op_type() != "MatMulNBitsMlp" ||
      node->domain() != kComMicrosoftDomain) {
    return std::nullopt;
  }
  if (node->input_size() < 8 || node->output_size() < 1) {
    return std::nullopt;
  }
  const std::string& a_name = node->input(0);
  const std::string& gate_b_name = node->input(3);
  const std::string& gate_scales_name = node->input(4);
  const std::string gate_bias_name = node->input(5);
  const std::string& up_b_name = node->input(6);
  const std::string& up_scales_name = node->input(7);
  const std::string up_bias_name =
      (node->input_size() > 8) ? node->input(8) : std::string();
  if (a_name.empty() || gate_b_name.empty() || gate_scales_name.empty() ||
      up_b_name.empty() || up_scales_name.empty()) {
    return std::nullopt;
  }
  if (gate_b_name == up_b_name) {
    return std::nullopt;  // Degenerate -- can't independently slice a shared
                          // weight.
  }

  const auto block_size_opt = MatMulNBitsIntAttr(*node, "block_size");
  const auto n_opt = MatMulNBitsIntAttr(*node, "N");
  const auto k_opt = MatMulNBitsIntAttr(*node, "K");
  if (!block_size_opt || !n_opt || !k_opt || *n_opt <= 0 || *k_opt <= 0) {
    return std::nullopt;
  }
  const int64_t block_size = *block_size_opt;
  const int64_t N = *n_opt;
  const int64_t K = *k_opt;
  if (!MatMulNBitsValidBlockSizes().count(block_size)) {
    return std::nullopt;
  }
  if (K % block_size != 0) {
    return std::nullopt;
  }
  const int64_t bits = MatMulNBitsIntAttrOr(*node, "bits", 4);
  if (bits != 4 && bits != 8) {
    return std::nullopt;
  }

  const int64_t k_blocks = K / block_size;
  const int64_t blob_size = block_size * bits / 8;

  auto resolve_branch = [&](const std::string& b_name,
                            const std::string& scales_name,
                            const std::string& bias_name)
      -> std::optional<
          std::tuple<std::string, std::string, std::optional<std::string>>> {
    auto b_it = init_map.find(b_name);
    auto s_it = init_map.find(scales_name);
    if (b_it == init_map.end() || s_it == init_map.end()) {
      return std::nullopt;  // Non-constant B/scales -- can't safely slice.
    }
    if (b_it->second->data_type() != onnx::TensorProto::UINT8) {
      return std::nullopt;
    }
    if (!MatMulNBitsDimsEqual(*b_it->second, {N, k_blocks, blob_size})) {
      return std::nullopt;
    }
    if (s_it->second->data_type() != onnx::TensorProto::FLOAT) {
      return std::nullopt;
    }
    if (!MatMulNBitsDimsEqual(*s_it->second, {N, k_blocks})) {
      return std::nullopt;
    }
    std::optional<std::string> bias_opt;
    if (!bias_name.empty()) {
      auto bias_it = init_map.find(bias_name);
      if (bias_it == init_map.end() ||
          bias_it->second->data_type() != s_it->second->data_type()) {
        return std::nullopt;
      }
      if (!MatMulNBitsDimsEqual(*bias_it->second, {N})) {
        return std::nullopt;
      }
      bias_opt = bias_name;
    }
    std::vector<std::string> shared = {b_name, scales_name};
    if (bias_opt) {
      shared.push_back(*bias_opt);
    }
    for (const auto& nm : shared) {
      if (ConsumerCount(consumers_of, nm) != 1) {
        return std::nullopt;  // Shared/tied tensor -- another node reads it
                              // too.
      }
    }
    return std::make_tuple(b_name, scales_name, bias_opt);
  };

  auto gate = resolve_branch(gate_b_name, gate_scales_name, gate_bias_name);
  if (!gate) {
    return std::nullopt;
  }
  auto up = resolve_branch(up_b_name, up_scales_name, up_bias_name);
  if (!up) {
    return std::nullopt;
  }

  MatMulNBitsMlpWeight w;
  w.node = node;
  w.gate_b_name = std::get<0>(*gate);
  w.gate_scales_name = std::get<1>(*gate);
  w.gate_bias_name = std::get<2>(*gate);
  w.up_b_name = std::get<0>(*up);
  w.up_scales_name = std::get<1>(*up);
  w.up_bias_name = std::get<2>(*up);
  w.N = N;
  w.K = K;
  w.bits = bits;
  w.block_size = block_size;
  w.k_blocks = k_blocks;
  return w;
}

// Wraps one branch (`gate=true`/`gate=false` for `up`) of a matched
// MatMulNBitsMlpWeight as a synthetic MatMulNBitsWeight -- purely so
// MatMulNBitsDequantized (IMPORTANCE RANKING ONLY, never written back) can
// be reused verbatim. `zero_points_name` is always unset, exactly what
// MatMulNBitsDequantized needs to fall back to the schema's own default
// zero point. Mirrors pruning.py's own `_matmul_nbits_mlp_branch_weight`.
MatMulNBitsWeight MlpBranchWeight(const MatMulNBitsMlpWeight& p, bool gate) {
  MatMulNBitsWeight w;
  w.node = p.node;
  w.b_name = gate ? p.gate_b_name : p.up_b_name;
  w.scales_name = gate ? p.gate_scales_name : p.up_scales_name;
  w.zero_points_name = std::nullopt;
  w.zero_points_packed = false;
  w.bias_name = gate ? p.gate_bias_name : p.up_bias_name;
  w.N = p.N;
  w.K = p.K;
  w.bits = p.bits;
  w.block_size = p.block_size;
  w.k_blocks = p.k_blocks;
  return w;
}

struct MatMulNBitsMlpChain {
  MatMulNBitsMlpWeight producer;
  std::vector<onnx::NodeProto*> chain_ops;
  NBitsSide consumer;
  int64_t n_channels;
};

// The `MatMulNBitsMlp` analogue of FindMatMulNBitsChains: every matched
// fused-gated-MLP node whose `Y` output feeds, through zero or more
// shape-preserving unary activations with no other consumer along the way,
// into a downstream `MatMulNBits`-or-plain-float consumer -- reusing
// WalkToMatMulNBitsConsumer verbatim, the exact same walk an ordinary
// single-`MatMulNBits`-producer chain uses. Mirrors pruning.py's own
// `_find_matmul_nbits_mlp_chains`.
std::vector<MatMulNBitsMlpChain> FindMatMulNBitsMlpChains(
    onnx::GraphProto* graph) {
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

  std::vector<MatMulNBitsMlpChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto w = MatchMatMulNBitsMlp(node, init_map, consumers_of);
    if (!w) {
      continue;
    }
    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    const int64_t n = w->N;
    auto found = WalkToMatMulNBitsConsumer(out_name, init_map, consumers_of,
                                           graph_outputs, n, kMaxChainHops);
    if (!found) {
      continue;
    }
    chains.push_back(MatMulNBitsMlpChain{std::move(*w),
                                         std::move(found->chain_ops),
                                         std::move(found->consumer), n});
  }
  return chains;
}

// Applies whole-output-channel pruning to every matched `MatMulNBitsMlp`
// chain in place -- co-slicing `gate`'s and `up`'s own shared `N` axis to one
// shared `keep` set (ranked by combined root-sum-square L2 row-norm
// importance, the block-quantized analogue of this file's own gated-pair
// criterion above), then the downstream consumer's own reduction axis
// (block-alignment decline for a `MatMulNBits` consumer, direct application
// for a plain-float one) -- mirrors pruning.py's own
// `_apply_matmul_nbits_mlp_chains`. Shares `touched` with
// ApplyStructuredPruning's every other chain family over this same graph
// (see this section's own top comment for why this chain kind, unlike
// MatMulNBitsQkv below, is wired in from ApplyStructuredPruning at all).
void ApplyMatMulNBitsMlpChains(onnx::GraphProto* graph,
                               std::vector<MatMulNBitsMlpChain>& chains,
                               double sparsity, TouchedState& touched) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }

  auto row_sq_norms = [](const std::vector<double>& w_nk, int64_t n) {
    const int64_t k = static_cast<int64_t>(w_nk.size()) / n;
    std::vector<double> out(static_cast<size_t>(n), 0.0);
    for (int64_t c = 0; c < n; ++c) {
      double sq = 0.0;
      for (int64_t j = 0; j < k; ++j) {
        const double v = w_nk[static_cast<size_t>(c * k + j)];
        sq += v * v;
      }
      out[static_cast<size_t>(c)] = sq;
    }
    return out;
  };

  for (auto& chain : chains) {
    auto& p = chain.producer;
    const std::string c_key = NBitsSideKey(chain.consumer);
    if (p.gate_b_name == c_key || p.up_b_name == c_key) {
      continue;  // Degenerate (the same weight in both roles).
    }
    if (touched.producer.count(p.gate_b_name) ||
        touched.producer.count(p.up_b_name) || touched.consumer.count(c_key)) {
      continue;  // A shared/tied weight another chain already resized.
    }

    const int64_t n = chain.n_channels;
    const int64_t keep_count = std::max<int64_t>(
        1, n - std::llround(static_cast<double>(n) * sparsity));
    if (keep_count >= n) {
      continue;  // Rounds down to nothing for this layer -- no-op.
    }

    const MatMulNBitsWeight gate_w = MlpBranchWeight(p, /*gate=*/true);
    const MatMulNBitsWeight up_w = MlpBranchWeight(p, /*gate=*/false);
    const std::vector<double> gate_nk =
        MatMulNBitsDequantized(gate_w, init_map);
    const std::vector<double> up_nk = MatMulNBitsDequantized(up_w, init_map);
    const std::vector<double> gate_sq = row_sq_norms(gate_nk, n);
    const std::vector<double> up_sq = row_sq_norms(up_nk, n);
    std::vector<double> importance(static_cast<size_t>(n));
    for (int64_t c = 0; c < n; ++c) {
      importance[static_cast<size_t>(c)] = std::sqrt(
          gate_sq[static_cast<size_t>(c)] + up_sq[static_cast<size_t>(c)]);
    }
    const std::vector<int64_t> keep =
        TopKIndicesAscending(importance, keep_count);

    std::vector<int64_t> consumer_keep;
    if (const auto* cw = std::get_if<MatMulNBitsWeight>(&chain.consumer)) {
      auto keep_blocks =
          MatMulNBitsBlockAlignedKeepBlocks(keep, cw->k_blocks, cw->block_size);
      if (!keep_blocks) {
        continue;  // Non-block-aligned request -- decline, see this file's
                   // own top-of-MatMulNBits-section comment.
      }
      consumer_keep = std::move(*keep_blocks);
    } else {
      consumer_keep = keep;  // Plain-float consumer -- no block structure.
    }

    SliceMatMulNBitsRowsNoZp(p.gate_b_name, p.gate_scales_name,
                             p.gate_bias_name, p.block_size, p.bits, p.k_blocks,
                             init_map, keep);
    SliceMatMulNBitsRowsNoZp(p.up_b_name, p.up_scales_name, p.up_bias_name,
                             p.block_size, p.bits, p.k_blocks, init_map, keep);
    SetOrAddIntAttr(p.node, "N", static_cast<int64_t>(keep.size()));
    SliceNBitsSideConsumer(chain.consumer, init_map, consumer_keep);

    touched.producer.insert(p.gate_b_name);
    touched.producer.insert(p.up_b_name);
    touched.consumer.insert(c_key);
    touched.stale_value_info.insert(p.node->output(0));
    for (auto* op : chain.chain_ops) {
      touched.stale_value_info.insert(op->output(0));
    }
  }
}

// A matched `com.microsoft::MatMulNBitsQkv` node's three block-quantized
// weight operands (`q`/`k`/`v`) -- mirrors pruning.py's own
// `_MatMulNBitsQkvWeight`. Like MatMulNBitsMlpWeight, no `zero_points` field
// at all (see that struct's own comment).
struct MatMulNBitsQkvWeight {
  onnx::NodeProto* node = nullptr;
  std::string q_b_name, q_scales_name;
  std::optional<std::string> q_bias_name;
  std::string k_b_name, k_scales_name;
  std::optional<std::string> k_bias_name;
  std::string v_b_name, v_scales_name;
  std::optional<std::string> v_bias_name;
  int64_t Nq = 0;
  int64_t Nkv = 0;
  int64_t K = 0;
  int64_t bits = 0;
  int64_t block_size = 0;
  int64_t k_blocks = 0;
};

// If `node` is a `com.microsoft::MatMulNBitsQkv` node matching every scope
// boundary this section's own top comment documents, returns the match --
// mirrors pruning.py's own `_match_matmul_nbits_qkv`. `skip` (input 1,
// optional) and `norm_scale` (input 2, always REQUIRED on this op's live
// schema -- unlike `MatMulNBitsMlp`'s own optional one) are checked for
// presence where the schema requires it, but never otherwise inspected: only
// the shared `Nq`/`Nkv` OUTPUT axes are ever pruned here, never the shared
// `K` INPUT axis.
std::optional<MatMulNBitsQkvWeight> MatchMatMulNBitsQkv(
    onnx::NodeProto* node, const InitMap& init_map,
    const ConsumerMap& consumers_of) {
  if (node->op_type() != "MatMulNBitsQkv" ||
      node->domain() != kComMicrosoftDomain) {
    return std::nullopt;
  }
  if (node->input_size() < 11 || node->output_size() < 3) {
    return std::nullopt;
  }
  const std::string& a_name = node->input(0);
  const std::string& norm_scale_name = node->input(2);
  if (a_name.empty() || norm_scale_name.empty()) {
    return std::nullopt;
  }
  const std::string& q_b_name = node->input(3);
  const std::string& q_scales_name = node->input(4);
  const std::string q_bias_name = node->input(5);
  const std::string& k_b_name = node->input(6);
  const std::string& k_scales_name = node->input(7);
  const std::string k_bias_name = node->input(8);
  const std::string& v_b_name = node->input(9);
  const std::string& v_scales_name = node->input(10);
  const std::string v_bias_name =
      (node->input_size() > 11) ? node->input(11) : std::string();
  if (q_b_name.empty() || q_scales_name.empty() || k_b_name.empty() ||
      k_scales_name.empty() || v_b_name.empty() || v_scales_name.empty()) {
    return std::nullopt;
  }
  if (q_b_name == k_b_name || q_b_name == v_b_name || k_b_name == v_b_name) {
    return std::nullopt;  // Degenerate -- can't independently slice a shared
                          // weight.
  }

  const auto block_size_opt = MatMulNBitsIntAttr(*node, "block_size");
  const auto nq_opt = MatMulNBitsIntAttr(*node, "Nq");
  const auto nkv_opt = MatMulNBitsIntAttr(*node, "Nkv");
  const auto k_opt = MatMulNBitsIntAttr(*node, "K");
  if (!block_size_opt || !nq_opt || !nkv_opt || !k_opt) {
    return std::nullopt;
  }
  const int64_t block_size = *block_size_opt;
  const int64_t Nq = *nq_opt;
  const int64_t Nkv = *nkv_opt;
  const int64_t K = *k_opt;
  if (Nq <= 0 || Nkv <= 0 || K <= 0) {
    return std::nullopt;
  }
  if (!MatMulNBitsValidBlockSizes().count(block_size)) {
    return std::nullopt;
  }
  if (K % block_size != 0) {
    return std::nullopt;
  }
  const int64_t bits = MatMulNBitsIntAttrOr(*node, "bits", 4);
  if (bits != 4 && bits != 8) {
    return std::nullopt;
  }

  const int64_t k_blocks = K / block_size;
  const int64_t blob_size = block_size * bits / 8;

  auto resolve_branch = [&](int64_t n, const std::string& b_name,
                            const std::string& scales_name,
                            const std::string& bias_name)
      -> std::optional<
          std::tuple<std::string, std::string, std::optional<std::string>>> {
    auto b_it = init_map.find(b_name);
    auto s_it = init_map.find(scales_name);
    if (b_it == init_map.end() || s_it == init_map.end()) {
      return std::nullopt;
    }
    if (b_it->second->data_type() != onnx::TensorProto::UINT8) {
      return std::nullopt;
    }
    if (!MatMulNBitsDimsEqual(*b_it->second, {n, k_blocks, blob_size})) {
      return std::nullopt;
    }
    if (s_it->second->data_type() != onnx::TensorProto::FLOAT) {
      return std::nullopt;
    }
    if (!MatMulNBitsDimsEqual(*s_it->second, {n, k_blocks})) {
      return std::nullopt;
    }
    std::optional<std::string> bias_opt;
    if (!bias_name.empty()) {
      auto bias_it = init_map.find(bias_name);
      if (bias_it == init_map.end() ||
          bias_it->second->data_type() != s_it->second->data_type()) {
        return std::nullopt;
      }
      if (!MatMulNBitsDimsEqual(*bias_it->second, {n})) {
        return std::nullopt;
      }
      bias_opt = bias_name;
    }
    std::vector<std::string> shared = {b_name, scales_name};
    if (bias_opt) {
      shared.push_back(*bias_opt);
    }
    for (const auto& nm : shared) {
      if (ConsumerCount(consumers_of, nm) != 1) {
        return std::nullopt;
      }
    }
    return std::make_tuple(b_name, scales_name, bias_opt);
  };

  auto q = resolve_branch(Nq, q_b_name, q_scales_name, q_bias_name);
  if (!q) {
    return std::nullopt;
  }
  auto k = resolve_branch(Nkv, k_b_name, k_scales_name, k_bias_name);
  if (!k) {
    return std::nullopt;
  }
  auto v = resolve_branch(Nkv, v_b_name, v_scales_name, v_bias_name);
  if (!v) {
    return std::nullopt;
  }

  MatMulNBitsQkvWeight w;
  w.node = node;
  w.q_b_name = std::get<0>(*q);
  w.q_scales_name = std::get<1>(*q);
  w.q_bias_name = std::get<2>(*q);
  w.k_b_name = std::get<0>(*k);
  w.k_scales_name = std::get<1>(*k);
  w.k_bias_name = std::get<2>(*k);
  w.v_b_name = std::get<0>(*v);
  w.v_scales_name = std::get<1>(*v);
  w.v_bias_name = std::get<2>(*v);
  w.Nq = Nq;
  w.Nkv = Nkv;
  w.K = K;
  w.bits = bits;
  w.block_size = block_size;
  w.k_blocks = k_blocks;
  return w;
}

// Wraps one branch (`'q'`/`'k'`/`'v'`) of a matched MatMulNBitsQkvWeight as a
// synthetic MatMulNBitsWeight -- the MatMulNBitsQkv analogue of
// MlpBranchWeight, so MatMulNBitsDequantized can be reused verbatim. Mirrors
// pruning.py's own `_matmul_nbits_qkv_branch_weight`.
MatMulNBitsWeight QkvBranchWeight(const MatMulNBitsQkvWeight& p, char branch) {
  MatMulNBitsWeight w;
  w.node = p.node;
  if (branch == 'q') {
    w.b_name = p.q_b_name;
    w.scales_name = p.q_scales_name;
    w.bias_name = p.q_bias_name;
    w.N = p.Nq;
  } else if (branch == 'k') {
    w.b_name = p.k_b_name;
    w.scales_name = p.k_scales_name;
    w.bias_name = p.k_bias_name;
    w.N = p.Nkv;
  } else {
    w.b_name = p.v_b_name;
    w.scales_name = p.v_scales_name;
    w.bias_name = p.v_bias_name;
    w.N = p.Nkv;
  }
  w.zero_points_name = std::nullopt;
  w.zero_points_packed = false;
  w.K = p.K;
  w.bits = p.bits;
  w.block_size = p.block_size;
  w.k_blocks = p.k_blocks;
  return w;
}

// From a matched `GroupQueryAttention`/plain `ai.onnx::Attention` node's own
// raw (`nv`-wide) output `start`, optionally through one `Reshape` hop
// (identical shape/single-use safety check to WalkToAttentionConsumer's
// own), to an output projection whose reduction dimension matches `nv` --
// either a `MatMulNBits` node (MatchMatMulNBits) or a plain-float
// MatMul/vanilla-Gemm (MatchPlainMatMulNBitsPeer), the same union
// WalkToMatMulNBitsConsumer already supports for an ordinary chain. Mirrors
// pruning.py's own `_walk_to_matmul_nbits_attention_consumer`.
std::pair<std::optional<NBitsSide>, std::vector<AttnChainOp>>
WalkToMatMulNBitsAttentionConsumer(
    const std::string& start, const InitMap& init_map,
    const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int64_t nv) {
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
    if (!last_dim || *last_dim != nv) {
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
    auto nc = consumers_of.find(cur);
    if (nc == consumers_of.end() || nc->second.size() != 1) {
      return {std::nullopt, chain_ops};
    }
    node = nc->second[0];
  }

  if (node->op_type() == "MatMulNBits" &&
      node->domain() == kComMicrosoftDomain && node->input_size() > 0 &&
      node->input(0) == cur) {
    auto w = MatchMatMulNBits(node, init_map, consumers_of);
    if (!w || w->K != nv) {
      return {std::nullopt, chain_ops};
    }
    return {NBitsSide(*w), chain_ops};
  }

  auto mm = MatchMatMulLikeRaw(*node);
  if (mm && mm->x_name == cur) {
    auto peer = MatchPlainMatMulNBitsPeer(node, init_map, consumers_of);
    if (!peer || peer->in_channels != nv) {
      return {std::nullopt, chain_ops};
    }
    return {NBitsSide(*peer), chain_ops};
  }

  return {std::nullopt, chain_ops};
}

struct MatMulNBitsQkvChain {
  MatMulNBitsQkvWeight producer;
  onnx::NodeProto* attn_node = nullptr;
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  int64_t head_size = 0;
  std::string num_heads_attr = "num_heads";
  std::vector<AttnChainOp> chain_ops;
  NBitsSide consumer;
};

// Match-time safety net for a `MatMulNBitsQkv` chain's own downstream
// attention node's `attention_bias`/`attn_mask` (`mask_idx`) and, for GQA
// only, `head_sink` (index 11) inputs -- see this file's own top-of-section
// comment for why this exists at all (this port has no dynamic-Gather-
// insertion machinery, unlike pruning.py's own `_slice_or_gather_head_bias`
// fix). `attention_bias`/`attn_mask`: safe (returns `true`) when absent, OR
// a constant whose shape provably can never carry a genuine per-head axis at
// all -- rank < 3 (no axis can ever land on the schema's own num_heads
// slot), or rank in [3, 4] with that slot's own size exactly 1 (an
// unconditional broadcast, mirrors pruning.py's own `_head_bias_axis`
// "-1" case) -- never touched by any head-count change, so leaving it
// exactly as-is stays correct; declined (`false`) otherwise -- a genuine
// per-head-sized constant (which this port could safely slice, but doesn't,
// to keep this new safety net small and obviously correct) and a dynamic
// tensor of any shape alike, rather than risk silently shipping a pruned
// graph with a stale per-head mask. `head_sink`: safe only when ABSENT --
// narrower still (pruning.py slices a genuine constant `(num_heads,)` one;
// this port declines it outright) since it has no `HeadColumnIndices`-shaped
// slice of its own to reuse here beyond one more bespoke branch this round
// chooses not to add.
bool MatMulNBitsQkvAttentionExtrasSafe(const onnx::NodeProto& attn,
                                       const InitMap& init_map, int mask_idx,
                                       std::optional<int> sink_idx) {
  if (attn.input_size() > mask_idx && !attn.input(mask_idx).empty()) {
    auto it = init_map.find(attn.input(mask_idx));
    if (it == init_map.end()) {
      return false;  // Dynamic -- no Gather-insertion machinery here.
    }
    const int64_t rank = it->second->dims_size();
    if (rank > 4) {
      return false;
    }
    const int64_t axis = rank - 3;
    if (axis >= 0) {
      const int64_t size = it->second->dims(static_cast<int>(axis));
      if (size != 1) {
        return false;  // Genuinely per-head (or unresolvable) -- decline
                       // rather than mis-slice or silently leave stale.
      }
    }
  }
  if (sink_idx && attn.input_size() > *sink_idx &&
      !attn.input(*sink_idx).empty()) {
    return false;  // Present at all -- this port never slices head_sink.
  }
  return true;
}

// Matches a `MatMulNBitsQkv` node whose `Q`/`K`/`V` outputs each feed --
// directly, single-consumer, no intermediate hop -- into the respective
// query/key/value inputs of one shared downstream `GroupQueryAttention`
// (MatchGqaProducer) or plain `ai.onnx::Attention` (MatchOnnxAttentionProducer)
// node, and whose own output in turn reaches a `MatMulNBits`-or-plain-float
// output projection (WalkToMatMulNBitsAttentionConsumer). Mirrors
// pruning.py's own `_find_matmul_nbits_qkv_chains`. Unlike
// `_find_separate_qkv_chains`'s own precedent, NO per-head Q/K-norm + RoPE
// pass-through hop is matched between this fused node's own `Q`/`K` outputs
// and the attention consumer -- a deliberate, documented scope boundary
// (this file's own top-of-section comment), not an oversight.
std::vector<MatMulNBitsQkvChain> FindMatMulNBitsQkvChains(
    onnx::GraphProto* graph) {
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

  std::vector<MatMulNBitsQkvChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto w = MatchMatMulNBitsQkv(node, init_map, consumers_of);
    if (!w) {
      continue;
    }

    const std::string& q_out = node->output(0);
    const std::string& k_out = node->output(1);
    const std::string& v_out = node->output(2);
    if (!(is_internal(q_out) && is_internal(k_out) && is_internal(v_out))) {
      continue;
    }
    onnx::NodeProto* attn = consumers_of.at(q_out)[0];
    if (consumers_of.at(k_out)[0] != attn ||
        consumers_of.at(v_out)[0] != attn) {
      continue;  // Q/K/V must feed the exact same downstream node.
    }
    if (attn->input_size() < 3) {
      continue;
    }
    if (attn->input(0) != q_out || attn->input(1) != k_out ||
        attn->input(2) != v_out) {
      continue;  // Must land on the consumer's own query/key/value inputs.
    }

    std::string num_heads_attr = "num_heads";
    int mask_idx = 3;
    std::optional<int> sink_idx;
    std::optional<HeadCountsMatch> info;
    const bool is_gqa = attn->domain() == kComMicrosoftDomain &&
                        attn->op_type() == "GroupQueryAttention";
    if (is_gqa) {
      info = MatchGqaProducer(*attn, init_map);
      mask_idx = 10;
      sink_idx = 11;
    } else if (attn->domain().empty() && attn->op_type() == "Attention") {
      info = MatchOnnxAttentionProducer(*attn, init_map);
      num_heads_attr = "q_num_heads";
      mask_idx = 3;
    }
    if (!info) {
      continue;
    }
    if (!MatMulNBitsQkvAttentionExtrasSafe(*attn, init_map, mask_idx,
                                           sink_idx)) {
      continue;
    }

    const int64_t num_heads = info->num_heads;
    const int64_t kv_num_heads = info->kv_num_heads;
    if (w->Nq % num_heads != 0 || w->Nkv % kv_num_heads != 0) {
      continue;
    }
    const int64_t head_size = w->Nq / num_heads;
    if (head_size <= 0 || w->Nkv / kv_num_heads != head_size) {
      continue;
    }

    const std::string& out_name = attn->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    const int64_t raw_out_width = num_heads * head_size;
    auto [consumer, chain_ops] = WalkToMatMulNBitsAttentionConsumer(
        out_name, init_map, consumers_of, graph_outputs, raw_out_width);
    if (!consumer) {
      continue;
    }

    chains.push_back(MatMulNBitsQkvChain{
        std::move(*w), attn, num_heads, kv_num_heads, head_size, num_heads_attr,
        std::move(chain_ops), std::move(*consumer)});
  }
  return chains;
}

// Applies whole-KV-group pruning to every matched `MatMulNBitsQkv` chain in
// place -- the fused-op analogue of ApplyOneGqaChain, reusing that
// function's own head-count/attribute-rewrite conventions directly against
// `chain.attn_node`, since that node itself owns no weight of its own to
// slice here -- only `q`'s/`k`'s/`v`'s own producer weights (this function's
// own job) and the downstream output-projection consumer
// (SliceNBitsSideConsumer, reused verbatim). Mirrors pruning.py's own
// `_apply_matmul_nbits_qkv_chains`, MINUS the dynamic-`attention_bias`-Gather
// path (see this file's own top-of-section comment for exactly why, and
// what IS still ported: constant `past_key`/`past_value` and GQA's own
// `k_scale`/`v_scale`). Uses its own dedicated producer/consumer-touched
// bookkeeping local to this one call -- see this file's own top-of-section
// comment for why that never risks a double-resize against
// ApplyAttentionChains's own chains over the same graph.
void ApplyMatMulNBitsQkvChains(onnx::GraphProto* graph,
                               std::vector<MatMulNBitsQkvChain>& chains,
                               double sparsity) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }
  std::unordered_set<std::string> producer_touched, consumer_touched,
      stale_value_info;

  for (auto& chain : chains) {
    auto& p = chain.producer;
    const std::unordered_set<std::string> p_keys = {p.q_b_name, p.k_b_name,
                                                    p.v_b_name};
    const std::string c_key = NBitsSideKey(chain.consumer);
    if (p_keys.count(c_key)) {
      continue;  // Degenerate (the same weight in both roles).
    }
    bool conflict = consumer_touched.count(c_key) != 0;
    for (const auto& k : p_keys) {
      if (producer_touched.count(k)) {
        conflict = true;
      }
    }
    if (conflict) {
      continue;  // A shared/tied weight another chain already resized.
    }

    const int64_t h = chain.kv_num_heads;
    const int64_t keep_count = std::max<int64_t>(
        1, h - std::llround(static_cast<double>(h) * sparsity));
    if (keep_count >= h) {
      continue;  // Rounds down to nothing for this block -- no-op.
    }

    const int64_t d = chain.head_size;
    const int64_t group_size = chain.num_heads / chain.kv_num_heads;

    const MatMulNBitsWeight q_w = QkvBranchWeight(p, 'q');
    const MatMulNBitsWeight k_w = QkvBranchWeight(p, 'k');
    const MatMulNBitsWeight v_w = QkvBranchWeight(p, 'v');
    const std::vector<double> q_nk = MatMulNBitsDequantized(q_w, init_map);
    const std::vector<double> k_nk = MatMulNBitsDequantized(k_w, init_map);
    const std::vector<double> v_nk = MatMulNBitsDequantized(v_w, init_map);

    // Combined per-KV-group importance -- mirrors pruning.py's own
    // `_matmul_nbits_qkv_group_importance` (L2 only, this port's own
    // established scope elsewhere): every KV group's own query-head rows,
    // K rows, and V rows are contiguous `head_size`-wide row ranges of
    // `q_nk`'s/`k_nk`'s/`v_nk`'s own `N` axis.
    std::vector<double> importance(static_cast<size_t>(h), 0.0);
    for (int64_t kv = 0; kv < h; ++kv) {
      double sq = 0.0;
      for (int64_t r = kv * group_size * d; r < (kv + 1) * group_size * d;
           ++r) {
        for (int64_t j = 0; j < p.K; ++j) {
          const double v = q_nk[static_cast<size_t>(r * p.K + j)];
          sq += v * v;
        }
      }
      for (int64_t r = kv * d; r < (kv + 1) * d; ++r) {
        for (int64_t j = 0; j < p.K; ++j) {
          const double v = k_nk[static_cast<size_t>(r * p.K + j)];
          sq += v * v;
        }
      }
      for (int64_t r = kv * d; r < (kv + 1) * d; ++r) {
        for (int64_t j = 0; j < p.K; ++j) {
          const double v = v_nk[static_cast<size_t>(r * p.K + j)];
          sq += v * v;
        }
      }
      importance[static_cast<size_t>(kv)] = std::sqrt(sq);
    }
    const std::vector<int64_t> keep_groups =
        TopKIndicesAscending(importance, keep_count);
    std::vector<int64_t> keep_q_heads;
    keep_q_heads.reserve(keep_groups.size() * static_cast<size_t>(group_size));
    for (int64_t g : keep_groups) {
      for (int64_t hh = g * group_size; hh < (g + 1) * group_size; ++hh) {
        keep_q_heads.push_back(hh);
      }
    }
    const std::vector<int64_t> q_idx = HeadColumnIndices(keep_q_heads, d);
    const std::vector<int64_t> kv_idx = HeadColumnIndices(keep_groups, d);

    std::vector<int64_t> consumer_keep;
    if (const auto* cw = std::get_if<MatMulNBitsWeight>(&chain.consumer)) {
      auto keep_blocks = MatMulNBitsBlockAlignedKeepBlocks(q_idx, cw->k_blocks,
                                                           cw->block_size);
      if (!keep_blocks) {
        continue;  // Non-block-aligned request -- decline.
      }
      consumer_keep = std::move(*keep_blocks);
    } else {
      consumer_keep = q_idx;  // Plain-float consumer -- no block structure.
    }

    SliceMatMulNBitsRowsNoZp(p.q_b_name, p.q_scales_name, p.q_bias_name,
                             p.block_size, p.bits, p.k_blocks, init_map, q_idx);
    SliceMatMulNBitsRowsNoZp(p.k_b_name, p.k_scales_name, p.k_bias_name,
                             p.block_size, p.bits, p.k_blocks, init_map,
                             kv_idx);
    SliceMatMulNBitsRowsNoZp(p.v_b_name, p.v_scales_name, p.v_bias_name,
                             p.block_size, p.bits, p.k_blocks, init_map,
                             kv_idx);
    SetOrAddIntAttr(p.node, "Nq", static_cast<int64_t>(q_idx.size()));
    SetOrAddIntAttr(p.node, "Nkv", static_cast<int64_t>(kv_idx.size()));

    const int64_t new_num_heads = keep_count * group_size;
    for (auto& attr : *chain.attn_node->mutable_attribute()) {
      if (attr.name() == chain.num_heads_attr) {
        attr.set_i(new_num_heads);
      } else if (attr.name() == "kv_num_heads") {
        attr.set_i(keep_count);
      }
    }

    const bool is_gqa = chain.attn_node->domain() == kComMicrosoftDomain &&
                        chain.attn_node->op_type() == "GroupQueryAttention";
    // Constant past_key/past_value -- guaranteed EMPTY (product-of-dims-0)
    // by MatchGqaProducer's own existing precondition whenever they're a
    // constant at all, so this is purely a dim-count metadata fix, never a
    // real data copy of any live cache values. Mirrors pruning.py's own
    // identical `past_kv_indices` handling.
    for (int idx :
         (is_gqa ? std::array<int, 2>{3, 4} : std::array<int, 2>{4, 5})) {
      if (chain.attn_node->input_size() <= idx ||
          chain.attn_node->input(idx).empty()) {
        continue;
      }
      auto it = init_map.find(chain.attn_node->input(idx));
      if (it != init_map.end() &&
          it->second->data_type() == onnx::TensorProto::FLOAT) {
        SliceFloatTensorAxis1(it->second, keep_groups);
      }
    }
    // GQA's own `k_scale`/`v_scale` (indices 12/13), constant, per-KV-head
    // shaped `[.., kv_num_heads, .., 1]` -- mirrors pruning.py's own
    // identical shape check.
    if (is_gqa) {
      for (int idx : {12, 13}) {
        if (chain.attn_node->input_size() <= idx ||
            chain.attn_node->input(idx).empty()) {
          continue;
        }
        auto it = init_map.find(chain.attn_node->input(idx));
        if (it == init_map.end() ||
            it->second->data_type() != onnx::TensorProto::FLOAT) {
          continue;
        }
        if (it->second->dims_size() == 4 && it->second->dims(1) == h) {
          SliceFloatTensorAxis1(it->second, keep_groups);
        }
      }
    }

    SliceNBitsSideConsumer(chain.consumer, init_map, consumer_keep);

    for (const auto& co : chain.chain_ops) {
      if (co.shape_name) {
        SetInt64TensorLastDim(init_map.at(*co.shape_name), new_num_heads * d);
      }
    }

    for (const auto& k : p_keys) {
      producer_touched.insert(k);
    }
    consumer_touched.insert(c_key);
    stale_value_info.insert(p.node->output(0));
    stale_value_info.insert(p.node->output(1));
    stale_value_info.insert(p.node->output(2));
    stale_value_info.insert(chain.attn_node->output(0));
    for (const auto& co : chain.chain_ops) {
      stale_value_info.insert(co.node->output(0));
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

// --- QDQ (quantized-weight) structured pruning, mirroring pruning.py's own
// "QDQ (quantized-weight) structured pruning" section (_QDQWeight through
// apply_structured_pruning_qdq) -------------------------------------------
//
// A Conv/ConvTranspose (group=1) or MatMul/vanilla-Gemm weight statically
// quantized in the QDQ pattern -- a constant int8/uint8 (per-tensor or
// per-channel) or int4/uint4 (opset 21+ blockwise) initializer fed through a
// single-consumer `DequantizeLinear` node -- can be structurally pruned the
// same way this file's plain-float chains already are, as long as the
// int8/int4 CODES are sliced directly and their own scale/zero-point are
// either co-sliced in lockstep (when they are indexed by the axis being cut)
// or left completely untouched (when they are not) -- NEVER dequantized,
// pruned as a float array, and requantized from scratch. That last approach
// would silently change every kept value's own quantization error (a fresh
// round-trip through a different candidate weight distribution), which is
// exactly what this pass must not do: the whole point of pruning a model
// that is already QDQ-quantized, rather than quantizing an already-pruned
// float model, is to preserve the original quantization exactly for every
// channel that survives. So: **slice the codes, don't recompute them** --
// the single most important correctness property of everything below, the
// direct analogue of this file's own FP16/BFloat16 "slice, don't recompute"
// precedent for a *quantization* metadata pair instead of a *dtype* one.
//
// Which of `Wq`'s own scale/zero-point get touched splits cleanly by role:
//   * Producer role (this weight's own OUTPUT channels are being cut): a
//     per-channel `Ws`/`Wzp` (each `[out_channels]`) is indexed by the exact
//     axis being cut, so it is co-sliced by the same `keep` set, in lockstep
//     with `Wq`. A per-tensor (scalar) `Ws`/`Wzp` isn't shaped by channel
//     count at all, so only `Wq` is sliced.
//   * Consumer role (this weight's own INPUT/reduction channels are being
//     cut): `Ws`/`Wzp` are indexed by the OUTPUT-channel axis, never the
//     input axis being cut here, so they are always left completely
//     untouched -- only `Wq` is sliced, along its reduction axis.
//   * Blockwise (opset 21+ INT4/UINT4, `block_size` on the reduction axis)
//     reverses which role is "simple": the producer's own output-channel
//     axis is never blocked, so `Wq`/`Ws`/`Wzp` co-slice exactly like the
//     per-channel producer case above, no alignment concern. The consumer's
//     own input axis IS the blocked axis, so an individual input channel
//     can't be dropped without re-quantizing its whole block -- out of
//     scope, this pass never invents new quantized values. Instead, the
//     candidate `keep` set (computed once per chain, shared with the
//     producer) is checked for block alignment (QdqBlockAlignedKeepBlocks):
//     every `block_size`-sized block of the consumer's own reduction axis
//     either wholly survives or wholly drops. When aligned, `Wq` is sliced
//     element-wise by `keep` (blocks stay contiguous by construction) and
//     `Ws`/`Wzp` are sliced by BLOCK index; when not aligned, the WHOLE
//     chain (producer and consumer alike) is left completely untouched
//     rather than forcing a partial-block re-quantization or a disagreeing
//     keep-set between producer and consumer.
//
// A `ConvTranspose` weight's own reversed layout (`[in_channels,
// out_channels/group, ...]`, `out_channels` on axis 1 rather than 0) needs
// only a different `expected_axis` threaded through the exact same
// MatchDequantizeLinearWeight[Blockwise]/slicing machinery Conv already
// uses -- see QdqWeightAxis. Every Conv/ConvTranspose/MatMul/Gemm match here
// is restricted to `group == 1` on both sides (a general grouped/depthwise
// QDQ Conv is out of scope -- already a materially bigger project for the
// plain-float case, before QDQ's extra scale/zero-point bookkeeping enters
// at all), and the forward walk (WalkToConsumerQdq) only ever crosses zero
// or more shape-preserving UNARY activations with no per-channel Add/Mul
// bias/scale hop -- narrower than the plain-float chain walk above, by
// design, for this section's own deliberately narrow first cut. A gated
// (SwiGLU/GeGLU) MatMul/Gemm pair IS matched (FindQdqGatedChains), with
// either or both producers and/or the consumer independently QDQ or plain
// float -- Conv/ConvTranspose never take part in a gated pair, mirroring
// FindGatedChains's own restriction.
//
// Only INT8/UINT8 (per-tensor or per-channel, via MatchDequantizeLinearWeight)
// and INT4/UINT4 blockwise-on-the-reduction-axis (via
// MatchDequantizeLinearWeightBlockwise) are matched; FLOAT8 codes, a
// non-default `output_dtype`, blocked quantization on the output-channel
// axis or with INT8/UINT8 codes, a non-exact-multiple block dimension, or
// any `x`/`x_scale`/`x_zero_point` read by more than one node (a shared/tied
// quantized tensor whose slicing here would silently corrupt whatever else
// reads it) are all declined outright by the matchers, never guessed at --
// the same conservative bar every matcher elsewhere in this file holds
// itself to. Unlike the plain-float MatMul/Gemm producer, this port matches
// only `"MatMul"`/`"Gemm"` (not `"FusedGemm"`/`"GemmFastGelu"`) and only
// `"Conv"` (not `"FusedConv"`) -- mirroring MatchMatMulLikeRaw/
// MatchConvProducer's own established scope in this file (which does not
// support those op types anywhere either), a narrower surface than
// pruning.py's own QDQ section but consistent with the rest of this port.
// Only a plain FLOAT (not FLOAT16/BFLOAT16) direct initializer is resolved
// on the non-QDQ side of a mixed chain, for the identical reason: this
// port's plain-float chains never support FLOAT16/BFLOAT16 either.
//
// This entire section is additive: FindQdqChains/FindQdqGatedChains only
// ever match a chain where at least one side is QDQ-quantized (a plain
// float/float pair is FindChains/FindGatedChains's own job, never
// duplicated here), so ApplyQdqChains below can never double-slice a tensor
// FindChains/FindGatedChains/FindConvChains already claimed -- but it still
// shares this graph's own single `TouchedState` with every other Apply*
// call in ApplyStructuredPruning, for the same "one shared conflict ledger
// per graph" reason those calls already share it with each other.

// --- Tensor <-> flat int64 code buffer, mirroring onnx.numpy_helper's own
// transparent INT8/UINT8/INT4/UINT4 (ml_dtypes) round-trip -----------------

int64_t TensorNumEl(const onnx::TensorProto& t) {
  int64_t n = 1;
  for (int64_t d : t.dims()) {
    n *= d;
  }
  return n;
}

bool DimsEqual(const google::protobuf::RepeatedField<int64_t>& a,
               const google::protobuf::RepeatedField<int64_t>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < a.size(); ++i) {
    if (a.Get(i) != b.Get(i)) {
      return false;
    }
  }
  return true;
}

// Returns `nbytes` raw bytes for `t`'s own codes, from `raw_data` when
// present or, for a hand-built (onnx.helper.make_tensor-style) tensor
// instead, from `int32_data` -- which ONNX's own wire format also uses to
// carry one *packed* byte per entry for INT4/UINT4 (mirroring `raw_data`'s
// own packing exactly, just one byte per `int32_data` slot instead of
// concatenated), so this one path handles every constant-tensor
// construction style uniformly for INT8/UINT8/INT4/UINT4 alike.
std::vector<uint8_t> RawOrPackedBytes(const onnx::TensorProto& t,
                                      int64_t nbytes) {
  std::vector<uint8_t> bytes(static_cast<size_t>(nbytes));
  if (t.has_raw_data()) {
    std::memcpy(bytes.data(), t.raw_data().data(), static_cast<size_t>(nbytes));
  } else {
    for (int64_t i = 0; i < nbytes; ++i) {
      bytes[static_cast<size_t>(i)] =
          static_cast<uint8_t>(t.int32_data(static_cast<int>(i)));
    }
  }
  return bytes;
}

// Unpacks an INT8/UINT8/INT4/UINT4 TensorProto's own codes into a flat,
// row-major (matching `t.dims()`) int64 buffer -- INT4/UINT4 two codes per
// byte, low nibble first (`byte = code[2i] | (code[2i+1] << 4)`), matching
// onnx.numpy_helper's own `_pack_4bitx2`/`_unpack_4bit` convention exactly
// (confirmed empirically -- see this section's own top comment).
std::vector<int64_t> ReadQuantCodes(const onnx::TensorProto& t) {
  const int64_t numel = TensorNumEl(t);
  const bool is4bit = t.data_type() == onnx::TensorProto::INT4 ||
                      t.data_type() == onnx::TensorProto::UINT4;
  const bool is_signed = t.data_type() == onnx::TensorProto::INT8 ||
                         t.data_type() == onnx::TensorProto::INT4;
  const int64_t nbytes = is4bit ? (numel + 1) / 2 : numel;
  const std::vector<uint8_t> bytes = RawOrPackedBytes(t, nbytes);
  std::vector<int64_t> out(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) {
    int64_t val;
    if (is4bit) {
      const uint8_t byte = bytes[static_cast<size_t>(i / 2)];
      val = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
      if (is_signed && val >= 8) {
        val -= 16;
      }
    } else {
      const uint8_t byte = bytes[static_cast<size_t>(i)];
      val = is_signed ? static_cast<int64_t>(static_cast<int8_t>(byte))
                      : static_cast<int64_t>(byte);
    }
    out[static_cast<size_t>(i)] = val;
  }
  return out;
}

// Overwrites `t` in place with a fresh INT8/UINT8/INT4/UINT4 tensor of
// `dims`/`codes` (`codes` already the target dtype's own two's-complement
// representation, e.g. -2 for INT8/INT4 alike), keeping its existing name --
// mirrors SetFloatTensorData's own "replace, don't mutate a live view"
// convention. Always writes `raw_data` (matching onnx.numpy_helper.
// from_array's own convention), regardless of how `t` was originally
// encoded.
void SetQuantCodes(onnx::TensorProto* t, int32_t dtype,
                   const std::vector<int64_t>& dims,
                   const std::vector<int64_t>& codes) {
  const bool is4bit =
      dtype == onnx::TensorProto::INT4 || dtype == onnx::TensorProto::UINT4;
  const int64_t numel = static_cast<int64_t>(codes.size());
  const std::string name = t->name();
  t->Clear();
  t->set_name(name);
  t->set_data_type(static_cast<onnx::TensorProto::DataType>(dtype));
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  if (is4bit) {
    std::string raw(static_cast<size_t>((numel + 1) / 2), '\0');
    for (int64_t i = 0; i < numel; ++i) {
      const uint8_t nibble =
          static_cast<uint8_t>(codes[static_cast<size_t>(i)]) & 0x0F;
      uint8_t& byte =
          reinterpret_cast<uint8_t&>(raw[static_cast<size_t>(i / 2)]);
      if (i % 2 == 0) {
        byte = nibble;
      } else {
        byte = static_cast<uint8_t>(byte | (nibble << 4));
      }
    }
    t->set_raw_data(std::move(raw));
  } else {
    std::string raw(static_cast<size_t>(numel), '\0');
    for (int64_t i = 0; i < numel; ++i) {
      raw[static_cast<size_t>(i)] = static_cast<char>(
          static_cast<uint8_t>(codes[static_cast<size_t>(i)]));
    }
    t->set_raw_data(std::move(raw));
  }
}

std::vector<int64_t> RowMajorStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size());
  int64_t acc = 1;
  for (int64_t i = static_cast<int64_t>(dims.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = acc;
    acc *= dims[static_cast<size_t>(i)];
  }
  return strides;
}

// Gathers `keep` (ascending indices) along `axis` of a row-major `dims`-
// shaped flat buffer -- the generic building block every QDQ slicer below
// uses, for both int64 codes and float32 scale/zero-point alike, at
// whichever axis/rank the specific role (producer/consumer, per-channel/
// blockwise) needs. Exactly `numpy`'s own `np.take(data, keep, axis=axis)`.
template <typename T>
std::vector<T> SliceAlongAxis(const std::vector<T>& data,
                              const std::vector<int64_t>& dims, int64_t axis,
                              const std::vector<int64_t>& keep) {
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= dims[static_cast<size_t>(i)];
  }
  const int64_t axis_dim = dims[static_cast<size_t>(axis)];
  int64_t inner = 1;
  for (size_t i = static_cast<size_t>(axis) + 1; i < dims.size(); ++i) {
    inner *= dims[i];
  }
  std::vector<T> out(static_cast<size_t>(outer) * keep.size() *
                     static_cast<size_t>(inner));
  for (int64_t o = 0; o < outer; ++o) {
    for (size_t k = 0; k < keep.size(); ++k) {
      const T* src = data.data() +
                     (static_cast<size_t>(o) * static_cast<size_t>(axis_dim) +
                      static_cast<size_t>(keep[k])) *
                         static_cast<size_t>(inner);
      T* dst = out.data() + (static_cast<size_t>(o) * keep.size() + k) *
                                static_cast<size_t>(inner);
      std::copy(src, src + inner, dst);
    }
  }
  return out;
}

// --- Dequantization for IMPORTANCE RANKING ONLY -- never written back to the
// graph. The actual mutation always slices the int8/int4 codes/scale/
// zero-point directly (see this section's own top comment); these two
// helpers exist purely so a QDQ producer's output channels can be ranked by
// the same L2-norm-of-dequantized-row criterion a plain float producer's
// already are (QdqChannelImportanceL2), without the ranking caring which
// quantization scheme the weight came from. -------------------------------

std::vector<double> PerChannelDequantFlat(const std::vector<int64_t>& codes,
                                          const std::vector<int64_t>& dims,
                                          const std::vector<float>& scale,
                                          const std::vector<int64_t>& zp,
                                          bool per_channel, int64_t axis) {
  const int64_t numel = static_cast<int64_t>(codes.size());
  std::vector<double> out(static_cast<size_t>(numel));
  if (!per_channel) {
    const double s = static_cast<double>(scale[0]);
    const double zpv = zp.empty() ? 0.0 : static_cast<double>(zp[0]);
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] =
          (static_cast<double>(codes[static_cast<size_t>(i)]) - zpv) * s;
    }
    return out;
  }
  const std::vector<int64_t> strides = RowMajorStrides(dims);
  const int64_t axis_dim = dims[static_cast<size_t>(axis)];
  const int64_t axis_stride = strides[static_cast<size_t>(axis)];
  for (int64_t i = 0; i < numel; ++i) {
    const int64_t a = (i / axis_stride) % axis_dim;
    const double zpv =
        zp.empty() ? 0.0 : static_cast<double>(zp[static_cast<size_t>(a)]);
    out[static_cast<size_t>(i)] =
        (static_cast<double>(codes[static_cast<size_t>(i)]) - zpv) *
        static_cast<double>(scale[static_cast<size_t>(a)]);
  }
  return out;
}

// The blockwise analogue: `scale`/`zp` are full-rank (same rank as `dims`),
// full-size on every axis except `block_axis`, where their own size is
// `num_blocks` -- each block's own scalar broadcasts to every element of its
// own `block_size`-sized span along `block_axis`.
std::vector<double> BlockwiseDequantFlat(const std::vector<int64_t>& codes,
                                         const std::vector<int64_t>& dims,
                                         const std::vector<float>& scale,
                                         const std::vector<int64_t>& zp,
                                         int64_t block_axis, int64_t block_size,
                                         int64_t num_blocks) {
  std::vector<int64_t> scale_dims = dims;
  scale_dims[static_cast<size_t>(block_axis)] = num_blocks;
  const std::vector<int64_t> strides = RowMajorStrides(dims);
  const std::vector<int64_t> scale_strides = RowMajorStrides(scale_dims);
  const int64_t numel = static_cast<int64_t>(codes.size());
  std::vector<double> out(static_cast<size_t>(numel));
  std::vector<int64_t> multi(dims.size());
  for (int64_t idx = 0; idx < numel; ++idx) {
    int64_t rem = idx;
    for (size_t d = 0; d < dims.size(); ++d) {
      multi[d] = rem / strides[d];
      rem %= strides[d];
    }
    int64_t sidx = 0;
    for (size_t d = 0; d < dims.size(); ++d) {
      const int64_t sd = (static_cast<int64_t>(d) == block_axis)
                             ? multi[d] / block_size
                             : multi[d];
      sidx += sd * scale_strides[d];
    }
    const double zpv =
        zp.empty() ? 0.0 : static_cast<double>(zp[static_cast<size_t>(sidx)]);
    out[static_cast<size_t>(idx)] =
        (static_cast<double>(codes[static_cast<size_t>(idx)]) - zpv) *
        static_cast<double>(scale[static_cast<size_t>(sidx)]);
  }
  return out;
}

// `[out_channels, in_channels, kH, kW]` -> `[N, K]` for a Conv weight (or
// its ConvTranspose analogue, `[in_channels, out_channels, kH, kW]`, moved
// to output-channel-first before the same flatten), or a plain 2-D
// MatMul/Gemm weight transposed to `[N, K]` when it isn't already -- the
// double-precision analogue of this file's own (float) TransposeFlat, for
// the exact same "the ranking machinery works over one directly comparable
// [N, K] layout" reason. Mirrors pruning.py's own `_weight_to_nk`.
std::vector<double> QdqWeightToNk(const std::vector<double>& data,
                                  const std::vector<int64_t>& dims,
                                  bool weight_transposed, bool is_conv,
                                  bool is_conv_transpose) {
  if (is_conv) {
    if (!is_conv_transpose) {
      return data;  // Already [Cout, rest] flattened row-major.
    }
    const int64_t cin = dims[0], cout = dims[1];
    const int64_t inner = dims[2] * dims[3];
    std::vector<double> out(data.size());
    for (int64_t ci = 0; ci < cin; ++ci) {
      for (int64_t co = 0; co < cout; ++co) {
        const double* src = data.data() + (ci * cout + co) * inner;
        double* dst = out.data() + (co * cin + ci) * inner;
        std::copy(src, src + inner, dst);
      }
    }
    return out;
  }
  const int64_t dim0 = dims[0], dim1 = dims[1];
  if (weight_transposed) {
    return data;  // Already [N, K].
  }
  std::vector<double> out(data.size());
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      out[static_cast<size_t>(j * dim0 + i)] =
          data[static_cast<size_t>(i * dim1 + j)];
    }
  }
  return out;
}

// L2 norm of each output channel's own (dequantized) weight row -- the
// single-producer QDQ analogue of ApplyChains's own inline importance
// computation. Only L2 is supported (this port, unlike pruning.py's own
// wider `importance_norm` parameter, has no L1 option anywhere -- neither
// does ApplyStructuredPruning's own plain-float path, so this is a
// consistent, not a novel, scope narrowing).
std::vector<double> QdqChannelImportanceL2(const std::vector<double>& w_nk,
                                           int64_t n, int64_t k) {
  std::vector<double> importance(static_cast<size_t>(n), 0.0);
  for (int64_t c = 0; c < n; ++c) {
    double sq = 0.0;
    for (int64_t j = 0; j < k; ++j) {
      const double v = w_nk[static_cast<size_t>(c * k + j)];
      sq += v * v;
    }
    importance[static_cast<size_t>(c)] = std::sqrt(sq);
  }
  return importance;
}

// The STABLE analogue of this file's own TopKIndicesAscending
// (std::partial_sort there is not stable for ties) -- mirrors pruning.py's
// own `np.argsort(-importance, kind="stable")[:keep_count]` exactly. Matters
// specifically here (unlike most other ranking call sites in this file):
// with exactly-tied importances, an unstable tie-break can select a keep set
// that no longer aligns to a blockwise consumer's own block boundaries
// (QdqBlockAlignedKeepBlocks), flipping a would-have-been-aligned chain to
// "declined" nondeterministically. A stable sort preserves ascending
// channel-index order among ties, matching this port's own test oracles
// (and pruning.py's own reference behavior) deterministically.
std::vector<int64_t> StableTopKIndicesAscending(
    const std::vector<double>& importance, int64_t keep_count) {
  const int64_t n = static_cast<int64_t>(importance.size());
  std::vector<int64_t> idx(static_cast<size_t>(n));
  std::iota(idx.begin(), idx.end(), int64_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
    return importance[static_cast<size_t>(a)] >
           importance[static_cast<size_t>(b)];
  });
  idx.resize(static_cast<size_t>(keep_count));
  std::sort(idx.begin(), idx.end());
  return idx;
}

// --- Matching: per-tensor/per-channel and blockwise DequantizeLinear-fed
// weights, mirroring _match_dequantize_linear_weight/
// _match_dequantize_linear_weight_blockwise/_resolve_weight_ref ------------

using DqMap = std::unordered_map<std::string, onnx::NodeProto*>;

struct QdqWeightMatch {
  onnx::NodeProto* dq_node;
  std::string q_name;
  std::string scale_name;
  std::optional<std::string> zp_name;
  int64_t axis;  // This weight's own output-channel axis.
  bool per_channel;
  std::vector<int64_t> dims;
};

// If `weight_name` is fed by a `DequantizeLinear` node from a constant
// int8/uint8 initializer, with a constant FLOAT scale that is either a
// scalar (per-tensor) or a 1-D vector of length `dims[expected_axis]`
// (per-channel, on exactly `expected_axis`) with a matching `axis`
// attribute, returns the match. `rank` is the expected weight rank (4 for
// Conv/ConvTranspose, 2 for MatMul/Gemm). Declines (nullopt) whenever
// anything is ambiguous -- see this section's own top comment.
std::optional<QdqWeightMatch> MatchDequantizeLinearWeight(
    const std::string& weight_name, int64_t rank, int64_t expected_axis,
    const InitMap& init_map, const DqMap& dq_of,
    const ConsumerMap& consumers_of) {
  auto dit = dq_of.find(weight_name);
  if (dit == dq_of.end()) {
    return std::nullopt;
  }
  onnx::NodeProto* dq = dit->second;
  if (dq->op_type() != "DequantizeLinear" || dq->output_size() != 1) {
    return std::nullopt;
  }
  if (ConsumerCount(consumers_of, weight_name) != 1) {
    return std::nullopt;  // DQ output must feed only this one weight use.
  }
  if (dq->input_size() != 2 && dq->input_size() != 3) {
    return std::nullopt;
  }
  const std::string& q_name = dq->input(0);
  const std::string& scale_name = dq->input(1);
  std::optional<std::string> zp_name;
  if (dq->input_size() == 3 && !dq->input(2).empty()) {
    zp_name = dq->input(2);
  }
  if (q_name.empty() || scale_name.empty()) {
    return std::nullopt;
  }

  auto qit = init_map.find(q_name);
  auto sit = init_map.find(scale_name);
  if (qit == init_map.end() || sit == init_map.end()) {
    return std::nullopt;  // Non-constant q/scale -- can't safely slice it.
  }
  const onnx::TensorProto* q_init = qit->second;
  const onnx::TensorProto* scale_init = sit->second;
  if (q_init->data_type() != onnx::TensorProto::INT8 &&
      q_init->data_type() != onnx::TensorProto::UINT8) {
    return std::nullopt;
  }
  if (scale_init->data_type() != onnx::TensorProto::FLOAT) {
    return std::nullopt;
  }
  if (q_init->dims_size() != rank) {
    return std::nullopt;
  }

  if (zp_name) {
    auto zit = init_map.find(*zp_name);
    if (zit == init_map.end()) {
      return std::nullopt;
    }
    const onnx::TensorProto* zp_init = zit->second;
    if (zp_init->data_type() != q_init->data_type()) {
      return std::nullopt;
    }
    if (!DimsEqual(zp_init->dims(), scale_init->dims())) {
      return std::nullopt;
    }
  }

  if (ConsumerCount(consumers_of, q_name) != 1 ||
      ConsumerCount(consumers_of, scale_name) != 1 ||
      (zp_name && ConsumerCount(consumers_of, *zp_name) != 1)) {
    return std::nullopt;  // Shared/tied quantized tensor.
  }

  for (const auto& attr : dq->attribute()) {
    if (attr.name() == "block_size" && attr.i() != 0) {
      return std::nullopt;  // Blocked quantization --
                            // MatchDequantizeLinearWeightBlockwise's own job.
    }
    if (attr.name() == "output_dtype" && attr.i() != 0) {
      return std::nullopt;
    }
  }

  int64_t axis = 1;  // DequantizeLinear's own schema default.
  for (const auto& attr : dq->attribute()) {
    if (attr.name() == "axis") {
      axis = attr.i();
      break;
    }
  }
  if (axis < 0) {
    axis += rank;
  }

  const std::vector<int64_t> q_dims(q_init->dims().begin(),
                                    q_init->dims().end());
  std::vector<int64_t> scale_dims(scale_init->dims().begin(),
                                  scale_init->dims().end());
  int64_t numel = 1;
  for (int64_t d : scale_dims) {
    numel *= d;
  }
  bool per_channel;
  if (numel == 1) {
    per_channel = false;
  } else if (scale_dims.size() == 1) {
    if (axis != expected_axis ||
        scale_dims[0] != q_dims[static_cast<size_t>(expected_axis)]) {
      return std::nullopt;
    }
    per_channel = true;
  } else {
    return std::nullopt;
  }

  return QdqWeightMatch{dq,          q_name, scale_name, zp_name, expected_axis,
                        per_channel, q_dims};
}

struct QdqBlockwiseWeightMatch {
  onnx::NodeProto* dq_node;
  std::string q_name;
  std::string scale_name;
  std::optional<std::string> zp_name;
  int64_t out_axis;
  int64_t block_axis;
  int64_t block_size;
  int64_t num_blocks;
  std::vector<int64_t> dims;
};

// The blockwise INT4/UINT4 analogue of MatchDequantizeLinearWeight -- see
// this section's own top comment. `block_axis` is always `1 - expected_axis`
// (this weight's own reduction axis, never its output-channel one).
std::optional<QdqBlockwiseWeightMatch> MatchDequantizeLinearWeightBlockwise(
    const std::string& weight_name, int64_t rank, int64_t expected_axis,
    const InitMap& init_map, const DqMap& dq_of,
    const ConsumerMap& consumers_of) {
  auto dit = dq_of.find(weight_name);
  if (dit == dq_of.end()) {
    return std::nullopt;
  }
  onnx::NodeProto* dq = dit->second;
  if (dq->op_type() != "DequantizeLinear" || dq->output_size() != 1) {
    return std::nullopt;
  }
  if (ConsumerCount(consumers_of, weight_name) != 1) {
    return std::nullopt;
  }
  if (dq->input_size() != 2 && dq->input_size() != 3) {
    return std::nullopt;
  }
  const std::string& q_name = dq->input(0);
  const std::string& scale_name = dq->input(1);
  std::optional<std::string> zp_name;
  if (dq->input_size() == 3 && !dq->input(2).empty()) {
    zp_name = dq->input(2);
  }
  if (q_name.empty() || scale_name.empty()) {
    return std::nullopt;
  }

  auto qit = init_map.find(q_name);
  auto sit = init_map.find(scale_name);
  if (qit == init_map.end() || sit == init_map.end()) {
    return std::nullopt;
  }
  const onnx::TensorProto* q_init = qit->second;
  const onnx::TensorProto* scale_init = sit->second;
  if (q_init->data_type() != onnx::TensorProto::INT4 &&
      q_init->data_type() != onnx::TensorProto::UINT4) {
    return std::nullopt;
  }
  if (scale_init->data_type() != onnx::TensorProto::FLOAT) {
    return std::nullopt;
  }
  if (q_init->dims_size() != rank) {
    return std::nullopt;
  }

  if (zp_name) {
    auto zit = init_map.find(*zp_name);
    if (zit == init_map.end()) {
      return std::nullopt;
    }
    const onnx::TensorProto* zp_init = zit->second;
    if (zp_init->data_type() != q_init->data_type()) {
      return std::nullopt;
    }
    if (!DimsEqual(zp_init->dims(), scale_init->dims())) {
      return std::nullopt;
    }
  }

  if (ConsumerCount(consumers_of, q_name) != 1 ||
      ConsumerCount(consumers_of, scale_name) != 1 ||
      (zp_name && ConsumerCount(consumers_of, *zp_name) != 1)) {
    return std::nullopt;
  }

  int64_t block_size = 0, output_dtype = 0;
  for (const auto& attr : dq->attribute()) {
    if (attr.name() == "block_size") {
      block_size = attr.i();
    } else if (attr.name() == "output_dtype") {
      output_dtype = attr.i();
    }
  }
  if (block_size <= 0) {
    return std::nullopt;  // Not blockwise -- MatchDequantizeLinearWeight's own
                          // job.
  }
  if (output_dtype != 0) {
    return std::nullopt;
  }

  int64_t axis = 1;
  for (const auto& attr : dq->attribute()) {
    if (attr.name() == "axis") {
      axis = attr.i();
      break;
    }
  }
  if (axis < 0) {
    axis += rank;
  }

  const int64_t block_axis = 1 - expected_axis;
  if (axis != block_axis) {
    return std::nullopt;
  }

  const std::vector<int64_t> dims(q_init->dims().begin(), q_init->dims().end());
  const int64_t block_dim = dims[static_cast<size_t>(block_axis)];
  if (block_dim % block_size != 0) {
    return std::nullopt;  // Padded/partial final block -- declined.
  }
  const int64_t num_blocks = block_dim / block_size;

  std::vector<int64_t> expected_scale_dims = dims;
  expected_scale_dims[static_cast<size_t>(block_axis)] = num_blocks;
  const std::vector<int64_t> scale_dims(scale_init->dims().begin(),
                                        scale_init->dims().end());
  if (scale_dims != expected_scale_dims) {
    return std::nullopt;
  }

  return QdqBlockwiseWeightMatch{dq,         q_name,        scale_name,
                                 zp_name,    expected_axis, block_axis,
                                 block_size, num_blocks,    dims};
}

// A Conv/ConvTranspose/MatMul/Gemm weight resolved from one of three
// sources -- a direct FLOAT initializer, a per-tensor/per-channel QDQ one,
// or a blockwise INT4/UINT4 QDQ one -- mirroring pruning.py's own
// `_WeightRef`/`_resolve_weight_ref`. Exactly one of `float_name`/`qdq`/
// `qdq_block` is set; `dims` is always populated regardless of which.
struct WeightRefMatch {
  std::optional<std::string> float_name;
  std::optional<QdqWeightMatch> qdq;
  std::optional<QdqBlockwiseWeightMatch> qdq_block;
  std::vector<int64_t> dims;

  bool is_qdq() const { return qdq.has_value() || qdq_block.has_value(); }

  // A name uniquely identifying the underlying tensor this resolves to --
  // the int8/int4 `q_init`'s own name for a QDQ weight (per-tensor/
  // per-channel or blockwise alike), or the float initializer's own name
  // otherwise. Used to detect a shared/tied weight playing the same role in
  // more than one chain -- mirrors pruning.py's own `_weight_ref_key`.
  std::string key() const {
    if (qdq) {
      return qdq->q_name;
    }
    if (qdq_block) {
      return qdq_block->q_name;
    }
    return *float_name;
  }
};

std::optional<WeightRefMatch> ResolveWeightRef(
    const std::string& weight_name, int64_t rank, int64_t expected_axis,
    const InitMap& init_map, const DqMap& dq_of,
    const ConsumerMap& consumers_of) {
  auto it = init_map.find(weight_name);
  if (it != init_map.end()) {
    const onnx::TensorProto* w = it->second;
    if (w->data_type() == onnx::TensorProto::FLOAT && w->dims_size() == rank) {
      WeightRefMatch ref;
      ref.float_name = weight_name;
      ref.dims.assign(w->dims().begin(), w->dims().end());
      return ref;
    }
    return std::nullopt;
  }
  auto qdq = MatchDequantizeLinearWeight(weight_name, rank, expected_axis,
                                         init_map, dq_of, consumers_of);
  if (qdq) {
    WeightRefMatch ref;
    ref.dims = qdq->dims;
    ref.qdq = std::move(qdq);
    return ref;
  }
  auto qdq_block = MatchDequantizeLinearWeightBlockwise(
      weight_name, rank, expected_axis, init_map, dq_of, consumers_of);
  if (qdq_block) {
    WeightRefMatch ref;
    ref.dims = qdq_block->dims;
    ref.qdq_block = std::move(qdq_block);
    return ref;
  }
  return std::nullopt;
}

// --- Per-op-family matchers built on ResolveWeightRef, mirroring
// _match_conv_qdq/_match_conv_transpose_qdq/_match_matmul_qdq --------------

struct QdqConvMatch {
  WeightRefMatch ref;
  std::optional<std::string> bias;
  int64_t out_channels;
  int64_t in_channels;
};

std::optional<QdqConvMatch> MatchConvQdq(const onnx::NodeProto& node,
                                         const InitMap& init_map,
                                         const DqMap& dq_of,
                                         const ConsumerMap& consumers_of) {
  if (node.op_type() != "Conv" || node.input_size() < 2) {
    return std::nullopt;
  }
  if (ConvGroupAttr(node) != 1) {
    return std::nullopt;  // Grouped/depthwise QDQ Conv -- out of scope.
  }
  auto ref =
      ResolveWeightRef(node.input(1), 4, 0, init_map, dq_of, consumers_of);
  if (!ref) {
    return std::nullopt;
  }
  const int64_t out_channels = ref->dims[0];
  const int64_t in_channels = ref->dims[1];
  std::optional<std::string> bias;
  if (node.input_size() == 3 && !node.input(2).empty()) {
    bias = node.input(2);
    auto bit = init_map.find(*bias);
    if (bit == init_map.end() ||
        bit->second->data_type() != onnx::TensorProto::FLOAT) {
      return std::nullopt;  // Non-constant bias -- can't safely slice it.
    }
  }
  return QdqConvMatch{*ref, bias, out_channels, in_channels};
}

struct QdqConvTransposeMatch {
  WeightRefMatch ref;
  std::optional<std::string> bias;
  int64_t out_channels;
  int64_t in_channels;
};

// The ConvTranspose analogue of MatchConvQdq -- see this section's own top
// comment for the reversed-layout ([in_channels, out_channels, kH, kW])
// reasoning. `expected_axis=1` is the ONE difference from MatchConvQdq.
std::optional<QdqConvTransposeMatch> MatchConvTransposeQdq(
    const onnx::NodeProto& node, const InitMap& init_map, const DqMap& dq_of,
    const ConsumerMap& consumers_of) {
  if (node.op_type() != "ConvTranspose" || node.input_size() < 2) {
    return std::nullopt;
  }
  if (ConvGroupAttr(node) != 1) {
    return std::nullopt;
  }
  auto ref =
      ResolveWeightRef(node.input(1), 4, 1, init_map, dq_of, consumers_of);
  if (!ref) {
    return std::nullopt;
  }
  const int64_t in_channels = ref->dims[0];
  const int64_t out_channels = ref->dims[1];
  std::optional<std::string> bias;
  if (node.input_size() == 3 && !node.input(2).empty()) {
    bias = node.input(2);
    auto bit = init_map.find(*bias);
    if (bit == init_map.end() ||
        bit->second->data_type() != onnx::TensorProto::FLOAT) {
      return std::nullopt;
    }
  }
  return QdqConvTransposeMatch{*ref, bias, out_channels, in_channels};
}

struct QdqMatmulMatch {
  std::string x_name;
  WeightRefMatch ref;
  bool weight_transposed;
  std::optional<std::string> bias;
  int64_t out_channels;
  int64_t in_channels;
};

std::optional<QdqMatmulMatch> MatchMatmulQdq(const onnx::NodeProto& node,
                                             const InitMap& init_map,
                                             const DqMap& dq_of,
                                             const ConsumerMap& consumers_of) {
  auto m = MatchMatMulLikeRaw(node);
  if (!m) {
    return std::nullopt;
  }
  const int64_t axis = m->weight_transposed ? 0 : 1;
  auto ref =
      ResolveWeightRef(m->w_name, 2, axis, init_map, dq_of, consumers_of);
  if (!ref) {
    return std::nullopt;
  }
  const int64_t out_channels = ref->dims[static_cast<size_t>(axis)];
  const int64_t in_channels = ref->dims[static_cast<size_t>(1 - axis)];
  std::optional<std::string> bias;
  if (node.op_type() == "Gemm" && node.input_size() == 3 &&
      !node.input(2).empty()) {
    bias = node.input(2);
    auto bit = init_map.find(*bias);
    if (bit == init_map.end() ||
        bit->second->data_type() != onnx::TensorProto::FLOAT) {
      return std::nullopt;
    }
  }
  return QdqMatmulMatch{m->x_name, *ref,         m->weight_transposed,
                        bias,      out_channels, in_channels};
}

// --- Chain data model + plain-chain walker/finder, mirroring _QDQProducer/
// _QDQConsumer/_QDQChain/_QDQGatedChain and _walk_to_consumer_qdq/
// _find_qdq_chains ----------------------------------------------------------

struct QdqProducer {
  onnx::NodeProto* node;
  WeightRefMatch ref;
  std::optional<std::string> bias;
  bool weight_transposed;
  bool is_conv;
  bool is_conv_transpose;
  // Activation nodes between this producer's raw output and the point it
  // combines with another producer -- a gated pair only, see
  // FindQdqGatedChains; empty for a plain single-producer chain.
  std::vector<onnx::NodeProto*> pre_ops;
};

struct QdqConsumerMatch {
  onnx::NodeProto* node;
  WeightRefMatch ref;
  bool weight_transposed;
  bool is_conv;
  bool is_conv_transpose = false;
};

struct QdqChain {
  QdqProducer producer;
  std::vector<onnx::NodeProto*> chain_ops;
  QdqConsumerMatch consumer;
  int64_t n_channels;
};

struct QdqGatedChain {
  QdqProducer producer_a;
  QdqProducer producer_b;
  std::vector<onnx::NodeProto*> chain_ops;
  QdqConsumerMatch consumer;
  int64_t n_channels;
};

// From tensor `start`, walks forward through shape-preserving unary
// activations (UnaryPassThroughOps) with no other consumer anywhere along
// the way, until a same-family (Conv/ConvTranspose-only or MatMul/Gemm-only,
// matching `is_conv`) consumer is found whose input-channel count matches
// `n_channels`. No per-channel Add/Mul bias/scale hop, no depthwise Conv
// pass-through, no branch -- narrower than WalkToConsumer by design, see
// this section's own top comment. Mirrors pruning.py's own
// `_walk_to_consumer_qdq` exactly, including trying a Conv consumer before a
// ConvTranspose one regardless of which family the producer itself was.
std::pair<std::optional<QdqConsumerMatch>, std::vector<onnx::NodeProto*>>
WalkToConsumerQdq(const std::string& start, bool is_conv,
                  const InitMap& init_map, const DqMap& dq_of,
                  const ConsumerMap& consumers_of,
                  const std::unordered_set<std::string>& graph_outputs,
                  int64_t n_channels, int max_hops) {
  std::vector<onnx::NodeProto*> chain_ops;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    auto cit = consumers_of.find(cur);
    if (cit == consumers_of.end() || cit->second.size() != 1) {
      return {std::nullopt, chain_ops};
    }
    onnx::NodeProto* nxt = cit->second[0];

    if (is_conv) {
      if (nxt->op_type() == "Conv" && nxt->input_size() > 0 &&
          nxt->input(0) == cur) {
        auto m = MatchConvQdq(*nxt, init_map, dq_of, consumers_of);
        if (!m || m->in_channels != n_channels) {
          return {std::nullopt, chain_ops};
        }
        return {QdqConsumerMatch{nxt, m->ref, false, true, false}, chain_ops};
      }
      if (nxt->op_type() == "ConvTranspose" && nxt->input_size() > 0 &&
          nxt->input(0) == cur) {
        auto m = MatchConvTransposeQdq(*nxt, init_map, dq_of, consumers_of);
        if (!m || m->in_channels != n_channels) {
          return {std::nullopt, chain_ops};
        }
        return {QdqConsumerMatch{nxt, m->ref, false, true, true}, chain_ops};
      }
    } else {
      auto mm = MatchMatmulQdq(*nxt, init_map, dq_of, consumers_of);
      if (mm && mm->x_name == cur) {
        if (mm->in_channels != n_channels) {
          return {std::nullopt, chain_ops};
        }
        return {
            QdqConsumerMatch{nxt, mm->ref, mm->weight_transposed, false, false},
            chain_ops};
      }
    }

    const bool is_unary = UnaryPassThroughOps().count(nxt->op_type()) != 0 &&
                          nxt->input_size() == 1 && nxt->input(0) == cur &&
                          nxt->output_size() == 1;
    if (!is_unary) {
      return {std::nullopt, chain_ops};
    }
    const std::string& out2 = nxt->output(0);
    auto oit = consumers_of.find(out2);
    if (oit == consumers_of.end() || oit->second.size() != 1 ||
        graph_outputs.count(out2)) {
      return {std::nullopt, chain_ops};
    }
    chain_ops.push_back(nxt);
    cur = out2;
  }
  return {std::nullopt, chain_ops};
}

std::vector<QdqChain> FindQdqChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  DqMap dq_of;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    if (n->op_type() == "DequantizeLinear" && n->output_size() == 1) {
      dq_of[n->output(0)] = n;
    }
  }
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  auto is_internal = [&](const std::string& name) {
    auto it = consumers_of.find(name);
    return it != consumers_of.end() && it->second.size() == 1 &&
           !graph_outputs.count(name);
  };

  std::vector<QdqChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    bool is_conv_transpose = false;
    std::optional<WeightRefMatch> ref;
    std::optional<std::string> bias_name;
    int64_t out_channels = 0;
    bool weight_transposed = false;
    bool is_conv;

    if (node->op_type() == "Conv") {
      auto m = MatchConvQdq(*node, init_map, dq_of, consumers_of);
      if (!m) {
        continue;
      }
      ref = m->ref;
      bias_name = m->bias;
      out_channels = m->out_channels;
      is_conv = true;
    } else if (node->op_type() == "ConvTranspose") {
      auto m = MatchConvTransposeQdq(*node, init_map, dq_of, consumers_of);
      if (!m) {
        continue;
      }
      ref = m->ref;
      bias_name = m->bias;
      out_channels = m->out_channels;
      is_conv = true;
      is_conv_transpose = true;
    } else {
      auto mm = MatchMatmulQdq(*node, init_map, dq_of, consumers_of);
      if (!mm) {
        continue;
      }
      ref = mm->ref;
      bias_name = mm->bias;
      out_channels = mm->out_channels;
      weight_transposed = mm->weight_transposed;
      is_conv = false;
    }

    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    auto [consumer, chain_ops] =
        WalkToConsumerQdq(out_name, is_conv, init_map, dq_of, consumers_of,
                          graph_outputs, out_channels, kMaxChainHops);
    if (!consumer) {
      continue;
    }
    if (!(ref->is_qdq() || consumer->ref.is_qdq())) {
      continue;  // Both plain float -- FindChains's own job.
    }

    QdqChain chain;
    chain.producer = QdqProducer{
        node, *ref, bias_name, weight_transposed, is_conv, is_conv_transpose,
        {}};
    chain.chain_ops = std::move(chain_ops);
    chain.consumer = *consumer;
    chain.n_channels = out_channels;
    chains.push_back(std::move(chain));
  }
  return chains;
}

// --- Gated (SwiGLU/GeGLU) chains, mirroring _trace_gate_producer_backward_qdq/
// _qdq_gated_channel_importance/_find_qdq_gated_chains ----------------------

struct QdqFullProducerMatch {
  onnx::NodeProto* node;
  WeightRefMatch ref;
  bool weight_transposed;
  std::optional<std::string> bias;
  int64_t n_channels;
};

// Walks backward from `tensor_name` through unary activation ops until it
// resolves to a QDQ-or-plain-float MatMul/vanilla-Gemm producer's raw
// output. Mirrors TraceGateProducerBackward's own float-only walk, kept
// separate (rather than shared) since `producer_infos` here holds a
// resolved WeightRefMatch rather than a bare weight name -- the same
// "duplicated for a structurally different producer_infos" choice
// pruning.py's own comment gives for its identical Python split.
std::optional<std::pair<QdqFullProducerMatch, std::vector<onnx::NodeProto*>>>
TraceGateProducerBackwardQdq(
    const std::string& tensor_name,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const std::unordered_map<std::string, QdqFullProducerMatch>& producer_infos,
    const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int max_hops) {
  std::vector<onnx::NodeProto*> pre_ops;
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

std::vector<QdqGatedChain> FindQdqGatedChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  DqMap dq_of;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    if (n->op_type() == "DequantizeLinear" && n->output_size() == 1) {
      dq_of[n->output(0)] = n;
    }
  }
  std::unordered_set<std::string> graph_outputs;
  for (const auto& o : graph->output()) {
    graph_outputs.insert(o.name());
  }
  auto is_internal = [&](const std::string& name) {
    auto it = consumers_of.find(name);
    return it != consumers_of.end() && it->second.size() == 1 &&
           !graph_outputs.count(name);
  };

  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  std::unordered_map<std::string, QdqFullProducerMatch> producer_infos;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
    }
    auto mm = MatchMatmulQdq(*node, init_map, dq_of, consumers_of);
    if (mm) {
      producer_infos[node->output(0)] = QdqFullProducerMatch{
          node, mm->ref, mm->weight_transposed, mm->bias, mm->out_channels};
    }
  }

  std::vector<QdqGatedChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    std::optional<QdqFullProducerMatch> info_a, info_b;
    std::vector<onnx::NodeProto*> pre_a, pre_b;

    if (node->op_type() == "Mul" && node->input_size() == 2 &&
        node->output_size() == 1) {
      const std::string& a_name = node->input(0);
      const std::string& b_name = node->input(1);
      if (a_name == b_name || init_map.count(a_name) ||
          init_map.count(b_name)) {
        continue;
      }
      auto trace_a = TraceGateProducerBackwardQdq(a_name, node_by_output,
                                                  producer_infos, consumers_of,
                                                  graph_outputs, kMaxChainHops);
      auto trace_b = TraceGateProducerBackwardQdq(b_name, node_by_output,
                                                  producer_infos, consumers_of,
                                                  graph_outputs, kMaxChainHops);
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
        WalkToConsumerQdq(out_name, false, init_map, dq_of, consumers_of,
                          graph_outputs, info_a->n_channels, kMaxChainHops);
    if (!consumer) {
      continue;
    }

    QdqProducer pa{info_a->node,      info_a->ref,
                   info_a->bias,      info_a->weight_transposed,
                   false /*is_conv*/, false,
                   std::move(pre_a)};
    QdqProducer pb{info_b->node,      info_b->ref,
                   info_b->bias,      info_b->weight_transposed,
                   false /*is_conv*/, false,
                   std::move(pre_b)};
    if (!(pa.ref.is_qdq() || pb.ref.is_qdq() || consumer->ref.is_qdq())) {
      continue;  // All plain float -- FindGatedChains's own job.
    }

    QdqGatedChain chain;
    chain.producer_a = std::move(pa);
    chain.producer_b = std::move(pb);
    chain.chain_ops = std::move(chain_ops);
    chain.consumer = *consumer;
    chain.n_channels = info_a->n_channels;
    chains.push_back(std::move(chain));
  }
  return chains;
}

// --- Apply: slicing, mirroring _slice_producer_weight_qdq[_block]/
// _slice_consumer_weight_qdq[_block]/_qdq_block_aligned_keep_blocks/
// apply_structured_pruning_qdq ----------------------------------------------

using MutInitMap = std::unordered_map<std::string, onnx::TensorProto*>;

MutInitMap BuildMutInitMap(onnx::GraphProto* graph) {
  MutInitMap m;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    m[t->name()] = t;
  }
  return m;
}

struct DequantResult {
  std::vector<double> data;
  std::vector<int64_t> dims;
};

// The full float64 array `ref` refers to, for IMPORTANCE RANKING ONLY -- see
// this section's own top comment. Never written back to the graph.
DequantResult DequantizeWeightRefForRanking(const WeightRefMatch& ref,
                                            const MutInitMap& init_map) {
  if (ref.float_name) {
    const onnx::TensorProto* t = init_map.at(*ref.float_name);
    const std::vector<float> f = ReadFloatTensor(*t);
    return {std::vector<double>(f.begin(), f.end()), ref.dims};
  }
  if (ref.qdq) {
    const QdqWeightMatch& q = *ref.qdq;
    const onnx::TensorProto* qt = init_map.at(q.q_name);
    const onnx::TensorProto* st = init_map.at(q.scale_name);
    const std::vector<int64_t> codes = ReadQuantCodes(*qt);
    const std::vector<float> scale = ReadFloatTensor(*st);
    std::vector<int64_t> zp;
    if (q.zp_name) {
      zp = ReadQuantCodes(*init_map.at(*q.zp_name));
    }
    return {
        PerChannelDequantFlat(codes, q.dims, scale, zp, q.per_channel, q.axis),
        q.dims};
  }
  const QdqBlockwiseWeightMatch& qb = *ref.qdq_block;
  const onnx::TensorProto* qt = init_map.at(qb.q_name);
  const onnx::TensorProto* st = init_map.at(qb.scale_name);
  const std::vector<int64_t> codes = ReadQuantCodes(*qt);
  const std::vector<float> scale = ReadFloatTensor(*st);
  std::vector<int64_t> zp;
  if (qb.zp_name) {
    zp = ReadQuantCodes(*init_map.at(*qb.zp_name));
  }
  return {BlockwiseDequantFlat(codes, qb.dims, scale, zp, qb.block_axis,
                               qb.block_size, qb.num_blocks),
          qb.dims};
}

// This weight's own output-channel axis (`producer_role`) or input/reduction
// axis (consumer role), within its OWN `q_init`/float initializer's dims --
// mirrors pruning.py's own identical branching in `_slice_producer_weight_qdq`/
// `_slice_consumer_weight_qdq`.
int64_t QdqWeightAxis(bool producer_role, bool weight_transposed, bool is_conv,
                      bool is_conv_transpose) {
  if (is_conv) {
    if (producer_role) {
      return is_conv_transpose ? 1 : 0;
    }
    return is_conv_transpose ? 0 : 1;
  }
  if (producer_role) {
    return weight_transposed ? 0 : 1;
  }
  return weight_transposed ? 1 : 0;
}

// Slices `ref`'s own output channels to `keep` (ascending indices) -- the
// producer role. See this section's own top comment for exactly which of
// Wq/Ws/Wzp get co-sliced for each of the three WeightRefMatch sources.
void SliceProducerWeightQdq(const WeightRefMatch& ref, bool weight_transposed,
                            const std::vector<int64_t>& keep, bool is_conv,
                            bool is_conv_transpose, MutInitMap& init_map) {
  if (ref.float_name) {
    onnx::TensorProto* wt = init_map.at(*ref.float_name);
    if (!is_conv_transpose) {
      SliceProducerWeight(wt, weight_transposed, keep, is_conv);
      return;
    }
    // ConvTranspose float producer: [Cin, Cout, kH, kW], axis 1.
    const std::vector<int64_t> dims(wt->dims().begin(), wt->dims().end());
    const std::vector<float> data = ReadFloatTensor(*wt);
    int64_t inner = 1;
    for (size_t i = 2; i < dims.size(); ++i) {
      inner *= dims[i];
    }
    const std::vector<float> out =
        SliceAxis1(data, dims[0], dims[1], inner, keep);
    std::vector<int64_t> new_dims = dims;
    new_dims[1] = static_cast<int64_t>(keep.size());
    SetFloatTensorData(wt, new_dims, out);
    return;
  }

  const int64_t axis =
      QdqWeightAxis(true, weight_transposed, is_conv, is_conv_transpose);
  if (ref.qdq) {
    const QdqWeightMatch& q = *ref.qdq;
    onnx::TensorProto* qt = init_map.at(q.q_name);
    const std::vector<int64_t> codes = ReadQuantCodes(*qt);
    const std::vector<int64_t> new_codes =
        SliceAlongAxis(codes, q.dims, axis, keep);
    std::vector<int64_t> new_dims = q.dims;
    new_dims[static_cast<size_t>(axis)] = static_cast<int64_t>(keep.size());
    SetQuantCodes(qt, qt->data_type(), new_dims, new_codes);

    if (q.per_channel) {
      SliceLastAxis(init_map.at(q.scale_name), keep);
      if (q.zp_name) {
        onnx::TensorProto* zt = init_map.at(*q.zp_name);
        const std::vector<int64_t> zdims(zt->dims().begin(), zt->dims().end());
        const std::vector<int64_t> zcodes = ReadQuantCodes(*zt);
        const std::vector<int64_t> new_z =
            SliceAlongAxis(zcodes, zdims, 0, keep);
        std::vector<int64_t> new_zdims = zdims;
        new_zdims[0] = static_cast<int64_t>(keep.size());
        SetQuantCodes(zt, zt->data_type(), new_zdims, new_z);
      }
    }
    return;
  }

  // Blockwise: out_axis never blocked, so codes/scale/zero_point co-slice by
  // `keep` in lockstep, no alignment concern (see this section's own top
  // comment).
  const QdqBlockwiseWeightMatch& qb = *ref.qdq_block;
  onnx::TensorProto* qt = init_map.at(qb.q_name);
  const std::vector<int64_t> codes = ReadQuantCodes(*qt);
  const std::vector<int64_t> new_codes =
      SliceAlongAxis(codes, qb.dims, qb.out_axis, keep);
  std::vector<int64_t> new_dims = qb.dims;
  new_dims[static_cast<size_t>(qb.out_axis)] =
      static_cast<int64_t>(keep.size());
  SetQuantCodes(qt, qt->data_type(), new_dims, new_codes);

  onnx::TensorProto* st = init_map.at(qb.scale_name);
  const std::vector<int64_t> sdims(st->dims().begin(), st->dims().end());
  const std::vector<float> scale = ReadFloatTensor(*st);
  const std::vector<float> new_scale =
      SliceAlongAxis(scale, sdims, qb.out_axis, keep);
  std::vector<int64_t> new_sdims = sdims;
  new_sdims[static_cast<size_t>(qb.out_axis)] =
      static_cast<int64_t>(keep.size());
  SetFloatTensorData(st, new_sdims, new_scale);

  if (qb.zp_name) {
    onnx::TensorProto* zt = init_map.at(*qb.zp_name);
    const std::vector<int64_t> zdims(zt->dims().begin(), zt->dims().end());
    const std::vector<int64_t> zcodes = ReadQuantCodes(*zt);
    const std::vector<int64_t> new_z =
        SliceAlongAxis(zcodes, zdims, qb.out_axis, keep);
    std::vector<int64_t> new_zdims = zdims;
    new_zdims[static_cast<size_t>(qb.out_axis)] =
        static_cast<int64_t>(keep.size());
    SetQuantCodes(zt, zt->data_type(), new_zdims, new_z);
  }
}

// Slices `ref`'s own input (reduction) channels to `keep` -- the consumer
// role for a plain float or per-tensor/per-channel QDQ weight only (never a
// blockwise one -- see SliceConsumerWeightQdqBlock below). Never touches
// scale/zero_point: they are indexed by the OUTPUT channel axis (or are a
// scalar), never by the input axis sliced here.
void SliceConsumerWeightQdq(const WeightRefMatch& ref, bool weight_transposed,
                            const std::vector<int64_t>& keep, bool is_conv,
                            bool is_conv_transpose, MutInitMap& init_map) {
  if (ref.float_name) {
    onnx::TensorProto* wt = init_map.at(*ref.float_name);
    if (!is_conv_transpose) {
      SliceConsumerWeight(wt, weight_transposed, keep, is_conv);
      return;
    }
    // ConvTranspose float consumer: [Cin, Cout, kH, kW], axis 0, flat (any
    // group).
    const std::vector<int64_t> dims(wt->dims().begin(), wt->dims().end());
    const std::vector<float> data = ReadFloatTensor(*wt);
    int64_t inner = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
      inner *= dims[i];
    }
    const std::vector<float> out = SliceAxis0(data, dims[0], inner, keep);
    std::vector<int64_t> new_dims = dims;
    new_dims[0] = static_cast<int64_t>(keep.size());
    SetFloatTensorData(wt, new_dims, out);
    return;
  }
  const QdqWeightMatch& q = *ref.qdq;
  const int64_t axis =
      QdqWeightAxis(false, weight_transposed, is_conv, is_conv_transpose);
  onnx::TensorProto* qt = init_map.at(q.q_name);
  const std::vector<int64_t> codes = ReadQuantCodes(*qt);
  const std::vector<int64_t> new_codes =
      SliceAlongAxis(codes, q.dims, axis, keep);
  std::vector<int64_t> new_dims = q.dims;
  new_dims[static_cast<size_t>(axis)] = static_cast<int64_t>(keep.size());
  SetQuantCodes(qt, qt->data_type(), new_dims, new_codes);
}

// Returns the ascending block indices `keep` corresponds to when every
// `block_size`-sized block of a `block_dim`-length axis is either wholly
// present in `keep` or wholly absent -- or nullopt when some block is only
// partially kept (this chain's consumer cannot be safely pruned to this
// exact `keep` set). Mirrors pruning.py's own `_qdq_block_aligned_keep_blocks`.
std::optional<std::vector<int64_t>> QdqBlockAlignedKeepBlocks(
    const std::vector<int64_t>& keep, int64_t block_dim, int64_t block_size) {
  const std::unordered_set<int64_t> keep_set(keep.begin(), keep.end());
  const int64_t num_blocks = block_dim / block_size;
  std::vector<int64_t> keep_blocks;
  for (int64_t b = 0; b < num_blocks; ++b) {
    const int64_t lo = b * block_size;
    bool all = true, any = false;
    for (int64_t p = lo; p < lo + block_size; ++p) {
      const bool in = keep_set.count(p) != 0;
      all = all && in;
      any = any || in;
    }
    if (all) {
      keep_blocks.push_back(b);
    } else if (any) {
      return std::nullopt;
    }
  }
  return keep_blocks;
}

// Slices `qb`'s own input/reduction (`block_axis`) axis -- the consumer role
// for a blockwise-quantized weight. Never called with a non-block-aligned
// `keep`; `keep_blocks` (QdqBlockAlignedKeepBlocks) is validated by the
// caller first. `q_init` is sliced element-wise by `keep`; `scale`/
// `zero_point` are indexed by BLOCK, so sliced by `keep_blocks` instead,
// along the same `block_axis`.
void SliceConsumerWeightQdqBlock(const QdqBlockwiseWeightMatch& qb,
                                 const std::vector<int64_t>& keep,
                                 const std::vector<int64_t>& keep_blocks,
                                 MutInitMap& init_map) {
  onnx::TensorProto* qt = init_map.at(qb.q_name);
  const std::vector<int64_t> codes = ReadQuantCodes(*qt);
  const std::vector<int64_t> new_codes =
      SliceAlongAxis(codes, qb.dims, qb.block_axis, keep);
  std::vector<int64_t> new_dims = qb.dims;
  new_dims[static_cast<size_t>(qb.block_axis)] =
      static_cast<int64_t>(keep.size());
  SetQuantCodes(qt, qt->data_type(), new_dims, new_codes);

  onnx::TensorProto* st = init_map.at(qb.scale_name);
  const std::vector<int64_t> sdims(st->dims().begin(), st->dims().end());
  const std::vector<float> scale = ReadFloatTensor(*st);
  const std::vector<float> new_scale =
      SliceAlongAxis(scale, sdims, qb.block_axis, keep_blocks);
  std::vector<int64_t> new_sdims = sdims;
  new_sdims[static_cast<size_t>(qb.block_axis)] =
      static_cast<int64_t>(keep_blocks.size());
  SetFloatTensorData(st, new_sdims, new_scale);

  if (qb.zp_name) {
    onnx::TensorProto* zt = init_map.at(*qb.zp_name);
    const std::vector<int64_t> zdims(zt->dims().begin(), zt->dims().end());
    const std::vector<int64_t> zcodes = ReadQuantCodes(*zt);
    const std::vector<int64_t> new_z =
        SliceAlongAxis(zcodes, zdims, qb.block_axis, keep_blocks);
    std::vector<int64_t> new_zdims = zdims;
    new_zdims[static_cast<size_t>(qb.block_axis)] =
        static_cast<int64_t>(keep_blocks.size());
    SetQuantCodes(zt, zt->data_type(), new_zdims, new_z);
  }
}

// The shared apply body for both plain and gated QDQ chains, mirroring
// `apply_structured_pruning_qdq`'s own main loop over
// `_find_qdq_chains`/`_find_qdq_gated_chains` -- see this section's own top
// comment for the producer/consumer co-slicing rules this enforces.
void ApplyQdqChains(onnx::GraphProto* graph, std::vector<QdqChain>& chains,
                    std::vector<QdqGatedChain>& gated_chains, double sparsity,
                    TouchedState& touched) {
  MutInitMap init_map = BuildMutInitMap(graph);
  auto& producer_touched = touched.producer;
  auto& consumer_touched = touched.consumer;
  auto& stale_value_info = touched.stale_value_info;

  auto mark_dq_stale = [&](const WeightRefMatch& ref) {
    if (ref.qdq) {
      stale_value_info.insert(ref.qdq->dq_node->output(0));
    } else if (ref.qdq_block) {
      stale_value_info.insert(ref.qdq_block->dq_node->output(0));
    }
  };

  for (const auto& chain : chains) {
    const QdqProducer& p = chain.producer;
    const QdqConsumerMatch& c = chain.consumer;
    const std::string p_key = p.ref.key();
    const std::string c_key = c.ref.key();
    if (p_key == c_key) {
      continue;  // Degenerate (the same weight in both roles).
    }
    if (producer_touched.count(p_key) || consumer_touched.count(c_key)) {
      continue;  // A shared/tied weight another chain already resized.
    }

    const int64_t n = chain.n_channels;
    const int64_t keep_count = std::max<int64_t>(
        1, n - std::llround(static_cast<double>(n) * sparsity));
    if (keep_count >= n) {
      continue;  // Rounds down to nothing for this layer -- no-op.
    }

    const DequantResult w = DequantizeWeightRefForRanking(p.ref, init_map);
    const std::vector<double> w_nk = QdqWeightToNk(
        w.data, w.dims, p.weight_transposed, p.is_conv, p.is_conv_transpose);
    const int64_t k = static_cast<int64_t>(w_nk.size()) / n;
    const std::vector<double> importance = QdqChannelImportanceL2(w_nk, n, k);
    // Stable tie-break matters here specifically -- see
    // StableTopKIndicesAscending's own comment: an unstable tie-break can
    // flip a would-have-been block-aligned consumer to "declined"
    // nondeterministically.
    const std::vector<int64_t> keep =
        StableTopKIndicesAscending(importance, keep_count);

    std::optional<std::vector<int64_t>> keep_blocks;
    if (c.ref.qdq_block) {
      keep_blocks =
          QdqBlockAlignedKeepBlocks(keep, n, c.ref.qdq_block->block_size);
      if (!keep_blocks) {
        continue;  // Non-block-aligned keep set -- decline the whole chain.
      }
    }

    SliceProducerWeightQdq(p.ref, p.weight_transposed, keep, p.is_conv,
                           p.is_conv_transpose, init_map);
    if (p.bias) {
      SliceLastAxis(init_map.at(*p.bias), keep);
    }
    if (keep_blocks) {
      SliceConsumerWeightQdqBlock(*c.ref.qdq_block, keep, *keep_blocks,
                                  init_map);
    } else {
      SliceConsumerWeightQdq(c.ref, c.weight_transposed, keep, c.is_conv,
                             c.is_conv_transpose, init_map);
    }

    producer_touched.insert(p_key);
    consumer_touched.insert(c_key);
    stale_value_info.insert(p.node->output(0));
    for (onnx::NodeProto* op : chain.chain_ops) {
      stale_value_info.insert(op->output(0));
    }
    mark_dq_stale(p.ref);
    mark_dq_stale(c.ref);
  }

  for (const auto& gchain : gated_chains) {
    const QdqProducer& pa = gchain.producer_a;
    const QdqProducer& pb = gchain.producer_b;
    const QdqConsumerMatch& c = gchain.consumer;
    const std::string pa_key = pa.ref.key();
    const std::string pb_key = pb.ref.key();
    const std::string c_key = c.ref.key();
    if (pa_key == pb_key || pa_key == c_key || pb_key == c_key) {
      continue;  // Degenerate (a weight tied across two roles).
    }
    if (producer_touched.count(pa_key) || producer_touched.count(pb_key) ||
        consumer_touched.count(c_key)) {
      continue;
    }

    const int64_t n = gchain.n_channels;
    const int64_t keep_count = std::max<int64_t>(
        1, n - std::llround(static_cast<double>(n) * sparsity));
    if (keep_count >= n) {
      continue;
    }

    const DequantResult wa = DequantizeWeightRefForRanking(pa.ref, init_map);
    const DequantResult wb = DequantizeWeightRefForRanking(pb.ref, init_map);
    const std::vector<double> wa_nk =
        QdqWeightToNk(wa.data, wa.dims, pa.weight_transposed, false, false);
    const std::vector<double> wb_nk =
        QdqWeightToNk(wb.data, wb.dims, pb.weight_transposed, false, false);
    const int64_t k = static_cast<int64_t>(wa_nk.size()) / n;
    const std::vector<double> imp_a = QdqChannelImportanceL2(wa_nk, n, k);
    const std::vector<double> imp_b = QdqChannelImportanceL2(wb_nk, n, k);
    std::vector<double> importance(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      importance[static_cast<size_t>(i)] = std::sqrt(
          imp_a[static_cast<size_t>(i)] * imp_a[static_cast<size_t>(i)] +
          imp_b[static_cast<size_t>(i)] * imp_b[static_cast<size_t>(i)]);
    }
    const std::vector<int64_t> keep =
        StableTopKIndicesAscending(importance, keep_count);

    std::optional<std::vector<int64_t>> keep_blocks;
    if (c.ref.qdq_block) {
      keep_blocks =
          QdqBlockAlignedKeepBlocks(keep, n, c.ref.qdq_block->block_size);
      if (!keep_blocks) {
        continue;
      }
    }

    SliceProducerWeightQdq(pa.ref, pa.weight_transposed, keep, false, false,
                           init_map);
    if (pa.bias) {
      SliceLastAxis(init_map.at(*pa.bias), keep);
    }
    SliceProducerWeightQdq(pb.ref, pb.weight_transposed, keep, false, false,
                           init_map);
    if (pb.bias) {
      SliceLastAxis(init_map.at(*pb.bias), keep);
    }
    if (keep_blocks) {
      SliceConsumerWeightQdqBlock(*c.ref.qdq_block, keep, *keep_blocks,
                                  init_map);
    } else {
      SliceConsumerWeightQdq(c.ref, c.weight_transposed, keep, false, false,
                             init_map);
    }

    producer_touched.insert(pa_key);
    producer_touched.insert(pb_key);
    consumer_touched.insert(c_key);
    stale_value_info.insert(pa.node->output(0));
    for (onnx::NodeProto* op : pa.pre_ops) {
      stale_value_info.insert(op->output(0));
    }
    stale_value_info.insert(pb.node->output(0));
    for (onnx::NodeProto* op : pb.pre_ops) {
      stale_value_info.insert(op->output(0));
    }
    for (onnx::NodeProto* op : gchain.chain_ops) {
      stale_value_info.insert(op->output(0));
    }
    mark_dq_stale(pa.ref);
    mark_dq_stale(pb.ref);
    mark_dq_stale(c.ref);
  }
}

// --- MatMulBnb4 (bitsandbytes FP4/NF4 block-quantized weight) structured
// --- pruning -------------------------------------------------------------
//
// Mirrors pruning.py's own "MatMulBnb4 (bitsandbytes FP4/NF4 block-quantized
// weight) structured pruning" section -- read that section's top comment
// first; it carries the full empirical schema/packing investigation (live
// onnxruntime schema introspection, `matmul_bnb4_quantizer.py`'s own source,
// and real `InferenceSession` round-trips) this whole section depends on,
// none of which is re-derived here. The load-bearing facts this port relies
// on, restated only as a checklist:
//
//   * `com.microsoft::MatMulBnb4` always has exactly 3 inputs (A, B,
//     absmax) -- no zero_points, no g_idx, no bias, unlike `MatMulNBits`.
//   * `B` is a FLAT 1-D `uint8` tensor of shape `[(N*K+1)//2]` -- the WHOLE
//     `[N, K]` weight flattened row-major and packed 2 codes/byte, NOT
//     `MatMulNBits`'s own `[N, k_blocks, blob_size]` (N kept as an explicit,
//     never-packed-across leading axis). Nibbles are packed HIGH nibble
//     first (flattened code `2*i` is byte `i`'s `>> 4`, code `2*i+1` is byte
//     `i`'s `& 0xF`) -- the OPPOSITE of `MatMulNBits`'s own confirmed
//     low-nibble-first convention, so the nibble unpack below is a
//     deliberately separate, non-shared helper rather than reusing
//     UnpackNibblesLastAxis with swapped arguments.
//   * `absmax` is a flat 1-D tensor of shape `[(N*K+block_size-1)//
//     block_size]`, one scale per flattened block.
//   * A `block_size`-sized block stays entirely inside one output row's own
//     `K` values -- and is therefore safe to keep/drop as a whole unit when
//     N-axis-pruning -- if and only if `K % block_size == 0` (see
//     pruning.py's own section comment for the full row-alignment argument,
//     both mathematical and empirical). `MatchMatMulBnb4Producer` declines
//     whenever this fails, mirroring `MatMulNBits`'s own analogous decline.
//   * Because every valid `block_size` is even (>= 16, a power of two) and
//     `K` is therefore always even too once that alignment check passes,
//     row `n`'s own packed bytes occupy EXACTLY `[n*(K/2), (n+1)*(K/2))` of
//     `B` -- a whole number of bytes, always -- so producer-side N-axis
//     pruning is a plain byte-range/scale-range row slice, no nibble-level
//     unpack/repack at all (simpler than `MatMulNBits`'s own producer-row
//     slice, which still has to handle an optional packed `zero_points`).
//
// Scope boundaries, mirroring pruning.py's own section comment exactly
// (each one restated there in more depth -- not re-argued here):
//   * PRODUCER role only -- a `MatMulBnb4` node is never matched as a
//     chain's CONSUMER (K-axis/input-channel pruning would need to
//     re-slice every one of the consumer's own N rows in lockstep, a
//     genuinely different problem left to a follow-up).
//   * The chain's consumer is always a plain-float (directly-constant
//     weight) `MatMul`/vanilla-`Gemm` -- `MatchPlainMatMulNBitsPeer`,
//     reused directly, unmodified, from the `MatMulNBits` section above
//     (fully generic despite its name).
//   * `transB=0` and `training_mode != 0` always declined -- unverified
//     semantics.
//   * `quant_type` outside `{0 (FP4), 1 (NF4)}` declined.
//   * `block_size` outside `{16, 32, 64, 128, 256}` declined (reuses
//     MatMulNBitsValidBlockSizes directly, the same live-CPU-kernel-
//     confirmed set).
//   * `K % block_size != 0` declined -- the whole chain left untouched.
//   * No grouped/gated (SwiGLU/GeGLU) pair -- only the plain single
//     producer -> [unary hops] -> single plain-float consumer topology,
//     mirroring `MatMulNBits`'s own identical scope decision.
//   * `absmax` admitted only as plain FLOAT (float32) here -- this file's
//     own float-tensor helpers have no FLOAT16/BFLOAT16 support anywhere
//     yet, mirroring `MatMulNBits`'s own identical, already-established
//     narrower-than-pruning.py restriction for its own `scales`/`bias`.
//   * A shared/tied `B`/`absmax` tensor (read by more than one node) is
//     declined by the matcher itself.
//
// Only L2 importance is supported here (reuses QdqChannelImportanceL2
// directly) -- this port, like every other quantized-weight family in this
// file, has no L1 option anywhere; StableTopKIndicesAscending (not the
// unstable TopKIndicesAscending) is used for the same cross-platform-
// determinism reason documented at its own definition above, so this
// section's own keep-set selection stays byte-identical to pruning.py's own
// `np.argsort(-importance, kind="stable")[:keep_count]` on exact ties.

constexpr int64_t kMatMulBnb4Fp4 = 0;
constexpr int64_t kMatMulBnb4Nf4 = 1;

bool MatMulBnb4QuantTypeValid(int64_t quant_type) {
  return quant_type == kMatMulBnb4Fp4 || quant_type == kMatMulBnb4Nf4;
}

// Empirically confirmed (see this section's own top comment) 16-level
// dequantization code tables, code (0-15) -> double value, for each
// `quant_type`. Used ONLY for producer-side importance ranking
// (MatMulBnb4Dequantized) -- never written back to the graph.
const std::array<double, 16>& MatMulBnb4Fp4Codes() {
  static const std::array<double, 16> kCodes = {
      0.0,
      0.005208333333333333,
      0.6666666666666666,
      1.0,
      0.3333333333333333,
      0.5,
      0.16666666666666666,
      0.25,
      -0.0,
      -0.005208333333333333,
      -0.6666666666666666,
      -1.0,
      -0.3333333333333333,
      -0.5,
      -0.16666666666666666,
      -0.25,
  };
  return kCodes;
}

const std::array<double, 16>& MatMulBnb4Nf4Codes() {
  static const std::array<double, 16> kCodes = {
      -1.0,
      -0.6961928009986877,
      -0.5250730514526367,
      -0.39491748809814453,
      -0.28444138169288635,
      -0.18477343022823334,
      -0.09105003625154495,
      0.0,
      0.07958029955625534,
      0.16093020141124725,
      0.24611230194568634,
      0.33791524171829224,
      0.44070982933044434,
      0.5626170039176941,
      0.7229568362236023,
      1.0,
  };
  return kCodes;
}

const std::array<double, 16>& MatMulBnb4CodeTable(int64_t quant_type) {
  return quant_type == kMatMulBnb4Fp4 ? MatMulBnb4Fp4Codes()
                                      : MatMulBnb4Nf4Codes();
}

// A `com.microsoft::MatMulBnb4` node's block-quantized weight operands,
// matched by MatchMatMulBnb4Producer -- see this section's own top comment
// for the schema facts and packing layout this depends on. `blocks_per_row`
// (`K / block_size`) and `row_bytes` (`K / 2`) are precomputed here since
// every producer-side slice needs both and MatchMatMulBnb4Producer has
// already verified `K % block_size == 0` (so both divide evenly, and
// `row_bytes` in particular is a whole number since `block_size` -- and
// therefore `K` -- is always even) by the time this is constructed.
struct MatMulBnb4Weight {
  onnx::NodeProto* node = nullptr;
  std::string b_name;
  std::string absmax_name;
  int64_t N = 0;
  int64_t K = 0;
  int64_t block_size = 0;
  int64_t quant_type = 0;
  int64_t blocks_per_row = 0;
  int64_t row_bytes = 0;
};

// If `node` is a `com.microsoft::MatMulBnb4` node matching every scope
// boundary this section's own top comment documents, returns the match --
// mirrors pruning.py's own `_match_matmul_bnb4_producer` exactly. `nullopt`
// whenever anything is ambiguous or out of scope, rather than guessing.
std::optional<MatMulBnb4Weight> MatchMatMulBnb4Producer(
    onnx::NodeProto* node, const InitMap& init_map,
    const ConsumerMap& consumers_of) {
  if (node->op_type() != "MatMulBnb4" ||
      node->domain() != kComMicrosoftDomain) {
    return std::nullopt;
  }
  if (node->input_size() != 3 || node->output_size() != 1) {
    return std::nullopt;
  }
  const std::string& a_name = node->input(0);
  const std::string& b_name = node->input(1);
  const std::string& absmax_name = node->input(2);
  if (a_name.empty() || b_name.empty() || absmax_name.empty()) {
    return std::nullopt;
  }

  if (MatMulNBitsIntAttrOr(*node, "transB", 1) != 1) {
    return std::nullopt;  // "backward pass" orientation -- unverified,
                          // declined.
  }
  if (MatMulNBitsIntAttrOr(*node, "training_mode", 0) != 0) {
    return std::nullopt;  // training-mode semantics unverified -- declined.
  }

  const auto quant_type_opt = MatMulNBitsIntAttr(*node, "quant_type");
  const auto n_opt = MatMulNBitsIntAttr(*node, "N");
  const auto k_opt = MatMulNBitsIntAttr(*node, "K");
  const auto block_size_opt = MatMulNBitsIntAttr(*node, "block_size");
  if (!quant_type_opt || !n_opt || !k_opt || !block_size_opt) {
    return std::nullopt;
  }
  const int64_t quant_type = *quant_type_opt;
  const int64_t N = *n_opt;
  const int64_t K = *k_opt;
  const int64_t block_size = *block_size_opt;
  if (!MatMulBnb4QuantTypeValid(quant_type)) {
    return std::nullopt;
  }
  if (N <= 0 || K <= 0) {
    return std::nullopt;
  }
  if (!MatMulNBitsValidBlockSizes().count(block_size)) {
    return std::nullopt;
  }
  if (K % block_size != 0) {
    return std::nullopt;  // A block would straddle two output rows.
  }

  const int64_t blocks_per_row = K / block_size;
  const int64_t row_bytes = K / 2;  // K is always even here.

  auto b_it = init_map.find(b_name);
  auto absmax_it = init_map.find(absmax_name);
  if (b_it == init_map.end() || absmax_it == init_map.end()) {
    return std::nullopt;  // Non-constant B/absmax -- can't safely slice them.
  }
  const onnx::TensorProto* b_init = b_it->second;
  const onnx::TensorProto* absmax_init = absmax_it->second;
  if (b_init->data_type() != onnx::TensorProto::UINT8) {
    return std::nullopt;
  }
  if (!MatMulNBitsDimsEqual(*b_init, {N * row_bytes})) {
    return std::nullopt;
  }
  // Scope: FLOAT32 only -- see this section's own top comment.
  if (absmax_init->data_type() != onnx::TensorProto::FLOAT) {
    return std::nullopt;
  }
  if (!MatMulNBitsDimsEqual(*absmax_init, {N * blocks_per_row})) {
    return std::nullopt;
  }

  for (const std::string& nm : {b_name, absmax_name}) {
    if (ConsumerCount(consumers_of, nm) != 1) {
      return std::nullopt;  // Shared/tied tensor -- another node reads it too.
    }
  }

  MatMulBnb4Weight w;
  w.node = node;
  w.b_name = b_name;
  w.absmax_name = absmax_name;
  w.N = N;
  w.K = K;
  w.block_size = block_size;
  w.quant_type = quant_type;
  w.blocks_per_row = blocks_per_row;
  w.row_bytes = row_bytes;
  return w;
}

// Unpacks `packed` (`[outer, row_bytes]`, `row_bytes == count / 2`) into
// `[outer, count]` codes in [0, 15], 2 per byte, HIGH NIBBLE FIRST -- see
// this section's own top comment for why this is the opposite nibble order
// from, and therefore not shared with, UnpackNibblesLastAxis
// (`MatMulNBits`'s own low-nibble-first convention). `count` is always even
// here (MatchMatMulBnb4Producer only ever constructs a `MatMulBnb4Weight`
// once `K % block_size == 0` -- and therefore `K` itself -- is confirmed
// even), so there is no odd-trailing-nibble case to handle, unlike
// UnpackNibblesLastAxis's own general form.
std::vector<uint8_t> UnpackBnb4CodesHighNibbleFirst(
    const std::vector<uint8_t>& packed, int64_t outer, int64_t row_bytes,
    int64_t count) {
  std::vector<uint8_t> out(static_cast<size_t>(outer * count));
  for (int64_t r = 0; r < outer; ++r) {
    const uint8_t* prow = packed.data() + r * row_bytes;
    uint8_t* orow = out.data() + r * count;
    for (int64_t j = 0; j < row_bytes; ++j) {
      const uint8_t byte = prow[j];
      orow[2 * j] = (byte >> 4) & 0x0F;
      orow[2 * j + 1] = byte & 0x0F;
    }
  }
  return out;
}

// The full float64 `[N, K]` dequantized weight matrix `w` refers to, for
// IMPORTANCE RANKING ONLY -- never written back to the graph (this file's
// own "slice codes directly, never dequant-requant" invariant, restated in
// this section's own top comment). `dequant[n, k] = table[code[n, k]] *
// absmax[n, k / block_size]` -- mirrors pruning.py's own
// `_matmul_bnb4_dequantized` exactly. `init_map` here is the Apply phase's
// own MUTABLE name->TensorProto* map (this function is only ever called
// from ApplyMatMulBnb4Chains, never during matching).
std::vector<double> MatMulBnb4Dequantized(
    const MatMulBnb4Weight& w,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map) {
  const onnx::TensorProto* b_init = init_map.at(w.b_name);
  const onnx::TensorProto* absmax_init = init_map.at(w.absmax_name);

  const std::vector<uint8_t> packed =
      ReadUint8Tensor(*b_init);  // [N*row_bytes]
  const std::vector<uint8_t> codes =
      UnpackBnb4CodesHighNibbleFirst(packed, w.N, w.row_bytes, w.K);  // [N, K]

  const std::vector<float> absmax_data =
      ReadFloatTensor(*absmax_init);  // [N * blocks_per_row]

  const std::array<double, 16>& table = MatMulBnb4CodeTable(w.quant_type);

  std::vector<double> out(static_cast<size_t>(w.N * w.K));
  for (int64_t n = 0; n < w.N; ++n) {
    for (int64_t kb = 0; kb < w.blocks_per_row; ++kb) {
      const double scale = static_cast<double>(
          absmax_data[static_cast<size_t>(n * w.blocks_per_row + kb)]);
      for (int64_t j = 0; j < w.block_size; ++j) {
        const int64_t idx = n * w.K + kb * w.block_size + j;
        const uint8_t code = codes[static_cast<size_t>(idx)];
        out[static_cast<size_t>(idx)] = table[code] * scale;
      }
    }
  }
  return out;
}

// Slices `w`'s own N (output-channel) axis to `keep` (ascending indices) --
// a plain byte-range/scale-range row-slice, no nibble-level unpack/repack at
// all (see this section's own top comment for why every row's own bytes/
// scales are already whole-byte/whole-block aligned once
// MatchMatMulBnb4Producer has confirmed `K % block_size == 0`). Both `B`
// and `absmax` are stored FLAT (1-D), unlike `MatMulNBits`'s own `[N,
// k_blocks, blob_size]`/`[N, k_blocks]`, so the sliced result is written
// back flat too -- mirrors pruning.py's own `_slice_matmul_bnb4_producer_
// rows` (`packed[keep].reshape(-1)`) exactly. Updates the node's own `N`
// attribute to `len(keep)`.
void SliceMatMulBnb4ProducerRows(
    const MatMulBnb4Weight& w,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    const std::vector<int64_t>& keep) {
  const int64_t kc = static_cast<int64_t>(keep.size());

  onnx::TensorProto* b = init_map.at(w.b_name);
  const std::vector<uint8_t> b_data = ReadUint8Tensor(*b);  // [N*row_bytes]
  std::vector<uint8_t> b_out(static_cast<size_t>(kc * w.row_bytes));
  for (int64_t i = 0; i < kc; ++i) {
    std::memcpy(b_out.data() + i * w.row_bytes,
                b_data.data() + keep[i] * w.row_bytes,
                static_cast<size_t>(w.row_bytes));
  }
  SetUint8TensorData(b, {kc * w.row_bytes}, b_out);

  onnx::TensorProto* absmax = init_map.at(w.absmax_name);
  const std::vector<float> absmax_data =
      ReadFloatTensor(*absmax);  // [N*blocks_per_row]
  std::vector<float> absmax_out(static_cast<size_t>(kc * w.blocks_per_row));
  for (int64_t i = 0; i < kc; ++i) {
    std::memcpy(absmax_out.data() + i * w.blocks_per_row,
                absmax_data.data() + keep[i] * w.blocks_per_row,
                static_cast<size_t>(w.blocks_per_row) * sizeof(float));
  }
  SetFloatTensorData(absmax, {kc * w.blocks_per_row}, absmax_out);

  SetOrAddIntAttr(w.node, "N", kc);
}

struct MatMulBnb4Chain {
  MatMulBnb4Weight producer;
  std::vector<onnx::NodeProto*> chain_ops;
  PlainMatMulNBitsPeer consumer;
  int64_t n_channels = 0;
};

// From tensor `start` (a `MatMulBnb4` producer's own output), walks forward
// through shape-preserving unary activations (UnaryPassThroughOps) with no
// other consumer anywhere along the way, until a plain-float (directly-
// constant weight) `MatMul`/vanilla-`Gemm` consumer
// (MatchPlainMatMulNBitsPeer -- reused directly here, fully generic despite
// its name) is found whose input-channel count matches `n_channels`. Unlike
// WalkToMatMulNBitsConsumer's own `MatMulNBits`/plain-float union, a
// `MatMulBnb4` CONSUMER is never matched here at all -- see this section's
// own top comment for why K-axis pruning of a `MatMulBnb4` weight remains
// out of scope. Returns `nullopt` if the walk runs out of hops, hits a
// branch, or never reaches such a consumer. Mirrors pruning.py's own
// `_walk_to_matmul_bnb4_consumer`.
std::optional<std::pair<PlainMatMulNBitsPeer, std::vector<onnx::NodeProto*>>>
WalkToMatMulBnb4Consumer(const std::string& start, const InitMap& init_map,
                         const ConsumerMap& consumers_of,
                         const std::unordered_set<std::string>& graph_outputs,
                         int64_t n_channels, int max_hops) {
  std::vector<onnx::NodeProto*> chain_ops;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    auto cit = consumers_of.find(cur);
    if (cit == consumers_of.end() || cit->second.size() != 1) {
      return std::nullopt;
    }
    onnx::NodeProto* nxt = cit->second[0];

    auto mm = MatchMatMulLikeRaw(*nxt);
    if (mm && mm->x_name == cur) {
      auto peer = MatchPlainMatMulNBitsPeer(nxt, init_map, consumers_of);
      if (!peer || peer->in_channels != n_channels) {
        return std::nullopt;
      }
      return std::make_pair(*peer, std::move(chain_ops));
    }

    if (!(UnaryPassThroughOps().count(nxt->op_type()) != 0 &&
          nxt->input_size() == 1 && nxt->input(0) == cur &&
          nxt->output_size() == 1)) {
      return std::nullopt;
    }
    const std::string& out2 = nxt->output(0);
    auto oc = consumers_of.find(out2);
    if (oc == consumers_of.end() || oc->second.size() != 1 ||
        graph_outputs.count(out2)) {
      return std::nullopt;
    }
    chain_ops.push_back(nxt);
    cur = out2;
  }
  return std::nullopt;
}

// Every `MatMulBnb4` producer -> [unary hops] -> plain-float consumer pair
// WalkToMatMulBnb4Consumer connects -- see this section's own top comment
// for why the consumer side is always plain-float (never another
// `MatMulBnb4`/`MatMulNBits` node). Mirrors pruning.py's own
// `_find_matmul_bnb4_chains`.
std::vector<MatMulBnb4Chain> FindMatMulBnb4Chains(onnx::GraphProto* graph) {
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

  std::vector<MatMulBnb4Chain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto producer = MatchMatMulBnb4Producer(node, init_map, consumers_of);
    if (!producer) {
      continue;
    }
    const std::string& out_name = node->output(0);
    if (!is_internal(out_name)) {
      continue;
    }
    auto found =
        WalkToMatMulBnb4Consumer(out_name, init_map, consumers_of,
                                 graph_outputs, producer->N, kMaxChainHops);
    if (!found) {
      continue;
    }
    chains.push_back(MatMulBnb4Chain{std::move(*producer),
                                     std::move(found->second),
                                     std::move(found->first), producer->N});
  }
  return chains;
}

// The actual apply step -- mirrors pruning.py's own
// `apply_structured_pruning_matmul_bnb4`'s own per-chain loop. Shares
// `touched` with every other chain family ApplyStructuredPruning already
// applies over this same graph, so a plain-float weight this pass also
// happens to match (as the consumer side of a chain) can never be
// double-resized by, or double-resize, an ordinary chain that already
// touched it.
void ApplyMatMulBnb4Chains(onnx::GraphProto* graph,
                           std::vector<MatMulBnb4Chain>& chains,
                           double sparsity, TouchedState& touched) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }

  for (auto& chain : chains) {
    const std::string& p_key = chain.producer.b_name;
    const std::string& c_key = chain.consumer.w_name;
    if (touched.producer.count(p_key) || touched.consumer.count(c_key)) {
      continue;  // A shared/tied weight another chain already resized.
    }

    const int64_t n = chain.n_channels;
    const int64_t keep_count = std::max<int64_t>(
        1, n - std::llround(static_cast<double>(n) * sparsity));
    if (keep_count >= n) {
      continue;  // Rounds down to nothing for this layer -- no-op.
    }

    const std::vector<double> w_nk =
        MatMulBnb4Dequantized(chain.producer, init_map);
    const std::vector<double> importance =
        QdqChannelImportanceL2(w_nk, n, chain.producer.K);
    const std::vector<int64_t> keep =
        StableTopKIndicesAscending(importance, keep_count);

    SliceMatMulBnb4ProducerRows(chain.producer, init_map, keep);
    SliceConsumerWeight(init_map.at(chain.consumer.w_name),
                        chain.consumer.weight_transposed, keep, false);

    touched.producer.insert(p_key);
    touched.consumer.insert(c_key);
    touched.stale_value_info.insert(chain.producer.node->output(0));
    for (onnx::NodeProto* op : chain.chain_ops) {
      touched.stale_value_info.insert(op->output(0));
    }
  }
}

// --- QMoE (com.microsoft, quantized-weight Mixture-of-Experts) expert-
// channel structured pruning -------------------------------------------
//
// Port of pruning.py's own "QMoE (quantized-weight MoE) pruning" section --
// see that section's own top comment there for the full schema/packing
// derivation (re-derived from QMoE's own live onnxruntime schema and real
// CPU `onnxruntime.InferenceSession` runs, not assumed from its name or
// from a superficially similar op). Only the EXPERT-CHANNEL half
// (`apply_qmoe_expert_channel_pruning`, narrowing every expert's own
// `inter_size` identically across all experts) is ported here -- the
// complementary WHOLE-EXPERT half (`apply_qmoe_whole_expert_pruning`)
// needs runtime calibration data (an ONNX Runtime inference session
// observing router activations), which this C++ port has no ONNX Runtime
// linked into at all (see CLAUDE.md: the wheel build never builds ORT) --
// so it stays out of scope here, exactly like every other calibration-
// driven pass this file's own established precedent already excludes
// (Wanda-style importance, whole-expert usage-based pruning, ...).
//
// *** THE CORE INVARIANT: SLICE CODES DIRECTLY, NEVER DEQUANT-REQUANT ***.
// Exactly like this file's own MatMulNBits/MatMulBnb4/QDQ sections above:
// `fc1_experts_weights`/`fc2_experts_weights` are never dequantized and
// re-quantized from a sliced float weight. QMoEDequantizeInt/
// QMoEDequantizeNvfp4 exist ONLY to rank channels by importance -- the
// actual channel removal (QMoESlice*Axis1/QMoESlice*Axis2, and the packed-
// axis unpack/select/repack helpers) always row/column-selects (and, for a
// sub-byte-packed axis, unpacks codes to bytes, re-selects, and re-packs)
// the EXISTING quantized codes. A pruned model's surviving channels must
// therefore produce BIT-IDENTICAL results to the unpruned model's own real
// quantization, not merely numerically close ones -- re-quantizing a
// sliced float weight from scratch would recompute a different scale
// wherever the dropped values happened to hold that row's own max (see
// pruning.py's own test suite, and tests/test_qmoe_pruning_cpp.py's own
// "hand-built presliced reference" tests here, which check exactly this:
// the pruned graph's own tensors must equal a HAND-SLICE of the ORIGINAL
// quantized codes, never a re-quantization of the sliced float weight).
//
// *** THE BLOCK-ALIGNMENT CONSTRAINT ***. With `block_size` set (groupwise
// quantization), `fc2_experts_weights`' own quantization blocks group
// along `inter_size` -- this pass's own pruned axis -- so every value in
// one `block_size`-sized group shares one `(scale, zero_point)` pair an
// individual channel cannot be dropped out of without re-quantizing the
// whole group (out of scope: this module never invents a new quantized
// value). So the keep-set is resolved to WHOLE `block_size`-sized groups
// from the very start (QMoEBlockAlignedKeep ranks and keeps *blocks*, not
// individual channels, by the same combined-L2-norm criterion aggregated
// one level up) -- never a channel-level keep-set checked for alignment
// after the fact, so the result is always block-aligned by construction,
// never partial. `fc1_experts_weights`' own blocks, by contrast, group
// along `hidden_size` (an axis this pass never touches), so `fc1`'s own
// slicing needs no such alignment check either way -- mirrors
// MatMulNBitsBlockAlignedKeepBlocks's own precedent for `MatMulNBits`'
// analogous K-axis case, just resolved eagerly here rather than validated
// against an already-chosen keep-set.
//
// Covers `quant_type='int'` (both with no `block_size` -- whole-row
// per-channel scale -- and with a groupwise `block_size`) and
// `quant_type='nvfp4'` (E2M1-packed weights, `float8e4m3fn` per-block
// scales, a required per-expert `float32` global scale, `block_size` fixed
// at 16 -- schema-derived only, since this environment's onnxruntime has
// no CPU kernel for any FP4/FP8 `quant_type` to verify against at all; see
// pruning.py's own section comment, point 9, for the full reasoning).
// `'fp4'`, `'fp8'`, and `'wfp4afp8'` remain explicitly out of scope, same
// as pruning.py's own Python reference.
//
// Two deliberate, narrower-than-pruning.py scope decisions specific to
// this C++ port (both consistent with this file's own established
// narrower-subset-of-pruning.py precedent elsewhere, e.g. MatMulNBits'
// own FLOAT-only scales/zero_points/bias above):
//   * `fc1_experts_scales`/`fc2_experts_scales` (quant_type='int') and
//     `fc1_experts_bias`/`fc2_experts_bias` (either quant_type) are
//     admitted only as plain FLOAT (float32) tensors -- this file's own
//     float-tensor helpers have no FLOAT16/BFLOAT16 support anywhere yet,
//     unlike pruning.py's own `_is_supported_float_dtype`. A FLOAT16/
//     BFLOAT16-quantized export is simply never matched here (declined,
//     not mis-handled).
//   * `expert_weight_bits` (`quant_type='int'`) is supported for every
//     value pruning.py's own reference supports (2, 4, 8) -- QMoE's own
//     sub-byte packing generalizes to `bits` in {2, 4, 8} unlike
//     MatMulNBits' own {4, 8}-only packing, so this section writes its own
//     generic QMoEUnpackSubbyte/QMoEPackSubbyte rather than reusing
//     MatMulNBits' UnpackNibblesLastAxis/UnpackCodesLastAxis (bits=4/8
//     only) -- see those functions' own comments.
//
// The E2M1 (`quant_type='nvfp4'`) magnitude table below (QMoEE2M1Table) is
// its OWN table, independently re-derived from pruning.py's own
// `_e2m1_decode`/`_E2M1_MAGNITUDE` -- NOT the same table as this file's own
// `MatMulBnb4Fp4Codes()` above (bitsandbytes' own FP4 format, a completely
// different, non-IEEE-754-shaped 16-value codebook -- {0, 0.0052, 0.667,
// 1.0, 0.333, 0.5, 0.167, 0.25, ...} -- verified by comparing the two
// tables' own raw values, not assumed compatible just because both are
// "4-bit float weights"). QMoE's own E2M1 format is the standard, publicly
// documented OCP Microscaling (MX) spec 4-bit float element: 1 sign + 2
// exponent + 1 mantissa bit, magnitude {0, 0.5, 1, 1.5, 2, 3, 4, 6} indexed
// by the code's own low 3 bits, sign the top bit -- a fresh table, since
// bnb4's own table is for an unrelated, differently-shaped format.
//
// FLOAT8E4M3FN (nvfp4's own block-scale dtype) has no existing support
// anywhere else in this file (no FP8-quantized pruning pass is ported to
// C++ yet) -- QMoEFloat8E4M3Table (256 entries, code -> double, generated
// once from the OCP FP8 E4M3FN spec: sign/4-bit-exponent(bias 7)/
// 3-bit-mantissa, no infinities, 0x7F/0xFF reserved for NaN) is this
// section's own, used ONLY for IMPORTANCE RANKING (QMoEDequantizeNvfp4) --
// the actual `fc1_scales`/`fc2_scales` bytes are always sliced as opaque
// raw bytes (QMoESliceFloat8Axis1/QMoESliceFloat8Axis2), never decoded and
// re-encoded, so a single-bit table error here could only ever bias which
// channels get RANKED higher, never corrupt what gets WRITTEN back.

enum class QMoEQuantType { kInt, kNvfp4 };

const std::unordered_set<std::string>& QMoEActivations() {
  static const std::unordered_set<std::string> kSet = {"relu", "identity",
                                                       "silu", "gelu"};
  return kSet;
}

std::string QMoEStringAttrOr(const onnx::NodeProto& node,
                             const std::string& name,
                             const std::string& default_value) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name) {
      return attr.s();
    }
  }
  return default_value;
}

bool QMoEDimsEqual(const onnx::TensorProto& t,
                   const std::vector<int64_t>& expected) {
  if (static_cast<size_t>(t.dims_size()) != expected.size()) {
    return false;
  }
  for (int i = 0; i < t.dims_size(); ++i) {
    if (t.dims(i) != expected[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

// `2 ** (bits - 1)` -- the schema's own documented default `zero_point`
// for whichever of `fc1_zero_points`/`fc2_zero_points` is absent.
int64_t QMoEDefaultZeroPoint(int64_t bits) { return int64_t{1} << (bits - 1); }

// Generic sub-byte pack/unpack, LOW INDEX IN LOW BITS -- QMoE's own raw
// (`weights_prepacked` in {-1, 0}) storage convention for
// `fc1_experts_weights`/`fc2_experts_weights` and `fc1_zero_points`/
// `fc2_zero_points` alike (empirically confirmed in pruning.py's own
// section comment, point 2), for `bits` in {2, 4, 8} -- unlike this file's
// own MatMulNBits UnpackNibblesLastAxis/UnpackCodesLastAxis (bits in
// {4, 8} only, since MatMulNBits itself never admits 2), QMoE's own
// `expert_weight_bits` also admits 2, so this is written fresh rather than
// reused. `outer` independent packed rows sharing one packing granularity
// (an expert's own N channels, or an (expert, channel) pair's own K
// values, depending on caller); mirrors pruning.py's own `_unpack_subbyte`
// exactly, including its own "drop the unused top slot" trim to
// `logical_len`.
std::vector<uint8_t> QMoEUnpackSubbyte(const std::vector<uint8_t>& packed,
                                       int64_t outer, int64_t packed_width,
                                       int64_t logical_len, int64_t bits) {
  const int64_t pack = 8 / bits;
  const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
  std::vector<uint8_t> out(static_cast<size_t>(outer * logical_len));
  for (int64_t r = 0; r < outer; ++r) {
    const uint8_t* prow = packed.data() + r * packed_width;
    uint8_t* orow = out.data() + r * logical_len;
    for (int64_t j = 0; j < logical_len; ++j) {
      const int64_t byte_idx = j / pack;
      const int64_t slot = j % pack;
      orow[j] = static_cast<uint8_t>((prow[byte_idx] >> (bits * slot)) & mask);
    }
  }
  return out;
}

// Inverse of QMoEUnpackSubbyte -- pads `count` up to a whole number of
// `8 / bits` values with zeros (the same "one wasted slot" case
// QMoEUnpackSubbyte trims away), mirrors pruning.py's own `_pack_subbyte`.
std::vector<uint8_t> QMoEPackSubbyte(const std::vector<uint8_t>& vals,
                                     int64_t outer, int64_t count,
                                     int64_t bits) {
  const int64_t pack = 8 / bits;
  const int64_t packed_width = (count + pack - 1) / pack;
  const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
  std::vector<uint8_t> out(static_cast<size_t>(outer * packed_width), 0);
  for (int64_t r = 0; r < outer; ++r) {
    const uint8_t* vrow = vals.data() + r * count;
    uint8_t* orow = out.data() + r * packed_width;
    for (int64_t j = 0; j < count; ++j) {
      const int64_t byte_idx = j / pack;
      const int64_t slot = j % pack;
      orow[byte_idx] = static_cast<uint8_t>(
          orow[byte_idx] | ((vrow[j] & mask) << (bits * slot)));
    }
  }
  return out;
}

// E2M1 (1 sign + 2 exponent + 1 mantissa bit) magnitude table, code's own
// low 3 bits -> magnitude, top bit -> sign -- the OCP Microscaling (MX)
// spec's own FP4 element format `quant_type` in {'fp4', 'nvfp4'} uses. See
// this section's own top comment for why this is a FRESH table, not
// `MatMulBnb4Fp4Codes()` (a completely different, bitsandbytes-specific
// 16-value codebook). Mirrors pruning.py's own `_E2M1_MAGNITUDE`/
// `_e2m1_decode` exactly.
const std::array<double, 16>& QMoEE2M1Table() {
  static const std::array<double, 16> kTable = {
      0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
  };
  return kTable;
}

// OCP FP8 E4M3FN (`float8e4m3fn`) code -> double, all 256 codes -- nvfp4's
// own `fc1_scales`/`fc2_scales` block-scale dtype. Generated once (offline,
// via `ml_dtypes.float8_e4m3fn`, the same reference library
// tests/test_qmoe_pruning_cpp.py itself uses to build test tensors) rather
// than computed from the sign/exponent/mantissa formula inline, so a
// transcription error here is directly checkable by inspection against
// that same reference rather than trusted to a from-scratch bit-twiddling
// implementation. NaN entries (codes 0x7F/0xFF -- E4M3FN has no infinity,
// unlike standard IEEE-754) use quiet_NaN(); this table is used ONLY for
// IMPORTANCE RANKING (QMoEDequantizeNvfp4), never written back, so a NaN
// input byte (never produced by a real quantizer) can only ever bias
// ranking, not corrupt output.
const std::array<double, 256>& QMoEFloat8E4M3Table() {
  static const double kNaN = std::numeric_limits<double>::quiet_NaN();
  static const std::array<double, 256> kTable = {
      0.0,          0.001953125,  0.00390625,   0.005859375,  0.0078125,
      0.009765625,  0.01171875,   0.013671875,  0.015625,     0.017578125,
      0.01953125,   0.021484375,  0.0234375,    0.025390625,  0.02734375,
      0.029296875,  0.03125,      0.03515625,   0.0390625,    0.04296875,
      0.046875,     0.05078125,   0.0546875,    0.05859375,   0.0625,
      0.0703125,    0.078125,     0.0859375,    0.09375,      0.1015625,
      0.109375,     0.1171875,    0.125,        0.140625,     0.15625,
      0.171875,     0.1875,       0.203125,     0.21875,      0.234375,
      0.25,         0.28125,      0.3125,       0.34375,      0.375,
      0.40625,      0.4375,       0.46875,      0.5,          0.5625,
      0.625,        0.6875,       0.75,         0.8125,       0.875,
      0.9375,       1.0,          1.125,        1.25,         1.375,
      1.5,          1.625,        1.75,         1.875,        2.0,
      2.25,         2.5,          2.75,         3.0,          3.25,
      3.5,          3.75,         4.0,          4.5,          5.0,
      5.5,          6.0,          6.5,          7.0,          7.5,
      8.0,          9.0,          10.0,         11.0,         12.0,
      13.0,         14.0,         15.0,         16.0,         18.0,
      20.0,         22.0,         24.0,         26.0,         28.0,
      30.0,         32.0,         36.0,         40.0,         44.0,
      48.0,         52.0,         56.0,         60.0,         64.0,
      72.0,         80.0,         88.0,         96.0,         104.0,
      112.0,        120.0,        128.0,        144.0,        160.0,
      176.0,        192.0,        208.0,        224.0,        240.0,
      256.0,        288.0,        320.0,        352.0,        384.0,
      416.0,        448.0,        kNaN,         -0.0,         -0.001953125,
      -0.00390625,  -0.005859375, -0.0078125,   -0.009765625, -0.01171875,
      -0.013671875, -0.015625,    -0.017578125, -0.01953125,  -0.021484375,
      -0.0234375,   -0.025390625, -0.02734375,  -0.029296875, -0.03125,
      -0.03515625,  -0.0390625,   -0.04296875,  -0.046875,    -0.05078125,
      -0.0546875,   -0.05859375,  -0.0625,      -0.0703125,   -0.078125,
      -0.0859375,   -0.09375,     -0.1015625,   -0.109375,    -0.1171875,
      -0.125,       -0.140625,    -0.15625,     -0.171875,    -0.1875,
      -0.203125,    -0.21875,     -0.234375,    -0.25,        -0.28125,
      -0.3125,      -0.34375,     -0.375,       -0.40625,     -0.4375,
      -0.46875,     -0.5,         -0.5625,      -0.625,       -0.6875,
      -0.75,        -0.8125,      -0.875,       -0.9375,      -1.0,
      -1.125,       -1.25,        -1.375,       -1.5,         -1.625,
      -1.75,        -1.875,       -2.0,         -2.25,        -2.5,
      -2.75,        -3.0,         -3.25,        -3.5,         -3.75,
      -4.0,         -4.5,         -5.0,         -5.5,         -6.0,
      -6.5,         -7.0,         -7.5,         -8.0,         -9.0,
      -10.0,        -11.0,        -12.0,        -13.0,        -14.0,
      -15.0,        -16.0,        -18.0,        -20.0,        -22.0,
      -24.0,        -26.0,        -28.0,        -30.0,        -32.0,
      -36.0,        -40.0,        -44.0,        -48.0,        -52.0,
      -56.0,        -60.0,        -64.0,        -72.0,        -80.0,
      -88.0,        -96.0,        -104.0,       -112.0,       -120.0,
      -128.0,       -144.0,       -160.0,       -176.0,       -192.0,
      -208.0,       -224.0,       -240.0,       -256.0,       -288.0,
      -320.0,       -352.0,       -384.0,       -416.0,       -448.0,
      kNaN,
  };
  return kTable;
}

// Overwrites `t` in place with a FLOAT8E4M3FN tensor of `dims`/`data` (raw
// bytes, one per element -- no endianness concerns, mirrors
// SetUint8TensorData exactly but for this dtype).
void SetFloat8E4M3TensorData(onnx::TensorProto* t,
                             const std::vector<int64_t>& dims,
                             const std::vector<uint8_t>& data) {
  const std::string name = t->name();
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT8E4M3FN);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  t->set_raw_data(
      std::string(reinterpret_cast<const char*>(data.data()), data.size()));
}

struct QMoEChannelChain {
  onnx::NodeProto* node = nullptr;
  std::string fc1_w;
  std::string fc1_scale;
  std::optional<std::string> fc1_bias;
  std::optional<std::string> fc1_zp;
  std::string fc2_w;
  std::string fc2_scale;
  std::optional<std::string> fc2_bias;
  std::optional<std::string> fc2_zp;
  int64_t num_experts = 0;
  int64_t inter_size = 0;
  int64_t hidden_size = 0;
  int64_t bits = 4;
  int64_t block_size = 0;
  QMoEQuantType quant_type = QMoEQuantType::kInt;
  std::optional<std::string> fc1_global_scale;
  std::optional<std::string> fc2_global_scale;
};

// {true, name} for a present-and-well-formed optional FLOAT input, {true,
// nullopt} for a genuinely absent one, {false, _} to decline the whole
// chain -- mirrors pruning.py's own `_optional_float` closure inside
// `_match_qmoe_producer` exactly (used for `fc1_experts_bias`/
// `fc2_experts_bias`, either quant_type).
bool QMoEOptionalFloatInput(const onnx::NodeProto& node, int idx,
                            const std::vector<int64_t>& expected_dims,
                            const InitMap& init_map,
                            const ConsumerMap& consumers_of,
                            std::optional<std::string>* out_name) {
  if (idx >= node.input_size() || node.input(idx).empty()) {
    *out_name = std::nullopt;
    return true;
  }
  const std::string& name = node.input(idx);
  auto it = init_map.find(name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::FLOAT ||
      !QMoEDimsEqual(*it->second, expected_dims) ||
      ConsumerCount(consumers_of, name) != 1) {
    return false;
  }
  *out_name = name;
  return true;
}

// The UINT8 analogue of QMoEOptionalFloatInput -- mirrors pruning.py's own
// `_optional_uint8` closure (used for `fc1_zero_points`/`fc2_zero_points`,
// `quant_type='int'` only).
bool QMoEOptionalUint8Input(const onnx::NodeProto& node, int idx,
                            const std::vector<int64_t>& expected_dims,
                            const InitMap& init_map,
                            const ConsumerMap& consumers_of,
                            std::optional<std::string>* out_name) {
  if (idx >= node.input_size() || node.input(idx).empty()) {
    *out_name = std::nullopt;
    return true;
  }
  const std::string& name = node.input(idx);
  auto it = init_map.find(name);
  if (it == init_map.end() ||
      it->second->data_type() != onnx::TensorProto::UINT8 ||
      !QMoEDimsEqual(*it->second, expected_dims) ||
      ConsumerCount(consumers_of, name) != 1) {
    return false;
  }
  *out_name = name;
  return true;
}

// If `node` is a `com.microsoft::QMoE` node this pass can safely prune the
// `inter_size` axis of, returns the matched QMoEChannelChain -- mirrors
// pruning.py's own `_match_qmoe_producer` exactly (see this section's own
// top comment, and pruning.py's own section comment points 1-5/9, for the
// full derivation of every check below).
std::optional<QMoEChannelChain> MatchQMoEProducer(
    onnx::NodeProto* node, const InitMap& init_map,
    const ConsumerMap& consumers_of) {
  if (node->domain() != kComMicrosoftDomain || node->op_type() != "QMoE") {
    return std::nullopt;
  }

  const std::string activation =
      QMoEStringAttrOr(*node, "activation_type", "relu");
  const int64_t swiglu_fusion = MatMulNBitsIntAttrOr(*node, "swiglu_fusion", 0);
  const std::string quant_type_s = QMoEStringAttrOr(*node, "quant_type", "int");
  int64_t bits = MatMulNBitsIntAttrOr(*node, "expert_weight_bits", 4);
  int64_t block_size = MatMulNBitsIntAttrOr(*node, "block_size", 0);
  const int64_t weights_prepacked =
      MatMulNBitsIntAttrOr(*node, "weights_prepacked", -1);

  if (!QMoEActivations().count(activation) || swiglu_fusion != 0) {
    return std::nullopt;
  }
  if (quant_type_s != "int" && quant_type_s != "nvfp4") {
    return std::nullopt;  // 'fp4'/'fp8'/'wfp4afp8' -- out of scope, see
    // this section's own top comment.
  }
  if (weights_prepacked != -1 && weights_prepacked != 0) {
    return std::nullopt;  // A CUTLASS mixed-GEMM prepacked byte layout, not
    // the raw [N, K/pack] storage this pass slices.
  }

  auto has_input = [&](int idx) {
    return idx < node->input_size() && !node->input(idx).empty();
  };
  if (has_input(8)) {
    return std::nullopt;  // fc3_experts_weights present -- no CPU oracle.
  }
  if (has_input(14)) {
    return std::nullopt;  // router_weights present -- DeepSeek-style
    // combine-weight topology, out of scope.
  }

  // Matches `fc1_experts_weights`/`fc2_experts_weights` (inputs 2/5): both
  // constant UINT8 rank-3 initializers, single consumer, agreeing on
  // `num_experts` and on `hidden_size`/`inter_size` once `pack` codes/byte
  // is accounted for -- shared by both quant_type paths below, since the
  // raw weight shape/packing convention is identical either way (`bits=4`,
  // `pack=2` for nvfp4; `pack = 8 / expert_weight_bits` for 'int').
  auto match_weight_pair =
      [&](int64_t pack) -> std::optional<std::array<int64_t, 3>> {
    if (node->input_size() < 7 || !has_input(2) || !has_input(5)) {
      return std::nullopt;
    }
    const std::string& fc1_name = node->input(2);
    const std::string& fc2_name = node->input(5);
    auto it1 = init_map.find(fc1_name);
    auto it2 = init_map.find(fc2_name);
    if (it1 == init_map.end() || it2 == init_map.end() ||
        it1->second->data_type() != onnx::TensorProto::UINT8 ||
        it2->second->data_type() != onnx::TensorProto::UINT8 ||
        it1->second->dims_size() != 3 || it2->second->dims_size() != 3 ||
        ConsumerCount(consumers_of, fc1_name) != 1 ||
        ConsumerCount(consumers_of, fc2_name) != 1) {
      return std::nullopt;
    }
    const int64_t num_experts = it1->second->dims(0);
    const int64_t inter_size = it1->second->dims(1);
    const int64_t fc1_k_packed = it1->second->dims(2);
    const int64_t fc2_num_experts = it2->second->dims(0);
    const int64_t hidden_size = it2->second->dims(1);
    const int64_t fc2_k_packed = it2->second->dims(2);
    if (fc2_num_experts != num_experts || fc1_k_packed * pack != hidden_size ||
        fc2_k_packed * pack != inter_size) {
      return std::nullopt;  // Also rules out a fused-swiglu fc1 (doubled
      // row count).
    }
    return std::array<int64_t, 3>{num_experts, inter_size, hidden_size};
  };

  if (quant_type_s == "int") {
    for (int idx = 15; idx <= 20; ++idx) {
      if (has_input(idx)) {
        return std::nullopt;  // FP4/FP8/global-scale/activation-scale
        // inputs -- naturally absent for quant_type='int', checked
        // explicitly rather than assumed.
      }
    }
    if (bits != 2 && bits != 4 && bits != 8) {
      return std::nullopt;
    }
    const int64_t pack = 8 / bits;
    if (block_size != 0) {
      if (block_size < 16) {
        return std::nullopt;  // The CPU kernel's own floor.
      }
      if (block_size % pack != 0) {
        return std::nullopt;  // Not byte-aligned to the sub-byte packing
        // width -- every real block_size this op's kernel accepts already
        // is, so this excludes no genuine configuration.
      }
    }

    auto weight_shape = match_weight_pair(pack);
    if (!weight_shape) {
      return std::nullopt;
    }
    const int64_t num_experts = (*weight_shape)[0];
    const int64_t inter_size = (*weight_shape)[1];
    const int64_t hidden_size = (*weight_shape)[2];
    if (block_size != 0 &&
        (hidden_size % block_size != 0 || inter_size % block_size != 0)) {
      return std::nullopt;  // Partial/padded final block -- declined; also
      // transitively enforces "hidden_size/inter_size divisible by
      // block_size * pack_size" via the zero_points shape checks below.
    }

    if (!has_input(3) || !has_input(6)) {
      return std::nullopt;  // fc1_scales/fc2_scales required by this pass,
      // even though the schema itself marks them optional.
    }
    const std::string& fc1_scale_name = node->input(3);
    const std::string& fc2_scale_name = node->input(6);

    int64_t hidden_blocks = 0, inter_blocks = 0;
    std::vector<int64_t> fc1_scale_dims, fc2_scale_dims;
    if (block_size != 0) {
      hidden_blocks = hidden_size / block_size;
      inter_blocks = inter_size / block_size;
      fc1_scale_dims = {num_experts, inter_size, hidden_blocks};
      fc2_scale_dims = {num_experts, hidden_size, inter_blocks};
    } else {
      fc1_scale_dims = {num_experts, inter_size};
      fc2_scale_dims = {num_experts, hidden_size};
    }
    auto s1 = init_map.find(fc1_scale_name);
    auto s2 = init_map.find(fc2_scale_name);
    if (s1 == init_map.end() || s2 == init_map.end() ||
        s1->second->data_type() != onnx::TensorProto::FLOAT ||
        s2->second->data_type() != onnx::TensorProto::FLOAT ||
        !QMoEDimsEqual(*s1->second, fc1_scale_dims) ||
        !QMoEDimsEqual(*s2->second, fc2_scale_dims) ||
        ConsumerCount(consumers_of, fc1_scale_name) != 1 ||
        ConsumerCount(consumers_of, fc2_scale_name) != 1) {
      return std::nullopt;
    }

    std::optional<std::string> fc1_bias, fc2_bias, fc1_zp, fc2_zp;
    if (!QMoEOptionalFloatInput(*node, 4, {num_experts, inter_size}, init_map,
                                consumers_of, &fc1_bias)) {
      return std::nullopt;
    }
    if (!QMoEOptionalFloatInput(*node, 7, {num_experts, hidden_size}, init_map,
                                consumers_of, &fc2_bias)) {
      return std::nullopt;
    }
    // `zero_points`' own packed axis moves from N (whole-row case) to the
    // trailing K-block axis (blockwise case) -- see pruning.py's own
    // section comment, point 3.
    std::vector<int64_t> fc1_zp_dims, fc2_zp_dims;
    if (block_size != 0) {
      fc1_zp_dims = {num_experts, inter_size, hidden_blocks / pack};
      fc2_zp_dims = {num_experts, hidden_size, inter_blocks / pack};
    } else {
      fc1_zp_dims = {num_experts, inter_size / pack};
      fc2_zp_dims = {num_experts, hidden_size / pack};
    }
    if (!QMoEOptionalUint8Input(*node, 11, fc1_zp_dims, init_map, consumers_of,
                                &fc1_zp)) {
      return std::nullopt;
    }
    if (!QMoEOptionalUint8Input(*node, 12, fc2_zp_dims, init_map, consumers_of,
                                &fc2_zp)) {
      return std::nullopt;
    }

    QMoEChannelChain chain;
    chain.node = node;
    chain.fc1_w = node->input(2);
    chain.fc1_scale = fc1_scale_name;
    chain.fc1_bias = fc1_bias;
    chain.fc1_zp = fc1_zp;
    chain.fc2_w = node->input(5);
    chain.fc2_scale = fc2_scale_name;
    chain.fc2_bias = fc2_bias;
    chain.fc2_zp = fc2_zp;
    chain.num_experts = num_experts;
    chain.inter_size = inter_size;
    chain.hidden_size = hidden_size;
    chain.bits = bits;
    chain.block_size = block_size;
    chain.quant_type = QMoEQuantType::kInt;
    return chain;
  }

  // quant_type == "nvfp4" -- see pruning.py's own section comment, point 9,
  // for the full schema-derived (never real-kernel-confirmed -- there is no
  // CPU kernel for this quant_type in this environment) reasoning below.
  if (bits != 4) {
    return std::nullopt;  // E2M1 is inherently 4-bit; must agree (or
    // default) for pack=2 to match "2 values per byte".
  }
  if (block_size != 0 && block_size != 16) {
    return std::nullopt;  // nvfp4 is normalized to block_size=16 regardless
    // of the attribute; declined defensively for any other explicit value.
  }
  block_size = 16;  // The effective value the schema says the kernel
  // always uses for nvfp4, even when the attribute itself is 0/absent.

  for (int idx : {11, 12, 13}) {  // fc1/fc2/fc3_zero_points -- no zero-point
    // concept for a signed E2M1 code.
    if (has_input(idx)) {
      return std::nullopt;
    }
  }
  for (int idx : {17, 18, 19, 20}) {  // FP8-activation-only inputs -- nvfp4
    // keeps float/fp16/bf16 activations.
    if (has_input(idx)) {
      return std::nullopt;
    }
  }

  auto weight_shape = match_weight_pair(2);
  if (!weight_shape) {
    return std::nullopt;
  }
  const int64_t num_experts = (*weight_shape)[0];
  const int64_t inter_size = (*weight_shape)[1];
  const int64_t hidden_size = (*weight_shape)[2];
  if (hidden_size % block_size != 0 || inter_size % block_size != 0) {
    return std::nullopt;  // Partial/padded final block -- declined,
    // mirroring the int blockwise case.
  }

  if (!has_input(3) || !has_input(6)) {
    return std::nullopt;
  }
  const std::string& fc1_scale_name = node->input(3);
  const std::string& fc2_scale_name = node->input(6);
  const int64_t hidden_blocks = hidden_size / block_size;
  const int64_t inter_blocks = inter_size / block_size;
  const std::vector<int64_t> fc1_scale_dims = {num_experts, inter_size,
                                               hidden_blocks};
  const std::vector<int64_t> fc2_scale_dims = {num_experts, hidden_size,
                                               inter_blocks};
  auto s1 = init_map.find(fc1_scale_name);
  auto s2 = init_map.find(fc2_scale_name);
  if (s1 == init_map.end() || s2 == init_map.end() ||
      s1->second->data_type() != onnx::TensorProto::FLOAT8E4M3FN ||
      s2->second->data_type() != onnx::TensorProto::FLOAT8E4M3FN ||
      !QMoEDimsEqual(*s1->second, fc1_scale_dims) ||
      !QMoEDimsEqual(*s2->second, fc2_scale_dims) ||
      ConsumerCount(consumers_of, fc1_scale_name) != 1 ||
      ConsumerCount(consumers_of, fc2_scale_name) != 1) {
    return std::nullopt;
  }

  if (node->input_size() <= 16 || !has_input(15) || !has_input(16)) {
    return std::nullopt;  // fc1/fc2_global_scale -- required present for
    // nvfp4 per the live schema's own input doc.
  }
  const std::string& fc1_gs_name = node->input(15);
  const std::string& fc2_gs_name = node->input(16);
  auto g1 = init_map.find(fc1_gs_name);
  auto g2 = init_map.find(fc2_gs_name);
  if (g1 == init_map.end() || g2 == init_map.end() ||
      g1->second->data_type() != onnx::TensorProto::FLOAT ||
      g2->second->data_type() != onnx::TensorProto::FLOAT ||
      !QMoEDimsEqual(*g1->second, {num_experts}) ||
      !QMoEDimsEqual(*g2->second, {num_experts}) ||
      ConsumerCount(consumers_of, fc1_gs_name) != 1 ||
      ConsumerCount(consumers_of, fc2_gs_name) != 1) {
    return std::nullopt;
  }

  std::optional<std::string> fc1_bias, fc2_bias;
  if (!QMoEOptionalFloatInput(*node, 4, {num_experts, inter_size}, init_map,
                              consumers_of, &fc1_bias)) {
    return std::nullopt;
  }
  if (!QMoEOptionalFloatInput(*node, 7, {num_experts, hidden_size}, init_map,
                              consumers_of, &fc2_bias)) {
    return std::nullopt;
  }

  QMoEChannelChain chain;
  chain.node = node;
  chain.fc1_w = node->input(2);
  chain.fc1_scale = fc1_scale_name;
  chain.fc1_bias = fc1_bias;
  chain.fc1_zp = std::nullopt;
  chain.fc2_w = node->input(5);
  chain.fc2_scale = fc2_scale_name;
  chain.fc2_bias = fc2_bias;
  chain.fc2_zp = std::nullopt;
  chain.num_experts = num_experts;
  chain.inter_size = inter_size;
  chain.hidden_size = hidden_size;
  chain.bits = 4;
  chain.block_size = block_size;
  chain.quant_type = QMoEQuantType::kNvfp4;
  chain.fc1_global_scale = fc1_gs_name;
  chain.fc2_global_scale = fc2_gs_name;
  return chain;
}

std::vector<QMoEChannelChain> FindQMoEChains(onnx::GraphProto* graph) {
  InitMap init_map;
  for (const auto& t : graph->initializer()) {
    init_map[t.name()] = &t;
  }
  ConsumerMap consumers_of = ConsumersOf(graph);
  std::vector<QMoEChannelChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    auto chain = MatchQMoEProducer(node, init_map, consumers_of);
    if (chain) {
      chains.push_back(std::move(*chain));
    }
  }
  return chains;
}

// Dequantizes a QMoE `quant_type='int'` fc1/fc2 weight tensor to a
// float64 `[E, N, K]` array, for IMPORTANCE RANKING ONLY -- never written
// back (this section's own top comment's "slice codes directly, never
// dequant-requant" invariant). Mirrors pruning.py's own `_qmoe_dequantize`
// exactly: `w` raw `[E, N, K/pack]` storage; `scale` `[E, N]`
// (block_size==0) or `[E, N, K/block_size]` (block_size set);
// `zero_points`, when present, packs along N (block_size==0) or the
// trailing K-block axis (block_size set) -- see pruning.py's own section
// comment, point 3.
std::vector<double> QMoEDequantizeInt(const onnx::TensorProto& w,
                                      const onnx::TensorProto& scale,
                                      const onnx::TensorProto* zero_points,
                                      int64_t bits, int64_t k,
                                      int64_t block_size) {
  const int64_t e = w.dims(0);
  const int64_t n = w.dims(1);
  const int64_t kp = w.dims(2);
  const std::vector<uint8_t> packed = ReadUint8Tensor(w);  // [E*N, kp]
  const std::vector<uint8_t> codes =
      QMoEUnpackSubbyte(packed, e * n, kp, k, bits);  // [E*N, K]
  const std::vector<float> scale_data = ReadFloatTensor(scale);
  const double default_zp = static_cast<double>(QMoEDefaultZeroPoint(bits));
  std::vector<double> out(static_cast<size_t>(e * n * k));

  if (block_size > 0) {
    const int64_t k_blocks = k / block_size;
    std::vector<double> zp(static_cast<size_t>(e * n * k_blocks), default_zp);
    if (zero_points != nullptr) {
      const int64_t zp_packed_width = zero_points->dims(2);
      const std::vector<uint8_t> zp_packed = ReadUint8Tensor(*zero_points);
      const std::vector<uint8_t> zp_codes =
          QMoEUnpackSubbyte(zp_packed, e * n, zp_packed_width, k_blocks, bits);
      for (size_t i = 0; i < zp.size(); ++i) {
        zp[i] = static_cast<double>(zp_codes[i]);
      }
    }
    for (int64_t en = 0; en < e * n; ++en) {
      for (int64_t kb = 0; kb < k_blocks; ++kb) {
        const double s = static_cast<double>(
            scale_data[static_cast<size_t>(en * k_blocks + kb)]);
        const double z = zp[static_cast<size_t>(en * k_blocks + kb)];
        for (int64_t j = 0; j < block_size; ++j) {
          const int64_t idx = en * k + kb * block_size + j;
          out[static_cast<size_t>(idx)] =
              (static_cast<double>(codes[static_cast<size_t>(idx)]) - z) * s;
        }
      }
    }
  } else {
    std::vector<double> zp(static_cast<size_t>(e * n), default_zp);
    if (zero_points != nullptr) {
      const int64_t zp_packed_width = zero_points->dims(1);
      const std::vector<uint8_t> zp_packed = ReadUint8Tensor(*zero_points);
      const std::vector<uint8_t> zp_codes =
          QMoEUnpackSubbyte(zp_packed, e, zp_packed_width, n, bits);  // [E, N]
      for (size_t i = 0; i < zp.size(); ++i) {
        zp[i] = static_cast<double>(zp_codes[i]);
      }
    }
    for (int64_t en = 0; en < e * n; ++en) {
      const double s = static_cast<double>(scale_data[static_cast<size_t>(en)]);
      const double z = zp[static_cast<size_t>(en)];
      for (int64_t j = 0; j < k; ++j) {
        const int64_t idx = en * k + j;
        out[static_cast<size_t>(idx)] =
            (static_cast<double>(codes[static_cast<size_t>(idx)]) - z) * s;
      }
    }
  }
  return out;
}

// Dequantizes a QMoE `quant_type='nvfp4'` fc1/fc2 weight tensor to a
// float64 `[E, N, K]` array, IMPORTANCE RANKING ONLY. Mirrors pruning.py's
// own `_qmoe_dequantize_nvfp4` exactly: `dequantized[e, n, k] =
// e2m1_decode(code[e, n, k]) * block_scale[e, n, k // 16] * global_scale[e]`
// -- `block_size` fixed at 16.
std::vector<double> QMoEDequantizeNvfp4(const onnx::TensorProto& w,
                                        const onnx::TensorProto& block_scale,
                                        const onnx::TensorProto& global_scale,
                                        int64_t k) {
  constexpr int64_t kBlockSize = 16;
  const int64_t e = w.dims(0);
  const int64_t n = w.dims(1);
  const int64_t kp = w.dims(2);
  const std::vector<uint8_t> packed = ReadUint8Tensor(w);
  const std::vector<uint8_t> codes =
      QMoEUnpackSubbyte(packed, e * n, kp, k, 4);  // [E*N, K] E2M1 codes.
  const int64_t k_blocks = k / kBlockSize;
  const std::vector<uint8_t> bs_raw = ReadUint8Tensor(block_scale);
  const std::vector<float> gs = ReadFloatTensor(global_scale);
  const auto& f8_table = QMoEFloat8E4M3Table();
  const auto& e2m1_table = QMoEE2M1Table();

  std::vector<double> out(static_cast<size_t>(e * n * k));
  for (int64_t ei = 0; ei < e; ++ei) {
    const double g = static_cast<double>(gs[static_cast<size_t>(ei)]);
    for (int64_t ni = 0; ni < n; ++ni) {
      const int64_t en = ei * n + ni;
      for (int64_t kb = 0; kb < k_blocks; ++kb) {
        const double bscale =
            f8_table[bs_raw[static_cast<size_t>(en * k_blocks + kb)]];
        for (int64_t j = 0; j < kBlockSize; ++j) {
          const int64_t idx = en * k + kb * kBlockSize + j;
          const uint8_t code = codes[static_cast<size_t>(idx)];
          out[static_cast<size_t>(idx)] = e2m1_table[code] * bscale * g;
        }
      }
    }
  }
  return out;
}

// Dispatches to QMoEDequantizeInt/QMoEDequantizeNvfp4 for `which` fc1/fc2 of
// `chain` -- mirrors pruning.py's own `_qmoe_dequantize_fc`.
std::vector<double> QMoEDequantizeFc(
    const QMoEChannelChain& chain,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map,
    bool is_fc1) {
  const std::string& w_name = is_fc1 ? chain.fc1_w : chain.fc2_w;
  const std::string& scale_name = is_fc1 ? chain.fc1_scale : chain.fc2_scale;
  const int64_t k = is_fc1 ? chain.hidden_size : chain.inter_size;
  const onnx::TensorProto* w = init_map.at(w_name);
  const onnx::TensorProto* scale = init_map.at(scale_name);
  if (chain.quant_type == QMoEQuantType::kNvfp4) {
    const std::string& gs_name =
        is_fc1 ? *chain.fc1_global_scale : *chain.fc2_global_scale;
    return QMoEDequantizeNvfp4(*w, *scale, *init_map.at(gs_name), k);
  }
  const std::optional<std::string>& zp_name =
      is_fc1 ? chain.fc1_zp : chain.fc2_zp;
  const onnx::TensorProto* zp = zp_name ? init_map.at(*zp_name) : nullptr;
  return QMoEDequantizeInt(*w, *scale, zp, chain.bits, k, chain.block_size);
}

// Combined (root-sum-square) L2-norm importance per `inter_size` channel
// index -- mirrors pruning.py's own `_qmoe_channel_importance` exactly:
// `fc1`'s own dequantized row (across every expert and `hidden_size` at
// once) plus `fc2`'s own dequantized column (same reduction), plus
// `fc1_experts_bias`'s own entry when present.
std::vector<double> QMoEChannelImportance(
    const QMoEChannelChain& chain,
    const std::unordered_map<std::string, onnx::TensorProto*>& init_map) {
  const int64_t e = chain.num_experts;
  const int64_t inter = chain.inter_size;
  const int64_t hidden = chain.hidden_size;
  const std::vector<double> fc1_dq =
      QMoEDequantizeFc(chain, init_map, true);  // [E, inter, hidden]
  const std::vector<double> fc2_dq =
      QMoEDequantizeFc(chain, init_map, false);  // [E, hidden, inter]

  std::vector<double> squared(static_cast<size_t>(inter), 0.0);
  for (int64_t ei = 0; ei < e; ++ei) {
    for (int64_t ni = 0; ni < inter; ++ni) {
      double sq = 0.0;
      const double* row = fc1_dq.data() + (ei * inter + ni) * hidden;
      for (int64_t h = 0; h < hidden; ++h) {
        sq += row[h] * row[h];
      }
      squared[static_cast<size_t>(ni)] += sq;
    }
  }
  for (int64_t ei = 0; ei < e; ++ei) {
    for (int64_t h = 0; h < hidden; ++h) {
      const double* row = fc2_dq.data() + (ei * hidden + h) * inter;
      for (int64_t ni = 0; ni < inter; ++ni) {
        squared[static_cast<size_t>(ni)] += row[ni] * row[ni];
      }
    }
  }
  if (chain.fc1_bias) {
    const std::vector<float> bias =
        ReadFloatTensor(*init_map.at(*chain.fc1_bias));
    for (int64_t ei = 0; ei < e; ++ei) {
      for (int64_t ni = 0; ni < inter; ++ni) {
        const double v =
            static_cast<double>(bias[static_cast<size_t>(ei * inter + ni)]);
        squared[static_cast<size_t>(ni)] += v * v;
      }
    }
  }
  std::vector<double> importance(static_cast<size_t>(inter));
  for (int64_t ni = 0; ni < inter; ++ni) {
    importance[static_cast<size_t>(ni)] =
        std::sqrt(squared[static_cast<size_t>(ni)]);
  }
  return importance;
}

// Resolves a target `inter_size` keep-set to whole `block_size`-sized
// groups: aggregates `importance` into one combined (root-sum-square)
// score per block, ranks BLOCKS (not individual channels) by that score,
// and keeps the top
// `max(1, n / block_size - round(n / block_size * sparsity))` of them --
// mirrors pruning.py's own `_qmoe_block_aligned_keep` exactly (see this
// section's own top comment, "THE BLOCK-ALIGNMENT CONSTRAINT"). Ranking at
// block granularity from the start (rather than computing a channel-level
// keep-set and checking alignment after the fact) means the result is
// ALWAYS block-aligned by construction. Returns nullopt when every block
// would be kept (rounds down to nothing to prune for this layer). The
// returned indices are ascending and, since blocks are visited in
// ascending block-id order, each kept block contributes exactly
// `block_size` CONSECUTIVE entries -- the caller (ApplyQMoEChannelChains)
// relies on this to recover `keep_block_idx` cheaply (`keep[i] /
// block_size` for `i` stepping by `block_size`) rather than a separate
// dedup pass.
std::optional<std::vector<int64_t>> QMoEBlockAlignedKeep(
    const std::vector<double>& importance, int64_t n, int64_t block_size,
    double sparsity) {
  const int64_t num_blocks = n / block_size;
  std::vector<double> block_importance(static_cast<size_t>(num_blocks));
  for (int64_t b = 0; b < num_blocks; ++b) {
    double sq = 0.0;
    for (int64_t j = 0; j < block_size; ++j) {
      const double v = importance[static_cast<size_t>(b * block_size + j)];
      sq += v * v;
    }
    block_importance[static_cast<size_t>(b)] = std::sqrt(sq);
  }
  const int64_t keep_blocks_count = std::max<int64_t>(
      1, num_blocks - std::llround(static_cast<double>(num_blocks) * sparsity));
  if (keep_blocks_count >= num_blocks) {
    return std::nullopt;
  }
  const std::vector<int64_t> keep_block_idx =
      StableTopKIndicesAscending(block_importance, keep_blocks_count);
  std::vector<int64_t> keep;
  keep.reserve(static_cast<size_t>(keep_blocks_count * block_size));
  for (int64_t b : keep_block_idx) {
    for (int64_t j = 0; j < block_size; ++j) {
      keep.push_back(b * block_size + j);
    }
  }
  return keep;
}

// Per-expert row (axis 1) index-select of a FLOAT tensor shaped `[E, N]`
// (`force_rank3 == false`, `row_width` always 1) or `[E, N, row_width]`
// (`force_rank3 == true`) -- used for `fc1/fc2_experts_bias` (always
// rank-2) and, for `quant_type='int'`, `fc1/fc2_scales` (rank-2 whole-row,
// rank-3 blockwise). `force_rank3` is passed explicitly by the caller
// rather than inferred from `row_width == 1` -- `row_width` (`hidden_blocks`
// for `fc1_scales`) can genuinely equal 1 in the BLOCKWISE case too
// (`hidden_size == block_size`), which must still come back rank-3 (a real
// shape this op's own schema documents, re-confirmed live in pruning.py's
// own section comment) rather than collapsing to the whole-row tensor's
// own rank-2 shape.
void QMoESliceFloatAxis1(onnx::TensorProto* t, int64_t e, int64_t n,
                         int64_t row_width, const std::vector<int64_t>& keep,
                         bool force_rank3) {
  const std::vector<float> data = ReadFloatTensor(*t);  // [E, N, row_width]
  const int64_t kc = static_cast<int64_t>(keep.size());
  std::vector<float> out(static_cast<size_t>(e * kc * row_width));
  for (int64_t ei = 0; ei < e; ++ei) {
    for (int64_t i = 0; i < kc; ++i) {
      std::memcpy(
          out.data() + (ei * kc + i) * row_width,
          data.data() + (ei * n + keep[static_cast<size_t>(i)]) * row_width,
          static_cast<size_t>(row_width) * sizeof(float));
    }
  }
  const std::vector<int64_t> dims = force_rank3
                                        ? std::vector<int64_t>{e, kc, row_width}
                                        : std::vector<int64_t>{e, kc};
  SetFloatTensorData(t, dims, out);
}

// The FLOAT8E4M3FN analogue of QMoESliceFloatAxis1 -- used for
// `quant_type='nvfp4'`'s own `fc1_scales` (always rank-3, `row_width` =
// `hidden_blocks`). Raw-byte slice, no decode/re-encode (1-byte dtype, no
// endianness concern either).
void QMoESliceFloat8Axis1(onnx::TensorProto* t, int64_t e, int64_t n,
                          int64_t row_width, const std::vector<int64_t>& keep) {
  const std::vector<uint8_t> data = ReadUint8Tensor(*t);
  const int64_t kc = static_cast<int64_t>(keep.size());
  std::vector<uint8_t> out(static_cast<size_t>(e * kc * row_width));
  for (int64_t ei = 0; ei < e; ++ei) {
    for (int64_t i = 0; i < kc; ++i) {
      std::memcpy(
          out.data() + (ei * kc + i) * row_width,
          data.data() + (ei * n + keep[static_cast<size_t>(i)]) * row_width,
          static_cast<size_t>(row_width));
    }
  }
  SetFloat8E4M3TensorData(t, {e, kc, row_width}, out);
}

// The UINT8 analogue -- used for `fc1_experts_weights` (`row_width` = the
// packed `K/pack` width, an axis-1 index-select since the pruned
// `inter_size` axis is never the packed one for `fc1`) and, when
// `block_size` is set, `fc1_zero_points` (whose own packed axis has moved
// off `inter_size` entirely, so it too is a plain index-select here).
void QMoESliceUint8Axis1(onnx::TensorProto* t, int64_t e, int64_t n,
                         int64_t row_width, const std::vector<int64_t>& keep,
                         const std::vector<int64_t>& out_dims) {
  const std::vector<uint8_t> data = ReadUint8Tensor(*t);
  const int64_t kc = static_cast<int64_t>(keep.size());
  std::vector<uint8_t> out(static_cast<size_t>(e * kc * row_width));
  for (int64_t ei = 0; ei < e; ++ei) {
    for (int64_t i = 0; i < kc; ++i) {
      std::memcpy(
          out.data() + (ei * kc + i) * row_width,
          data.data() + (ei * n + keep[static_cast<size_t>(i)]) * row_width,
          static_cast<size_t>(row_width));
    }
  }
  SetUint8TensorData(t, out_dims, out);
}

// Per-expert, per-`hidden_size`-row, LAST-axis (block-index) select of a
// FLOAT tensor shaped `[E, hidden, num_blocks]` -- `quant_type='int'`'s own
// `fc2_scales` in the blockwise case, sliced by *block* index
// (`keep_block_idx`), not channel index, since `fc2`'s own blocks group
// along `inter_size` (this pass's own pruned axis) -- see this section's
// own top comment.
void QMoESliceFloatAxis2(onnx::TensorProto* t, int64_t e, int64_t hidden,
                         int64_t num_blocks,
                         const std::vector<int64_t>& keep_block_idx) {
  const std::vector<float> data =
      ReadFloatTensor(*t);  // [E, hidden, num_blocks]
  const int64_t kb = static_cast<int64_t>(keep_block_idx.size());
  std::vector<float> out(static_cast<size_t>(e * hidden * kb));
  for (int64_t r = 0; r < e * hidden; ++r) {
    for (int64_t i = 0; i < kb; ++i) {
      out[static_cast<size_t>(r * kb + i)] = data[static_cast<size_t>(
          r * num_blocks + keep_block_idx[static_cast<size_t>(i)])];
    }
  }
  SetFloatTensorData(t, {e, hidden, kb}, out);
}

// The FLOAT8E4M3FN analogue of QMoESliceFloatAxis2 -- `quant_type='nvfp4'`'s
// own `fc2_scales`.
void QMoESliceFloat8Axis2(onnx::TensorProto* t, int64_t e, int64_t hidden,
                          int64_t num_blocks,
                          const std::vector<int64_t>& keep_block_idx) {
  const std::vector<uint8_t> data = ReadUint8Tensor(*t);
  const int64_t kb = static_cast<int64_t>(keep_block_idx.size());
  std::vector<uint8_t> out(static_cast<size_t>(e * hidden * kb));
  for (int64_t r = 0; r < e * hidden; ++r) {
    for (int64_t i = 0; i < kb; ++i) {
      out[static_cast<size_t>(r * kb + i)] = data[static_cast<size_t>(
          r * num_blocks + keep_block_idx[static_cast<size_t>(i)])];
    }
  }
  SetFloat8E4M3TensorData(t, {e, hidden, kb}, out);
}

// The actual apply step -- mirrors pruning.py's own
// `_apply_qmoe_channel_chains` exactly, including its own two distinct
// slicing shapes (whole-row/`pack`-floored vs. block-aligned) and every
// per-tensor packing special case documented on each call site below. A
// standalone application pass (not sharing state with
// ApplyStructuredPruning's own combined TouchedState -- `com.microsoft::
// QMoE`'s own weight names can never collide with any plain-float/
// MatMulNBits/QDQ/MatMulBnb4 chain this file's other passes match, so a
// local `touched` set mirroring pruning.py's own is all that's needed).
void ApplyQMoEChannelChains(onnx::GraphProto* graph,
                            std::vector<QMoEChannelChain>& chains,
                            double sparsity) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }
  std::unordered_set<std::string> touched;

  for (auto& chain : chains) {
    std::vector<std::string> weight_names = {chain.fc1_w, chain.fc2_w,
                                             chain.fc1_scale};
    if (chain.fc1_bias) weight_names.push_back(*chain.fc1_bias);
    if (chain.fc1_zp) weight_names.push_back(*chain.fc1_zp);
    if (chain.block_size) {
      // Blockwise fc2_scales -- and fc2_zero_points, if present -- are now
      // also mutated below (their own trailing axis is inter_size-block-
      // indexed), unlike the whole-row case where neither depends on
      // inter_size at all.
      weight_names.push_back(chain.fc2_scale);
      if (chain.fc2_zp) weight_names.push_back(*chain.fc2_zp);
    }
    bool already_touched = false;
    for (const auto& nm : weight_names) {
      if (touched.count(nm)) {
        already_touched = true;
        break;
      }
    }
    if (already_touched) {
      continue;  // A shared/tied initializer another QMoE node already
      // resized.
    }
    for (const auto& nm : weight_names) {
      touched.insert(nm);
    }

    const int64_t n = chain.inter_size;
    const int64_t pack = 8 / chain.bits;
    const int64_t block_size = chain.block_size;
    const int64_t e = chain.num_experts;
    const int64_t hidden = chain.hidden_size;

    std::vector<int64_t> keep;
    std::vector<int64_t> keep_block_idx;
    if (block_size > 0) {
      // Resolved at block granularity from the start -- see
      // QMoEBlockAlignedKeep. A block is always a multiple of `pack` (the
      // matcher itself requires `block_size % pack == 0`), so this also
      // transitively satisfies the same "survivor count is a multiple of
      // `pack`" constraint the whole-row case floors to separately below.
      const std::vector<double> importance =
          QMoEChannelImportance(chain, init_map);
      auto keep_opt = QMoEBlockAlignedKeep(importance, n, block_size, sparsity);
      if (!keep_opt) {
        continue;  // Rounds down to nothing for this layer -- no-op.
      }
      keep = std::move(*keep_opt);
      keep_block_idx.reserve(keep.size() / static_cast<size_t>(block_size));
      for (size_t i = 0; i < keep.size();
           i += static_cast<size_t>(block_size)) {
        keep_block_idx.push_back(keep[i] / block_size);
      }
    } else {
      int64_t keep_count = std::max<int64_t>(
          1, n - std::llround(static_cast<double>(n) * sparsity));
      // The survivor count must itself stay an exact multiple of `pack`:
      // this environment's own QMoE kernel derives `inter_size` purely
      // from `fc2_experts_weights`' own packed last axis length times
      // `pack` -- it has no way to represent "the last packed byte's own
      // high nibble is unused padding". So a survivor count that isn't
      // itself a multiple of `pack` is rounded DOWN to the nearest one
      // here (floored at `pack`, never below it), mirroring pruning.py's
      // own `_apply_qmoe_channel_chains` exactly.
      keep_count = std::max<int64_t>(pack, (keep_count / pack) * pack);
      if (keep_count >= n) {
        continue;  // Rounds down to nothing for this layer -- no-op.
      }
      const std::vector<double> importance =
          QMoEChannelImportance(chain, init_map);
      keep = StableTopKIndicesAscending(importance, keep_count);
    }

    // fc1_experts_weights: [E, inter_size, hidden_size/pack] -- the pruned
    // axis (1) is the UNPACKED one (packing lives on axis 2, untouched), so
    // this is a plain per-expert row index-select, no unpack/repack needed
    // at all -- unaffected by block_size (fc1's own blocks, if any, group
    // along hidden_size, an axis this pass never touches).
    {
      onnx::TensorProto* fc1_w = init_map.at(chain.fc1_w);
      const int64_t kp = fc1_w->dims(2);
      const int64_t kc = static_cast<int64_t>(keep.size());
      QMoESliceUint8Axis1(fc1_w, e, n, kp, keep, {e, kc, kp});
    }

    // fc1_scales: [E, inter_size] (whole-row) or [E, inter_size,
    // hidden_blocks] (blockwise) -- `keep` always indexes axis 1 either
    // way. FLOAT for quant_type='int', FLOAT8E4M3FN for quant_type='nvfp4'.
    {
      onnx::TensorProto* fc1_scale = init_map.at(chain.fc1_scale);
      const int64_t row_width = block_size > 0 ? (hidden / block_size) : 1;
      if (chain.quant_type == QMoEQuantType::kNvfp4) {
        QMoESliceFloat8Axis1(fc1_scale, e, n, row_width, keep);
      } else {
        QMoESliceFloatAxis1(fc1_scale, e, n, row_width, keep, block_size > 0);
      }
    }

    if (chain.fc1_bias) {
      onnx::TensorProto* fc1_bias = init_map.at(*chain.fc1_bias);
      QMoESliceFloatAxis1(fc1_bias, e, n, 1, keep, /*force_rank3=*/false);
    }

    if (chain.fc1_zp) {
      onnx::TensorProto* fc1_zp = init_map.at(*chain.fc1_zp);
      const int64_t kc = static_cast<int64_t>(keep.size());
      if (block_size > 0) {
        // fc1_zero_points: [E, inter_size, hidden_blocks/pack] -- blockwise
        // moves the packed axis off of N entirely, so slicing N (axis 1)
        // is now a plain index-select, no unpack/repack needed at all.
        const int64_t hb = hidden / block_size;
        QMoESliceUint8Axis1(fc1_zp, e, n, hb / pack, keep, {e, kc, hb / pack});
      } else {
        // fc1_zero_points: [E, inter_size/pack] -- packed along the SAME
        // axis being pruned, so this genuinely needs unpack/select/repack.
        const int64_t zp_packed_width = fc1_zp->dims(1);
        const std::vector<uint8_t> packed = ReadUint8Tensor(*fc1_zp);
        const std::vector<uint8_t> unpacked = QMoEUnpackSubbyte(
            packed, e, zp_packed_width, n, chain.bits);  // [E, N]
        std::vector<uint8_t> selected(static_cast<size_t>(e * kc));
        for (int64_t ei = 0; ei < e; ++ei) {
          for (int64_t i = 0; i < kc; ++i) {
            selected[static_cast<size_t>(ei * kc + i)] =
                unpacked[static_cast<size_t>(ei * n +
                                             keep[static_cast<size_t>(i)])];
          }
        }
        const std::vector<uint8_t> repacked =
            QMoEPackSubbyte(selected, e, kc, chain.bits);
        const int64_t new_pw = (kc + pack - 1) / pack;
        SetUint8TensorData(fc1_zp, {e, new_pw}, repacked);
      }
    }

    // fc2_experts_weights: [E, hidden_size, inter_size/pack] -- the pruned
    // axis (inter_size) IS the packed one here, so this needs a real
    // unpack/select/repack round trip. Packing is flat/block-boundary-
    // oblivious (confirmed empirically in pruning.py's own section
    // comment, point 3), so this is unaffected by block_size beyond `keep`
    // itself already being block-aligned.
    {
      onnx::TensorProto* fc2_w = init_map.at(chain.fc2_w);
      const int64_t packed_width = fc2_w->dims(2);
      const int64_t kc = static_cast<int64_t>(keep.size());
      const std::vector<uint8_t> packed = ReadUint8Tensor(*fc2_w);
      const std::vector<uint8_t> unpacked = QMoEUnpackSubbyte(
          packed, e * hidden, packed_width, n, chain.bits);  // [E*hidden, n]
      std::vector<uint8_t> selected(static_cast<size_t>(e * hidden * kc));
      for (int64_t r = 0; r < e * hidden; ++r) {
        for (int64_t i = 0; i < kc; ++i) {
          selected[static_cast<size_t>(r * kc + i)] =
              unpacked[static_cast<size_t>(r * n +
                                           keep[static_cast<size_t>(i)])];
        }
      }
      const std::vector<uint8_t> repacked =
          QMoEPackSubbyte(selected, e * hidden, kc, chain.bits);
      const int64_t new_pw = (kc + pack - 1) / pack;
      SetUint8TensorData(fc2_w, {e, hidden, new_pw}, repacked);
    }

    if (block_size > 0) {
      // fc2_scales/fc2_zero_points: [E, hidden_size, inter_blocks(/pack)]
      // -- unlike the whole-row case, these now DO depend on inter_size,
      // via the block axis, so they must be cut too -- by BLOCK index
      // (`keep_block_idx`), not channel index (`keep`).
      onnx::TensorProto* fc2_scale = init_map.at(chain.fc2_scale);
      const int64_t inter_blocks = n / block_size;
      if (chain.quant_type == QMoEQuantType::kNvfp4) {
        QMoESliceFloat8Axis2(fc2_scale, e, hidden, inter_blocks,
                             keep_block_idx);
      } else {
        QMoESliceFloatAxis2(fc2_scale, e, hidden, inter_blocks, keep_block_idx);
      }
      if (chain.fc2_zp) {
        onnx::TensorProto* fc2_zp = init_map.at(*chain.fc2_zp);
        const int64_t zp_packed_width = fc2_zp->dims(2);
        const std::vector<uint8_t> packed = ReadUint8Tensor(*fc2_zp);
        const std::vector<uint8_t> unpacked = QMoEUnpackSubbyte(
            packed, e * hidden, zp_packed_width, inter_blocks, chain.bits);
        const int64_t kb = static_cast<int64_t>(keep_block_idx.size());
        std::vector<uint8_t> selected(static_cast<size_t>(e * hidden * kb));
        for (int64_t r = 0; r < e * hidden; ++r) {
          for (int64_t i = 0; i < kb; ++i) {
            selected[static_cast<size_t>(r * kb + i)] =
                unpacked[static_cast<size_t>(
                    r * inter_blocks + keep_block_idx[static_cast<size_t>(i)])];
          }
        }
        const std::vector<uint8_t> repacked =
            QMoEPackSubbyte(selected, e * hidden, kb, chain.bits);
        const int64_t new_pw = (kb + pack - 1) / pack;
        SetUint8TensorData(fc2_zp, {e, hidden, new_pw}, repacked);
      }
    }
    // fc2_experts_bias always indexes hidden_size, fc2's own OUTPUT axis --
    // unaffected by an inter_size cut regardless of block_size, the same
    // reasoning plain MoE's own fc2 bias already gets -- never sliced
    // here. fc1/fc2_global_scale (nvfp4 only) are per-EXPERT, not
    // per-channel -- expert-channel pruning never touches these either. In
    // the whole-row case (block_size == 0), fc2_scales/fc2_zero_points are
    // the same "indexes hidden_size only" shape and are likewise never
    // sliced.
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
    std::vector<MatMulNBitsChain> matmul_nbits_chains =
        FindMatMulNBitsChains(graph);
    std::vector<MatMulNBitsGatedChain> matmul_nbits_gated_chains =
        FindMatMulNBitsGatedChains(graph);
    std::vector<MatMulNBitsMlpChain> matmul_nbits_mlp_chains =
        FindMatMulNBitsMlpChains(graph);
    std::vector<QdqChain> qdq_chains = FindQdqChains(graph);
    std::vector<QdqGatedChain> qdq_gated_chains = FindQdqGatedChains(graph);
    std::vector<MatMulBnb4Chain> matmul_bnb4_chains =
        FindMatMulBnb4Chains(graph);

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
    if (!matmul_nbits_chains.empty() || !matmul_nbits_gated_chains.empty()) {
      // Another separate application pass -- see the "MatMulNBits
      // (com.microsoft, block-quantized weight) structured pruning" section
      // comment above. Shares `touched` with every call above for the same
      // reason ApplySplitGatedChains does: a plain-float weight this pass
      // also happens to match (as the OTHER side of a mixed chain) can never
      // be double-resized by, or double-resize, an ordinary chain that
      // already touched it.
      ApplyMatMulNBitsChains(graph, matmul_nbits_chains,
                             matmul_nbits_gated_chains, sparsity, touched);
    }
    if (!matmul_nbits_mlp_chains.empty()) {
      // The fused `MatMulNBitsMlp` variant -- see the "MatMulNBitsMlp/
      // MatMulNBitsQkv (fused block-quantized weight) structured pruning"
      // section comment above ApplyMatMulNBitsChains for why this chain
      // kind (unlike MatMulNBitsQkv) is wired in right here, sharing
      // `touched` with every chain family above for the same reason
      // ApplyMatMulNBitsChains itself does.
      ApplyMatMulNBitsMlpChains(graph, matmul_nbits_mlp_chains, sparsity,
                                touched);
    }
    if (!qdq_chains.empty() || !qdq_gated_chains.empty()) {
      // QDQ-quantized chains -- see this file's own "QDQ (quantized-weight)
      // structured pruning" section comment. Additive and (per
      // FindQdqChains/FindQdqGatedChains's own "at least one side QDQ"
      // requirement) never targets a tensor any plain-float pass above could
      // also match, but still shares `touched` with them for the same
      // "one shared conflict ledger per graph" reason ApplySplitGatedChains
      // does.
      ApplyQdqChains(graph, qdq_chains, qdq_gated_chains, sparsity, touched);
    }
    if (!matmul_bnb4_chains.empty()) {
      // The sixth and final application pass over this graph -- see this
      // file's own "MatMulBnb4 (bitsandbytes FP4/NF4 block-quantized
      // weight) structured pruning" section comment. Shares `touched` with
      // every call above for the same "one shared conflict ledger per
      // graph" reason ApplyMatMulNBitsChains/ApplyQdqChains both do: a
      // plain-float consumer weight this pass matches can never be
      // double-resized by, or double-resize, an ordinary chain/MatMulNBits/
      // QDQ chain that already touched it.
      ApplyMatMulBnb4Chains(graph, matmul_bnb4_chains, sparsity, touched);
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
    // The fused `com.microsoft::MatMulNBitsQkv` variant -- see the
    // "MatMulNBitsMlp/MatMulNBitsQkv (fused block-quantized weight)
    // structured pruning" section comment above ApplyMatMulNBitsChains
    // (onnxsim's own "MatMulNBits" section) for why this chain kind is
    // wired in HERE rather than alongside MatMulNBitsMlp in
    // ApplyStructuredPruning: pruning a whole KV group needs this
    // function's own GQA/plain-Attention head-count matching machinery
    // (MatchGqaProducer/MatchOnnxAttentionProducer/HeadColumnIndices), which
    // MatMulNBitsMlp's plain producer/consumer pruning unit never needs at
    // all. Uses its own dedicated producer/consumer-touched bookkeeping,
    // never shared with ApplyAttentionChains's own above -- always safe,
    // since a `MatMulNBitsQkv` node's own `q_b`/`k_b`/`v_b` weight names can
    // never collide with a plain/GQA chain's own separately-matched Q/K/V
    // producer weight names (structurally different node/weight shapes).
    std::vector<MatMulNBitsQkvChain> matmul_nbits_qkv_chains =
        FindMatMulNBitsQkvChains(graph);
    if (!matmul_nbits_qkv_chains.empty()) {
      ApplyMatMulNBitsQkvChains(graph, matmul_nbits_qkv_chains, sparsity);
    }
  }
  return out;
}

// QMoE expert-channel pruning: removes intermediate (`inter_size`) channels
// from every expert of a matched `com.microsoft::QMoE` node at once -- the
// quantized-weight counterpart of ApplyStructuredPruning's own plain-float
// channel pruning, targeting QMoE's own packed `uint8` `fc1`/`fc2` weights
// (plus their `scales`/`zero_points`/`global_scale`, co-sliced in lockstep)
// instead of plain floats. See this file's own "QMoE (com.microsoft,
// quantized-weight Mixture-of-Experts) expert-channel structured pruning"
// section comment above (MatchQMoEProducer/ApplyQMoEChannelChains) for the
// exact matched topology, every empirically-confirmed decline, and why
// whole-expert removal (needing runtime calibration data this C++ port has
// no ONNX Runtime linked in to gather) stays out of scope. A standalone
// entry point -- unlike MatMulNBits/MatMulBnb4/QDQ above, QMoE is not
// folded into ApplyStructuredPruning's own combined pass, since it targets
// a wholly disjoint node type (`com.microsoft::QMoE`) that can never
// collide with anything that pass already matches; mirrors pruning.py's
// own separate `apply_qmoe_expert_channel_pruning` top-level function.
onnx::ModelProto ApplyQMoEExpertChannelPruning(const onnx::ModelProto& model,
                                               double sparsity) {
  if (!(sparsity >= 0.0 && sparsity < 1.0)) {
    throw std::invalid_argument(
        "ApplyQMoEExpertChannelPruning: sparsity must be in [0, 1), got " +
        std::to_string(sparsity));
  }
  onnx::ModelProto out = model;

  // Subgraph-aware (IterSubgraphs, see this file's own "Subgraph
  // recursion" section comment above): every `QMoE` node is matched and
  // pruned inside a nested If/Loop/Scan/BeamSearch-family subgraph, at any
  // nesting depth, exactly as if that subgraph were its own top-level
  // graph -- each returned GraphProto* gets its own fresh FindQMoEChains
  // call and its own local `touched` state inside ApplyQMoEChannelChains,
  // mirroring pruning.py's own `apply_qmoe_expert_channel_pruning` loop
  // over `_iter_subgraphs(out.graph)` exactly.
  for (onnx::GraphProto* graph : IterSubgraphs(out.mutable_graph())) {
    std::vector<QMoEChannelChain> chains = FindQMoEChains(graph);
    if (!chains.empty()) {
      ApplyQMoEChannelChains(graph, chains, sparsity);
    }
  }
  return out;
}

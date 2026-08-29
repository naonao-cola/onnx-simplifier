// SPDX-License-Identifier: Apache-2.0
//
// C++ port of pruning.py's own apply_structured_pruning -- see that
// function's docstring for the full technique description (this is quoted
// here only where it constrains scope). This port covers all five of
// pruning.py's own chain finders: a MatMul/vanilla-Gemm producer ->
// consumer pair (_find_chains), a Conv producer -> consumer pair, including
// depthwise pass-through hops and general grouped Conv on either side
// (_find_conv_chains), the gated-FFN SwiGLU/GeGLU pattern -- two producers
// combined by Mul (or ONNX opset-28+'s native SwiGLU node) feeding one
// consumer, both pruned to the same channel indices (_find_gated_chains) --
// and Conv/MatMul residual (skip-connection) chains, a bounded slice of the
// general dependency-graph-grouping problem: a channel-preserving
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
// every other chain finder here already holds. For MatMul/Gemm specifically,
// a fused com.microsoft::SkipLayerNormalization/
// SkipSimplifiedLayerNormalization node -- what onnxruntime's transformer
// optimizer collapses a bare residual `Add` plus the following LayerNorm
// into, and so what a fully-optimized transformer's residual connections
// typically look like -- is also recognized as an eligible merge point
// (mirroring pruning.py's own _match_matmul_residual_merge), its own
// gamma/beta/bias constants riding along as a per-channel affine hop on the
// resolved chain. Conv residual chains only ever see a bare `Add` -- there
// is no Conv analogue of that fused op.
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
      "Relu", "LeakyRelu", "Elu",      "Selu", "Sigmoid",
      "Tanh", "Softplus",  "Softsign", "Gelu", "HardSigmoid",
      "Mish", "Identity",  "Cast",
  };
  return kOps;
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
    int max_hops) {
  std::vector<ChainOp> chain_ops;
  std::optional<ConsumerMatch> consumer;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    auto cit = consumers_of.find(cur);
    if (cit == consumers_of.end() || cit->second.size() != 1) {
      break;
    }
    onnx::NodeProto* nxt = cit->second[0];

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

struct ConvConsumerResult {
  onnx::NodeProto* node;
  std::string weight;
  int64_t group;
};

std::tuple<std::optional<ConvConsumerResult>, std::vector<ChainOp>,
           std::vector<ConvPassThrough>>
WalkToConvConsumer(const std::string& start, const InitMap& init_map,
                   const ConsumerMap& consumers_of,
                   const std::unordered_set<std::string>& graph_outputs,
                   int64_t n_channels, int max_hops) {
  std::vector<ChainOp> chain_ops;
  std::vector<ConvPassThrough> pass_through;
  std::optional<ConvConsumerResult> consumer;
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    auto cit = consumers_of.find(cur);
    if (cit == consumers_of.end() || cit->second.size() != 1) {
      break;
    }
    onnx::NodeProto* nxt = cit->second[0];

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
  return {consumer, chain_ops, pass_through};
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
    auto [consumer, chain_ops, pass_through] =
        WalkToConvConsumer(out_name, init_map, consumers_of, graph_outputs,
                           info->out_channels, kMaxChainHops);
    if (!consumer) {
      continue;
    }
    if (info->group > 1 && consumer->group > 1 &&
        info->group != consumer->group) {
      continue;  // Both sides grouped with mismatched group counts: declined.
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
    chains.push_back(std::move(chain));
  }
  return chains;
}

// --- Residual (Add-merged) chains, mirroring _is_eligible_add_merge/
// _walk_conv_producer_backward/_find_conv_residual_chains and
// _walk_matmul_producer_backward/_find_matmul_residual_chains. Scope note:
// this only recognizes a bare Add merge point, not pruning.py's own
// com.microsoft::SkipLayerNormalization/SkipSimplifiedLayerNormalization
// fusion (see _match_matmul_residual_merge's own comment in pruning.py) --
// a model whose residual connections have already been fused into that op
// by onnxruntime's transformer optimizer is left untouched by this port's
// MatMul residual finder; the pure-Python implementation remains the
// full-featured reference for that case.

enum class BackwardEdgeKind { kFail, kProducer, kAdd };

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

ConvBackwardEdge WalkConvProducerBackward(
    const std::string& start,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const InitMap& init_map, const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int max_hops) {
  std::vector<ConvPassThrough> pass_through;  // Backward order.
  std::vector<onnx::NodeProto*> unary_ops;    // Backward order.
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    if (ConsumerCount(consumers_of, cur) != 1 || graph_outputs.count(cur)) {
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
      if (prod_info->group != 1) {
        // General grouped Conv is out of scope for the residual/Add-merge
        // case -- see WalkConvProducerBackward's Python counterpart.
        return ConvBackwardEdge{};
      }
      ConvBackwardEdge edge;
      edge.kind = BackwardEdgeKind::kProducer;
      edge.producer =
          Producer{node, prod_info->weight, false, prod_info->bias, true, 1};
      edge.n_channels = prod_info->out_channels;
      std::reverse(pass_through.begin(), pass_through.end());
      std::reverse(unary_ops.begin(), unary_ops.end());
      edge.pass_through = std::move(pass_through);
      edge.unary_ops = std::move(unary_ops);
      return edge;
    }

    auto dw = MatchConvPassThroughSelf(*node, init_map);
    if (dw) {
      pass_through.push_back(ConvPassThrough{node, dw->weight, dw->bias});
      cur = node->input(0);
      continue;
    }

    if (UnaryPassThroughOps().count(node->op_type()) != 0 &&
        node->input_size() == 1) {
      unary_ops.push_back(node);
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
      return edge;
    }

    return ConvBackwardEdge{};
  }
  return ConvBackwardEdge{};
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
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };
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

    for (int idx : members) {
      for (auto& edge : edge_results[static_cast<size_t>(idx)]) {
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

    const std::string& sink_out = sink_add->output(0);
    if (!is_internal(sink_out)) {
      continue;
    }

    auto [consumer, fwd_chain_ops, fwd_pass_through] =
        WalkToConvConsumer(sink_out, init_map, consumers_of, graph_outputs,
                           n_channels, kMaxChainHops);
    if (!consumer) {
      continue;
    }
    if (consumer->group != 1) {
      continue;
    }

    std::vector<ChainOp> chain_ops;
    for (auto* op : unary_ops) {
      chain_ops.push_back(ChainOp{op, std::nullopt});
    }
    for (int idx : members) {
      chain_ops.push_back(
          ChainOp{eligible_adds[static_cast<size_t>(idx)], std::nullopt});
    }
    for (auto& co : fwd_chain_ops) {
      chain_ops.push_back(std::move(co));
    }

    Chain chain;
    chain.producers = std::move(leaf_producers);
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = false;
    chain.n_channels = n_channels;
    chain.consumer_is_conv = true;
    pass_through.insert(pass_through.end(), fwd_pass_through.begin(),
                        fwd_pass_through.end());
    chain.conv_pass_through = std::move(pass_through);
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

constexpr char kComMicrosoftDomain[] = "com.microsoft";

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
// FindMatmulResidualChains.
struct MatMulBackwardEdge {
  BackwardEdgeKind kind = BackwardEdgeKind::kFail;
  Producer producer;
  int64_t n_channels = 0;
  onnx::NodeProto* add_node = nullptr;
  std::vector<ChainOp> chain_ops;  // Forward order.
};

MatMulBackwardEdge WalkMatmulProducerBackward(
    const std::string& start,
    const std::unordered_map<std::string, onnx::NodeProto*>& node_by_output,
    const InitMap& init_map, const ConsumerMap& consumers_of,
    const std::unordered_set<std::string>& graph_outputs, int max_hops) {
  std::vector<ChainOp> chain_ops;  // Backward order.
  std::string cur = start;
  for (int hop = 0; hop < max_hops; ++hop) {
    if (ConsumerCount(consumers_of, cur) != 1 || graph_outputs.count(cur)) {
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
      return edge;
    }

    if (UnaryPassThroughOps().count(node->op_type()) != 0 &&
        node->input_size() == 1) {
      chain_ops.push_back(ChainOp{node, std::nullopt});
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
          cur = other;
          continue;
        }
        return MatMulBackwardEdge{};
      }
      // Both operands constant (degenerate) or both non-constant: for `Add`
      // the latter is exactly IsEligibleAddMerge's own shape, handled
      // below; for `Mul` it's a gated (SwiGLU/GeGLU) combine point this
      // walk doesn't try to pick a branch through -- either way, falling
      // through to the merge check or "fail" is correct.
    }

    if (MatchResidualMerge(node, init_map, consumers_of, graph_outputs)) {
      MatMulBackwardEdge edge;
      edge.kind = BackwardEdgeKind::kAdd;
      edge.add_node = node;
      std::reverse(chain_ops.begin(), chain_ops.end());
      edge.chain_ops = std::move(chain_ops);
      return edge;
    }

    return MatMulBackwardEdge{};
  }
  return MatMulBackwardEdge{};
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
  auto is_internal = [&](const std::string& name) {
    return ConsumerCount(consumers_of, name) == 1 && !graph_outputs.count(name);
  };
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    for (const auto& out : node->output()) {
      node_by_output[out] = node;
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
          kMaxChainHops);
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

    for (int idx : members) {
      pre_chain_ops.insert(pre_chain_ops.end(),
                           merges[static_cast<size_t>(idx)].extra_ops.begin(),
                           merges[static_cast<size_t>(idx)].extra_ops.end());
      for (auto& edge : edge_results[static_cast<size_t>(idx)]) {
        pre_chain_ops.insert(pre_chain_ops.end(), edge.chain_ops.begin(),
                             edge.chain_ops.end());
        if (edge.kind == BackwardEdgeKind::kProducer) {
          leaf_producers.push_back(edge.producer);
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
    if (!is_internal(sink_out)) {
      continue;
    }

    auto [consumer, fwd_chain_ops] =
        WalkToConsumer(sink_out, init_map, consumers_of, graph_outputs,
                       n_channels, kMaxChainHops);
    if (!consumer) {
      continue;
    }

    std::vector<ChainOp> chain_ops = std::move(pre_chain_ops);
    for (int idx : members) {
      chain_ops.push_back(
          ChainOp{merges[static_cast<size_t>(idx)].node, std::nullopt});
    }
    for (auto& co : fwd_chain_ops) {
      chain_ops.push_back(std::move(co));
    }

    Chain chain;
    chain.producers = std::move(leaf_producers);
    chain.chain_ops = std::move(chain_ops);
    chain.consumer_node = consumer->node;
    chain.consumer_weight = consumer->weight;
    chain.consumer_weight_transposed = consumer->weight_transposed;
    chain.n_channels = n_channels;
    chains.push_back(std::move(chain));
  }
  return chains;
}

int64_t ChainGroup(const Chain& chain) {
  if (chain.producers.size() == 1 && chain.producers[0].group > 1) {
    return chain.producers[0].group;
  }
  return chain.consumer_group;
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

void ApplyChains(onnx::GraphProto* graph, std::vector<Chain>& chains,
                 double sparsity) {
  std::unordered_map<std::string, onnx::TensorProto*> init_map;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    onnx::TensorProto* t = graph->mutable_initializer(i);
    init_map[t->name()] = t;
  }

  std::unordered_set<std::string> producer_touched, consumer_touched,
      const_touched, conv_hop_touched, stale_value_info;

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
    std::unordered_set<std::string> conv_hop_weights;
    for (const auto& h : chain.conv_pass_through) {
      if (!conv_hop_weights.insert(h.weight).second) {
        degenerate = true;
        break;
      }
    }
    if (degenerate) {
      continue;
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

    bool conflict = consumer_touched.count(chain.consumer_weight) != 0;
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
      SetOrAddIntAttr(hop.node, "group", static_cast<int64_t>(keep.size()));
    }
    if (chain.consumer_is_conv && chain.consumer_group > 1) {
      SliceGroupedConsumerConvWeight(init_map.at(chain.consumer_weight), keep,
                                     chain.consumer_group, n);
    } else {
      SliceConsumerWeight(init_map.at(chain.consumer_weight),
                          chain.consumer_weight_transposed, keep,
                          chain.consumer_is_conv);
    }

    for (const auto& w : producer_weights) {
      producer_touched.insert(w);
    }
    consumer_touched.insert(chain.consumer_weight);
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

}  // namespace

onnx::ModelProto ApplyStructuredPruning(const onnx::ModelProto& model,
                                        double sparsity) {
  if (!(sparsity >= 0.0 && sparsity < 1.0)) {
    throw std::invalid_argument(
        "ApplyStructuredPruning: sparsity must be in [0, 1), got " +
        std::to_string(sparsity));
  }
  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

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
  if (!chains.empty()) {
    ApplyChains(graph, chains, sparsity);
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
  onnx::GraphProto* graph = out.mutable_graph();

  std::vector<AttnChain> chains = FindAttentionChains(graph);
  std::vector<AttnChain> gqa_chains = FindGqaChains(graph);
  std::vector<AttnChain> onnx_attn_chains = FindOnnxAttentionChains(graph);
  chains.insert(chains.end(), std::make_move_iterator(gqa_chains.begin()),
                std::make_move_iterator(gqa_chains.end()));
  chains.insert(chains.end(), std::make_move_iterator(onnx_attn_chains.begin()),
                std::make_move_iterator(onnx_attn_chains.end()));
  if (!chains.empty()) {
    ApplyAttentionChains(graph, chains, sparsity);
  }
  return out;
}

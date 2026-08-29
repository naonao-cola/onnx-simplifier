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
// backward walk plus union-find grouping across such eligible Add merge
// points (mirroring pruning.py's own _walk_conv_producer_backward/
// _find_conv_residual_chains and _walk_matmul_producer_backward/
// _find_matmul_residual_chains) covers not just a single `Add(x, f(x))` but
// a whole chain of such merges transitively sharing one spine channel
// count; a group with any branch that fails to resolve, or whose leaf
// producers disagree on channel count, is declined in its entirety, never
// partially pruned -- the same conservative "no branch-following" boundary
// every other chain finder here already holds. One piece of pruning.py's
// own residual support is *not* ported here: recognizing a fused
// com.microsoft::SkipLayerNormalization/SkipSimplifiedLayerNormalization
// node (what onnxruntime's transformer optimizer collapses a bare residual
// Add plus the following LayerNorm into) as an eligible merge point in its
// own right -- see _match_matmul_residual_merge's own comment in
// pruning.py for why that matters for a fully-optimized transformer. A
// model whose residual connections have already been fused that way is
// left untouched by this port's MatMul residual finder; the pure-Python
// implementation remains the full-featured reference for that case.
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

    if (IsEligibleAddMerge(*node, init_map)) {
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
// instead of WalkConvProducerBackward. See this section's own comment
// above for the scope note (bare Add merge points only, not the fused
// SkipLayerNormalization case pruning.py also recognizes).
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

  std::vector<onnx::NodeProto*> merges;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    if (IsEligibleAddMerge(*node, init_map)) {
      merges.push_back(node);
    }
  }
  if (merges.empty()) {
    return {};
  }
  std::unordered_map<onnx::NodeProto*, int> merge_index;
  for (size_t i = 0; i < merges.size(); ++i) {
    merge_index[merges[i]] = static_cast<int>(i);
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
    for (const auto& operand : merges[idx]->input()) {
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
    onnx::NodeProto* sink_node = merges[static_cast<size_t>(sinks[0])];

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
          ChainOp{merges[static_cast<size_t>(idx)], std::nullopt});
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

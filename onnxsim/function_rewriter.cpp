/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * A data-driven, onnxscript.rewriter-style pattern rewriter that lives in the
 * onnxsim C++ core so *every* binding (Python, the C ABI, Rust) can use it: a
 * rule is a pair of ``onnx.FunctionProto`` (a pattern to match + a
 * replacement), i.e. pure protobuf data, rather than a Python callback that
 * cannot cross the FFI boundary.
 *
 * Supported (the "richer" matcher):
 *   * pattern-function inputs as wildcards binding to host-graph values;
 *   * arbitrary connected DAG patterns with one or more outputs;
 *   * exact attribute matching, plus attribute *wildcards* via
 * ``ref_attr_name`` (bound and substituted into the replacement);
 *   * commutative reordering for the 2-input commutative ops (Add, Mul, ...);
 *   * constant-value matching (a pattern ``Constant`` node matches a host
 *     ``Constant`` node or an initializer with a byte-equal tensor);
 *   * a safety rule: matched interior values must not be consumed outside the
 *     match, so a rewrite never breaks unrelated parts of the graph.
 *
 * Documented limitations (first version):
 *   * only the main graph is traversed -- patterns are not matched inside
 *     If/Loop/Scan subgraph bodies;
 *   * input arity must match exactly -- variadic / optional-input ops are not
 *     specially handled, and commutative permutation is only tried for the
 *     2-input case, not N-ary (>2) operands;
 *   * attributes are matched by equality or bound as wildcards -- there is no
 *     attribute *predicate* (e.g. "perm is a reverse");
 *   * the replacement is spliced in at the position of the last matched node,
 *     so a match whose outputs are consumed by an earlier node is rejected.
 */
#include "function_rewriter.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "profiler.h"

namespace onnxsim {
namespace {

// Splits [0, n) into contiguous chunks run across a small thread pool spawned
// fresh for this call, or serially in the calling thread when there is no
// benefit (n too small, or only one hardware thread) or no pthread runtime
// (Emscripten's default build -- see profiler.cpp's sampler thread for the
// same guard). ``f`` must be safe to call concurrently for distinct indices;
// it is never called concurrently for the *same* index.
template <typename Fn>
void ParallelFor(int n, const Fn& f) {
#if !defined(__EMSCRIPTEN__)
  constexpr int kMinPerThread =
      64;  // below this, threading overhead isn't worth it.
  constexpr int kMaxThreads = 8;
  const unsigned hw = std::thread::hardware_concurrency();
  const int max_threads = hw > 1 ? static_cast<int>(hw) : 1;
  const int num_threads =
      std::min({max_threads, n / kMinPerThread, kMaxThreads});
  if (num_threads > 1) {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));
    const int chunk = (n + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
      const int begin = t * chunk;
      const int end = std::min(n, begin + chunk);
      if (begin >= end) {
        break;
      }
      workers.emplace_back([&f, begin, end]() {
        for (int i = begin; i < end; ++i) {
          f(i);
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
    return;
  }
#endif
  for (int i = 0; i < n; ++i) {
    f(i);
  }
}

// The default ONNX domain is spelled either "" or "ai.onnx"; treat them as one.
std::string NormDomain(const std::string& domain) {
  return domain == "ai.onnx" ? std::string() : domain;
}

bool SameOp(const onnx::NodeProto& a, const onnx::NodeProto& b) {
  return a.op_type() == b.op_type() &&
         NormDomain(a.domain()) == NormDomain(b.domain());
}

const onnx::AttributeProto* FindAttr(const onnx::NodeProto& node,
                                     const std::string& name) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name) {
      return &attr;
    }
  }
  return nullptr;
}

// Structural attribute equality. onnxsim may be built against the protobuf lite
// runtime (ONNX_USE_LITE_PROTO), where MessageDifferencer is unavailable, so
// compare the deterministic wire encodings instead. Two AttributeProtos that
// are field-for-field equal serialize identically.
bool AttrEqual(const onnx::AttributeProto& a, const onnx::AttributeProto& b) {
  return a.SerializeAsString() == b.SerializeAsString();
}

// Tensor value equality ignoring the (irrelevant) tensor name.
bool TensorValueEqual(onnx::TensorProto a, onnx::TensorProto b) {
  a.clear_name();
  b.clear_name();
  return a.SerializeAsString() == b.SerializeAsString();
}

// The commutative binary ops of the default ONNX domain whose two operands may
// be matched in either order. (Variadic ops such as Sum/Max/Min are only
// reordered when they happen to have exactly two inputs -- see the caller.)
bool IsCommutative(const onnx::NodeProto& node) {
  if (!NormDomain(node.domain()).empty()) {
    return false;
  }
  static const std::set<std::string> kCommutative = {
      "Add", "Mul", "And",  "Or",         "Xor",       "Max",
      "Min", "Sum", "Mean", "BitwiseAnd", "BitwiseOr", "BitwiseXor"};
  return kCommutative.count(node.op_type()) > 0;
}

// A read-only index over the host graph built once per rewrite scan.
struct HostIndex {
  // value name -> (node index producing it, output slot).
  std::unordered_map<std::string, std::pair<int, int>> producer;
  // value name -> indices of nodes consuming it.
  std::unordered_map<std::string, std::vector<int>> consumers;
  // initializer name -> tensor.
  std::unordered_map<std::string, const onnx::TensorProto*> initializer;
  std::unordered_set<std::string> graph_outputs;
  std::unordered_set<std::string> value_names;  // every name in use anywhere.
  // Node *names* -- a separate namespace from value names, but onnxruntime
  // still rejects two nodes sharing one, so freshly-minted replacement node
  // names need to avoid these too (see FreshPrefix).
  std::unordered_set<std::string> node_names;

  explicit HostIndex(const onnx::GraphProto& graph) {
    for (int i = 0; i < graph.node_size(); ++i) {
      const auto& node = graph.node(i);
      if (!node.name().empty()) {
        node_names.insert(node.name());
      }
      for (int s = 0; s < node.output_size(); ++s) {
        const std::string& out = node.output(s);
        if (!out.empty()) {
          producer[out] = {i, s};
          value_names.insert(out);
        }
      }
      for (const auto& in : node.input()) {
        if (!in.empty()) {
          consumers[in].push_back(i);
          value_names.insert(in);
        }
      }
    }
    for (const auto& init : graph.initializer()) {
      initializer[init.name()] = &init;
      value_names.insert(init.name());
    }
    for (const auto& vi : graph.input()) {
      value_names.insert(vi.name());
    }
    for (const auto& vi : graph.output()) {
      graph_outputs.insert(vi.name());
      value_names.insert(vi.name());
    }
  }
};

// Precomputed view of a pattern FunctionProto.
struct PatternIndex {
  const onnx::FunctionProto* fn;
  std::unordered_set<std::string> inputs;  // wildcard names.
  // pattern value name -> (pattern node index, output slot).
  std::unordered_map<std::string, std::pair<int, int>> producer;

  explicit PatternIndex(const onnx::FunctionProto& f) : fn(&f) {
    for (const auto& in : f.input()) {
      inputs.insert(in);
    }
    for (int i = 0; i < f.node_size(); ++i) {
      const auto& node = f.node(i);
      for (int s = 0; s < node.output_size(); ++s) {
        if (!node.output(s).empty()) {
          producer[node.output(s)] = {i, s};
        }
      }
    }
  }
};

// The bindings accumulated while matching a single occurrence of a pattern.
struct MatchState {
  std::unordered_map<std::string, std::string>
      vbind;  // wildcard -> host value.
  std::unordered_map<std::string, onnx::AttributeProto> abind;  // ref -> value.
  std::map<int, int> node_map;  // pattern node index -> host node index.
};

class Matcher {
 public:
  Matcher(const onnx::GraphProto& graph, const HostIndex& host,
          const PatternIndex& pat)
      : graph_(graph), host_(host), pat_(pat) {}

  // Match a pattern value against a host value, extending ``state``.
  bool MatchValue(const std::string& pval, const std::string& hval,
                  MatchState& state) {
    if (pat_.inputs.count(pval)) {
      // Wildcard: bind consistently.
      auto it = state.vbind.find(pval);
      if (it != state.vbind.end()) {
        return it->second == hval;
      }
      state.vbind[pval] = hval;
      return true;
    }
    auto pit = pat_.producer.find(pval);
    if (pit == pat_.producer.end()) {
      // A pattern value that is neither an input nor produced by the body
      // cannot be matched.
      return false;
    }
    const int p_node = pit->second.first;
    const int p_slot = pit->second.second;

    auto hprod = host_.producer.find(hval);
    if (hprod == host_.producer.end()) {
      // Host value is a graph input or initializer. The only pattern node that
      // can match it is a Constant whose tensor equals the initializer.
      const onnx::NodeProto& pnode = pat_.fn->node(p_node);
      if (pnode.op_type() == "Constant" && NormDomain(pnode.domain()).empty()) {
        auto init = host_.initializer.find(hval);
        const onnx::AttributeProto* value = FindAttr(pnode, "value");
        if (init != host_.initializer.end() && value != nullptr &&
            value->has_t() && TensorValueEqual(value->t(), *init->second)) {
          return true;  // constant matched against an initializer.
        }
      }
      return false;
    }
    if (hprod->second.second != p_slot) {
      return false;  // produced at a different output slot.
    }
    return MatchNode(p_node, hprod->second.first, state);
  }

  // Match a pattern node against a host node, extending ``state``.
  bool MatchNode(int p_idx, int h_idx, MatchState& state) {
    auto existing = state.node_map.find(p_idx);
    if (existing != state.node_map.end()) {
      return existing->second == h_idx;  // shared subexpression: be consistent.
    }
    // A host node may fill at most one pattern role (injective mapping).
    for (const auto& kv : state.node_map) {
      if (kv.second == h_idx) {
        return false;
      }
    }

    const onnx::NodeProto& p = pat_.fn->node(p_idx);
    const onnx::NodeProto& h = graph_.node(h_idx);
    if (!SameOp(p, h) || p.input_size() != h.input_size()) {
      return false;
    }
    if (!MatchAttributes(p, h, state)) {
      return false;
    }

    state.node_map[p_idx] = h_idx;

    // Try operand orders (identity, plus the swap for 2-input commutative ops).
    std::vector<std::vector<int>> orders;
    std::vector<int> identity(p.input_size());
    for (int i = 0; i < p.input_size(); ++i) {
      identity[i] = i;
    }
    orders.push_back(identity);
    if (p.input_size() == 2 && IsCommutative(p)) {
      orders.push_back({1, 0});
    }

    for (const auto& order : orders) {
      MatchState trial = state;
      bool ok = true;
      for (int i = 0; i < p.input_size(); ++i) {
        const std::string& pin = p.input(i);
        const std::string& hin = h.input(order[i]);
        if (pin.empty() || hin.empty()) {
          ok =
              (pin.empty() == hin.empty());  // optional-input slots must align.
          if (!ok) break;
          continue;
        }
        if (!MatchValue(pin, hin, trial)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        state = trial;
        return true;
      }
    }
    return false;
  }

 private:
  bool MatchAttributes(const onnx::NodeProto& p, const onnx::NodeProto& h,
                       MatchState& state) {
    for (const auto& pa : p.attribute()) {
      if (!pa.ref_attr_name().empty()) {
        // Attribute wildcard: bind the host node's attribute named pa.name().
        const onnx::AttributeProto* ha = FindAttr(h, pa.name());
        if (ha == nullptr) {
          return false;
        }
        onnx::AttributeProto bound = *ha;
        bound.clear_name();  // compare/store the value only.
        auto it = state.abind.find(pa.ref_attr_name());
        if (it != state.abind.end()) {
          if (!AttrEqual(it->second, bound)) {
            return false;
          }
        } else {
          state.abind[pa.ref_attr_name()] = std::move(bound);
        }
      } else {
        // Concrete attribute: the host must carry an equal one.
        const onnx::AttributeProto* ha = FindAttr(h, pa.name());
        if (ha == nullptr || !AttrEqual(pa, *ha)) {
          return false;
        }
      }
    }
    return true;
  }

  const onnx::GraphProto& graph_;
  const HostIndex& host_;
  const PatternIndex& pat_;
};

// Choose a prefix that collides with nothing already in the graph (as either
// a value name or a node name -- two separate ONNX namespaces, but checked
// together here since that's only ever more conservative) nor in ``reserved``
// -- names already claimed by other matches instantiated earlier in the same
// batch (see ApplyMatchesBatch) -- so a replacement's interior values *and*
// its nodes' own names are guaranteed fresh even when several matches of the
// same rule (which typically share the same interior/node names, e.g. always
// "t", or "node0" for an unnamed single-node replacement) are instantiated
// together against one frozen HostIndex.
std::string FreshPrefix(const HostIndex& host,
                        const std::unordered_set<std::string>& reserved,
                        const std::vector<std::string>& names) {
  for (int k = 0;; ++k) {
    std::string prefix = "onnxsim_fnrw" + std::to_string(k) + "_";
    bool ok = true;
    for (const auto& name : names) {
      const std::string full = prefix + name;
      if (host.value_names.count(full) || host.node_names.count(full) ||
          reserved.count(full)) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return prefix;
    }
  }
}

// Ensure ``model`` imports every operator-set domain the replacement nodes use.
void EnsureOpsetImports(onnx::ModelProto& model,
                        const onnx::FunctionProto& replacement) {
  std::unordered_set<std::string> present;
  for (const auto& imp : model.opset_import()) {
    present.insert(NormDomain(imp.domain()));
  }
  for (const auto& node : replacement.node()) {
    const std::string dom = NormDomain(node.domain());
    if (present.count(dom)) {
      continue;
    }
    int version = 0;
    for (const auto& imp : replacement.opset_import()) {
      if (NormDomain(imp.domain()) == dom) {
        version = static_cast<int>(imp.version());
        break;
      }
    }
    if (version <= 0) {
      version = 1;
    }
    auto* imp = model.add_opset_import();
    imp->set_domain(dom);
    imp->set_version(version);
    present.insert(dom);
  }
}

// A single validated, not-yet-applied match of a rule's pattern.
struct MatchResult {
  MatchState state;
  std::vector<std::string> out_host;
  std::set<int> matched;
};

// Instantiate ``rule``'s replacement for one validated match: build the
// renamed, attribute-filled replacement nodes. Does not touch the graph's
// node list -- ApplyMatchesBatch does that once for a whole batch of matches,
// rather than once per match. ``reserved_names`` accumulates every fresh
// interior name minted so far this batch (see FreshPrefix). Returns false (no
// change) if the replacement cannot be instantiated cleanly.
bool InstantiateReplacement(const HostIndex& host,
                            const FunctionRewriteRule& rule,
                            const MatchState& state,
                            const std::vector<std::string>& out_host,
                            std::unordered_set<std::string>& reserved_names,
                            std::vector<onnx::NodeProto>& new_nodes) {
  const onnx::FunctionProto& rep = rule.replacement;
  if (rep.output_size() != static_cast<int>(out_host.size())) {
    return false;
  }

  std::unordered_set<std::string> rep_inputs(rep.input().begin(),
                                             rep.input().end());
  std::unordered_map<std::string, std::string> rep_outputs;
  for (int i = 0; i < rep.output_size(); ++i) {
    rep_outputs[rep.output(i)] = out_host[i];
  }

  // Collect the replacement's interior value names (neither input nor output).
  std::vector<std::string> interior;
  std::unordered_set<std::string> seen_interior;
  auto note_interior = [&](const std::string& name) {
    if (name.empty() || rep_inputs.count(name) || rep_outputs.count(name) ||
        seen_interior.count(name)) {
      return;
    }
    seen_interior.insert(name);
    interior.push_back(name);
  };
  for (const auto& node : rep.node()) {
    for (const auto& v : node.output()) note_interior(v);
    for (const auto& v : node.input()) note_interior(v);
  }

  // Node *names* need the same freshness as interior values (see FreshPrefix)
  // -- computed here, ahead of the node-building loop below, so both use the
  // exact same fallback name per replacement node.
  std::vector<std::string> node_name_suffixes;
  node_name_suffixes.reserve(rep.node_size());
  for (int i = 0; i < rep.node_size(); ++i) {
    const auto& rnode = rep.node(i);
    node_name_suffixes.push_back(
        rnode.name().empty() ? "node" + std::to_string(i) : rnode.name());
  }

  std::vector<std::string> fresh_names = interior;
  fresh_names.insert(fresh_names.end(), node_name_suffixes.begin(),
                     node_name_suffixes.end());
  const std::string prefix = FreshPrefix(host, reserved_names, fresh_names);
  for (const auto& name : fresh_names) {
    reserved_names.insert(prefix + name);
  }

  auto rename = [&](const std::string& name) -> std::string {
    if (name.empty()) {
      return name;
    }
    auto rin = rep_inputs.find(name);
    if (rin != rep_inputs.end()) {
      auto b = state.vbind.find(name);
      // A replacement input must be a bound pattern wildcard.
      return b != state.vbind.end() ? b->second : std::string();
    }
    auto rout = rep_outputs.find(name);
    if (rout != rep_outputs.end()) {
      return rout->second;
    }
    return prefix + name;
  };

  // Build the replacement nodes with renamed edges and bound attributes.
  new_nodes.clear();
  new_nodes.reserve(rep.node_size());
  for (int node_counter = 0; node_counter < rep.node_size(); ++node_counter) {
    const auto& rnode = rep.node(node_counter);
    onnx::NodeProto n;
    n.set_op_type(rnode.op_type());
    if (!rnode.domain().empty()) {
      n.set_domain(rnode.domain());
    }
    n.set_name(prefix + node_name_suffixes[node_counter]);
    for (const auto& in : rnode.input()) {
      const std::string mapped = rename(in);
      if (!in.empty() && mapped.empty()) {
        return false;  // an input referenced an unbound name.
      }
      n.add_input(mapped);
    }
    for (const auto& out : rnode.output()) {
      n.add_output(rename(out));
    }
    for (const auto& a : rnode.attribute()) {
      if (!a.ref_attr_name().empty()) {
        auto it = state.abind.find(a.ref_attr_name());
        if (it == state.abind.end()) {
          return false;  // replacement references an unbound attribute.
        }
        onnx::AttributeProto filled = it->second;
        filled.set_name(a.name());
        filled.clear_ref_attr_name();
        *n.add_attribute() = std::move(filled);
      } else {
        *n.add_attribute() = a;
      }
    }
    new_nodes.push_back(std::move(n));
  }
  return true;
}

// Instantiates and splices in every match's replacement in ONE combined
// node-list rebuild, rather than rebuilding the whole node list once per
// match: with many independent, non-conflicting matches per round (the
// common case -- e.g. the same fusion applying to N structurally-identical
// blocks), this turns O(node count x matches) copying into O(node count) for
// the whole round.
bool ApplyMatchesBatch(onnx::ModelProto& model, onnx::GraphProto& graph,
                       const HostIndex& host, const FunctionRewriteRule& rule,
                       std::vector<MatchResult>& matches) {
  struct Pending {
    std::set<int> matched;
    std::vector<onnx::NodeProto> new_nodes;
  };
  std::unordered_set<std::string> reserved_names;
  std::vector<Pending> pending;
  pending.reserve(matches.size());
  for (auto& m : matches) {
    Pending p;
    p.matched = std::move(m.matched);
    if (InstantiateReplacement(host, rule, m.state, m.out_host, reserved_names,
                               p.new_nodes)) {
      pending.push_back(std::move(p));
    }
    // Instantiation failure (e.g. an unbound replacement input/attribute) is a
    // rule-authoring issue, not specific to this host location; skip just this
    // match, matching the pre-batching behavior of moving on to the next
    // candidate.
  }
  if (pending.empty()) {
    return false;
  }

  EnsureOpsetImports(model, rule.replacement);

  // Map each match's splice point (its last matched node index) to its
  // instantiated replacement, and union every matched index across the batch.
  std::unordered_map<int, size_t> splice_at;
  std::set<int> all_matched;
  size_t new_node_total = 0;
  for (size_t i = 0; i < pending.size(); ++i) {
    splice_at[*pending[i].matched.rbegin()] = i;
    all_matched.insert(pending[i].matched.begin(), pending[i].matched.end());
    new_node_total += pending[i].new_nodes.size();
  }

  std::vector<onnx::NodeProto> rebuilt;
  rebuilt.reserve(graph.node_size() - all_matched.size() + new_node_total);
  for (int i = 0; i < graph.node_size(); ++i) {
    if (all_matched.count(i)) {
      auto it = splice_at.find(i);
      if (it != splice_at.end()) {
        for (auto& n : pending[it->second].new_nodes) {
          rebuilt.push_back(std::move(n));
        }
      }
      continue;
    }
    rebuilt.push_back(graph.node(i));
  }
  graph.clear_node();
  for (auto& n : rebuilt) {
    *graph.add_node() = std::move(n);
  }
  return true;
}

// Scan every host node for ``rule``, collecting every valid match that
// doesn't conflict with one already collected -- no two collected matches
// share a graph node, and no wildcard in one match binds to a value produced
// by a node another collected match is about to delete. ApplyMatchesBatch
// then rewrites every collected match in one combined pass. A match that
// conflicts with an earlier one this round is simply skipped -- it'll be
// found again (against the then-current graph) on the next round, once
// HostIndex is rebuilt.
std::vector<MatchResult> CollectNonConflictingMatches(
    const onnx::GraphProto& graph, const HostIndex& host,
    const FunctionRewriteRule& rule) {
  std::vector<MatchResult> results;
  const onnx::FunctionProto& pattern = rule.pattern;
  if (pattern.output_size() == 0 || pattern.node_size() == 0) {
    return results;
  }
  PatternIndex pat(pattern);

  // Anchor on the node producing the pattern's first output.
  auto root_it = pat.producer.find(pattern.output(0));
  if (root_it == pat.producer.end()) {
    return results;  // output(0) is a pass-through input -- nothing to match.
  }
  const int root_pat_node = root_it->second.first;
  const onnx::NodeProto& root = pattern.node(root_pat_node);

  // Phase 1 (parallel, read-only): independently try to match the pattern
  // rooted at each host node and run every safety check that does NOT depend
  // on which other matches this round end up accepted -- that part
  // (`claimed`, built incrementally in phase 2) is inherently sequential
  // since acceptance order matters, but the DAG search (Matcher::MatchNode)
  // and the rest of the checks only read `graph`/`host`/`pat`, all const and
  // already shared read-only across every host node in the original
  // single-threaded scan. Splitting the work this way changes only the
  // scheduling, not the result: phase 2 below resolves candidates in the same
  // node order, and with the same conflict rules, the fully-sequential
  // version used.
  std::vector<std::optional<MatchResult>> candidates(
      static_cast<size_t>(graph.node_size()));
  ParallelFor(graph.node_size(), [&](int h) {
    if (!SameOp(root, graph.node(h))) {
      return;
    }
    // "Match" covers the DAG search (Matcher::MatchNode) plus the bookkeeping
    // that decides whether a found match is safe to apply -- everything up to
    // (but not including) instantiating the replacement.
    ProfiledScope match_scope("RuleMatch");
    MatchState state;
    Matcher matcher(graph, host, pat);
    if (!matcher.MatchNode(root_pat_node, h, state)) {
      return;
    }

    // Resolve every pattern output to a concrete host value.
    std::vector<std::string> out_host;
    out_host.reserve(pattern.output_size());
    for (const auto& pout : pattern.output()) {
      if (pat.inputs.count(pout)) {
        auto b = state.vbind.find(pout);
        if (b == state.vbind.end()) {
          return;
        }
        out_host.push_back(b->second);
        continue;
      }
      auto pit = pat.producer.find(pout);
      if (pit == pat.producer.end() ||
          !state.node_map.count(pit->second.first)) {
        return;
      }
      const onnx::NodeProto& hn =
          graph.node(state.node_map.at(pit->second.first));
      if (pit->second.second >= hn.output_size()) {
        return;
      }
      out_host.push_back(hn.output(pit->second.second));
    }

    std::set<int> matched;
    for (const auto& kv : state.node_map) {
      matched.insert(kv.second);
    }
    if (matched.empty()) {
      return;
    }

    const int max_idx = *matched.rbegin();
    std::unordered_set<std::string> out_set(out_host.begin(), out_host.end());

    // Safety: an interior value (produced inside the match, not a pattern
    // output) must be consumed only within the match and must not be a
    // graph output -- otherwise the rewrite would break an outside consumer.
    for (int idx : matched) {
      for (const auto& v : graph.node(idx).output()) {
        if (v.empty() || out_set.count(v)) {
          continue;
        }
        if (host.graph_outputs.count(v)) {
          return;
        }
        auto cit = host.consumers.find(v);
        if (cit != host.consumers.end()) {
          for (int c : cit->second) {
            if (!matched.count(c)) {
              return;
            }
          }
        }
      }
    }

    // Placement safety: every consumer of a (reproduced) pattern-output value
    // must come after the splice point.
    for (const std::string& v : out_host) {
      auto cit = host.consumers.find(v);
      if (cit != host.consumers.end()) {
        for (int c : cit->second) {
          if (!matched.count(c) && c < max_idx) {
            return;
          }
        }
      }
    }

    candidates[static_cast<size_t>(h)] =
        MatchResult{std::move(state), std::move(out_host), std::move(matched)};
  });

  // Phase 2 (serial): resolve conflicts against `claimed` in host-node order,
  // exactly as the fully-sequential version did.
  std::set<int> claimed;  // union of matched-node-sets already collected.
  for (int h = 0; h < graph.node_size(); ++h) {
    std::optional<MatchResult>& cand = candidates[static_cast<size_t>(h)];
    if (!cand || claimed.count(h)) {
      continue;
    }

    // Reject if this match shares a node with an already-collected match --
    // it'll be re-examined next round.
    bool conflicts = false;
    for (int idx : cand->matched) {
      if (claimed.count(idx)) {
        conflicts = true;
        break;
      }
    }
    if (conflicts) {
      continue;
    }

    // No replacement input may be produced by a node about to be deleted --
    // by this match or another one already claimed this round.
    bool safe = true;
    for (const auto& kv : cand->state.vbind) {
      auto pit = host.producer.find(kv.second);
      if (pit != host.producer.end() &&
          (cand->matched.count(pit->second.first) ||
           claimed.count(pit->second.first))) {
        safe = false;  // a wildcard bound to a value about to be deleted.
        break;
      }
    }
    if (!safe) {
      continue;
    }

    claimed.insert(cand->matched.begin(), cand->matched.end());
    results.push_back(std::move(*cand));
  }
  return results;
}

// A GraphRewriter that applies a fixed set of FunctionRewriteRules. Kept
// private to this translation unit: callers obtain it only via
// MakeFunctionProtoRewriter, as a base GraphRewriter pointer, so no other
// shared object references this concrete type's vtable.
class FunctionProtoRewriterImpl final : public GraphRewriter {
 public:
  explicit FunctionProtoRewriterImpl(std::vector<FunctionRewriteRule> rules)
      : rules_(std::move(rules)) {}

  bool _Run(onnx::ModelProto& model) const override {
    if (rules_.empty() || !model.has_graph()) {
      return false;
    }
    onnx::GraphProto& graph = *model.mutable_graph();
    bool changed = false;
    // Keep applying rules until a full pass over every rule changes nothing.
    // Every successful round -- possibly many non-conflicting matches of one
    // rule, rewritten together -- invalidates the node indices, so rebuild
    // the index and restart the scan. A generous iteration cap guards against
    // a pathological rule whose replacement re-creates its own pattern (which
    // would otherwise loop forever); normal rule sets converge far below it.
    const long long kMaxRewrites =
        static_cast<long long>(graph.node_size()) * 100 + 100000;
    long long applied = 0;
    bool progress = true;
    while (progress && applied < kMaxRewrites) {
      progress = false;
      HostIndex host(graph);
      for (const auto& rule : rules_) {
        std::vector<MatchResult> matches =
            CollectNonConflictingMatches(graph, host, rule);
        if (matches.empty()) {
          continue;
        }
        ProfiledScope rewrite_scope("RuleApply");
        if (ApplyMatchesBatch(model, graph, host, rule, matches)) {
          changed = true;
          progress = true;
          applied += static_cast<long long>(matches.size());
          break;  // indices changed -- rebuild the index and rescan.
        }
      }
    }
    return changed;
  }

 private:
  std::vector<FunctionRewriteRule> rules_;
};

}  // namespace

std::shared_ptr<GraphRewriter> MakeFunctionProtoRewriter(
    std::vector<FunctionRewriteRule> rules) {
  return std::make_shared<FunctionProtoRewriterImpl>(std::move(rules));
}

}  // namespace onnxsim

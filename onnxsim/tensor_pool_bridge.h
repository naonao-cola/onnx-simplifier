/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bridges onnx::TensorProto <-> TensorPool, and ties TensorPool's
 * safetensors file format to onnx's own external-data mechanism
 * (TensorProto's data_location/external_data fields) so a model's weights
 * can live in one canonical, ecosystem-standard .safetensors file instead of
 * being embedded in the .onnx file itself or dumped in onnx's ad hoc raw
 * external-data format.
 *
 * Two directions:
 *   * ExportModelWithSafetensors -- turns an ordinary, fully in-memory model
 *     into one whose initializers are EXTERNAL references into a
 *     freshly-written safetensors file. Each initializer's raw_data is moved
 *     into the pool via TensorProto::release_raw_data() (no copy), then
 *     every reference is patched with the real, absolute file offset/length
 *     the tensor's bytes ended up at -- so the result is simultaneously a
 *     valid safetensors file (openable by the `safetensors` Python package,
 *     HF transformers/diffusers, etc. with no onnxsim involved) AND correct,
 *     spec-compliant classic onnx external data (openable by vanilla
 *     onnx/onnxruntime, which only look at offset/length and don't care
 *     what bytes -- here, the JSON header -- precede them in the file).
 *   * ImportModelWithSafetensors -- the reverse: loads a .safetensors file
 *     into a pool (one read, zero-copy views; see tensor_pool.h) and, by
 *     default, hydrates every matching initializer back to an ordinary
 *     in-memory (non-EXTERNAL) TensorProto, so the resulting model works
 *     with every existing onnxsim pass with no caveats. Hydration is one
 *     copy per tensor -- unavoidable, because onnx::TensorProto physically
 *     owns raw_data as a plain std::string (no borrowed/Cord-backed bytes
 *     type) -- but it's the *only* copy: the pool itself loads zero-copy,
 *     and only the tensors a caller actually asks to hydrate ever pay it
 *     (pass hydrate_all=false to leave the rest as lazy EXTERNAL pool
 *     references and hydrate individual ones on demand via
 *     HydrateTensorProto -- see that function's caveat first, though).
 *
 * NOT covered here: keeping initializers EXTERNAL as onnxsim's *internal*
 * in-flight representation during Simplify()'s fixed point, to dodge
 * ModelProto copy costs mid-pipeline. That would collide with existing logic
 * that already treats EXTERNAL as "we don't have the bytes, skip this
 * tensor" -- onnxsim.cpp's IsAllZeroTensor and its integer-tensor extraction
 * helper, and dlpack_bridge.h's FromTensorProtoBorrowing, all bail out on
 * EXTERNAL tensors today. Using TensorPool as a lazy, on-demand backing
 * store for those call sites (hydrate a tensor from the pool only when a
 * pass actually needs its bytes, instead of skipping it outright) is a
 * natural follow-up, but it means auditing and updating each of those call
 * sites, not just adding a pool -- so hydrate_all=false below is a low-level
 * building block for that future work, not something to wire into Simplify()
 * as-is.
 */
#ifndef ONNXSIM_TENSOR_POOL_BRIDGE_H_
#define ONNXSIM_TENSOR_POOL_BRIDGE_H_

#include <onnx/onnx_pb.h>

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensor_pool.h"

namespace onnxsim {
namespace tensor_pool {

namespace detail {

template <typename Fn>
void ForEachTensor(onnx::GraphProto& graph, Fn&& fn);

// Recurse into one node attribute: its own tensor (`t`), its repeated
// `tensors`, and any subgraph(s) it carries (`g` / `graphs`, e.g. an `If`
// branch or a `Loop` body). `path` is a stable, human-readable fallback name
// stem for tensors that have no `.name()` of their own (attribute tensors
// frequently don't).
template <typename Fn>
void ForEachTensorInAttr(onnx::AttributeProto& attr, const std::string& path,
                         Fn&& fn) {
  if (attr.has_t()) {
    auto* t = attr.mutable_t();
    fn(t->name().empty() ? path + "/t" : t->name(), *t);
  }
  for (int i = 0; i < attr.tensors_size(); ++i) {
    auto* t = attr.mutable_tensors(i);
    fn(t->name().empty() ? path + "/tensors" + std::to_string(i) : t->name(),
       *t);
  }
  if (attr.has_g()) ForEachTensor(*attr.mutable_g(), fn);
  for (auto& g : *attr.mutable_graphs()) ForEachTensor(g, fn);
}

// Recurse into every TensorProto a graph carries -- initializers and node
// attribute tensors, recursively through subgraphs -- calling fn(name,
// tensor) for each. `name` is the tensor's own `.name()` when non-empty,
// else a positional fallback, so every tensor gets a distinct pool key even
// when unnamed.
template <typename Fn>
void ForEachTensor(onnx::GraphProto& graph, Fn&& fn) {
  for (int i = 0; i < graph.initializer_size(); ++i) {
    auto* init = graph.mutable_initializer(i);
    fn(init->name().empty() ? "initializer" + std::to_string(i) : init->name(),
       *init);
  }
  for (int ni = 0; ni < graph.node_size(); ++ni) {
    auto* node = graph.mutable_node(ni);
    std::string node_path =
        node->name().empty() ? "node" + std::to_string(ni) : node->name();
    for (int ai = 0; ai < node->attribute_size(); ++ai) {
      ForEachTensorInAttr(*node->mutable_attribute(ai),
                          node_path + "/attr" + std::to_string(ai), fn);
    }
  }
}

}  // namespace detail

// Move `tensor`'s inline raw_data into `pool` under `name` via
// TensorProto::release_raw_data() (no copy). Returns false, leaving `tensor`
// untouched, when there's nothing eligible to move: no raw_data (a typed
// repeated field, or an empty tensor), already EXTERNAL, or STRING (no raw
// byte layout at all).
//
// CAUTION: on success, `tensor` is left in an intermediate, not-yet-
// externalized state -- its bytes are gone but data_location is still
// DEFAULT, not EXTERNAL -- because the pool doesn't know the tensor's final
// file offset until the whole pool has been written (offsets depend on every
// other tensor's size too). Callers must follow up by marking it EXTERNAL
// with the real location/offset/length once that's known; do not leave (or
// serialize) a model with tensors in this half-adopted state.
// ExportModelWithSafetensors below does this correctly; use it rather than
// calling AdoptFromTensorProto directly unless you need to compose your own
// export sequence.
inline bool AdoptFromTensorProto(const std::string& name,
                                 onnx::TensorProto& tensor, TensorPool& pool) {
  if (tensor.data_location() == onnx::TensorProto::EXTERNAL) return false;
  if (tensor.data_type() == onnx::TensorProto::STRING) return false;
  if (!tensor.has_raw_data()) return false;

  std::vector<int64_t> shape(tensor.dims().begin(), tensor.dims().end());
  std::unique_ptr<std::string> bytes(tensor.release_raw_data());
  pool.Add(name, tensor.data_type(), std::move(shape), std::move(*bytes));
  return true;
}

// Fill `tensor`'s raw_data by copying `name`'s bytes out of `pool` (this copy
// is unavoidable: onnx::TensorProto physically owns raw_data as a plain
// std::string) and clear any EXTERNAL / external_data state, so the result is
// an ordinary, self-contained in-memory tensor safe for every existing
// onnxsim pass. Returns false, leaving `tensor` untouched, if `name` isn't in
// `pool`.
inline bool HydrateTensorProto(const std::string& name,
                               onnx::TensorProto& tensor,
                               const TensorPool& pool) {
  const Entry* entry = pool.Find(name);
  if (entry == nullptr) return false;
  tensor.set_data_location(onnx::TensorProto::DEFAULT);
  tensor.clear_external_data();
  tensor.set_raw_data(entry->data.data(), entry->data.size());
  return true;
}

// Rewrite `model` so every eligible tensor (graph initializers and node-
// attribute tensors, recursing through subgraphs) becomes an EXTERNAL
// reference into a freshly-written `safetensors_path`, and write that file.
// Returns the number of tensors externalized. `pool` is an out-parameter: it
// ends up holding every externalized tensor, so a caller can inspect or
// reuse them (e.g. across several models sharing weights) without re-reading
// the file this just wrote.
inline size_t ExportModelWithSafetensors(onnx::ModelProto& model,
                                         const std::string& safetensors_path,
                                         TensorPool& pool) {
  // Keyed by the *pool key* ForEachTensor assigned (name, or a positional
  // fallback for an unnamed attribute tensor) -- NOT by `t->name()`, which
  // may be empty. Using t->name() to look the tensor back up after pooling
  // would silently drop every unnamed attribute tensor: it'd be adopted (its
  // bytes moved into the pool) but never re-marked EXTERNAL, leaving it
  // half-adopted (see AdoptFromTensorProto's caution note).
  std::vector<std::pair<std::string, onnx::TensorProto*>> exported;
  detail::ForEachTensor(*model.mutable_graph(),
                        [&](const std::string& name, onnx::TensorProto& t) {
                          if (AdoptFromTensorProto(name, t, pool)) {
                            exported.emplace_back(name, &t);
                          }
                        });

  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  pool.SaveSafetensors(safetensors_path, &offsets);
  const uint64_t prefix = HeaderPrefixSize(safetensors_path);

  for (auto& [pool_key, t] : exported) {
    const auto& [begin, end] = offsets.at(pool_key);
    t->set_data_location(onnx::TensorProto::EXTERNAL);
    t->clear_external_data();
    auto set_kv = [&](const std::string& k, const std::string& v) {
      auto* e = t->add_external_data();
      e->set_key(k);
      e->set_value(v);
    };
    set_kv("location", safetensors_path);
    set_kv("offset", std::to_string(prefix + begin));
    set_kv("length", std::to_string(end - begin));
  }
  return exported.size();
}

// Load `safetensors_path` into `pool` and, for every graph initializer whose
// name matches a pooled tensor, either hydrate it in place (hydrate_all, the
// default -- an ordinary in-memory tensor ready for any onnxsim pass) or
// leave it as a lazy EXTERNAL reference (hydrate_all=false) for a caller
// that will call HydrateTensorProto itself on just the tensors it ends up
// needing. See this header's top comment for why hydrate_all=false is not
// yet safe to feed straight into onnxsim's Simplify(). Returns the number of
// initializers matched.
inline size_t ImportModelWithSafetensors(onnx::ModelProto& model,
                                         const std::string& safetensors_path,
                                         TensorPool& pool,
                                         bool hydrate_all = true) {
  pool.LoadSafetensors(safetensors_path);
  size_t matched = 0;
  for (auto& init : *model.mutable_graph()->mutable_initializer()) {
    const Entry* entry = pool.Find(init.name());
    if (entry == nullptr) continue;
    ++matched;
    if (hydrate_all) {
      HydrateTensorProto(init.name(), init, pool);
    } else {
      init.set_data_location(onnx::TensorProto::EXTERNAL);
      init.clear_external_data();
      auto* e = init.add_external_data();
      e->set_key("location");
      e->set_value(safetensors_path);
    }
  }
  return matched;
}

}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_TENSOR_POOL_BRIDGE_H_

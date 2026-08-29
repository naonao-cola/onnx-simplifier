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
 * A third pair, EmbedModel/ExtractModel plus the format-specific
 * SaveModelAsSafetensors/LoadModelFromSafetensors below, goes one step
 * further: instead of a small .onnx file with EXTERNAL references pointing
 * at a separate weights file, the graph ITSELF is embedded inside the
 * weights file too, as one more pooled entry (name "model.onnx", a raw
 * byte blob) alongside the tensors -- one self-describing archive, not two
 * files. The embedded copy's own initializers use a SYMBOLIC EXTERNAL
 * reference (location = the archive path, no offset/length -- the same
 * convention hydrate_all=false above already uses), resolved by name
 * lookup into the archive's own pool, not by byte offset. That sidesteps a
 * real chicken-and-egg problem real offsets would have here (the model's
 * serialized size depends on the offsets, which depend on where the model
 * blob itself lands in the file, which depends on the model's serialized
 * size) at the cost of the extracted blob not being standalone-loadable by
 * a vanilla onnx tool on its own -- only onnxsim's own
 * LoadModelFromSafetensors/LoadModelFromGGUF (via ExtractModel) know to
 * resolve it. Giving the embedded copy real, byte-accurate offsets instead
 * (so an extracted model.onnx is standalone-usable elsewhere too) is a
 * tracked follow-up, not implemented here.
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

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mmap_file.h"
#include "tensor_pool.h"
#include "tensor_pool_dtype.h"
#include "tensor_pool_hash.h"

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

// Value of `t`'s external_data record keyed `key` (e.g. "location",
// "offset", "length" -- https://onnx.ai/onnx/repo-docs/ExternalData.html),
// or `default_value` if `t` carries no such key.
inline std::string ExternalDataValue(const onnx::TensorProto& t,
                                     const std::string& key,
                                     const std::string& default_value = "") {
  for (const auto& e : t.external_data()) {
    if (e.key() == key) return e.value();
  }
  return default_value;
}

// Resolves an external_data "location" value against the directory
// containing the .onnx file that referenced it, matching onnx's own
// external-data convention: an absolute location is used as-is, and a
// relative one is joined to `base_dir`.
inline std::string ResolveExternalDataPath(const std::string& base_dir,
                                           const std::string& location) {
  bool is_absolute =
      !location.empty() && (location[0] == '/' || location[0] == '\\' ||
                            (location.size() > 1 && location[1] == ':'));
  if (is_absolute || base_dir.empty()) return location;
  char sep = base_dir.back();
  return (sep == '/' || sep == '\\') ? base_dir + location
                                     : base_dir + "/" + location;
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

// --- classic ONNX external data, loaded via mmap into a TensorPool --------
//
// Unlike ExportModelWithSafetensors/ImportModelWithSafetensors above (which
// speak onnxsim's own safetensors-backed external-data convention),
// LoadModelWithTensorPool reads an *ordinary* .onnx file's own external-data
// mechanism (TensorProto::EXTERNAL plus a "location"/"offset"/"length"
// external_data record -- https://onnx.ai/onnx/repo-docs/ExternalData.html),
// the format any exporter (PyTorch, HuggingFace optimum, ...) or plain
// onnx.save(..., save_as_external_data=True) produces.
//
// Motivation: onnx's own external-data loader opens and reads each
// referenced file once per tensor, even when many initializers share one
// weights file (the common all_tensors_to_one_file=True case, which is
// exactly what a model over a few hundred MB tends to use). This mmaps each
// *distinct* external file exactly once (via mmap_file.h's TryMmapFile) and
// gives every tensor a zero-copy view into that one mapping -- so loading a
// large (>100MB) external-data model pays one mmap() per weights file, not
// one open+seek+read per tensor. A file this process can't mmap (no mapping
// support on this platform, or the OS call itself fails) falls back to an
// ordinary seek + read of just that tensor's own byte range, so no tensor is
// ever left unloaded because of it.
//
// `model` is populated by parsing `onnx_path` directly (this is the load,
// there is no separate "structure-only" step: an EXTERNAL tensor carries no
// inline bytes to skip in the first place). Returns the number of EXTERNAL
// tensors resolved into `pool`. As with ImportModelWithSafetensors,
// `hydrate_all` (the default) turns every resolved tensor into an ordinary
// in-memory TensorProto ready for any onnxsim pass; pass false to leave
// them as lazy EXTERNAL references for on-demand HydrateTensorProto, keyed
// by the same name ForEachTensor assigned (see that function's doc comment
// for the unnamed-attribute-tensor fallback naming). Throws
// std::runtime_error if `onnx_path` can't be read or parsed as a
// ModelProto; a malformed or out-of-range individual external_data record
// is skipped (that tensor is left as an unresolved EXTERNAL reference)
// rather than failing the whole load.
inline size_t LoadModelWithTensorPool(const std::string& onnx_path,
                                      onnx::ModelProto* model, TensorPool& pool,
                                      bool hydrate_all = true) {
  {
    std::ifstream in(onnx_path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("TensorPool: cannot open '" + onnx_path + "'");
    }
    if (!model->ParseFromIstream(&in)) {
      throw std::runtime_error("TensorPool: '" + onnx_path +
                               "' is not a valid serialized ModelProto");
    }
  }

  std::string base_dir;
  {
    auto pos = onnx_path.find_last_of("/\\");
    base_dir =
        (pos == std::string::npos) ? std::string() : onnx_path.substr(0, pos);
  }

  // Cache of already-mapped external files, keyed by resolved path -- so a
  // model with many tensors backed by one shared weights file mmaps that
  // file exactly once, however many tensors reference it.
  std::map<std::string, std::pair<std::shared_ptr<const char[]>, uint64_t>>
      mapped;

  size_t loaded = 0;
  detail::ForEachTensor(*model->mutable_graph(), [&](const std::string& name,
                                                     onnx::TensorProto& t) {
    if (t.data_location() != onnx::TensorProto::EXTERNAL) return;
    std::string location = detail::ExternalDataValue(t, "location");
    if (location.empty()) return;  // malformed; leave untouched
    std::string full_path = detail::ResolveExternalDataPath(base_dir, location);

    auto it = mapped.find(full_path);
    if (it == mapped.end()) {
      it = mapped.emplace(full_path, TryMmapFile(full_path)).first;
    }
    const std::shared_ptr<const char[]>& owner = it->second.first;
    const uint64_t file_size = it->second.second;

    uint64_t offset = 0;
    {
      std::string offset_str = detail::ExternalDataValue(t, "offset");
      if (!offset_str.empty()) offset = std::stoull(offset_str);
    }
    std::string length_str = detail::ExternalDataValue(t, "length");
    std::vector<int64_t> shape(t.dims().begin(), t.dims().end());

    if (owner != nullptr) {
      uint64_t length = length_str.empty()
                            ? (file_size > offset ? file_size - offset : 0)
                            : std::stoull(length_str);
      if (offset + length > file_size) return;  // out-of-range record
      std::string_view view(owner.get() + offset, length);
      pool.Add(name, t.data_type(), shape, owner, view);
    } else {
      // mmap unavailable/failed for this file -- fall back to reading
      // just this tensor's own byte range, same as onnx's own
      // external-data loader.
      std::ifstream data_in(full_path, std::ios::binary);
      if (!data_in) return;  // leave this tensor's EXTERNAL ref as-is
      uint64_t length;
      if (length_str.empty()) {
        data_in.seekg(0, std::ios::end);
        std::streamoff end = data_in.tellg();
        length = end > static_cast<std::streamoff>(offset)
                     ? static_cast<uint64_t>(end) - offset
                     : 0;
      } else {
        length = std::stoull(length_str);
      }
      data_in.seekg(static_cast<std::streamoff>(offset));
      std::string bytes(length, '\0');
      data_in.read(bytes.data(), static_cast<std::streamsize>(length));
      pool.Add(name, t.data_type(), shape, std::move(bytes));
    }
    ++loaded;
    if (hydrate_all) HydrateTensorProto(name, t, pool);
  });
  return loaded;
}

// Export `pool`'s per-tensor ContentHash (see tensor_pool.h) into
// `model`'s graph-level metadata_props, one entry per pooled tensor keyed
// "<algorithm>:<tensor name>" (e.g. "blake3:conv1.weight"), plus one
// "<algorithm>:__all__" entry -- a digest, under the same HashAlgorithm, of
// every "<name>\0<hex digest>\n" line concatenated in name-sorted order --
// so a consumer can check the whole set with a single recomputation
// instead of re-deriving and comparing every metadata_props entry itself.
// A downstream verifier needs only tensor_pool_hash.h's ContentDigest (or
// an equivalent reimplementation of its framing) and the tensor's own
// dtype/shape/bytes -- not onnxsim or TensorPool -- to recompute and check
// either value. Call this AFTER `pool` holds its final tensor bytes (e.g.
// right after ExportModelWithSafetensors, which is also the point at which
// `model`'s initializers already carry the file references this hash lets
// a caller confirm weren't corrupted or substituted); calling it earlier
// hashes whatever `pool` happens to contain at the time.
inline void AttachContentHashMetadata(const TensorPool& pool,
                                      onnx::ModelProto& model) {
  const std::string algo_name(HashAlgorithmName(pool.GetHashAlgorithm()));
  auto* graph = model.mutable_graph();
  std::string concat;
  for (const auto& [name, hex] : pool.AllContentHashes()) {
    auto* kv = graph->add_metadata_props();
    kv->set_key(algo_name + ":" + name);
    kv->set_value(hex);
    concat += name;
    concat.push_back('\0');
    concat += hex;
    concat.push_back('\n');
  }
  auto* all_kv = graph->add_metadata_props();
  all_kv->set_key(algo_name + ":__all__");
  all_kv->set_value(ToHex(RawDigest(pool.GetHashAlgorithm(), concat)));
}

// --- self-describing single-file archives (see this header's top comment) -

// Pool key under which a self-describing archive's own graph is stored.
inline const std::string kEmbeddedModelKey = "model.onnx";  // NOLINT

// TensorPool::Add silently overwrites an existing entry of the same name,
// which would be actively dangerous here: a model with a real initializer
// named "model.onnx" (unlikely, but not forbidden by the ONNX spec) would
// have that weight silently clobbered by the embedded-model blob with no
// error. Call this right before adding kEmbeddedModelKey and it throws
// instead.
inline void CheckNoEmbeddedModelKeyCollision(const TensorPool& pool) {
  if (pool.Find(kEmbeddedModelKey) != nullptr) {
    throw std::runtime_error(
        "TensorPool: a tensor is already named '" + kEmbeddedModelKey +
        "', which collides with the reserved key this archive format "
        "embeds the graph itself under; rename that tensor before "
        "embedding");
  }
}

// Move every eligible tensor's bytes out of `model` into `pool` (as
// AdoptFromTensorProto does), mark each as a symbolic EXTERNAL reference
// into `archive_path`, then serialize the now-byte-free `model` and add
// those bytes to `pool` under kEmbeddedModelKey. After this, `pool` (once
// saved to `archive_path`) is a complete, self-describing archive; `model`
// itself is mutated into the same byte-free, symbolically-external state
// AdoptFromTensorProto leaves individual tensors in -- reload it via
// ExtractModel/LoadModelFromSafetensors/LoadModelFromGGUF rather than using
// `model` directly afterwards.
//
// The embedded copy's EXTERNAL references are symbolic (see this header's
// top comment) -- for real, byte-accurate offsets that make the extracted
// model.onnx blob standalone-loadable elsewhere, use
// SaveModelAsSafetensorsStandalone / SaveModelAsGGUFStandalone instead.
inline void EmbedModel(onnx::ModelProto& model, const std::string& archive_path,
                       TensorPool& pool) {
  detail::ForEachTensor(*model.mutable_graph(),
                        [&](const std::string& name, onnx::TensorProto& t) {
                          if (!AdoptFromTensorProto(name, t, pool)) return;
                          t.set_data_location(onnx::TensorProto::EXTERNAL);
                          t.clear_external_data();
                          auto* e = t.add_external_data();
                          e->set_key("location");
                          e->set_value(archive_path);
                        });
  CheckNoEmbeddedModelKeyCollision(pool);

  std::string bytes;
  if (!model.SerializeToString(&bytes)) {
    throw std::runtime_error(
        "TensorPool: EmbedModel: failed to serialize the model");
  }
  int64_t nbytes = static_cast<int64_t>(bytes.size());
  pool.Add(kEmbeddedModelKey, ONNX_INT8, {nbytes}, std::move(bytes));
}

// Reverse of EmbedModel: find kEmbeddedModelKey in `pool`, parse it into
// `*model`, and hydrate its initializers from `pool` by name (or, with
// hydrate_all=false, leave them as the symbolic EXTERNAL references
// EmbedModel wrote, for on-demand HydrateTensorProto -- same caveat as
// ImportModelWithSafetensors's hydrate_all=false). Returns false, leaving
// `*model` untouched, if `pool` has no embedded model. Throws
// std::runtime_error if the embedded bytes aren't a valid ModelProto.
inline bool ExtractModel(const TensorPool& pool, onnx::ModelProto* model,
                         bool hydrate_all = true) {
  const Entry* blob = pool.Find(kEmbeddedModelKey);
  if (blob == nullptr) return false;
  if (!model->ParseFromArray(blob->data.data(),
                             static_cast<int>(blob->data.size()))) {
    throw std::runtime_error("TensorPool: ExtractModel: embedded '" +
                             kEmbeddedModelKey +
                             "' is not a valid serialized ModelProto");
  }
  if (hydrate_all) {
    for (auto& init : *model->mutable_graph()->mutable_initializer()) {
      HydrateTensorProto(init.name(), init, pool);
    }
  }
  return true;
}

// Convenience wrapper: EmbedModel then pool.SaveSafetensors(path). `model`
// ends up mutated exactly as EmbedModel leaves it (see that function's
// note) -- reload with LoadModelFromSafetensors rather than reusing `model`.
inline void SaveModelAsSafetensors(onnx::ModelProto& model,
                                   const std::string& path, TensorPool& pool) {
  EmbedModel(model, path, pool);
  pool.SaveSafetensors(path);
}

// Convenience wrapper: pool.LoadSafetensors(path) then ExtractModel. Returns
// false if `path`'s pool has no embedded model (e.g. it's a plain weights-
// only safetensors file with no onnxsim-authored archive on top).
inline bool LoadModelFromSafetensors(const std::string& path,
                                     onnx::ModelProto* model, TensorPool& pool,
                                     bool hydrate_all = true) {
  pool.LoadSafetensors(path);
  return ExtractModel(pool, model, hydrate_all);
}

// --- standalone archives: real, byte-accurate self-referencing offsets ----
//
// EmbedModel's embedded copy is symbolic -- fine for onnxsim's own
// round-trip (ExtractModel resolves by name, never touches offset/length),
// but useless to a vanilla onnx/onnxruntime given just the extracted
// model.onnx blob. Making the embedded copy's offsets byte-accurate has a
// real chicken-and-egg problem: the model's serialized size depends on the
// digit-string offset/length values patched into it, which depend on where
// the model blob itself ends up in the archive, which depends on the
// model's serialized size.
//
// Broken with a fixed-width field: every offset/length value is written as
// exactly kOffsetFieldWidth decimal digits (zero-padded -- ordinary decimal
// parsers, including onnx's own external-data reader, skip leading zeros
// with no issue), so re-patching the *real* value in later never changes
// the model's serialized byte length versus an all-zero placeholder. That
// makes the whole thing a single clean pass: adopt tensors with placeholder
// offsets -> serialize -> pool it -> ask TensorPool for the real layout ->
// patch the *same-length* real values in -> re-serialize (guaranteed
// identical length) -> overwrite just the already-written model blob's
// bytes in place. Every weight tensor is still written to disk exactly
// once; only the small model blob is touched twice.

inline constexpr int kOffsetFieldWidth = 20;  // covers up to ~10**20 bytes

inline std::string FixedWidthDecimal(uint64_t v) {
  std::string s = std::to_string(v);
  if (s.size() > static_cast<size_t>(kOffsetFieldWidth)) {
    throw std::runtime_error("TensorPool: offset/length " + s +
                             " exceeds the reserved field width (" +
                             std::to_string(kOffsetFieldWidth) + " digits)");
  }
  return std::string(static_cast<size_t>(kOffsetFieldWidth) - s.size(), '0') +
         s;
}

// First half of the standalone-embed algorithm: adopt every eligible
// tensor's bytes into `pool` and mark each EXTERNAL with `archive_path` and
// fixed-width placeholder (all-zero) offset/length, ready for
// PatchStandaloneOffsets once the pool's real layout is known. Returns
// each adopted tensor's pool key alongside its TensorProto*, exactly like
// ExportModelWithSafetensors's internal bookkeeping, so the second half can
// find and re-patch them without a second graph traversal.
//
// `external_tensor_bytes`, when non-null, supplies the adopted bytes
// directly (keyed the same way ForEachTensor derives pool keys) instead of
// pulling them out of each TensorProto's own raw_data via
// AdoptFromTensorProto -- entries are moved out of the map as they're
// consumed. This is for a caller (see cpp2py_export.cc's
// export_safetensors/export_gguf) that already has the tensor bytes
// separately and stripped `model`'s own raw_data fields before crossing it
// into C++, to avoid paying a full protobuf encode/decode of the
// (potentially huge) tensor data on top of the copies pool.Add and the
// eventual archive write already make. A tensor with no raw_data left (or
// none to begin with, e.g. it already used a typed field) and no matching
// entry in the map is simply not adopted, same as AdoptFromTensorProto's
// own "nothing eligible" case.
inline std::vector<std::pair<std::string, onnx::TensorProto*>>
AdoptAllWithPlaceholderOffsets(
    onnx::ModelProto& model, const std::string& archive_path, TensorPool& pool,
    std::map<std::string, std::string>* external_tensor_bytes = nullptr) {
  std::vector<std::pair<std::string, onnx::TensorProto*>> adopted;
  detail::ForEachTensor(*model.mutable_graph(), [&](const std::string& name,
                                                    onnx::TensorProto& t) {
    bool ok;
    if (external_tensor_bytes != nullptr) {
      auto it = external_tensor_bytes->find(name);
      if (it == external_tensor_bytes->end()) {
        ok = false;
      } else {
        std::vector<int64_t> shape(t.dims().begin(), t.dims().end());
        pool.Add(name, t.data_type(), std::move(shape), std::move(it->second));
        external_tensor_bytes->erase(it);
        t.clear_raw_data();
        ok = true;
      }
    } else {
      ok = AdoptFromTensorProto(name, t, pool);
    }
    if (!ok) return;
    t.set_data_location(onnx::TensorProto::EXTERNAL);
    t.clear_external_data();
    auto set_kv = [&](const std::string& k, const std::string& v) {
      auto* e = t.add_external_data();
      e->set_key(k);
      e->set_value(v);
    };
    set_kv("location", archive_path);
    set_kv("offset", FixedWidthDecimal(0));
    set_kv("length", FixedWidthDecimal(0));
    adopted.emplace_back(name, &t);
  });
  return adopted;
}

// Second half: overwrite each adopted tensor's placeholder offset/length
// (in place, same field width) with its real value from `offsets`
// (relative to the data section) and `data_section_start` (absolute) --
// both exactly what TensorPool::SaveSafetensors's/SaveGGUF's own
// data_offsets_out/data_section_start_out out-params provide.
inline void PatchStandaloneOffsets(
    std::vector<std::pair<std::string, onnx::TensorProto*>>& adopted,
    const std::map<std::string, std::pair<uint64_t, uint64_t>>& offsets,
    uint64_t data_section_start) {
  for (auto& [name, t] : adopted) {
    const auto& [begin, end] = offsets.at(name);
    for (auto& e : *t->mutable_external_data()) {
      if (e.key() == "offset") {
        e.set_value(FixedWidthDecimal(data_section_start + begin));
      } else if (e.key() == "length") {
        e.set_value(FixedWidthDecimal(end - begin));
      }
    }
  }
}

// Overwrite `bytes.size()` bytes of the already-written file at `path`,
// starting at absolute byte `offset`, without touching anything else in the
// file (no truncation, no resize) -- the final step of the standalone-embed
// algorithm, patching the model blob's placeholder bytes with the real,
// offset-patched serialization in place.
inline void PatchFileBytes(const std::string& path, uint64_t offset,
                           const std::string& bytes) {
  std::fstream out(path, std::ios::in | std::ios::out | std::ios::binary);
  if (!out) {
    throw std::runtime_error("TensorPool: cannot reopen '" + path +
                             "' to patch the embedded model's bytes");
  }
  out.seekp(static_cast<std::streamoff>(offset));
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) {
    throw std::runtime_error(
        "TensorPool: failed to patch the embedded model's bytes into '" + path +
        "'");
  }
}

// Standalone counterpart of SaveModelAsSafetensors: same one-file result,
// but the embedded copy's initializers carry real, byte-accurate
// external_data offset/length into `path` -- extracting the "model.onnx"
// tensor's bytes and saving them separately produces an ordinary, fully
// standalone .onnx file loadable by vanilla onnx/onnxruntime, no onnxsim
// involved. Every weight tensor is written to disk exactly once; the model
// blob itself is written, then patched in place (same algorithm as
// SaveModelAsGGUFStandalone) -- see this section's top comment.
inline void SaveModelAsSafetensorsStandalone(
    onnx::ModelProto& model, const std::string& path, TensorPool& pool,
    std::map<std::string, std::string>* external_tensor_bytes = nullptr) {
  auto adopted =
      AdoptAllWithPlaceholderOffsets(model, path, pool, external_tensor_bytes);
  CheckNoEmbeddedModelKeyCollision(pool);

  std::string placeholder_bytes;
  if (!model.SerializeToString(&placeholder_bytes)) {
    throw std::runtime_error(
        "TensorPool: SaveModelAsSafetensorsStandalone: failed to serialize "
        "the model");
  }
  pool.Add(kEmbeddedModelKey, ONNX_INT8,
           {static_cast<int64_t>(placeholder_bytes.size())},
           std::string(placeholder_bytes));

  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  pool.SaveSafetensors(path, &offsets);
  uint64_t data_section_start = HeaderPrefixSize(path);

  PatchStandaloneOffsets(adopted, offsets, data_section_start);

  std::string final_bytes;
  if (!model.SerializeToString(&final_bytes)) {
    throw std::runtime_error(
        "TensorPool: SaveModelAsSafetensorsStandalone: failed to "
        "re-serialize the model");
  }
  if (final_bytes.size() != placeholder_bytes.size()) {
    // Should be unreachable given FixedWidthDecimal's fixed width; a hard
    // check rather than a silently-corrupt file if the invariant ever
    // breaks (e.g. a future edit to what EmbedModel-style code patches).
    throw std::runtime_error(
        "TensorPool: SaveModelAsSafetensorsStandalone: internal error -- "
        "re-serializing with real offsets changed the model's size (" +
        std::to_string(placeholder_bytes.size()) + " -> " +
        std::to_string(final_bytes.size()) + " bytes)");
  }

  const auto& model_range = offsets.at(kEmbeddedModelKey);
  PatchFileBytes(path, data_section_start + model_range.first, final_bytes);
  pool.Add(kEmbeddedModelKey, ONNX_INT8,
           {static_cast<int64_t>(final_bytes.size())}, std::move(final_bytes));
}

}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_TENSOR_POOL_BRIDGE_H_

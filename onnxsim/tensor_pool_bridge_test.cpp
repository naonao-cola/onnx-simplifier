/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises tensor_pool_bridge.h's onnx::TensorProto <-> TensorPool glue and
 * its safetensors export/import round trip. Needs onnx configured (unlike
 * tensor_pool_test.cpp / tensor_pool_dtype_test.cpp), so it's built against
 * the fully-configured `onnxsim` CMake target rather than standalone.
 */
#include "tensor_pool_bridge.h"

#include <onnx/onnx_pb.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <string>

using namespace onnxsim::tensor_pool;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

std::string TempPath(const std::string& suffix) {
  return "/tmp/onnxsim_tensor_pool_bridge_test_" + std::to_string(::getpid()) +
         suffix;
}

std::string ReadFileRange(const std::string& path, uint64_t offset,
                          uint64_t length) {
  std::ifstream in(path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(offset));
  std::string out(length, '\0');
  in.read(out.data(), static_cast<std::streamsize>(length));
  return out;
}

onnx::TensorProto MakeInitializer(const std::string& name,
                                  onnx::TensorProto::DataType dtype,
                                  const std::vector<int64_t>& dims,
                                  const std::string& raw) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(dtype);
  for (int64_t d : dims) t.add_dims(d);
  t.set_raw_data(raw);
  return t;
}

void TestAdoptAndHydrateSingleTensor() {
  onnx::TensorProto t =
      MakeInitializer("w", onnx::TensorProto::FLOAT, {2}, std::string(8, 'x'));
  TensorPool pool;
  bool adopted = AdoptFromTensorProto("w", t, pool);
  Check(adopted, "AdoptFromTensorProto succeeds for an inline-raw_data tensor");
  Check(!t.has_raw_data(),
        "raw_data is gone from the TensorProto after adopting (moved, not "
        "copied)");
  const Entry* e = pool.Find("w");
  Check(e != nullptr && e->nbytes() == 8, "pool holds the adopted bytes");

  // Adopting an already-adopted (raw_data-less) tensor is a no-op.
  Check(!AdoptFromTensorProto("w", t, pool),
        "re-adopting a tensor with no raw_data returns false");

  bool hydrated = HydrateTensorProto("w", t, pool);
  Check(hydrated, "HydrateTensorProto succeeds");
  Check(t.has_raw_data() && t.raw_data() == std::string(8, 'x'),
        "hydrated raw_data matches the original bytes exactly");
  Check(t.data_location() == onnx::TensorProto::DEFAULT,
        "hydrated tensor is DEFAULT-located, not EXTERNAL");

  Check(!HydrateTensorProto("missing", t, pool),
        "hydrating an unpooled name returns false");
}

void TestStringTensorIsNotAdopted() {
  onnx::TensorProto t;
  t.set_name("s");
  t.set_data_type(onnx::TensorProto::STRING);
  t.add_string_data("hello");
  TensorPool pool;
  Check(!AdoptFromTensorProto("s", t, pool),
        "STRING tensors are never adopted (no raw byte layout)");
  Check(pool.empty(), "pool stays empty");
}

void TestExportModelWithSafetensors() {
  onnx::ModelProto model;
  auto* graph = model.mutable_graph();
  *graph->add_initializer() = MakeInitializer(
      "weight", onnx::TensorProto::FLOAT, {2, 2}, std::string(16, '\x11'));
  *graph->add_initializer() = MakeInitializer("bias", onnx::TensorProto::INT64,
                                              {2}, std::string(16, '\x22'));

  // A node-attribute tensor (e.g. as Constant's `value` attribute uses) --
  // exercises the recursion into node attributes, not just top-level
  // initializers.
  auto* node = graph->add_node();
  node->set_name("const_node");
  auto* attr = node->add_attribute();
  attr->set_name("value");
  *attr->mutable_t() = MakeInitializer("const_tensor", onnx::TensorProto::UINT8,
                                       {4}, std::string(4, '\x33'));

  std::string path = TempPath("_export.safetensors");
  TensorPool pool;
  size_t n = ExportModelWithSafetensors(model, path, pool);
  Check(n == 3, "all three tensors (2 initializers + 1 attribute) exported");

  for (const auto& name : {"weight", "bias", "const_tensor"}) {
    const onnx::TensorProto* t = nullptr;
    for (const auto& init : graph->initializer()) {
      if (init.name() == name) t = &init;
    }
    if (t == nullptr && graph->node(0).attribute(0).t().name() == name) {
      t = &graph->node(0).attribute(0).t();
    }
    Check(t != nullptr, std::string(name) + ": found in model");
    if (t == nullptr) continue;
    Check(t->data_location() == onnx::TensorProto::EXTERNAL,
          std::string(name) + ": is EXTERNAL after export");
    Check(!t->has_raw_data(),
          std::string(name) + ": no inline raw_data left after export");

    std::string location, offset_str, length_str;
    for (const auto& e : t->external_data()) {
      if (e.key() == "location") location = e.value();
      if (e.key() == "offset") offset_str = e.value();
      if (e.key() == "length") length_str = e.value();
    }
    Check(location == path,
          std::string(name) + ": location == safetensors path");
    Check(!offset_str.empty() && !length_str.empty(),
          std::string(name) + ": offset/length are set");

    uint64_t offset = std::stoull(offset_str);
    uint64_t length = std::stoull(length_str);
    std::string on_disk = ReadFileRange(path, offset, length);
    const Entry* pooled = pool.Find(name);
    Check(pooled != nullptr && on_disk == pooled->data,
          std::string(name) +
              ": bytes at the recorded external_data offset match the pool "
              "(and so the original tensor)");
  }

  // The file this wrote is a valid, standalone safetensors file too -- a
  // fresh TensorPool can load it with no onnxsim/onnx involvement.
  TensorPool reloaded;
  reloaded.LoadSafetensors(path);
  Check(reloaded.size() == 3,
        "the exported file is independently loadable as safetensors");

  std::remove(path.c_str());
}

void TestUnnamedAttributeTensor() {
  // Regression test: an attribute tensor with no .name() of its own must
  // still round-trip through export, keyed by ForEachTensor's positional
  // fallback rather than being silently dropped (see
  // ExportModelWithSafetensors's comment on why it tracks pool keys
  // alongside each TensorProto* instead of re-deriving the key from
  // t->name() after the fact).
  onnx::ModelProto model;
  auto* node = model.mutable_graph()->add_node();
  node->set_name("n0");
  auto* attr = node->add_attribute();
  attr->set_name("value");
  auto* t = attr->mutable_t();
  t->set_data_type(onnx::TensorProto::FLOAT);
  t->add_dims(1);
  t->set_raw_data(std::string(4, '\x77'));
  // t->name() is deliberately left empty.

  std::string path = TempPath("_unnamed_attr.safetensors");
  TensorPool pool;
  size_t n = ExportModelWithSafetensors(model, path, pool);
  Check(n == 1, "the unnamed attribute tensor is exported");

  const auto& exported_t = model.graph().node(0).attribute(0).t();
  Check(exported_t.data_location() == onnx::TensorProto::EXTERNAL,
        "unnamed attribute tensor is EXTERNAL after export");
  std::string location, offset_str, length_str;
  for (const auto& e : exported_t.external_data()) {
    if (e.key() == "location") location = e.value();
    if (e.key() == "offset") offset_str = e.value();
    if (e.key() == "length") length_str = e.value();
  }
  Check(!location.empty() && !offset_str.empty() && !length_str.empty(),
        "unnamed attribute tensor got real location/offset/length, not left "
        "half-adopted");
  if (!offset_str.empty() && !length_str.empty()) {
    std::string on_disk =
        ReadFileRange(path, std::stoull(offset_str), std::stoull(length_str));
    Check(on_disk == std::string(4, '\x77'),
          "bytes at the recorded offset match the original tensor");
  }
  std::remove(path.c_str());
}

void TestImportModelWithSafetensorsHydrateAll() {
  // Build and export a small model, matching TestExportModelWithSafetensors
  // but round-tripping all the way back through import.
  onnx::ModelProto original;
  auto* g = original.mutable_graph();
  *g->add_initializer() = MakeInitializer("w1", onnx::TensorProto::FLOAT, {3},
                                          std::string(12, '\x44'));
  *g->add_initializer() = MakeInitializer("w2", onnx::TensorProto::INT32, {2},
                                          std::string(8, '\x55'));

  std::string path = TempPath("_import.safetensors");
  TensorPool export_pool;
  ExportModelWithSafetensors(original, path, export_pool);

  // A fresh model with the same initializer names/dtypes/shapes but no
  // bytes -- the situation a real loader would be in after reading a
  // .onnx whose initializers are external_data placeholders.
  onnx::ModelProto fresh;
  auto* fg = fresh.mutable_graph();
  auto* w1 = fg->add_initializer();
  w1->set_name("w1");
  w1->set_data_type(onnx::TensorProto::FLOAT);
  w1->add_dims(3);
  auto* w2 = fg->add_initializer();
  w2->set_name("w2");
  w2->set_data_type(onnx::TensorProto::INT32);
  w2->add_dims(2);

  TensorPool import_pool;
  size_t matched = ImportModelWithSafetensors(fresh, path, import_pool);
  Check(matched == 2, "both initializers matched on import");

  Check(fg->initializer(0).has_raw_data() &&
            fg->initializer(0).raw_data() == std::string(12, '\x44'),
        "w1 hydrated to the original bytes");
  Check(fg->initializer(0).data_location() == onnx::TensorProto::DEFAULT,
        "w1 is DEFAULT-located after hydration");
  Check(fg->initializer(1).has_raw_data() &&
            fg->initializer(1).raw_data() == std::string(8, '\x55'),
        "w2 hydrated to the original bytes");

  std::remove(path.c_str());
}

void TestImportModelWithSafetensorsLazy() {
  onnx::ModelProto original;
  *original.mutable_graph()->add_initializer() = MakeInitializer(
      "lazy", onnx::TensorProto::FLOAT, {1}, std::string(4, '\x66'));
  std::string path = TempPath("_lazy.safetensors");
  TensorPool export_pool;
  ExportModelWithSafetensors(original, path, export_pool);

  onnx::ModelProto fresh;
  auto* init = fresh.mutable_graph()->add_initializer();
  init->set_name("lazy");
  init->set_data_type(onnx::TensorProto::FLOAT);
  init->add_dims(1);

  TensorPool import_pool;
  size_t matched = ImportModelWithSafetensors(fresh, path, import_pool,
                                              /*hydrate_all=*/false);
  Check(matched == 1, "lazy import still reports the match");
  Check(!fresh.graph().initializer(0).has_raw_data(),
        "lazy import leaves raw_data unset");
  Check(fresh.graph().initializer(0).data_location() ==
            onnx::TensorProto::EXTERNAL,
        "lazy import marks the tensor EXTERNAL for on-demand hydration");

  // The caller can still hydrate it explicitly, on demand, straight from the
  // already-loaded pool -- no second file read.
  bool hydrated = HydrateTensorProto(
      "lazy", *fresh.mutable_graph()->mutable_initializer(0), import_pool);
  Check(hydrated &&
            fresh.graph().initializer(0).raw_data() == std::string(4, '\x66'),
        "on-demand HydrateTensorProto recovers the bytes");

  std::remove(path.c_str());
}

void TestAttachContentHashMetadata() {
  onnx::ModelProto model;
  auto* graph = model.mutable_graph();
  *graph->add_initializer() = MakeInitializer(
      "weight", onnx::TensorProto::FLOAT, {2, 2}, std::string(16, '\x11'));
  *graph->add_initializer() = MakeInitializer("bias", onnx::TensorProto::INT64,
                                              {2}, std::string(16, '\x22'));

  std::string path = TempPath("_hash.safetensors");
  TensorPool pool;
  ExportModelWithSafetensors(model, path, pool);
  AttachContentHashMetadata(pool, model);

  std::map<std::string, std::string> props;
  for (const auto& kv : graph->metadata_props()) {
    props[kv.key()] = kv.value();
  }
  Check(props.size() == 3,
        "one metadata_props entry per tensor plus one __all__ summary");
  Check(props.count("blake3:weight") == 1 && props.count("blake3:bias") == 1,
        "per-tensor entries are keyed \"<algorithm>:<name>\"");
  Check(props.at("blake3:weight") == pool.ContentHash("weight"),
        "exported digest matches TensorPool::ContentHash");
  Check(props.at("blake3:weight").size() == 64,
        "exported digest is a 32-byte hex string");
  Check(props.count("blake3:__all__") == 1,
        "a whole-pool summary digest is also exported");

  // Recomputed independently of AttachContentHashMetadata's own bookkeeping,
  // the way an external verifier would: sort by name, concatenate
  // "<name>\0<hex>\n", hash under the same algorithm.
  std::string concat;
  concat += "bias";
  concat.push_back('\0');
  concat += props.at("blake3:bias");
  concat.push_back('\n');
  concat += "weight";
  concat.push_back('\0');
  concat += props.at("blake3:weight");
  concat.push_back('\n');
  Check(props.at("blake3:__all__") ==
            ToHex(RawDigest(HashAlgorithm::kBlake3, concat)),
        "__all__ is independently reproducible from the per-tensor digests");

  // A different HashAlgorithm changes both the key prefix and the values.
  onnx::ModelProto model2 = model;
  model2.mutable_graph()->clear_metadata_props();
  pool.SetHashAlgorithm(HashAlgorithm::kSha256);
  AttachContentHashMetadata(pool, model2);
  bool has_sha_key = false;
  for (const auto& kv : model2.graph().metadata_props()) {
    if (kv.key() == "sha256:weight") has_sha_key = true;
  }
  Check(has_sha_key, "switching HashAlgorithm changes the exported key prefix");

  std::remove(path.c_str());
}

}  // namespace

int main() {
  TestAdoptAndHydrateSingleTensor();
  TestStringTensorIsNotAdopted();
  TestExportModelWithSafetensors();
  TestUnnamedAttributeTensor();
  TestImportModelWithSafetensorsHydrateAll();
  TestImportModelWithSafetensorsLazy();
  TestAttachContentHashMetadata();

  if (g_failures == 0) {
    std::printf("tensor_pool_bridge_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "tensor_pool_bridge_test: %d failure(s)\n", g_failures);
  return 1;
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises the self-describing single-file archive functions
 * (EmbedModel/ExtractModel in tensor_pool_bridge.h, and the format-specific
 * SaveModelAsSafetensors/LoadModelFromSafetensors there plus
 * SaveModelAsGGUF/LoadModelFromGGUF in tensor_pool_gguf_bridge.h). Needs
 * onnx configured, so it's built against the fully-configured `onnxsim`
 * CMake target, same as tensor_pool_bridge_test.cpp and
 * tensor_pool_gguf_bridge_test.cpp.
 */
#include <onnx/onnx_pb.h>
#include <unistd.h>

#include <cstdio>
#include <string>
#include <vector>

#include "tensor_pool_gguf_bridge.h"

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
  return "/tmp/onnxsim_tensor_pool_archive_test_" + std::to_string(::getpid()) +
         suffix;
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

void TestSafetensorsRoundTrip() {
  onnx::ModelProto model;
  auto* g = model.mutable_graph();
  *g->add_initializer() = MakeInitializer("w1", onnx::TensorProto::FLOAT, {2},
                                          std::string(8, '\x11'));
  *g->add_initializer() = MakeInitializer("w2", onnx::TensorProto::INT64, {1},
                                          std::string(8, '\x22'));

  std::string path = TempPath(".safetensors");
  TensorPool save_pool;
  SaveModelAsSafetensors(model, path, save_pool);

  // The archive holds the weights AND the embedded model blob, all under
  // one path -- no separate .onnx file involved anywhere in this test.
  Check(save_pool.Find("w1") != nullptr, "w1 pooled");
  Check(save_pool.Find("w2") != nullptr, "w2 pooled");
  Check(save_pool.Find(kEmbeddedModelKey) != nullptr, "model blob pooled");

  TensorPool load_pool;
  onnx::ModelProto loaded;
  bool ok = LoadModelFromSafetensors(path, &loaded, load_pool);
  Check(ok, "LoadModelFromSafetensors succeeds");
  Check(loaded.graph().initializer_size() == 2, "both initializers present");
  Check(loaded.graph().initializer(0).name() == "w1" &&
            loaded.graph().initializer(0).has_raw_data() &&
            loaded.graph().initializer(0).raw_data() == std::string(8, '\x11'),
        "w1 hydrated with correct bytes");
  Check(loaded.graph().initializer(1).name() == "w2" &&
            loaded.graph().initializer(1).raw_data() == std::string(8, '\x22'),
        "w2 hydrated with correct bytes");
  Check(loaded.graph().initializer(0).data_location() ==
            onnx::TensorProto::DEFAULT,
        "hydrated tensor is DEFAULT-located, not left EXTERNAL");

  // A plain weights-only file (no archive on top) has no embedded model --
  // must report false, not throw or silently return an empty model.
  TensorPool bare_pool;
  bare_pool.Add("just_a_tensor", ONNX_FLOAT, {1}, std::string(4, '\0'));
  std::string bare_path = TempPath("_bare.safetensors");
  bare_pool.SaveSafetensors(bare_path);
  TensorPool p2;
  onnx::ModelProto m2;
  Check(!LoadModelFromSafetensors(bare_path, &m2, p2),
        "a plain weights-only file has no embedded model -> false");

  std::remove(path.c_str());
  std::remove(bare_path.c_str());
}

void TestGGUFRoundTrip() {
  onnx::ModelProto model;
  *model.mutable_graph()->add_initializer() = MakeInitializer(
      "w1", onnx::TensorProto::FLOAT, {3}, std::string(12, '\x33'));

  std::string path = TempPath(".gguf");
  TensorPool save_pool;
  SaveModelAsGGUF(model, path, save_pool,
                  {{"general.architecture", "onnxsim-test"}});

  TensorPool load_pool;
  onnx::ModelProto loaded;
  std::vector<std::string> skipped;
  bool ok = LoadModelFromGGUF(path, &loaded, load_pool, true, &skipped);
  Check(ok, "LoadModelFromGGUF succeeds");
  Check(skipped.empty(),
        "nothing skipped -- the model blob is a raw I8 "
        "entry, never quantized");
  Check(loaded.graph().initializer_size() == 1, "initializer present");
  Check(loaded.graph().initializer(0).raw_data() == std::string(12, '\x33'),
        "w1 hydrated with correct bytes");

  std::remove(path.c_str());
}

void TestLazyExtract() {
  // hydrate_all=false: the extracted model keeps EmbedModel's own symbolic
  // EXTERNAL markers (location-only, no offset/length -- see
  // tensor_pool_bridge.h's top comment on why), resolved on demand.
  onnx::ModelProto model;
  *model.mutable_graph()->add_initializer() = MakeInitializer(
      "lazy", onnx::TensorProto::FLOAT, {1}, std::string(4, '\x44'));
  std::string path = TempPath("_lazy.safetensors");
  TensorPool save_pool;
  SaveModelAsSafetensors(model, path, save_pool);

  TensorPool load_pool;
  onnx::ModelProto loaded;
  LoadModelFromSafetensors(path, &loaded, load_pool, /*hydrate_all=*/false);
  Check(!loaded.graph().initializer(0).has_raw_data(),
        "lazy extract leaves raw_data unset");
  Check(loaded.graph().initializer(0).data_location() ==
            onnx::TensorProto::EXTERNAL,
        "lazy extract leaves the symbolic EXTERNAL marker in place");

  bool hydrated = HydrateTensorProto(
      "lazy", *loaded.mutable_graph()->mutable_initializer(0), load_pool);
  Check(hydrated &&
            loaded.graph().initializer(0).raw_data() == std::string(4, '\x44'),
        "on-demand HydrateTensorProto still recovers the bytes afterwards");
  std::remove(path.c_str());
}

}  // namespace

int main() {
  TestSafetensorsRoundTrip();
  TestGGUFRoundTrip();
  TestLazyExtract();

  if (g_failures == 0) {
    std::printf("tensor_pool_archive_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "tensor_pool_archive_test: %d failure(s)\n", g_failures);
  return 1;
}

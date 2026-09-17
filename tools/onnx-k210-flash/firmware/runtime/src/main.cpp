// Generic K210 kmodel runtime -- the K210 counterpart to
// onnx-cardputer-flash's firmware/runtime: build and flash this ONCE,
// then every model swap after that is a Web Serial write of a plain
// .kmodel file's bytes at a fixed flash offset, no rebuild.
//
// Unlike ESP32-S3 (esp_partition_mmap lets a running app read flash
// in-place), K210 has no flash-mapped execution model at all -- the mask
// ROM copies the whole firmware image into the K210's 8MB SRAM and runs it
// from there (see onnx-k210-flash's own ISP work: SRAM_STUB_ADDRESS
// 0x80000000 is SRAM, not flash). So this reads the model off SPI flash
// into a RAM buffer at boot via w25qxx (the same driver, same read call,
// kendryte-standalone-demo/kpu's own main.c uses for its
// LOAD_KMODEL_FROM_FLASH path), rather than mapping it.
//
// This calls nncase's own nncase::runtime::interpreter C++ API directly
// (not the simplified kpu_load_kmodel/kpu_run_kmodel C wrapper in
// lib/nncase/nncase.cpp) specifically so it can ask the loaded model its
// own input shape/dtype (interp.input_shape(0)/input_desc(0)) instead of
// needing a model-specific input size hardcoded at build time -- the same
// "runtime introspects the model, not the other way around" property
// onnx-cardputer-flash's TFLite Micro runtime has via
// interpreter->input(0)->dims. nncase_v1.cpp (this SDK's own C-wrapper
// implementation) is what this is modeled on -- it calls the exact same
// interpreter API internally, just without exposing the shape query.
//
// Status: compiled and linked for real against kendryte-standalone-sdk
// (toolchain: kendryte-gnu-toolchain 8.2.0, riscv64-unknown-elf) -- see
// ../README.md for the exact build/verification. NOT run on a real K210
// board: no device attached to the environment this was written in.

#include <cstdio>
#include <cstring>

// <nncase/runtime/interpreter.h>, matching how nncase_v1.cpp itself
// includes it: needs lib/nncase/v1/include on the include path. This
// SDK's own top-level CMakeLists (marked "DO NOT MODIFY") never adds
// that exact directory -- its recursive header_directories() glob adds
// each *.h's own containing directory instead, and lib/nncase/v1's
// internal headers use this <nncase/...> angle-bracket form among
// themselves throughout (not consistently quoted-relative, so this
// file can't dodge it as a v1-only include the way it could avoid the
// v0/v1 runtime/interpreter.h name collision alone). See
// ../README.md for the -DCMAKE_CXX_FLAGS=... this needs at configure
// time to add that one missing include root.
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

#include "plic.h"
#include "sysctl.h"
#include "uarths.h"
extern "C" {
#include "w25qxx.h" // vendored as-is (kendryte-standalone-demo/kpu's own copy) -- it has no extern "C" guard of its own, hence the wrapper here rather than editing the vendored file.
}

using namespace nncase;
using namespace nncase::runtime;

namespace {

constexpr uint32_t kModelFlashAddress = 0x00C00000; // Kendryte's own kfpkg convention (e.g. kendryte-standalone-demo/kpu's flash-list.json) -- reused rather than inventing a new offset.
constexpr size_t kModelBufferSize = 2 * 1024 * 1024; // must fit the real kmodel; every candidate in ../README.md's table does with room to spare.
constexpr uint32_t PLL0_OUTPUT_FREQ = 800000000UL;

alignas(256) uint8_t g_model_buffer[kModelBufferSize];

const char *datatype_name(datatype_t t) {
  switch (t) {
    case dt_uint8: return "uint8";
    case dt_uint16: return "uint16";
    case dt_uint32: return "uint32";
    case dt_int8: return "int8";
    case dt_int16: return "int16";
    case dt_int32: return "int32";
    case dt_float32: return "float32";
    default: return "?";
  }
}

void print_shape(const runtime_shape_t &shape) {
  for (size_t i = 0; i < shape.size(); i++) {
    printf("%zu", shape[i]);
    if (i + 1 < shape.size()) printf("x");
  }
}

} // namespace

int main() {
  sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
  uarths_init();
  plic_init();
  sysctl_enable_irq();

  printf("onnx-k210-flash runtime\n");
  printf("reading model from flash @0x%08x (%zu bytes)...\n", kModelFlashAddress, kModelBufferSize);

  w25qxx_init(3, 0);
  w25qxx_enable_quad_mode();
  w25qxx_read_data(kModelFlashAddress, g_model_buffer, kModelBufferSize, W25QXX_QUAD_FAST);

  // nncase's own kmodel identifier ('KMDL' packed as a little-endian u32)
  // -- checked before ever handing the buffer to the interpreter, since
  // right after flashing this runtime (before any model is flashed) the
  // model region is just erased/unrelated flash content.
  uint32_t identifier;
  memcpy(&identifier, g_model_buffer, sizeof(identifier));
  if (identifier != 'KMDL') {
    printf("no model flashed yet at 0x%08x (identifier 0x%08x, expected 0x4b4d444c)\n",
           kModelFlashAddress, identifier);
    printf("flash a .kmodel there over Web Serial, then reset.\n");
    while (1) {}
  }

  interpreter interp;
  auto load_result = interp.load_model({reinterpret_cast<const gsl::byte *>(g_model_buffer), kModelBufferSize});
  if (!load_result.is_ok()) {
    printf("model load failed: %s\n", load_result.unwrap_err().message().c_str());
    while (1) {}
  }

  printf("model loaded ok. inputs: %zu  outputs: %zu\n", interp.inputs_size(), interp.outputs_size());

  // Sanity inference over zeroed input(s) for every input tensor, sized
  // from the model's own declared shape/dtype -- proves the load-from-
  // flash + interpreter path actually executes this model on this
  // device. Says nothing about accuracy; that needs real sensor data,
  // wired up per-model (this runtime is deliberately generic).
  for (size_t i = 0; i < interp.inputs_size(); i++) {
    auto &shape = interp.input_shape(i);
    auto type = interp.input_desc(i).datatype;
    printf("  in[%zu]: ", i);
    print_shape(shape);
    printf(" type=%s\n", datatype_name(type));

    size_t nbytes = get_bytes(type, shape);
    static uint8_t input_buf[512 * 1024]; // generous fixed scratch; covers every candidate model's input tensor
    if (nbytes > sizeof(input_buf)) {
      printf("  input %zu needs %zu bytes, more than this runtime's %zu-byte scratch buffer\n",
             i, nbytes, sizeof(input_buf));
      while (1) {}
    }
    memset(input_buf, 0, nbytes);

    auto tensor_result = hrt::create(type, shape, {reinterpret_cast<gsl::byte *>(input_buf), nbytes}, false, hrt::pool_shared);
    if (!tensor_result.is_ok()) {
      printf("  input tensor creation failed: %s\n", tensor_result.unwrap_err().message().c_str());
      while (1) {}
    }
    auto &input_tensor = tensor_result.unwrap();
    if (auto r = hrt::sync(input_tensor, hrt::sync_write_back); !r.is_ok()) {
      printf("  input tensor sync failed: %s\n", r.unwrap_err().message().c_str());
      while (1) {}
    }
    if (auto r = interp.input_tensor(i, input_tensor); !r.is_ok()) {
      printf("  binding input tensor %zu failed: %s\n", i, r.unwrap_err().message().c_str());
      while (1) {}
    }
  }

  auto run_result = interp.run();
  if (!run_result.is_ok()) {
    printf("test inference: FAILED (%s)\n", run_result.unwrap_err().message().c_str());
    while (1) {}
  }
  printf("test inference: OK\n");

  for (size_t i = 0; i < interp.outputs_size(); i++) {
    auto tensor_result = interp.output_tensor(i);
    if (!tensor_result.is_ok()) continue;
    auto host_result = tensor_result.unwrap().as_host();
    if (!host_result.is_ok()) continue;
    auto map_result = hrt::map(host_result.unwrap(), hrt::map_read);
    if (!map_result.is_ok()) continue;
    auto buffer = map_result.unwrap().buffer();
    printf("  out[%zu]: %zu bytes\n", i, buffer.size_bytes());
  }

  while (1) {}
  return 0;
}

// onnx-deploy: drive an optimum-onnx export directory's encoder/decoder(-with-past)
// ONNX files through a greedy autoregressive generate() loop, in plain C++,
// via the onnx_deploy C ABI (onnx_deploy_c_api.h) -- the same swappable-libort
// interface any other language would use. No tokenizer -- pass/receive token
// ids directly (get them from transformers.AutoTokenizer separately). See
// ../README.md for the design and its scope.

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "onnx_deploy/onnx_deploy_c_api.h"

namespace {

struct Args {
  std::string libort_path;
  std::string model_dir;
  std::vector<int64_t> input_ids;
  int64_t max_new_tokens = 32;
  int64_t eos_token_id = -1;
  int64_t decoder_start_token_id = 0;
  std::string execution_provider = "cpu";
  int cuda_device_id = 0;
};

[[noreturn]] void Usage(const char* prog) {
  std::fprintf(stderr,
      "usage: %s --libort PATH <export_dir> <id1,id2,...> [--max-new-tokens N]\n"
      "          [--eos-token-id N] [--decoder-start-token-id N]\n"
      "          [--execution-provider cpu|cuda|webgpu] [--cuda-device-id N]\n\n"
      "--libort PATH points at the libonnxruntime shared library to load at\n"
      "runtime (e.g. an extracted onnxruntime-linux-x64-*.tgz's lib/libonnxruntime.so) --\n"
      "any build works, nothing about this tool is compiled against a specific one.\n\n"
      "--execution-provider cuda requires --libort to point at a CUDA-enabled ORT\n"
      "build (e.g. an onnxruntime-linux-x64-gpu-*.tgz release) and a CUDA-capable\n"
      "GPU/driver -- otherwise this fails cleanly with ORT's own error, not a crash.\n"
      "--execution-provider webgpu requires --libort to point at an ORT build with\n"
      "the native WebGPU EP compiled in (--use_webgpu from source as of ORT 1.23.0 --\n"
      "the plain prebuilt release tarballs don't include it) and a GPU; this is ORT's\n"
      "own native WebGPU EP (Dawn), not onnxruntime-web's browser one the wasm/ build\n"
      "uses -- same clean-failure behavior if unavailable.\n\n"
      "<export_dir> must contain decoder_model.onnx + decoder_with_past_model.onnx\n"
      "(and encoder_model.onnx for seq2seq models) -- the optimum-onnx\n"
      "no_post_process=True export shape. <id1,id2,...> are token ids (from a\n"
      "tokenizer run separately), comma-separated, no spaces.\n",
      prog);
  std::exit(1);
}

std::vector<int64_t> ParseIds(const std::string& csv) {
  std::vector<int64_t> ids;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) ids.push_back(std::stoll(item));
  return ids;
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  std::vector<std::string> positional;

  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) Usage(argv[0]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--libort") a.libort_path = need(i);
    else if (arg == "--max-new-tokens") a.max_new_tokens = std::stoll(need(i));
    else if (arg == "--eos-token-id") a.eos_token_id = std::stoll(need(i));
    else if (arg == "--decoder-start-token-id") a.decoder_start_token_id = std::stoll(need(i));
    else if (arg == "--execution-provider") a.execution_provider = need(i);
    else if (arg == "--cuda-device-id") a.cuda_device_id = std::stoi(need(i));
    else if (arg == "-h" || arg == "--help") Usage(argv[0]);
    else if (!arg.empty() && arg[0] == '-') {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      Usage(argv[0]);
    } else {
      positional.push_back(arg);
    }
  }
  if (a.libort_path.empty() || positional.size() != 2) Usage(argv[0]);
  a.model_dir = positional[0];
  a.input_ids = ParseIds(positional[1]);
  if (a.input_ids.empty()) Usage(argv[0]);
  return a;
}

[[noreturn]] void Die(const char* what, char* err) {
  std::fprintf(stderr, "error: %s: %s\n", what, err ? err : "(no message)");
  onnx_deploy_free_string(err);
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);

  char* err = nullptr;
  if (onnx_deploy_load_ort(args.libort_path.c_str(), &err) != ONNX_DEPLOY_OK) Die("onnx_deploy_load_ort", err);

  OnnxDeployPipeline* pipeline =
      onnx_deploy_create_ex(args.model_dir.c_str(), args.execution_provider.c_str(), args.cuda_device_id, &err);
  if (!pipeline) Die("onnx_deploy_create_ex", err);
  std::printf("loaded %s pipeline from %s (via %s, execution provider: %s)\n",
              onnx_deploy_is_seq2seq(pipeline) ? "seq2seq" : "decoder-only", args.model_dir.c_str(),
              args.libort_path.c_str(), args.execution_provider.c_str());

  int64_t* out_ids = nullptr;
  size_t out_count = 0;
  OnnxDeployStatus status =
      onnx_deploy_generate(pipeline, args.input_ids.data(), args.input_ids.size(), args.max_new_tokens,
                            args.eos_token_id, args.decoder_start_token_id, &out_ids, &out_count, &err);
  if (status != ONNX_DEPLOY_OK) {
    onnx_deploy_destroy(pipeline);
    Die("onnx_deploy_generate", err);
  }

  std::printf("generated %zu token(s):", out_count);
  for (size_t i = 0; i < out_count; ++i) std::printf(" %lld", static_cast<long long>(out_ids[i]));
  std::printf("\n");

  onnx_deploy_free_ids(out_ids);
  onnx_deploy_destroy(pipeline);
  return 0;
}

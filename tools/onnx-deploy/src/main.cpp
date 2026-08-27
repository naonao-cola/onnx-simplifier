// onnx-deploy: drive an optimum-onnx export directory's encoder/decoder(-with-past)
// ONNX files through a greedy autoregressive generate() loop, in plain C++
// via onnx_deploy::KvCachePipeline. No tokenizer -- pass/receive token ids
// directly (get them from transformers.AutoTokenizer separately). See
// ../README.md for the design and its scope.

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "onnx_deploy/kv_cache_pipeline.h"

namespace {

struct Args {
  std::string model_dir;
  std::vector<int64_t> input_ids;
  int64_t max_new_tokens = 32;
  int64_t eos_token_id = -1;
  int64_t decoder_start_token_id = 0;
};

[[noreturn]] void Usage(const char* prog) {
  std::fprintf(stderr,
      "usage: %s <export_dir> <id1,id2,...> [--max-new-tokens N]\n"
      "          [--eos-token-id N] [--decoder-start-token-id N]\n\n"
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
  if (argc < 3) Usage(argv[0]);
  Args a;
  a.model_dir = argv[1];
  a.input_ids = ParseIds(argv[2]);

  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) Usage(argv[0]);
    return argv[++i];
  };
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--max-new-tokens") a.max_new_tokens = std::stoll(need(i));
    else if (arg == "--eos-token-id") a.eos_token_id = std::stoll(need(i));
    else if (arg == "--decoder-start-token-id") a.decoder_start_token_id = std::stoll(need(i));
    else if (arg == "-h" || arg == "--help") Usage(argv[0]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      Usage(argv[0]);
    }
  }
  if (a.input_ids.empty()) Usage(argv[0]);
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-deploy");

  try {
    onnx_deploy::KvCachePipeline pipeline(env, args.model_dir);
    std::printf("loaded %s pipeline from %s\n", pipeline.is_seq2seq() ? "seq2seq" : "decoder-only",
                args.model_dir.c_str());

    onnx_deploy::GenerationConfig config;
    config.max_new_tokens = args.max_new_tokens;
    config.eos_token_id = args.eos_token_id;
    config.decoder_start_token_id = args.decoder_start_token_id;

    std::vector<int64_t> generated = pipeline.Generate(args.input_ids, config);

    std::printf("generated %zu token(s):", generated.size());
    for (int64_t id : generated) std::printf(" %lld", static_cast<long long>(id));
    std::printf("\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}

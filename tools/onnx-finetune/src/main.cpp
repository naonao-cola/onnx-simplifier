// onnx-finetune: run an ONNX Runtime on-device training loop against
// pre-generated training artifacts (see scripts/generate_artifacts.py) and
// export the result as a normal inference-ready ONNX model.
//
// No Python at runtime: this links only onnxruntime's training C++ API.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "onnxruntime_training_cxx_api.h"

namespace {

struct Args {
  std::string artifacts_dir;
  std::string train_input;
  std::string train_target;
  int64_t input_dim = 0;
  int64_t target_dim = 0;
  int64_t num_samples = 0;
  int64_t batch_size = 8;
  int64_t epochs = 10;
  double lr = 1e-3;
  std::string output_model;
  std::string output_names;   // comma-separated
  std::string save_checkpoint;
  int log_every = 50;
};

[[noreturn]] void Usage(const char* prog) {
  std::fprintf(stderr,
      "usage: %s --artifacts-dir DIR --train-input FILE --train-target FILE\n"
      "          --input-dim N --target-dim N --num-samples N\n"
      "          --output-model FILE --output-names name1,name2,...\n"
      "          [--batch-size N] [--epochs N] [--lr F] [--save-checkpoint FILE]\n"
      "          [--log-every N]\n\n"
      "Trains against artifacts produced by scripts/generate_artifacts.py.\n"
      "--train-input/--train-target are raw contiguous float32 files:\n"
      "  input:  num_samples * input_dim  floats\n"
      "  target: num_samples * target_dim floats\n",
      prog);
  std::exit(1);
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) out.push_back(item);
  return out;
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) Usage(argv[0]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--artifacts-dir") a.artifacts_dir = need(i);
    else if (arg == "--train-input") a.train_input = need(i);
    else if (arg == "--train-target") a.train_target = need(i);
    else if (arg == "--input-dim") a.input_dim = std::stoll(need(i));
    else if (arg == "--target-dim") a.target_dim = std::stoll(need(i));
    else if (arg == "--num-samples") a.num_samples = std::stoll(need(i));
    else if (arg == "--batch-size") a.batch_size = std::stoll(need(i));
    else if (arg == "--epochs") a.epochs = std::stoll(need(i));
    else if (arg == "--lr") a.lr = std::stod(need(i));
    else if (arg == "--output-model") a.output_model = need(i);
    else if (arg == "--output-names") a.output_names = need(i);
    else if (arg == "--save-checkpoint") a.save_checkpoint = need(i);
    else if (arg == "--log-every") a.log_every = std::stoi(need(i));
    else if (arg == "-h" || arg == "--help") Usage(argv[0]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      Usage(argv[0]);
    }
  }
  if (a.artifacts_dir.empty() || a.train_input.empty() || a.train_target.empty() ||
      a.input_dim <= 0 || a.target_dim <= 0 || a.num_samples <= 0 ||
      a.output_model.empty() || a.output_names.empty()) {
    Usage(argv[0]);
  }
  return a;
}

std::vector<float> ReadRawFloats(const std::string& path, size_t expected_count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<float> data(expected_count);
  f.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
  if (!f) {
    std::fprintf(stderr, "error: %s is shorter than expected (%zu floats)\n", path.c_str(), expected_count);
    std::exit(1);
  }
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  std::vector<std::string> output_names_vec = Split(args.output_names, ',');

  std::vector<float> inputs = ReadRawFloats(args.train_input, static_cast<size_t>(args.num_samples) * args.input_dim);
  std::vector<float> targets = ReadRawFloats(args.train_target, static_cast<size_t>(args.num_samples) * args.target_dim);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-finetune");
  Ort::SessionOptions session_options;

  auto checkpoint_state = Ort::CheckpointState::LoadCheckpoint(args.artifacts_dir + "/checkpoint");
  Ort::TrainingSession train_session(
      env, session_options, checkpoint_state,
      args.artifacts_dir + "/training_model.onnx",
      args.artifacts_dir + "/eval_model.onnx",
      args.artifacts_dir + "/optimizer_model.onnx");

  train_session.SetLearningRate(static_cast<float>(args.lr));

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> order(args.num_samples);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(42);

  std::vector<float> batch_input(static_cast<size_t>(args.batch_size) * args.input_dim);
  std::vector<float> batch_target(static_cast<size_t>(args.batch_size) * args.target_dim);

  int64_t global_step = 0;
  for (int64_t epoch = 0; epoch < args.epochs; ++epoch) {
    std::shuffle(order.begin(), order.end(), rng);

    for (int64_t start = 0; start + args.batch_size <= args.num_samples; start += args.batch_size) {
      for (int64_t b = 0; b < args.batch_size; ++b) {
        int64_t src = order[start + b];
        std::copy_n(inputs.begin() + src * args.input_dim, args.input_dim,
                    batch_input.begin() + b * args.input_dim);
        std::copy_n(targets.begin() + src * args.target_dim, args.target_dim,
                    batch_target.begin() + b * args.target_dim);
      }

      std::vector<int64_t> in_shape = {args.batch_size, args.input_dim};
      std::vector<int64_t> tgt_shape = {args.batch_size, args.target_dim};
      std::vector<Ort::Value> step_inputs;
      step_inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info, batch_input.data(), batch_input.size(), in_shape.data(), in_shape.size()));
      step_inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info, batch_target.data(), batch_target.size(), tgt_shape.data(), tgt_shape.size()));

      auto step_outputs = train_session.TrainStep(step_inputs);
      train_session.OptimizerStep();
      train_session.LazyResetGrad();

      if (args.log_every > 0 && global_step % args.log_every == 0 && !step_outputs.empty()) {
        float loss = *step_outputs[0].GetTensorData<float>();
        std::printf("epoch %lld step %lld loss %.6f\n",
                    static_cast<long long>(epoch), static_cast<long long>(global_step), loss);
      }
      ++global_step;
    }
  }

  train_session.ExportModelForInferencing(args.output_model, output_names_vec);
  std::printf("wrote fine-tuned inference model -> %s\n", args.output_model.c_str());

  if (!args.save_checkpoint.empty()) {
    Ort::CheckpointState::SaveCheckpoint(checkpoint_state, args.save_checkpoint, /*include_optimizer_state=*/true);
    std::printf("wrote checkpoint -> %s\n", args.save_checkpoint.c_str());
  }

  return 0;
}

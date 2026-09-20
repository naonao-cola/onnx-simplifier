#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: onnxsim_android_runner MODEL.onnx INPUT.f32 OUTPUT.f32\n";
    return 2;
  }

  try {
    std::ifstream input_file(argv[2], std::ios::binary);
    if (!input_file) throw std::runtime_error("cannot open input tensor");
    std::vector<char> input((std::istreambuf_iterator<char>(input_file)), {});
    if (input.size() != 4 * sizeof(float)) {
      throw std::runtime_error("input tensor must contain exactly four float32 values");
    }
    std::array<float, 4> input_values{};
    std::memcpy(input_values.data(), input.data(), sizeof(input_values));

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxsim-android-smoke");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    Ort::Session session(env, argv[1], options);
    const std::array<int64_t, 2> shape{1, 4};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, input_values.data(),
                                                   input_values.size(), shape.data(),
                                                   shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                               output_names, 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
      throw std::runtime_error("model did not return one tensor");
    }
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != 4) {
      throw std::runtime_error("model output must be four float32 values");
    }
    const float* values = outputs[0].GetTensorData<float>();
    std::ofstream output_file(argv[3], std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(values), 4 * sizeof(float));
    if (!output_file) throw std::runtime_error("failed to write output tensor");
    std::cerr << "PASS cpu " << argv[1] << '\n';
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
  }
  return 1;
}

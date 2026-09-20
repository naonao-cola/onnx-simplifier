#include <onnxruntime_cxx_api.h>

#include <algorithm>
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
    if (input.size() % sizeof(float) != 0) {
      throw std::runtime_error("input tensor byte count is not float32 aligned");
    }
    std::vector<float> input_values(input.size() / sizeof(float));
    std::memcpy(input_values.data(), input.data(), input.size());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxsim-android-smoke");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    Ort::Session session(env, argv[1], options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
      throw std::runtime_error("only single-input, single-output models are supported");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto shape = input_info.GetShape();
    for (auto& dimension : shape) {
      if (dimension < 0) dimension = 1;
    }
    size_t expected_input_count = 1;
    for (const auto dimension : shape) expected_input_count *= static_cast<size_t>(dimension);
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        expected_input_count != input_values.size()) {
      throw std::runtime_error("model input must be a fixed-shape float32 tensor matching input.f32");
    }
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, input_values.data(),
                                                   input_values.size(), shape.data(),
                                                   shape.size());
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                               output_names, 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
      throw std::runtime_error("model did not return one tensor");
    }
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("model output must be a float32 tensor");
    }
    const float* values = outputs[0].GetTensorData<float>();
    std::ofstream output_file(argv[3], std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(values),
                      info.GetElementCount() * sizeof(float));
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

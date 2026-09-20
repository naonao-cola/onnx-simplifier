cmake_minimum_required(VERSION 3.22)
project(onnxsim_device_test LANGUAGES CXX)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a/libonnxruntime.so")

add_library(onnxsim_device_test SHARED jni_runner.cpp)
target_include_directories(onnxsim_device_test PRIVATE headers)
target_compile_features(onnxsim_device_test PRIVATE cxx_std_17)
target_link_libraries(onnxsim_device_test PRIVATE onnxruntime log neuralnetworks)

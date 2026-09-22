cmake_minimum_required(VERSION 3.22)
project(onnxsim_rpc_server LANGUAGES CXX)

# ORT_INCLUDE_DIR / ORT_LIBRARY: the extracted onnxruntime-android AAR (headers/ and
# jni/arm64-v8a/libonnxruntime.so), or any onnxruntime C++ SDK.
if(NOT DEFINED ORT_INCLUDE_DIR OR NOT DEFINED ORT_LIBRARY)
  message(FATAL_ERROR "Set ORT_INCLUDE_DIR and ORT_LIBRARY")
endif()

add_executable(onnxsim_rpc_server main.cpp)
target_compile_features(onnxsim_rpc_server PRIVATE cxx_std_17)
target_include_directories(onnxsim_rpc_server PRIVATE "${ORT_INCLUDE_DIR}")
find_package(Threads REQUIRED)
target_link_libraries(onnxsim_rpc_server PRIVATE "${ORT_LIBRARY}" Threads::Threads)
set_target_properties(onnxsim_rpc_server PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")

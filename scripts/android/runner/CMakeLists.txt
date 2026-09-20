cmake_minimum_required(VERSION 3.22)
project(onnxsim_android_runner LANGUAGES CXX)

if(NOT DEFINED ORT_INCLUDE_DIR OR NOT DEFINED ORT_LIBRARY)
  message(FATAL_ERROR "Set ORT_INCLUDE_DIR and ORT_LIBRARY to the extracted Android AAR contents")
endif()

add_executable(onnxsim_android_runner main.cpp)
target_compile_features(onnxsim_android_runner PRIVATE cxx_std_17)
target_include_directories(onnxsim_android_runner PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(onnxsim_android_runner PRIVATE "${ORT_LIBRARY}" log)
set_target_properties(onnxsim_android_runner PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

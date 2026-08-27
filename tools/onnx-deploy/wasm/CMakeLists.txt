cmake_minimum_required(VERSION 3.18)
project(onnx-deploy-wasm CXX)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure this with an Emscripten toolchain, e.g.:\n  emcmake cmake -B build\n  cmake --build build\nSee ../README.md's WASM section.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# No ONNX Runtime dependency at all here -- unlike tools/onnx-finetune/wasm
# (which statically compiles ONNX Runtime from source into the module), this
# target has zero ORT C/C++ code: every tensor computation happens in JS via
# onnxruntime-web, reached through src/onnx_deploy_wasm.cpp's Asyncify calls.
# "Swappable libort" here means the JS host picks which onnxruntime-web
# build/version/execution-provider implements Module.onnxDeployRunSession --
# see ../test/ort_web_runtime.mjs for the reference implementation.
add_executable(onnx_deploy_wasm src/onnx_deploy_wasm.cpp)
target_link_options(onnx_deploy_wasm PRIVATE
  "-lembind"
  "-sASYNCIFY=1"
  "-sMODULARIZE=1"
  "-sEXPORT_ES6=1"
  "-sEXPORT_NAME=createOnnxDeployWasmModule"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sENVIRONMENT=web,node"
)

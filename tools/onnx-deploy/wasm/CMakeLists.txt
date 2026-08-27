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
# Exception CATCHING must be enabled at both compile and link time.
# onnx_deploy_wasm.cpp's Generate() wraps GenerateImpl() in a C++ try/catch
# to turn a JS-side failure (e.g. requesting the "webgpu" EP with no
# navigator.gpu) into an explicit Promise.reject() return -- letting an
# uncaught exception escape an Asyncify-wrapped exported function was
# observed to crash the whole process instead of rejecting
# Module.generate()'s Promise, no matter how it was thrown; ThrowIfJsError
# throwing in ordinary (already-resumed) code after a successful .await()
# and being caught by that try/catch, entirely within one synchronous C++
# call stack, is what actually works reliably. Without exception catching
# enabled, that catch block would never run either. See
# ../test/run_test.mjs's webgpu-unavailable case, and ort_web_runtime.mjs's
# errorResult/__onnxDeployError for the JS-side half of this.
#
# This has to be the legacy JS-based emulation (-sDISABLE_EXCEPTION_CATCHING=0),
# NOT modern native wasm exceptions (-fwasm-exceptions) -- Emscripten
# explicitly rejects that combination ("ASYNCIFY=1 is not compatible with
# -fwasm-exceptions"), since ASYNCIFY=1 is required for val::await() itself.
target_compile_options(onnx_deploy_wasm PRIVATE "-sDISABLE_EXCEPTION_CATCHING=0")
target_link_options(onnx_deploy_wasm PRIVATE
  "-lembind"
  "-sDISABLE_EXCEPTION_CATCHING=0"
  "-sASYNCIFY=1"
  "-sMODULARIZE=1"
  "-sEXPORT_ES6=1"
  "-sEXPORT_NAME=createOnnxDeployWasmModule"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sENVIRONMENT=web,node"
)

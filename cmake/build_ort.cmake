# For MessageDifferencer::Equals
option(onnxruntime_USE_FULL_PROTOBUF "" ON)
if (EMSCRIPTEN)
  if (NOT DEFINED ONNX_CUSTOM_PROTOC_EXECUTABLE)
    message(FATAL_ERROR "ONNX_CUSTOM_PROTOC_EXECUTABLE must be set for emscripten")
  endif()

  option(onnxruntime_BUILD_WEBASSEMBLY "" ON)
  option(onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB "" ON)
  option(onnxruntime_ENABLE_WEBASSEMBLY_SIMD "" OFF)
  option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING "" ON)
  option(onnxruntime_ENABLE_WEBASSEMBLY_THREADS "" OFF)
  option(onnxruntime_BUILD_UNIT_TESTS "" OFF)
  set(onnxruntime_EMSCRIPTEN_SETTINGS "MALLOC=dlmalloc")
  set(onnxruntime_ENABLE_WEBASSEMBLY_THREADS ON)
  set(onnxruntime_ENABLE_WEBASSEMBLY_SIMD ON)

  # For custom onnx target in onnx optimizer
  set(ONNX_TARGET_NAME onnxruntime_webassembly)
else()
  # Option 1 (mirror the Python wheel): build ONNX Runtime -- and, below, the
  # whole onnx/protobuf stack -- as STATIC libraries so they are absorbed into
  # the single onnxsim_c shared object instead of living in a separate
  # libonnxruntime.so. That leaves exactly one copy of onnx in the process (so
  # protobuf's generated descriptors register once) while letting
  # onnx::ModelProto's vtable/RTTI resolve inside onnxsim_c at link time rather
  # than needing to be exported across a shared-library boundary -- which ONNX
  # 1.23's hidden visibility no longer allows, and is what makes the Rust native
  # build fail with "undefined symbol: typeinfo for onnx::ModelProto".
  #
  # The old "linked twice" hazard (onnx pulled into both libonnxruntime.so and
  # onnxsim) does not apply here: with ORT static there is no separate ORT
  # shared object, so onnx is absorbed once, into onnxsim_c.
  option(onnxruntime_BUILD_SHARED_LIB "" OFF)
endif()
set(ONNXRUNTIME_INCLUDE_DIR third_party/onnxruntime-1.28.0/include)
add_subdirectory(third_party/onnxruntime-1.28.0/cmake)

# EMSCRIPTEN keeps its bundled-static-archive flow above; the native path now
# builds everything static (CMAKE_POSITION_INDEPENDENT_CODE is already ON, so
# the static objects still link into the shared onnxsim_c). Do NOT set
# BUILD_SHARED_LIBS ON here anymore.

# --- OPT1-DIAG: one-time probe so CI reveals how ORT exposes itself when built
# static (the ORT source can't be inspected in the dev sandbox). Remove once the
# link wiring below is settled.
if (NOT EMSCRIPTEN)
  foreach(_c onnxruntime onnxruntime_session onnxruntime_framework onnxruntime_graph
             onnxruntime_optimizer onnxruntime_providers onnxruntime_mlas
             onnxruntime_common onnxruntime_flatbuffers onnxruntime_util onnxruntime_lora)
    if (TARGET ${_c})
      get_target_property(_t ${_c} TYPE)
      message(STATUS "OPT1-DIAG target ${_c} = ${_t}")
    endif()
  endforeach()
  message(STATUS "OPT1-DIAG onnxruntime_INTERNAL_LIBRARIES=[${onnxruntime_INTERNAL_LIBRARIES}]")
  message(STATUS "OPT1-DIAG onnxruntime_EXTERNAL_LIBRARIES=[${onnxruntime_EXTERNAL_LIBRARIES}]")
  message(STATUS "OPT1-DIAG onnxruntime_libs=[${onnxruntime_libs}]")
endif()

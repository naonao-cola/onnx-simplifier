/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace onnxsim {

// Import operator schemas for a set of ONNX Runtime "com.microsoft" contrib
// operators (QLinearAdd and related quantized/QNN ops) into the ONNX operator
// schema registry.
//
// ONNX's own shape inference only knows about operators that have a schema
// registered in the ONNX registry. Quantized models produced by ONNX Runtime
// use contrib operators such as QLinearAdd, whose schemas live outside ONNX,
// so shape deduction stops as soon as it reaches one of them
// (https://github.com/onnxsim/onnxsim/issues/245). Registering the schemas
// here lets onnx::shape_inference::InferShapes propagate shapes and types
// through these ops, which in turn unlocks further simplification.
//
// The registration is performed at most once per process and never overrides a
// schema that is already registered (for example when onnxsim is built with the
// built-in ONNX Runtime, which registers its own contrib schemas). It is safe
// to call this function multiple times and from any of the simplification entry
// points.
void RegisterContribOpSchemas();

}  // namespace onnxsim

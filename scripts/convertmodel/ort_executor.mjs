// The JavaScript half of the onnxruntime-web constant-folding executor.
//
// When onnxsim's WASM module is built with ONNXSIM_WASM_ORT_WEB, its C++
// constant folder does not link ONNX Runtime; instead JsModelExecutor::_Run
// (js_model_executor.cpp) calls a runner registered on the Emscripten Module as
// `Module.onnxsimOrtWebRun`. This file builds that runner on top of an
// already-loaded `onnxruntime-web` module.
//
// Contract (must match js_model_executor.cpp):
//   runner(modelBytes: Uint8Array, inputs: InTensor[]) => Promise<OutMap>
//     InTensor = { name: string, dataType: number, dims: number[], data: Uint8Array }
//     OutMap   = { [outputName: string]: { dataType, dims: number[], data: Uint8Array } }
// `dataType` is the ONNX TensorProto.DataType enum value; `data` is the tensor
// content as raw little-endian bytes of the element type.

// ONNX TensorProto.DataType enum value -> [onnxruntime-web tensor type string,
// TypedArray constructor for that type]. Kept in lock-step with the dtype set
// TensorProtoRawBytes() in js_model_executor.cpp emits/accepts.
const ONNX_DTYPE_TO_ORT = {
  1: ["float32", Float32Array], // FLOAT
  2: ["uint8", Uint8Array], // UINT8
  3: ["int8", Int8Array], // INT8
  4: ["uint16", Uint16Array], // UINT16
  5: ["int16", Int16Array], // INT16
  6: ["int32", Int32Array], // INT32
  7: ["int64", BigInt64Array], // INT64
  9: ["bool", Uint8Array], // BOOL (onnxruntime-web bool data is a Uint8Array)
  11: ["float64", Float64Array], // DOUBLE
  13: ["uint64", BigUint64Array], // UINT64
};

// onnxruntime-web tensor type string -> ONNX TensorProto.DataType enum value.
const ORT_TYPE_TO_ONNX = {
  float32: 1,
  uint8: 2,
  int8: 3,
  uint16: 4,
  int16: 5,
  int32: 6,
  int64: 7,
  bool: 9,
  float64: 11,
  uint64: 13,
};

// Reinterpret a Uint8Array's bytes as `Ctor` elements. The C++ side hands us
// freshly-allocated (0-offset) buffers, but fall back to a copy if an offset is
// ever unaligned for the element size.
function typedFromBytes(Ctor, u8) {
  const bpe = Ctor.BYTES_PER_ELEMENT;
  if (u8.byteOffset % bpe === 0 && u8.byteLength % bpe === 0) {
    return new Ctor(u8.buffer, u8.byteOffset, u8.byteLength / bpe);
  }
  const copy = u8.slice(); // fresh, 0-offset ArrayBuffer
  return new Ctor(copy.buffer, 0, copy.byteLength / bpe);
}

// Raw little-endian bytes backing a TypedArray, without copying.
function bytesFromTyped(ta) {
  return new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
}

// Build the runner that JsModelExecutor calls. `ort` is a loaded onnxruntime-web
// module. Sessions are created per fold group with graph optimization disabled,
// mirroring the built-in CppModelExecutor (ORT_DISABLE_ALL): we want ORT to
// execute the sub-model as-is, not rewrite it.
//
// `providers` lets a browser prefer e.g. WebGPU and fall back to wasm; folding
// correctness does not depend on the provider.
export function makeOrtRunner(ort, { providers = ["wasm"] } = {}) {
  return async function onnxsimOrtWebRun(modelBytes, inputs) {
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: providers,
      graphOptimizationLevel: "disabled",
    });

    const feeds = {};
    for (const inp of inputs) {
      const entry = ONNX_DTYPE_TO_ORT[inp.dataType];
      if (!entry) {
        throw new Error(
          `onnxruntime-web executor: unsupported input dtype ${inp.dataType} for '${inp.name}'`,
        );
      }
      const [type, Ctor] = entry;
      feeds[inp.name] = new ort.Tensor(type, typedFromBytes(Ctor, inp.data), inp.dims);
    }

    const results = await session.run(feeds);

    const out = {};
    for (const name of Object.keys(results)) {
      const t = results[name];
      const onnxType = ORT_TYPE_TO_ONNX[t.type];
      if (onnxType === undefined) {
        throw new Error(
          `onnxruntime-web executor: unsupported output dtype '${t.type}' for '${name}'`,
        );
      }
      out[name] = {
        dataType: onnxType,
        dims: Array.from(t.dims),
        data: bytesFromTyped(t.data),
      };
    }

    // Free the session's native resources; folding creates one per group.
    if (typeof session.release === "function") {
      await session.release();
    }
    return out;
  };
}

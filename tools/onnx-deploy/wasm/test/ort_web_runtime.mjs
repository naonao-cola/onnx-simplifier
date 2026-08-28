// Reference implementation of the Module.onnxDeployCreateSession /
// Module.onnxDeployRunSession functions onnx_deploy_wasm.cpp calls via
// Asyncify -- see that file's header comment for the exact contract. Backed
// by onnxruntime-web, so swapping this file (or its `ort` import) for a
// different onnxruntime-web build/version/execution-provider is the whole
// "swappable libort" story on the WASM side, mirroring the native side's
// dlopen(libort_path) swap at the JS/wasm boundary instead of the OS loader.
//
// installRuntime(moduleConfig) mutates and returns the Emscripten module
// config object (the one passed to createOnnxDeployWasmModule({...})),
// adding these two functions to it -- they must be present before any
// exported wasm function that calls them runs.

import * as ort from "onnxruntime-web";

// A rejected Promise returned to val::await() does not reliably surface as
// a catchable C++ exception across the Asyncify boundary in practice -- it
// has been observed instead as an uncaught exception that crashes the
// whole process (Node), bypassing every try/catch on both the JS and C++
// sides (reproduced with a "webgpu" execution provider request in a
// navigator.gpu-less environment). Instead, Module.onnxDeployCreateSession/
// Module.onnxDeployRunSession NEVER reject: on failure they resolve to
// `{ __onnxDeployError: message }`, a sentinel onnx_deploy_wasm.cpp checks
// for after every successful (non-rejecting) .await() and turns into a
// normal, synchronously-thrown C++ exception -- ordinary exception
// propagation from there, no Asyncify unwind involved, which is the part
// that actually works reliably. See ../src/onnx_deploy_wasm.cpp's
// CreateSession/RunSession and ../../README.md's WASM section.
function errorResult(err) {
  return { __onnxDeployError: (err && err.message) || String(err) };
}

export function installRuntime(moduleConfig) {
  const sessions = new Map();
  let nextHandle = 1;

  // executionProviders: array of onnxruntime-web EP names, e.g. ["webgpu"]
  // or ["wasm"] (the default if omitted/empty). "webgpu" requires the host
  // to actually expose navigator.gpu (a real browser, or Node started with
  // WebGPU support) -- if it doesn't, ort.InferenceSession.create rejects,
  // caught below and turned into the __onnxDeployError sentinel, not a
  // crash. preferredOutputLocation: "cpu" is set unconditionally so output
  // tensors are always synchronously readable via `.data` below regardless
  // of which EP actually ran the graph (a WebGPU-resident output otherwise
  // needs an async readback before `.data` is valid) -- see
  // ../../README.md's WASM section.
  moduleConfig.onnxDeployCreateSession = async (bytes, executionProviders) => {
    try {
      const eps = executionProviders && executionProviders.length > 0 ? Array.from(executionProviders) : undefined;
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: eps,
        preferredOutputLocation: "cpu",
      });
      const handle = nextHandle++;
      sessions.set(handle, session);
      return { handle, inputNames: session.inputNames, outputNames: session.outputNames };
    } catch (err) {
      return errorResult(err);
    }
  };

  moduleConfig.onnxDeployRunSession = async (handle, inputs, outputNames) => {
    try {
      const session = sessions.get(handle);
      if (!session) throw new Error(`onnxDeployRunSession: unknown session handle ${handle}`);

      const feeds = {};
      for (const t of inputs) {
        const dims = t.shape.map((d) => Number(d));
        if (t.dtype === "int64") {
          feeds[t.name] = new ort.Tensor("int64", BigInt64Array.from(t.data.map((v) => BigInt(Math.trunc(v)))), dims);
        } else if (t.dtype === "float32") {
          feeds[t.name] = new ort.Tensor("float32", Float32Array.from(t.data), dims);
        } else {
          throw new Error(`onnxDeployRunSession: unsupported dtype ${t.dtype} for input ${t.name}`);
        }
      }

      const results = await session.run(feeds, outputNames);

      return outputNames.map((name) => {
        const t = results[name];
        const data = t.type === "int64" ? Array.from(t.data, (v) => Number(v)) : Array.from(t.data);
        return { name, dtype: t.type === "int64" ? "int64" : "float32", shape: Array.from(t.dims), data };
      });
    } catch (err) {
      return errorResult(err);
    }
  };

  return moduleConfig;
}

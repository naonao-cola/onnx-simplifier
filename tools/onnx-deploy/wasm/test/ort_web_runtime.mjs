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

export function installRuntime(moduleConfig) {
  const sessions = new Map();
  let nextHandle = 1;

  moduleConfig.onnxDeployCreateSession = async (bytes) => {
    const session = await ort.InferenceSession.create(bytes);
    const handle = nextHandle++;
    sessions.set(handle, session);
    return { handle, inputNames: session.inputNames, outputNames: session.outputNames };
  };

  moduleConfig.onnxDeployRunSession = async (handle, inputs, outputNames) => {
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
  };

  return moduleConfig;
}

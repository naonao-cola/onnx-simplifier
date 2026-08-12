// Node.js wrapper around onnxsim's WebAssembly build.
//
// This package ships the ORT-web variant (see docs/wasm_ort_web.md in the
// onnxsim repository): the wasm module links no ONNX Runtime and instead
// delegates constant folding to onnxruntime-web, which is an ordinary npm
// dependency here rather than a second copy of ONNX Runtime compiled into
// the module. onnxsim.cjs / onnxsim.wasm / ort_executor.mjs are build
// artifacts staged in by scripts/stage_npm_package.sh (see that script and
// .github/workflows/static.yml) — they are not checked into git.

import createOnnxsim from "./onnxsim.cjs";
import { makeOrtRunner } from "./ort_executor.mjs";

// Matches onnxsim's Python CLI default (DEFAULT_TENSOR_SIZE_THRESHOLDHOLD in
// onnxsim/onnx_simplifier.py): skip folding a constant larger than this many
// bytes, trading a smaller "no large tensor" optimization for a smaller
// model / faster fold.
const DEFAULT_TENSOR_SIZE_THRESHOLD = 1.5 * 1024 ** 3;

let runtimePromise;

function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      let runtime;
      try {
        runtime = await createOnnxsim({
          print: (str) => console.log(str),
          printErr: (str) => console.error(str),
        });
      } catch (err) {
        // onnxsim.cjs is Emscripten's minified, single-physical-line output;
        // an uncaught throw from inside it makes Node's default formatter
        // dump the *entire* file as "source context" instead of a useful
        // message. Catch and rethrow a short, readable error instead.
        const detail = (err && err.message) || err;
        throw new Error(`onnxsim: failed to initialize the wasm module: ${detail}`, { cause: err });
      }
      if (
        typeof runtime.onnxsim_needs_ort_web === "function" &&
        runtime.onnxsim_needs_ort_web()
      ) {
        const ort = await import("onnxruntime-web");
        runtime.onnxsimOrtWebRun = makeOrtRunner(ort.default ?? ort);
      }
      return runtime;
    })();
  }
  return runtimePromise;
}

function toBytes(model) {
  if (model instanceof Uint8Array) return model;
  if (model instanceof ArrayBuffer) return new Uint8Array(model);
  throw new TypeError(
    "onnxsim: model must be a Uint8Array or ArrayBuffer of serialized ONNX bytes",
  );
}

/**
 * Simplify a serialized ONNX model.
 *
 * @param {Uint8Array|ArrayBuffer} model - serialized `onnx.ModelProto` bytes.
 * @param {object} [options]
 * @param {string[]} [options.skipOptimizers] - onnx-optimizer pass names to skip.
 * @param {boolean} [options.constantFolding=true]
 * @param {boolean} [options.shapeInference=true]
 * @param {number} [options.tensorSizeThreshold] - skip folding constants
 *   larger than this many bytes (default 1.5 GiB, matching the Python CLI).
 * @param {number} [options.targetOpsetVersion=-1] - <=0 keeps the model's opset.
 * @param {boolean} [options.profile=false] - also return a Chrome trace JSON.
 * @param {boolean} [options.annotateModelInfo=false] - bake MAC/FLOP counts
 *   into the model's metadata_props.
 * @returns {Promise<{model: Uint8Array, trace: string}>}
 */
export async function simplify(model, options = {}) {
  const bytes = toBytes(model);
  const runtime = await getRuntime();
  const {
    skipOptimizers = [],
    constantFolding = true,
    shapeInference = true,
    tensorSizeThreshold = DEFAULT_TENSOR_SIZE_THRESHOLD,
    targetOpsetVersion = -1,
    profile = false,
    annotateModelInfo = false,
  } = options;

  let result = runtime.onnxsimplify_export(
    bytes,
    skipOptimizers,
    constantFolding,
    shapeInference,
    tensorSizeThreshold,
    targetOpsetVersion,
    profile,
    annotateModelInfo,
  );
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: simplify failed (see stderr output for details)");
  }
  return { model: new Uint8Array(result.model), trace: result.trace || "" };
}

/** onnxsim / onnx-optimizer version strings baked into this build. */
export async function versions() {
  const runtime = await getRuntime();
  return runtime.onnxsim_versions();
}

export default { simplify, versions };

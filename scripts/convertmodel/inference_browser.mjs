// Browser glue for the "run inference" panel on the converter page.
//
// Loads onnxruntime-web on demand from a CDN, builds dummy inputs from the
// model's own input metadata, and runs a few inference iterations through the
// shared inference_core.mjs (the same code the Node smoke test drives). The
// execution provider defaults to WebGPU and falls back to wasm when WebGPU is
// unavailable, so it works on any browser.

import { runInference } from "./inference_core.mjs";

const ORT_VERSION = "1.27.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ortPromise = null;
function loadOrt() {
  if (!ortPromise) {
    ortPromise = import(/* @vite-ignore */ `${ORT_BASE}ort.min.mjs`).then((m) => {
      const ort = m.default ?? m;
      // Pull the matching wasm binaries from the same CDN directory.
      ort.env.wasm.wasmPaths = ORT_BASE;
      return ort;
    });
  }
  return ortPromise;
}

const TYPED_ARRAY = {
  float32: Float32Array,
  float64: Float64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
  bool: Uint8Array,
  int64: typeof BigInt64Array !== "undefined" ? BigInt64Array : null,
  uint64: typeof BigUint64Array !== "undefined" ? BigUint64Array : null,
};

// A symbolic or non-positive dimension (dynamic axis) is materialized as 1 so we
// can construct a concrete tensor to run.
function concreteShape(shape) {
  return (shape || []).map((d) => (typeof d === "number" && d > 0 ? d : 1));
}

function makeDummyInputs(ort, session) {
  const feeds = {};
  for (const meta of session.inputMetadata) {
    if (!meta.isTensor) {
      throw new Error(`input '${meta.name}' is not a tensor; cannot auto-generate`);
    }
    const dims = concreteShape(meta.shape);
    const count = dims.reduce((a, b) => a * b, 1);
    const Ctor = TYPED_ARRAY[meta.type];
    if (!Ctor) throw new Error(`unsupported input type '${meta.type}'`);
    feeds[meta.name] = new ort.Tensor(meta.type, new Ctor(count), dims);
  }
  return feeds;
}

async function runOnModel(modelBytes, { iterations, preferWebGPU }, log) {
  const ort = await loadOrt();
  const providers = preferWebGPU ? ["webgpu", "wasm"] : ["wasm"];

  // Build inputs from a throwaway wasm session's metadata (cheap and always
  // available), then run the real iterations through the chosen provider.
  const metaSession = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  const feeds = makeDummyInputs(ort, metaSession);
  const inputName = metaSession.inputNames[0];

  const res = await runInference(ort, {
    model: modelBytes,
    inputName,
    input: feeds[inputName],
    providers,
    iterations,
    onLog: log,
  });
  return res;
}

export function initInferencePanel() {
  const btn = document.getElementById("run-inference");
  const out = document.getElementById("inference-output");
  const fileInput = document.getElementById("file-input");
  const itersInput = document.getElementById("inference-iters");
  const epSelect = document.getElementById("inference-ep");
  if (!btn) return;

  const log = (msg) => {
    out.textContent += msg + "\n";
  };

  btn.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      out.textContent = "Pick an .onnx file first.\n";
      return;
    }
    btn.disabled = true;
    out.textContent = "";
    try {
      const iterations = Math.max(1, parseInt(itersInput.value, 10) || 5);
      const preferWebGPU = epSelect.value === "webgpu";
      const bytes = new Uint8Array(await file.arrayBuffer());
      log(`loading onnxruntime-web ${ORT_VERSION}…`);
      const res = await runOnModel(bytes, { iterations, preferWebGPU }, log);
      log(
        `PASS: ${res.iterations} iterations on '${res.ep}', deterministic ` +
          `(avg ${res.avgMs.toFixed(2)} ms/iter)`,
      );
    } catch (e) {
      log("FAIL: " + (e && e.message ? e.message : String(e)));
    } finally {
      btn.disabled = false;
    }
  });
}

initInferencePanel();

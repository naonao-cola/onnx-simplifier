# WebNN in the WASM converter UI

**Status: experimental.** The converter page's **Run inference** panel can now
run a model through [WebNN](https://www.w3.org/TR/webnn/) as an onnxruntime-web
execution provider, alongside the existing WebGPU and WebAssembly options. This
document records the state of WebNN as of onnxsim's move to onnxruntime-web
1.27 (early 2026) and how the panel uses it.

## What WebNN is

WebNN (the W3C Web Neural Network API, exposed as `navigator.ml`) lets a web
page run a neural network on the platform's own machine-learning stack —
DirectML on Windows, Core ML on macOS, and the platform NN API elsewhere —
reaching the GPU and, where present, a dedicated NPU, without shipping a
compute backend in the page.

## Status (early 2026)

- **Specification:** WebNN is a W3C Candidate Recommendation, still evolving.
- **Browsers:** available in recent Chromium-based browsers (Chrome, Edge, and
  relatives) behind the *"Enables WebNN API"* flag —
  `chrome://flags/#web-machine-learning-neural-network`. Coverage is broadest on
  **Windows**; GPU and NPU paths on other platforms are still maturing. Not yet
  in Firefox or Safari stable.
- **onnxruntime-web:** the WebNN backend is an **experimental** execution
  provider. It is not part of the default `ort.min.mjs` bundle — it ships in the
  "all" bundle (`ort.all.min.mjs`).

Because of that, WebNN is **usable but not guaranteed** on any given visitor's
browser: treat it as an acceleration opportunity with a required fallback, not a
baseline.

## How the panel uses it

`scripts/convertmodel/webnn.mjs` holds the browser-free pieces (all unit-tested
in `test/webnn.test.mjs`):

- **`detectWebnn(navigator)`** probes `navigator.ml` at page load, attempting to
  create an `MLContext` for each device type (`gpu`, `npu`, `cpu`). The panel
  renders the result as a live **WebNN status** line under the inference
  controls, so a visitor can see whether WebNN will run before selecting it.
- **`providersForEp(value)`** maps each EP dropdown choice to an ordered
  onnxruntime-web provider list. The WebNN choices are:

  | dropdown value | providers tried (in order)                                   |
  | -------------- | ------------------------------------------------------------ |
  | `webnn-gpu`    | `{ name: "webnn", deviceType: "gpu", powerPreference: … }`, `wasm` |
  | `webnn-npu`    | `{ name: "webnn", deviceType: "npu", powerPreference: … }`, `wasm` |
  | `webnn-cpu`    | `{ name: "webnn", deviceType: "cpu" }`, `wasm`               |

  Every WebNN (and WebGPU) choice ends in `wasm`, so onnxruntime-web
  transparently falls back to WebAssembly when the WebNN context — or a
  particular operator — is unsupported. `inference_core.mjs` reports which
  provider actually won (e.g. `webnn:gpu` or the `wasm` fallback).

- The WebNN provider lives only in the "all" onnxruntime-web bundle, so the
  panel loads `ort.all.min.mjs` on demand the first time a WebNN EP is selected,
  and keeps the smaller default bundle for the common WebGPU/WASM path. Both
  bundles pull the same JSEP wasm artifacts, so the CDN `wasmPaths` are
  unchanged.

Nothing about the model conversion (Simplify / Optimize) path changes — WebNN is
only an option for the post-conversion inference check.

## References

- WebNN API — https://www.w3.org/TR/webnn/
- onnxruntime-web WebNN EP —
  https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html
- WebNN operator support in onnxruntime-web —
  https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webnn-operators.md

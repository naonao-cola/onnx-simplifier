// Browser glue for the "run inference" panel on the converter page.
//
// Loads onnxruntime-web on demand from a CDN, builds dummy inputs from the
// model's own input metadata, and runs a few inference iterations through the
// shared inference_core.mjs (the same code the Node smoke test drives). The
// execution provider defaults to WebGPU and falls back to wasm when WebGPU is
// unavailable, so it works on any browser.
//
// The panel can run either the original uploaded model or the converted
// (simplify/optimize) result. The converter page publishes the converted bytes
// on `window.__onnxsimConverted` (set in index.html when a conversion
// finishes); "original" reads the file input directly.

import { runInference } from "./inference_core.mjs";
import { summarizeOrtTrace } from "./trace_build.mjs";
import { renderTrace } from "./trace_viewer.mjs";
import {
  readAnnotations,
  perOpSummary,
  humanNum,
  humanBytes,
  humanDensity,
  throughput,
  evalMetric,
  isSymbolicStr,
} from "./macs.mjs";

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

// True for a symbolic (string) or non-positive dimension — i.e. a dynamic axis
// with no fixed size baked into the model.
function isDynamicDim(d) {
  return !(typeof d === "number" && d > 0);
}

// Materialize a model input shape into concrete integers. Fixed dims are kept;
// a dynamic leading (batch) axis becomes `batch`, and any other dynamic axis
// becomes 1, so we can construct a runnable tensor. With batch === 1 this
// matches the model's own default sizing.
function concreteShape(shape, batch = 1) {
  return (shape || []).map((d, i) => {
    if (!isDynamicDim(d)) return d;
    return i === 0 ? batch : 1;
  });
}

function makeDummyInputs(ort, session, batch = 1) {
  const feeds = {};
  for (const meta of session.inputMetadata) {
    if (!meta.isTensor) {
      throw new Error(`input '${meta.name}' is not a tensor; cannot auto-generate`);
    }
    const dims = concreteShape(meta.shape, batch);
    const count = dims.reduce((a, b) => a * b, 1);
    const Ctor = TYPED_ARRAY[meta.type];
    if (!Ctor) throw new Error(`unsupported input type '${meta.type}'`);
    feeds[meta.name] = new ort.Tensor(meta.type, new Ctor(count), dims);
  }
  return feeds;
}

async function runOnModel(modelBytes, { iterations, batch, preferWebGPU, profile }, log) {
  const ort = await loadOrt();
  const providers = preferWebGPU ? ["webgpu", "wasm"] : ["wasm"];

  // Build inputs from a throwaway wasm session's metadata (cheap and always
  // available), then run the real iterations through the chosen provider.
  const metaSession = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  const feeds = makeDummyInputs(ort, metaSession, batch);
  const inputName = metaSession.inputNames[0];
  log(`input '${inputName}' dims [${feeds[inputName].dims.join(", ")}]`);

  const res = await runInference(ort, {
    model: modelBytes,
    inputName,
    input: feeds[inputName],
    providers,
    iterations,
    profile,
    onLog: log,
  });
  // How much the annotated (per-sample, dynamic-dims=1) work was scaled up by:
  // only a dynamic leading axis is driven by `batch`, so a fixed-shape input
  // leaves the metrics untouched (batch is a no-op there).
  const leadMeta = metaSession.inputMetadata.find((mm) => mm.name === inputName);
  res.batchScale = isDynamicDim(leadMeta?.shape?.[0]) ? batch : 1;
  return res;
}

// Resolve the model bytes for the chosen source. "converted" uses the bytes the
// converter page published on window.__onnxsimConverted; "original" reads the
// file input. Returns { bytes, label } or throws with a user-facing message.
async function resolveModelBytes(source, fileInput) {
  if (source === "converted") {
    const converted = window.__onnxsimConverted;
    if (!converted || !converted.bytes) {
      throw new Error(
        "no converted model yet — pick a file above to run a conversion " +
          "first, or switch the model dropdown to 'original'.",
      );
    }
    return { bytes: converted.bytes, label: `converted (${converted.name})` };
  }
  const file = fileInput.files && fileInput.files[0];
  if (!file) throw new Error("Pick an .onnx file first.");
  return {
    bytes: new Uint8Array(await file.arrayBuffer()),
    label: `original (${file.name})`,
  };
}

// Render the onnxsim MAC/FLOP metrics (onnxsim PR #527) that travel inside the
// model's `metadata_props`, plus the throughput implied by the measured average
// latency. `res` may be null (metrics shown before/without a run). When the
// model carries no onnxsim.* metrics the container is cleared and a hint is
// logged instead.
function renderMacs(container, log, bytes, res) {
  if (!container) return;
  let ann;
  try {
    ann = readAnnotations(bytes);
  } catch (e) {
    container.innerHTML = "";
    log(`could not read onnxsim metrics: ${e && e.message ? e.message : e}`);
    return;
  }
  if (!ann.annotated) {
    container.innerHTML = "";
    log(
      "no onnxsim MAC metrics in this model — annotate it first with " +
        "onnxsim.model_info.annotate_metadata (see onnxsim PR #527).",
    );
    return;
  }

  const m = ann.model;
  // The metrics are per-sample (dynamic dims -> 1); when the run fed a larger
  // batch on a dynamic leading axis, scale FLOPs/MACs up so the throughput
  // reflects the aggregate work done in that latency.
  const batchScale = res && res.batchScale > 0 ? res.batchScale : 1;
  const tp = res ? throughput(m, res.avgMs) : null;
  if (tp && batchScale !== 1) {
    tp.gflops *= batchScale;
    tp.gmacs *= batchScale;
  }
  // Any dynamic dimension makes the stored metrics symbolic formulas; we show
  // them with every symbol substituted by 1 (the per-sample value).
  const symbolic = [m.macs, m.flops, m.mem_access, m.memory_footprint].some(
    isSymbolicStr,
  );
  const macs = evalMetric(m.macs);
  const cards = [
    ["MACs", humanNum(macs), macs != null ? macs.toLocaleString() : ""],
    ["FLOPs", humanNum(evalMetric(m.flops)), "= 2 × MACs"],
    ["Memory access", humanBytes(evalMetric(m.mem_access)), "read + written"],
    ["Peak footprint", humanBytes(evalMetric(m.memory_footprint)), "resident"],
    ["Compute density",
      m.compute_density != null ? humanDensity(evalMetric(m.compute_density)) : "—", ""],
    ["Model size", humanBytes(evalMetric(m.model_size)), ""],
  ];
  if (tp) {
    const batchNote = batchScale !== 1 ? ` × batch ${batchScale}` : "";
    cards.push(["Throughput", `${tp.gflops.toFixed(2)} GFLOP/s`, `avg ${res.avgMs.toFixed(2)} ms on '${res.ep}'${batchNote}`]);
    cards.push(["MAC rate", `${tp.gmacs.toFixed(2)} GMAC/s`, ""]);
  }

  const { rows, totalMacs } = perOpSummary(ann.nodes);
  const cardHtml = cards
    .map(
      ([k, v, sub]) =>
        `<div class="macs-card"><div class="macs-k">${k}</div>` +
        `<div class="macs-v">${v}${sub ? ` <small>${sub}</small>` : ""}</div></div>`,
    )
    .join("");
  const opRows = rows
    .map((r) => {
      const pct = totalMacs ? (100 * r.macs) / totalMacs : 0;
      return (
        `<tr><td>${r.opType}</td><td>${r.count}</td>` +
        `<td>${humanNum(r.macs)}</td><td>${humanNum(r.macs * 2)}</td>` +
        `<td>${pct.toFixed(1)}%</td></tr>`
      );
    })
    .join("");
  const substNote = symbolic
    ? " Dynamic dimensions substituted with 1 (per-sample values)."
    : "";
  container.innerHTML =
    `<div class="macs-cards">${cardHtml}</div>` +
    `<table class="macs-table"><thead><tr><th>Operator</th><th>Nodes</th>` +
    `<th>MACs</th><th>FLOPs</th><th>% of MACs</th></tr></thead><tbody>${opRows}</tbody></table>` +
    `<p class="macs-note">Metrics read from the model's <code>metadata_props</code> ` +
    `(onnxsim <a href="https://github.com/onnxsim/onnxsim/pull/527" target="_blank" rel="noopener">PR #527</a>). ` +
    `Throughput = model FLOPs ÷ average onnxruntime-web latency.${substNote}</p>`;

  const macsStr = macs != null ? macs.toLocaleString() : (m.macs ?? "?");
  const suffix = symbolic ? " (per-sample, dims=1)" : "";
  if (tp) {
    log(`MACs: ${macsStr}${suffix} → ${tp.gflops.toFixed(2)} GFLOP/s at ${res.avgMs.toFixed(2)} ms/iter (from PR #527 metadata)`);
  } else {
    log(`MACs: ${macsStr}${suffix} (from onnxsim metadata_props, PR #527)`);
  }
}

export function initInferencePanel() {
  const btn = document.getElementById("run-inference");
  const out = document.getElementById("inference-output");
  const fileInput = document.getElementById("file-input");
  const itersInput = document.getElementById("inference-iters");
  const batchInput = document.getElementById("inference-batch");
  const epSelect = document.getElementById("inference-ep");
  const sourceSelect = document.getElementById("inference-source");
  const profileChk = document.getElementById("inference-profile");
  const traceContainer = document.getElementById("inference-trace");
  const macsContainer = document.getElementById("inference-macs");
  if (!btn) return;

  const log = (msg) => {
    out.textContent += msg + "\n";
  };

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    out.textContent = "";
    if (traceContainer) traceContainer.innerHTML = "";
    if (macsContainer) macsContainer.innerHTML = "";
    try {
      const source = sourceSelect ? sourceSelect.value : "original";
      const { bytes, label } = await resolveModelBytes(source, fileInput);
      const iterations = Math.max(1, parseInt(itersInput.value, 10) || 5);
      const batch = Math.max(1, parseInt(batchInput ? batchInput.value : "1", 10) || 1);
      const preferWebGPU = epSelect.value === "webgpu";
      const profile = !profileChk || profileChk.checked;
      log(`running ${label}`);
      // Static MACs/FLOPs from the model's onnxsim metadata_props (PR #527),
      // shown before the run so they appear even if inference fails.
      renderMacs(macsContainer, log, bytes, null);
      log(`loading onnxruntime-web ${ORT_VERSION}…`);
      const res = await runOnModel(
        bytes,
        { iterations, batch, preferWebGPU, profile },
        log,
      );
      log(
        `PASS: ${res.iterations} iterations on '${res.ep}', deterministic ` +
          `(avg ${res.avgMs.toFixed(2)} ms/iter)`,
      );
      // Re-render with the measured latency to add achieved throughput.
      renderMacs(macsContainer, () => {}, bytes, res);
      if (profile && res.trace && traceContainer) {
        const runs = res.timings.map((durMs, index) => ({ index, startMs: 0, durMs }));
        const s = summarizeOrtTrace({ runs, kernels: res.kernels || [] });
        if (s.kernels > 0) {
          log(`captured ${s.kernels} GPU kernel spans (${s.gpuMs.toFixed(2)} ms on device)`);
        } else {
          log(
            "no per-kernel GPU profiling for this provider; showing " +
              "per-iteration wall spans. Run on WebGPU for op-level detail.",
          );
        }
        renderTrace(traceContainer, res.trace, {
          title: `onnxruntime-web ${res.ep} inference`,
          filename: `onnxruntime-web.${res.ep}.trace.json`,
        });
      }
    } catch (e) {
      log("FAIL: " + (e && e.message ? e.message : String(e)));
    } finally {
      btn.disabled = false;
    }
  });
}

initInferencePanel();

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

import { runInference, compareInference } from "./inference_core.mjs";
import { summarizeOrtTrace } from "./trace_build.mjs";
import { renderTrace } from "./trace_viewer.mjs";
import { downloadText } from "./download.mjs";
import { providersForEp, isWebnnEp, detectWebnn, formatWebnnStatus } from "./webnn.mjs";
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

// onnxruntime-web ships several EP bundles. The default (`ort.min.mjs`) carries
// WASM + WebGPU; WebNN lives only in the "all" bundle (`ort.all.min.mjs`). Both
// use the same JSEP wasm artifacts from ORT_BASE, so switching bundles keeps
// wasmPaths valid. We load the WebNN bundle on demand (only when a WebNN EP is
// picked) so the common WebGPU/WASM path keeps its smaller download.
const ORT_BUNDLE = { default: "ort.min.mjs", all: "ort.all.min.mjs" };
const ortPromises = {};
function loadOrt(variant = "default") {
  if (!ortPromises[variant]) {
    const file = ORT_BUNDLE[variant] || ORT_BUNDLE.default;
    ortPromises[variant] = import(/* @vite-ignore */ `${ORT_BASE}${file}`).then((m) => {
      const ort = m.default ?? m;
      // Pull the matching wasm binaries from the same CDN directory.
      ort.env.wasm.wasmPaths = ORT_BASE;
      return ort;
    });
  }
  return ortPromises[variant];
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

// Fill a float typed-array in place with small deterministic pseudo-random
// values in [-1, 1) (a seeded xorshift32, so every call produces the same
// sequence). Used by the before/after comparison: all-zero inputs can mask a
// divergence introduced by a graph change (many ops map 0 -> 0), whereas varied
// inputs actually exercise the graph. Integer/bool inputs are left at zero,
// which is the safe default for index-like tensors (Gather, etc.) where an
// arbitrary value could go out of bounds. Determinism matters because
// runInference() re-feeds the same tensor every iteration and asserts the output
// never changes.
function fillPseudoRandom(arr, type) {
  if (type !== "float32" && type !== "float64") return; // zeros for ints/bool
  let s = 0x2545f491; // fixed non-zero seed
  for (let i = 0; i < arr.length; i++) {
    // xorshift32 (bitwise ops keep `s` a 32-bit signed int throughout).
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    arr[i] = s / 0x80000000; // 32-bit signed / 2^31 -> [-1, 1)
  }
}

// Build a feeds object for every model input. `fill` is "zero" (the default,
// matching a plain single-model run) or "random" (deterministic non-zero float
// values, used by the compare path so the before/after diff is meaningful).
function makeDummyInputs(ort, session, batch = 1, fill = "zero") {
  const feeds = {};
  for (const meta of session.inputMetadata) {
    if (!meta.isTensor) {
      throw new Error(`input '${meta.name}' is not a tensor; cannot auto-generate`);
    }
    const dims = concreteShape(meta.shape, batch);
    const count = dims.reduce((a, b) => a * b, 1);
    const Ctor = TYPED_ARRAY[meta.type];
    if (!Ctor) throw new Error(`unsupported input type '${meta.type}'`);
    const buf = new Ctor(count);
    if (fill === "random") fillPseudoRandom(buf, meta.type);
    feeds[meta.name] = new ort.Tensor(meta.type, buf, dims);
  }
  return feeds;
}

// Build the dummy inputs for a model from a throwaway wasm session's metadata
// (cheap and always available) and report the leading (batch) input. Shared by
// the single-model and compare paths so the compare path can build ONE set of
// inputs and feed the SAME tensor to both models. Returns
// { feeds, inputName, leadMeta, leadingDynamic }.
async function prepareInputs(ort, modelBytes, batch, fill, log) {
  const metaSession = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  const feeds = makeDummyInputs(ort, metaSession, batch, fill);
  const inputName = metaSession.inputNames[0];
  const leadMeta = metaSession.inputMetadata.find((mm) => mm.name === inputName);
  const leadingDynamic = isDynamicDim(leadMeta?.shape?.[0]);
  // Explain a dropped batch request: onnxruntime enforces a model's declared
  // static shapes, so a fixed leading dimension can't be resized here.
  if (batch > 1 && !leadingDynamic) {
    log(
      `note: batch ${batch} ignored — '${inputName}' has a fixed first ` +
        `dimension (${leadMeta?.shape?.[0]}); re-export the model with a ` +
        `dynamic batch axis to change it.`,
    );
  }
  return { feeds, inputName, leadMeta, leadingDynamic };
}

async function runOnModel(modelBytes, { iterations, batch, providers, needWebnn, profile }, log) {
  const ort = await loadOrt(needWebnn ? "all" : "default");

  const { feeds, inputName, leadingDynamic } = await prepareInputs(
    ort,
    modelBytes,
    batch,
    "zero",
    log,
  );
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
  res.batchScale = leadingDynamic ? batch : 1;
  return res;
}

// Run the "compare (before vs after)" path: feed ONE shared, deterministic
// input to both the original and converted models and report how far their
// outputs diverge plus the speed difference. Building the input from the
// original model (whose I/O the conversion preserves) and reusing the exact same
// tensor for both isolates the effect of the graph change.
async function runCompare(
  originalBytes,
  convertedBytes,
  { iterations, batch, providers, needWebnn, profile },
  log,
) {
  const ort = await loadOrt(needWebnn ? "all" : "default");

  const { feeds, inputName, leadingDynamic } = await prepareInputs(
    ort,
    originalBytes,
    batch,
    "random",
    log,
  );
  const input = feeds[inputName];
  log(`shared input '${inputName}' dims [${input.dims.join(", ")}] (deterministic)`);

  const cmp = await compareInference(ort, {
    before: { model: originalBytes, label: "original" },
    after: { model: convertedBytes, label: "converted" },
    inputName,
    input,
    providers,
    iterations,
    profile,
    onLog: log,
  });
  const batchScale = leadingDynamic ? batch : 1;
  cmp.before.batchScale = batchScale;
  cmp.after.batchScale = batchScale;
  return cmp;
}

// Resolve the model bytes for the chosen source. "converted" uses the bytes the
// converter page published on window.__onnxsimConverted; "original" reads the
// file input, falling back to a model loaded from Hugging Face (published on
// window.__onnxsimOriginal, which has no file-input entry). Returns
// { bytes, label } or throws with a user-facing message.
async function resolveModelBytes(source, fileInput) {
  if (source === "converted") {
    const converted = window.__onnxsimConverted;
    if (!converted || !converted.bytes) {
      throw new Error(
        "no converted model yet — pick a file (or load one from Hugging " +
          "Face) above to run a conversion first, or switch the model " +
          "dropdown to 'original'.",
      );
    }
    return { bytes: converted.bytes, label: `converted (${converted.name})` };
  }
  const file = fileInput.files && fileInput.files[0];
  // Resolve the "original" bytes + a display name from either a local file
  // upload or a Hugging Face download (which has no file-input entry, and
  // publishes itself on window.__onnxsimOriginal).
  let name = null;
  let readBytes = null;
  if (file) {
    name = file.name;
    readBytes = async () => new Uint8Array(await file.arrayBuffer());
  } else {
    const original = window.__onnxsimOriginal;
    if (original && original.bytes) {
      name = original.name;
      readBytes = async () => original.bytes;
    }
  }
  if (!readBytes) {
    throw new Error("Pick an .onnx file or load one from Hugging Face first.");
  }
  // Prefer the MAC-annotated copy of this original that the converter produced
  // (window.__onnxsimOriginalAnnotated, published in index.html when a
  // conversion finishes with "annotate model info" on). It is the same model
  // plus onnxsim.* metadata_props, so it runs identically but lets the panel
  // report MACs/throughput — enabling an original-vs-converted speed
  // comparison. The name guard avoids serving a stale annotation for a model
  // that has since been swapped out.
  const annotated = window.__onnxsimOriginalAnnotated;
  if (annotated && annotated.bytes && annotated.name === name) {
    return {
      bytes: annotated.bytes,
      label: `original (${name}, MAC-annotated)`,
    };
  }
  return { bytes: await readBytes(), label: `original (${name})` };
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

// Read a model's annotated per-sample MACs (onnxsim PR #527 metadata_props),
// or null when the model carries none / can't be parsed.
function modelMacs(bytes) {
  try {
    const ann = readAnnotations(bytes);
    return ann.annotated ? evalMetric(ann.model.macs) : null;
  } catch {
    return null;
  }
}

// Render the before/after comparison summary into `container`: a verdict on
// whether the conversion preserved the numerics (max |Δ| vs tolerance), the
// per-iteration latency of each model with the resulting speedup, and — when
// both models are MAC-annotated — the MAC reduction. `cmp` is the object from
// runCompare()/compareInference(). Also logs a compact text summary.
function renderCompare(container, log, originalBytes, convertedBytes, cmp, tolerance) {
  const { before, after, maxAbsDiff, dimsMatch, comparable, equivalent, speedup } = cmp;

  const diffStr = comparable ? maxAbsDiff.toExponential(3) : "n/a";
  let verdictText;
  let verdictClass;
  if (!comparable) {
    verdictText = "outputs not comparable (shape/length differ)";
    verdictClass = "cmp-warn";
  } else if (equivalent) {
    verdictText = `equivalent — max |Δ| = ${diffStr} ≤ ${tolerance}`;
    verdictClass = "cmp-ok";
  } else {
    verdictText = `differs — max |Δ| = ${diffStr} > ${tolerance}`;
    verdictClass = "cmp-bad";
  }

  const speedStr =
    speedup != null
      ? `${speedup.toFixed(2)}× ${speedup >= 1 ? "faster" : "slower"}`
      : "—";
  const dimsStr = `[${(before.dims || []).join(", ")}]${
    dimsMatch ? "" : ` vs [${(after.dims || []).join(", ")}]`
  }`;

  const macsBefore = modelMacs(originalBytes);
  const macsAfter = modelMacs(convertedBytes);
  const macsCell =
    macsBefore != null && macsAfter != null
      ? `${humanNum(macsBefore)} → ${humanNum(macsAfter)}` +
        (macsBefore > 0
          ? ` (${(((macsBefore - macsAfter) / macsBefore) * 100).toFixed(1)}% fewer)`
          : "")
      : "annotate model info to compare";

  const rows = [
    ["Outputs", verdictText, verdictClass],
    ["Output dims", `${dimsStr}${dimsMatch ? " (match)" : " (differ)"}`, ""],
    [
      "Latency",
      `before ${before.avgMs.toFixed(2)} ms, after ${after.avgMs.toFixed(2)} ms → ${speedStr}`,
      "",
    ],
    ["MACs", macsCell, ""],
  ];
  container.innerHTML =
    `<table class="macs-table cmp-table"><tbody>` +
    rows
      .map(
        ([k, v, cls]) =>
          `<tr><th>${k}</th><td class="${cls}">${v}</td></tr>`,
      )
      .join("") +
    `</tbody></table>` +
    `<p class="macs-note">Both models run on the <b>same</b> deterministic input ` +
    `(dynamic axes sized from the batch control); a small max |Δ| means the ` +
    `conversion preserved the model's numerics. Speedup = before ÷ after average ` +
    `<code>session.run</code> latency on '${after.ep}'.</p>`;

  log("── comparison (before vs after conversion) ──");
  log(`outputs: ${verdictText}`);
  log(`output dims: ${dimsStr}${dimsMatch ? " (match)" : " (differ)"}`);
  log(
    `speed: before ${before.avgMs.toFixed(2)} ms/iter, after ` +
      `${after.avgMs.toFixed(2)} ms/iter → ${speedStr}`,
  );
  if (macsBefore != null && macsAfter != null) {
    log(`MACs: ${macsCell}`);
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
  const dlLogBtn = document.getElementById("download-inference-log-button");
  const webnnStatusEl = document.getElementById("webnn-status");
  if (!btn) return;

  // The inference output is a readonly textarea (mirroring the simplify console
  // log above), so append to `.value` and keep it scrolled to the newest line.
  const log = (msg) => {
    out.value += msg + "\n";
    out.scrollTop = out.scrollHeight;
  };

  // Probe the WebNN API once at load and show a live status next to the EP
  // picker, so a user can see whether the WebNN options will actually run
  // before selecting one. detectWebnn never throws.
  if (webnnStatusEl) {
    webnnStatusEl.textContent = "WebNN status: probing…";
    detectWebnn()
      .then((report) => {
        webnnStatusEl.textContent = formatWebnnStatus(report);
        webnnStatusEl.title = report.apiPresent
          ? `device availability — ${["gpu", "npu", "cpu"]
              .map((d) => `${d}: ${report.devices[d] ? "yes" : "no"}`)
              .join(", ")}`
          : "";
      })
      .catch(() => {
        webnnStatusEl.textContent =
          "WebNN status: ❌ could not probe navigator.ml.";
      });
  }

  if (dlLogBtn) {
    dlLogBtn.addEventListener("click", () => {
      downloadText((out && out.value) || "", "onnxsim-inference.log");
    });
  }

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    out.value = "";
    if (traceContainer) traceContainer.innerHTML = "";
    if (macsContainer) macsContainer.innerHTML = "";
    try {
      const source = sourceSelect ? sourceSelect.value : "original";
      const iterations = Math.max(1, parseInt(itersInput.value, 10) || 5);
      const batch = Math.max(1, parseInt(batchInput ? batchInput.value : "1", 10) || 1);
      const epValue = epSelect.value;
      const { providers, needWebnn } = providersForEp(epValue);
      const profile = !profileChk || profileChk.checked;

      if (isWebnnEp(epValue)) {
        log(
          "WebNN is experimental; falling back to WASM if the WebNN device " +
            "or an operator is unsupported.",
        );
      }

      // "compare (before vs after)" runs both the original and converted models
      // on one shared input and reports how far their outputs diverge plus the
      // speed difference — the verification the rest of the panel can't give
      // from a single model.
      if (source === "compare") {
        const original = await resolveModelBytes("original", fileInput);
        const converted = await resolveModelBytes("converted", fileInput);
        const tolerance = 1e-3;
        log(`comparing ${original.label} vs ${converted.label}`);
        log(`loading onnxruntime-web ${ORT_VERSION}…`);
        const cmp = await runCompare(
          original.bytes,
          converted.bytes,
          { iterations, batch, providers, needWebnn, profile },
          log,
        );
        log(
          `PASS: ran both models ${cmp.after.iterations} iterations on ` +
            `'${cmp.after.ep}'`,
        );
        if (macsContainer) {
          renderCompare(macsContainer, log, original.bytes, converted.bytes, cmp, tolerance);
        }
        if (profile && cmp.after.trace && traceContainer) {
          renderTrace(traceContainer, cmp.after.trace, {
            title: `onnxruntime-web ${cmp.after.ep} inference (converted)`,
            filename: `onnxruntime-web.${cmp.after.ep}.converted.trace.json`,
          });
        }
        return;
      }

      const { bytes, label } = await resolveModelBytes(source, fileInput);
      log(`running ${label}`);
      // Static MACs/FLOPs from the model's onnxsim metadata_props (PR #527),
      // shown before the run so they appear even if inference fails.
      renderMacs(macsContainer, log, bytes, null);
      log(`loading onnxruntime-web ${ORT_VERSION}…`);
      const res = await runOnModel(
        bytes,
        { iterations, batch, providers, needWebnn, profile },
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

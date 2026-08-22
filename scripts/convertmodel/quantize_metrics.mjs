// Result-quality metrics for a quantized model, shown in the "Quantize a
// model" panel (quantize_ui.mjs) after a successful quantize: how much
// smaller the model got, and how far its outputs drift from the float
// baseline on the same random input.
//
// The drift metrics -- relative L2 error and SQNR (signal-to-quantization-
// noise ratio) -- mirror the aggregate-error check onnxsim's own Python
// quantization test suite uses (e.g. tests/test_qoperator_quantize_matmul.py's
// _assert_close: relative L2 error, not a tight per-element bound, since
// quantization is lossy by design and a handful of elements at the rounding
// boundary are expected), computed here in the browser via onnxruntime-web
// instead of onnxruntime.

import { loadOrt, makeDummyInputs } from "./inference_browser.mjs";
import { compareInference } from "./inference_core.mjs";

// Relative L2 error (||f - q|| / ||f||) and SQNR in dB
// (20*log10(||f|| / ||f-q||)) between two same-length flat arrays.
function errorMetrics(floatData, quantData) {
  let sumSqDiff = 0;
  let sumSqFloat = 0;
  let maxAbs = 0;
  const n = Math.min(floatData.length, quantData.length);
  for (let i = 0; i < n; i++) {
    const f = Number(floatData[i]);
    const q = Number(quantData[i]);
    const d = f - q;
    sumSqDiff += d * d;
    sumSqFloat += f * f;
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  const l2Float = Math.sqrt(sumSqFloat);
  const l2Diff = Math.sqrt(sumSqDiff);
  const relL2 = l2Diff / Math.max(l2Float, 1e-6);
  const sqnrDb = l2Diff > 0 ? 20 * Math.log10(Math.max(l2Float, 1e-12) / l2Diff) : Infinity;
  return { relL2, sqnrDb, maxAbsDiff: maxAbs };
}

// Runs `floatBytes` (the original model) and `quantBytes` (the quantized
// result) on the same synthetic random input through onnxruntime-web and
// returns a result-quality summary:
// { sizeBefore, sizeAfter, sizeReductionPct, ep, relL2, sqnrDb, maxAbsDiff,
//   dimsMatch }.
// Best-effort by design: the caller should treat a thrown error (e.g. no
// usable execution provider, or an input onnxruntime-web can't auto-fill)
// as "quality metrics unavailable" and not block the quantized model's
// availability on it -- the model is already valid and downloadable either
// way.
export async function computeQuantizationQuality(floatBytes, quantBytes, log = () => {}) {
  const ort = await loadOrt();
  const metaSession = await ort.InferenceSession.create(floatBytes, {
    executionProviders: ["wasm"],
  });
  const feeds = await makeDummyInputs(ort, metaSession, 1, "random", null, null, log);

  const cmp = await compareInference(ort, {
    before: { model: floatBytes, label: "float" },
    after: { model: quantBytes, label: "quantized" },
    feeds,
    providers: ["wasm"],
    iterations: 1,
    warmup: 0,
    // compareInference's own "equivalent" (maxAbsDiff <= tolerance) isn't
    // useful here -- quantization is lossy by design, so it's expected to
    // read false under any tight tolerance. The metrics below (relative L2,
    // SQNR) are the actual quality signal; Infinity just keeps "equivalent"
    // from reading as a spurious failure if a caller ever inspects it.
    tolerance: Infinity,
    onLog: log,
  });

  if (!cmp.comparable) {
    throw new Error("quantized model's output shape/length doesn't match the float model's");
  }

  const { relL2, sqnrDb, maxAbsDiff } = errorMetrics(cmp.before.output, cmp.after.output);
  const sizeBefore = floatBytes.length;
  const sizeAfter = quantBytes.length;
  return {
    sizeBefore,
    sizeAfter,
    sizeReductionPct: sizeBefore > 0 ? (1 - sizeAfter / sizeBefore) * 100 : 0,
    ep: cmp.after.ep,
    relL2,
    sqnrDb,
    maxAbsDiff,
    dimsMatch: cmp.dimsMatch,
  };
}

function humanBytes(n) {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${units[u]}`;
}

// Renders computeQuantizationQuality()'s result as a small HTML table
// (a plain string -- the caller assigns it to an element's innerHTML). All
// interpolated values are our own computed numbers/labels, never
// user-supplied text.
export function renderQuantizationQuality(q) {
  const sizeRow = `${humanBytes(q.sizeBefore)} → ${humanBytes(q.sizeAfter)} (${q.sizeReductionPct >= 0 ? "−" : "+"}${Math.abs(q.sizeReductionPct).toFixed(1)}%)`;
  const sqnrText = Number.isFinite(q.sqnrDb) ? `${q.sqnrDb.toFixed(1)} dB` : "∞ dB (bit-exact on this sample)";
  return `
    <table style="border-collapse: collapse; font-size: 0.85em;">
      <tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted);">Model size</td><td>${sizeRow}</td></tr>
      <tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted);">Execution provider</td><td>${q.ep}</td></tr>
      <tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted);">Relative L2 error</td><td>${(q.relL2 * 100).toFixed(2)}%</td></tr>
      <tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted);">SQNR</td><td>${sqnrText}</td></tr>
      <tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted);">Max abs diff</td><td>${q.maxAbsDiff.toPrecision(4)}</td></tr>
    </table>
    <p class="tool-note" style="margin-top: 0.3em;">
      Measured on one synthetic random input (not representative calibration
      data) via onnxruntime-web -- a quick sanity check, not a rigorous
      accuracy evaluation. Relative L2 error and SQNR are computed the same
      way onnxsim's own Python quantization tests check result quality.
    </p>`;
}

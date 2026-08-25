// Static, calibration-free INT8-quantization risk pre-check for the
// "Quantize a model" panel (quantize_ui.mjs): runs onnxsim's
// estimate_model_quantization_drop analysis -- a C++ port of
// onnxsim.precision_estimator (onnxsim/precision_estimator.h/.cpp), kept in
// exact sync with the Python module it mirrors -- entirely via the page's own
// WASM module. No model execution and no calibration data are needed, so
// this is available the instant a model is picked, before the user has even
// chosen a quantize method or run anything through onnxruntime-web (unlike
// quantize_metrics.mjs's post-quantize quality report, which does need
// onnxruntime-web to actually run the models).
//
// Loaded as its own top-level module (a <script type="module"> tag right
// after quantize_ui.mjs's own in index.html), not imported by it.
// resolveQuantizeInput/getRuntime are imported lazily, inside the click
// handler below, rather than at module scope: quantize_ui.mjs pulls in
// inference_browser.mjs, which touches `document` unconditionally at module
// scope (no browser-environment guard), so a static top-level import here
// would make this module -- and therefore its pure, DOM-free helpers below --
// impossible to load under plain Node (see test/quantize_risk_estimate.test.mjs,
// which imports only those helpers and needs this module's own top-level code
// to run without a DOM).

// Runs onnxsim_estimate_quantization_drop on `bytes` via `runtime` (the
// page's WASM module) and returns the parsed result. Throws if the model
// fails to parse or the analysis otherwise fails (see the console for the
// WASM module's own error message in that case).
export function estimateQuantizationRisk(runtime, bytes) {
  const buf = bytes.slice().buffer;
  const result = runtime.onnxsim_estimate_quantization_drop(buf);
  if (!result) {
    throw new Error("quantization risk estimate failed (see console for details)");
  }
  return result;
}

const RISK_LABELS = {
  safe: { text: "safe", color: "#2a9d3f" },
  degraded: { text: "degraded", color: "#c98a10" },
  unsafe: { text: "unsafe", color: "#c0392b" },
};

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

function nodeListHtml(names, max = 6) {
  if (names.length === 0) return "none";
  const shown = names.slice(0, max).map(esc).join(", ");
  return names.length > max
    ? `${shown}, … (+${names.length - max} more)`
    : shown;
}

// Renders estimateQuantizationRisk()'s result as a small HTML summary (a
// plain string -- the caller assigns it to an element's innerHTML).
export function renderQuantizationRisk(est) {
  const label = RISK_LABELS[est.riskLevel] || { text: est.riskLevel, color: "inherit" };
  const errorText = Number.isFinite(est.estimatedRelativeError)
    ? `${(est.estimatedRelativeError * 100).toFixed(2)}%`
    : "n/a (accumulator overflow -- see unsafe nodes below)";

  const rows = [
    `<tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted);">Estimated relative error</td><td>${errorText}</td></tr>`,
  ];
  if (Number.isFinite(est.worstOutlierRatio)) {
    rows.push(
      `<tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted);">Worst channel outlier ratio</td><td>${est.worstOutlierRatio.toFixed(1)}×</td></tr>`,
    );
  }
  if (est.unsafeNodes.length > 0) {
    rows.push(
      `<tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted); vertical-align: top;">Unsafe (accumulator overflow)</td><td>${nodeListHtml(est.unsafeNodes)}</td></tr>`,
    );
  }
  if (est.outlierRiskNodes.length > 0) {
    rows.push(
      `<tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted); vertical-align: top;">Outlier-risk nodes</td><td>${nodeListHtml(est.outlierRiskNodes)}</td></tr>`,
    );
  }

  return `
    <p style="margin: 0.3em 0;"><b>INT8 quantization risk:</b>
      <span style="color: ${label.color}; font-weight: 600;">${esc(label.text)}</span>
      (${est.totalNodesAnalyzed} MatMul/Gemm/Conv/Attention node(s) analyzed)</p>
    <table style="border-collapse: collapse; font-size: 0.85em;">${rows.join("")}</table>
    <p class="tool-note" style="margin-top: 0.3em;">
      Static analysis of this model's own weights and shapes -- no execution
      or calibration data needed. This is a heuristic screening signal (a
      whole-model rollup of onnxsim's per-node accumulator-overflow and
      outlier-ratio checks), not a certified bound -- use "report result
      quality" below after quantizing for a real, data-driven measurement on
      this model.
    </p>`;
}

function initQuantizeRiskPanel() {
  const btn = document.getElementById("quantize-risk-button");
  if (!btn) return;
  const statusEl = document.getElementById("quantize-risk-status");
  const resultEl = document.getElementById("quantize-risk-result");

  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };

  btn.addEventListener("click", async () => {
    const sourceEl = document.querySelector('input[name="quantize-source"]:checked');
    const source = sourceEl ? sourceEl.value : "original";
    const fileInput = document.getElementById("file-input");
    btn.disabled = true;
    if (resultEl) {
      resultEl.style.display = "none";
      resultEl.innerHTML = "";
    }
    try {
      const { getRuntime, resolveQuantizeInput } = await import("./quantize_ui.mjs");
      setStatus("loading model…");
      const { bytes } = await resolveQuantizeInput(source, fileInput);

      setStatus("analyzing…");
      const runtime = await getRuntime();
      const est = estimateQuantizationRisk(runtime, bytes);

      if (resultEl) {
        resultEl.innerHTML = renderQuantizationRisk(est);
        resultEl.style.display = "";
      }
      setStatus("done.");
    } catch (err) {
      setStatus("risk check error: " + (err && err.message ? err.message : err));
    } finally {
      btn.disabled = false;
    }
  });
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  initQuantizeRiskPanel();
}

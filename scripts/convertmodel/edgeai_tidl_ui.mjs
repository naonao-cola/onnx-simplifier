// Browser glue for the "Check Edge AI (TI TIDL) compatibility" panel.
//
// Runs edgeai_tidl_check.mjs's static analysis on either the original
// upload or the Convert section's own simplify/optimize output (an
// "edgeai-source" radio picks which, same convention as
// quantize_ui.mjs's "Quantize input" radio) -- entirely client-side, no
// model execution and no download, so results are available the instant a
// model is picked.
//
// Loaded as its own top-level module (a <script type="module"> tag in
// index.html), not imported by anything else. resolveOriginalModelBytes is
// imported lazily, inside the click handler, for the same reason
// quantize_risk_estimate.mjs does: inference_browser.mjs touches `document`
// unconditionally at module scope, so importing it eagerly here would make
// this file -- and therefore edgeai_tidl_check.mjs's pure, DOM-free
// helpers it re-exports nothing from -- impossible to load under plain
// Node. (edgeai_tidl_check.mjs itself has no such import, which is exactly
// why test/edgeai_tidl_check.test.mjs can import it directly with no DOM.)
//
// This is a static heuristic, not a real TIDL compile -- see
// edgeai_tidl_check.mjs's header comment and scripts/edgeai/tidl_ops.py's
// docstring (the Python module this ports) for exactly what that does and
// doesn't confirm, and scripts/edgeai/real_compile.py for what an actual
// compile via TI's real onnxruntime_tidl/tidl_tools binaries looks like
// (Python-only; there is no in-browser or WASM equivalent -- those are
// native x86 binaries downloaded from TI's own servers, not something a
// WASM build could embed).

import { analyzeTidlCompat, PRECISION_NOTES } from "./edgeai_tidl_check.mjs";

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

function opTypeListHtml(opTypeCounts, max = 8) {
  const entries = [...opTypeCounts.entries()].sort((a, b) => b[1] - a[1]);
  const shown = entries
    .slice(0, max)
    .map(([opType, count]) => `${esc(opType)}×${count}`)
    .join(", ");
  return entries.length > max ? `${shown}, … (+${entries.length - max} more)` : shown;
}

// Renders analyzeTidlCompat()'s result as a small HTML summary (a plain
// string -- the caller assigns it to an element's innerHTML).
export function renderTidlCompat(result) {
  const isFull = result.coverage === "full";
  const label = isFull
    ? { text: "full (no known blocker found)", color: "#2a9d3f" }
    : { text: "partial (see below)", color: "#c98a10" };

  const rows = [];
  if (result.hasDynamicShape) {
    rows.push(
      `<tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted); vertical-align: top;">Dynamic shape</td>` +
        `<td>a graph input has a symbolic or unranked dimension; TIDL requires fully static input shapes ` +
        `(fix with onnxsim's <code>overwrite_input_shapes</code>)</td></tr>`,
    );
  }
  if (result.blockers.length > 0) {
    const items = result.blockers
      .map((b) => `<li><code>${esc(b.opType)}</code> — ${esc(b.reason)}</li>`)
      .join("");
    rows.push(
      `<tr><td style="padding: 2px 0.8em 2px 0; color: var(--muted); vertical-align: top;">Blocking op type(s)</td>` +
        `<td><ul style="margin: 0; padding-left: 1.2em;">${items}</ul></td></tr>`,
    );
  }

  return `
    <p style="margin: 0.3em 0;"><b>TIDL coverage:</b>
      <span style="color: ${label.color}; font-weight: 600;">${esc(label.text)}</span>
      (${result.opTypeCounts.size} distinct op type(s): ${opTypeListHtml(result.opTypeCounts)})</p>
    ${rows.length > 0 ? `<table style="border-collapse: collapse; font-size: 0.85em;">${rows.join("")}</table>` : ""}
    <p class="tool-note" style="margin-top: 0.3em;">
      Static heuristic ported from
      <a href="https://github.com/onnxsim/onnxsim/blob/master/scripts/edgeai/tidl_ops.py" target="_blank" rel="noopener">scripts/edgeai/tidl_ops.py</a>,
      checked against edgeai-tidl-tools' own published <code>docs/operators.md</code> --
      not a real TIDL compile. "Full" means no known blocker was found, not a
      guarantee this exact graph compiles on the real toolchain (per-op
      attribute limits, e.g. supported <code>Resize</code> modes or
      <code>Conv</code> group/dilation ranges, aren't checked here).
    </p>`;
}

function renderPrecisionNote(precision) {
  const note = PRECISION_NOTES[precision] || "";
  return note
    ? `<p class="tool-note" style="margin-top: 0.3em;">${esc(note)}</p>`
    : "";
}

function initEdgeaiTidlPanel() {
  const btn = document.getElementById("edgeai-tidl-button");
  if (!btn) return;
  const fileInput = document.getElementById("file-input");
  const statusEl = document.getElementById("edgeai-tidl-status");
  const resultEl = document.getElementById("edgeai-tidl-result");
  const precisionEl = document.getElementById("edgeai-tidl-precision");
  const precisionNoteEl = document.getElementById("edgeai-tidl-precision-note");

  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };

  const updatePrecisionNote = () => {
    if (!precisionNoteEl) return;
    precisionNoteEl.innerHTML = renderPrecisionNote(precisionEl ? precisionEl.value : "int8");
  };
  if (precisionEl) precisionEl.addEventListener("change", updatePrecisionNote);
  updatePrecisionNote();

  btn.addEventListener("click", async () => {
    const sourceEl = document.querySelector('input[name="edgeai-tidl-source"]:checked');
    const source = sourceEl ? sourceEl.value : "original";
    btn.disabled = true;
    if (resultEl) {
      resultEl.style.display = "none";
      resultEl.innerHTML = "";
    }
    try {
      setStatus("loading model…");
      let bytes;
      if (source === "converted") {
        const converted = window.__onnxsimConverted;
        if (!converted || !converted.bytes) {
          throw new Error(
            "no onnxsim-simplified result yet -- run Simplify/Optimize in the " +
              "Convert section above first, or switch back to 'Original upload'.",
          );
        }
        bytes = converted.bytes;
      } else {
        const { resolveOriginalModelBytes } = await import("./inference_browser.mjs");
        ({ bytes } = await resolveOriginalModelBytes(fileInput));
      }

      setStatus("analyzing…");
      const result = analyzeTidlCompat(bytes);

      if (resultEl) {
        resultEl.innerHTML = renderTidlCompat(result);
        resultEl.style.display = "";
      }
      setStatus("done.");
    } catch (err) {
      setStatus("edge AI check error: " + (err && err.message ? err.message : err));
    } finally {
      btn.disabled = false;
    }
  });
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  initEdgeaiTidlPanel();
}

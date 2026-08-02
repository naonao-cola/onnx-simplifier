// Browser glue for the converter page's "Dynamic dimensions (dim_param)" panel.
//
// Lists a model's symbolic axis names (ONNX `dim_param`s) on its graph inputs
// and outputs, before vs after simplify/optimize, so you can see how the
// conversion changed the model's dynamic axes -- e.g. a "batch"/"sequence" axis
// preserved, or a symbolic dimension folded to a fixed size.
//
// Wiring mirrors the Netron panel (netron_view.mjs): the "before" side watches
// the file input directly and exposes a `window.dimParamsShowBefore(bytes,
// name)` hook for Hugging Face loads (which bypass the file input); the "after"
// side is driven by `window.dimParamsShowAfter(dataUrl, name)`, called from the
// converter worker glue in index.html once a conversion finishes.

import { dataUrlToArrayBuffer } from "./netron.mjs";
import {
  readShapes,
  formatShape,
  isParamDim,
  dimParamNames,
  diffDimParams,
} from "./shapes.mjs";

// Latest parsed shapes for each side; null until a model arrives there.
const state = { before: null, after: null };

// Escape text destined for innerHTML. Tensor names and dim_param names come
// from a user-supplied model, so never interpolate them raw.
function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Render one shape with its symbolic axes emphasized, as safe HTML.
function shapeHtml(shape) {
  if (shape == null) return "<span class=\"dp-dim\">(unknown rank)</span>";
  if (shape.length === 0) return "<span class=\"dp-dim\">scalar</span>";
  const parts = shape.map((dim) => {
    const tok = esc(formatShapeToken(dim));
    return isParamDim(dim) ? `<span class="dp-param">${tok}</span>` : `<span class="dp-dim">${tok}</span>`;
  });
  return "[" + parts.join(", ") + "]";
}

// One axis token (kept in sync with shapes.mjs formatDim, but per-dim so the
// symbolic ones can be wrapped for emphasis above).
function formatShapeToken(dim) {
  if (dim == null) return "?";
  if (isParamDim(dim)) return dim.param;
  if (dim.value != null) return String(dim.value);
  return "?";
}

// A table of a model's input/output tensors and their shapes.
function tensorTableHtml(shapes) {
  const row = (t, io) =>
    `<tr><td>${esc(io)}</td><td><code>${esc(t.name)}</code></td><td>${shapeHtml(t.shape)}</td></tr>`;
  const rows = [
    ...shapes.inputs.map((t) => row(t, "input")),
    ...shapes.outputs.map((t) => row(t, "output")),
  ].join("");
  return (
    `<table class="dp-table"><thead><tr><th>I/O</th><th>tensor</th><th>shape</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>`
  );
}

// The dim_param name chips for one model ("(none)" when fully static).
function namesHtml(shapes) {
  const names = dimParamNames(shapes);
  if (names.length === 0) return "<em>(none — fully static)</em>";
  return names.map((n) => `<span class="dp-param">${esc(n)}</span>`).join(" ");
}

// One side ("before" | "after") of the comparison.
function paneHtml(title, shapes) {
  if (!shapes) {
    return `<div class="dp-pane"><div class="dp-head">${esc(title)}</div><em>—</em></div>`;
  }
  return (
    `<div class="dp-pane"><div class="dp-head">${esc(title)}</div>` +
    `<div class="dp-names">dim_params: ${namesHtml(shapes)}</div>` +
    tensorTableHtml(shapes) +
    `</div>`
  );
}

// A one-line summary of what the conversion did to the dynamic axes.
function diffHtml() {
  if (!state.before || !state.after) return "";
  const d = diffDimParams(state.before, state.after);
  const chips = (names) => names.map((n) => `<span class="dp-param">${esc(n)}</span>`).join(" ");
  const bits = [];
  if (d.removed.length) bits.push(`made static: ${chips(d.removed)}`);
  if (d.added.length) bits.push(`newly symbolic: ${chips(d.added)}`);
  if (!bits.length) {
    bits.push(
      d.common.length
        ? `dim_params unchanged (${chips(d.common)})`
        : "both models are fully static",
    );
  }
  return `<div class="dp-diff">After simplify/optimize — ${bits.join("; ")}.</div>`;
}

function render() {
  const container = document.getElementById("dim-params-content");
  if (!container) return;
  container.innerHTML =
    diffHtml() +
    `<div class="dp-panes">${paneHtml("Before", state.before)}${paneHtml("After", state.after)}</div>`;
}

// Parse model bytes (Uint8Array | ArrayBuffer) into shapes for one side and
// re-render. Never throws out to the caller — a parse failure just clears that
// side, so a malformed model can't take the panel (or its host page) down.
function setSide(which, data) {
  try {
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    state[which] = readShapes(bytes);
  } catch (err) {
    console.error(`dim_params (${which}):`, err);
    state[which] = null;
  }
  render();
}

function initShapesPanel() {
  render(); // draw the empty placeholder up front

  const fileInput = document.getElementById("file-input");
  if (fileInput) {
    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      // A new upload invalidates the previous "after" listing until the
      // conversion for this model finishes.
      state.after = null;
      try {
        setSide("before", new Uint8Array(await file.arrayBuffer()));
      } catch (err) {
        console.error("dim_params (before):", err);
      }
    });
  }

  // Hugging Face downloads bypass the file input; the loader drives us here.
  window.dimParamsShowBefore = (data, _name) => {
    state.after = null;
    setSide("before", data);
  };

  // Called by the converter worker glue in index.html when a result is ready.
  window.dimParamsShowAfter = (dataUrl, _name) => {
    try {
      setSide("after", dataUrlToArrayBuffer(dataUrl));
    } catch (err) {
      console.error("dim_params (after):", err);
    }
  };
}

initShapesPanel();

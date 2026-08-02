// Parse the converter page's input configuration from URL query parameters, so a
// model (and the conversion options) can be driven straight from a shareable
// link — e.g. `?model=https://…/foo.onnx` or `?model=onnxmodelzoo/resnet18d_Opset18`.
//
// Kept pure (no DOM/network) so the parsing is unit-tested; hf_load.mjs imports
// these and does the browser wiring (prefilling controls and driving the load).
//
// Recognized parameters (aliases in parentheses):
//   model (hf, url, input) — a Hugging Face repo id or a direct .onnx URL; the
//     same reference the "Load from Hugging Face" box accepts.
//   backend (testcase, test) — an ONNX backend test-case URL to prefill the
//     "Run an ONNX backend test case" panel.
//   autoload (run, convert) — whether to start the conversion automatically once
//     a model is given (default: true).
//   optimizer (opt) — "simplify" | "optimize" | "optimize_fixed" | "inline" |
//     the single-pass debug modes.
//   constant_fold (cf), shape_inference (si) — booleans.
//   inline_functions (inline) — boolean: inline local functions before the
//     simplify / optimize / fixed-optimize transform.
//   tensor_size_threshold (tst), target_opset (opset) — integers.

// Parse a boolean-ish value: an empty value (a bare `?cf`) means "on"; returns
// null for absent or unrecognized so callers can tell "not set" from false.
function toBool(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim().toLowerCase();
  if (s === "" || s === "1" || s === "true" || s === "yes" || s === "on") return true;
  if (s === "0" || s === "false" || s === "no" || s === "off") return false;
  return null;
}

function toInt(v) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

// The converter's mode radio values: the three optimizers plus the single-pass
// debug modes (each runs one of Simplify's fixed-point steps once).
const OPTIMIZERS = [
  "simplify",
  "optimize",
  "optimize_fixed",
  "inline",
  "infer_shapes",
  "data_propagation",
  "fold_constant",
];

// Parse a `location.search` (or any query string) into a normalized config.
// Fields are null when the parameter is absent, except `autoload` which
// defaults to true (only meaningful when a model is given).
export function parseInputParams(search) {
  const p = new URLSearchParams(search || "");
  const first = (...keys) => {
    for (const k of keys) {
      const v = p.get(k);
      if (v !== null) return v;
    }
    return null;
  };
  const trimmed = (v) => (v === null ? null : v.trim() || null);

  // The conversion processor / mode: optimizer (opt) is the canonical key;
  // mode / processor are friendlier aliases.
  const optimizerRaw = trimmed(first("optimizer", "opt", "mode", "processor"));
  const optimizer = optimizerRaw && OPTIMIZERS.includes(optimizerRaw) ? optimizerRaw : null;
  const autoloadRaw = toBool(first("autoload", "run", "convert"));

  return {
    model: trimmed(first("model", "hf", "url", "input")),
    backend: trimmed(first("backend", "testcase", "test")),
    // The ONNX text graph to parse (onnx.parser syntax). Not trimmed of interior
    // whitespace — only null when the key is absent or empty — since the graph
    // body's own formatting matters.
    graph: first("graph", "text"),
    autoload: autoloadRaw === null ? true : autoloadRaw,
    optimizer,
    constantFold: toBool(first("constant_fold", "cf")),
    shapeInference: toBool(first("shape_inference", "si")),
    // Whether to inline local functions before simplify / optimize / fixed
    // optimize (the "inline local functions first" checkbox).
    inlineFunctions: toBool(first("inline_functions", "inline")),
    tensorSizeThreshold: toInt(first("tensor_size_threshold", "tst")),
    targetOpset: toInt(first("target_opset", "opset")),
  };
}

// Every query key that names an input source; setting one canonical input
// clears all of these so the URL always reflects a single current input.
const INPUT_KEYS = [
  "model", "hf", "url", "input",
  "backend", "testcase", "test",
  "graph", "text",
];

// The canonical query key for each input kind.
const INPUT_KEY = { backend: "backend", graph: "graph", model: "model" };

// Return a new query string (leading "?" or "") that names `value` as the input,
// under the canonical key for `kind` ("backend"/"graph", else "model"), dropping
// every other input key while preserving option params (optimizer, cf, …).
// Pure, for unit testing.
export function setInputParam(search, kind, value) {
  const p = new URLSearchParams(search || "");
  for (const k of INPUT_KEYS) p.delete(k);
  p.set(INPUT_KEY[kind] || "model", value);
  const s = p.toString();
  return s ? `?${s}` : "";
}

// Set (or, for an empty value, delete) a single non-input param, preserving
// everything else — used to reflect the conversion processor (?optimizer=).
// Pure, for unit testing.
export function setUrlParam(search, key, value) {
  const p = new URLSearchParams(search || "");
  if (value === null || value === undefined || value === "") {
    p.delete(key);
  } else {
    p.set(key, value);
  }
  const s = p.toString();
  return s ? `?${s}` : "";
}

// Reflect a single param in the address bar (history.replaceState). Best-effort.
export function syncUrlParam(key, value) {
  try {
    if (typeof window === "undefined" || !window.history) return;
    const search = setUrlParam(window.location.search, key, value);
    window.history.replaceState(null, "", window.location.pathname + search + window.location.hash);
  } catch {
    // ignore — a failed URL update must not break anything.
  }
}

// Reflect a just-set input model in the address bar (history.replaceState, so no
// navigation / history entry), so the link is shareable and reproducible. Used
// for the Hugging Face / URL loader ("model") and the backend-test panel
// ("backend"); uploaded local files have no URL representation and are skipped
// by their callers. Best-effort: never throws into the caller.
export function syncInputUrl(kind, value) {
  try {
    if (typeof window === "undefined" || !window.history || !value) return;
    const search = setInputParam(window.location.search, kind, value);
    const url = window.location.pathname + search + window.location.hash;
    window.history.replaceState(null, "", url);
  } catch {
    // ignore — a failed URL update must not break loading.
  }
}

// Prefill the converter's option controls from a parsed config, so both an
// auto-run and a later manual run honor the link's options. `doc` is the
// document (injected for testability); only set fields are touched. Returns the
// list of control ids it changed (handy for tests / logging).
export function applyOptionParams(params, doc) {
  const changed = [];
  const setChecked = (id, val) => {
    if (val === null || val === undefined) return;
    const el = doc.getElementById(id);
    if (el) {
      el.checked = val;
      changed.push(id);
    }
  };
  const setValue = (id, val) => {
    if (val === null || val === undefined) return;
    const el = doc.getElementById(id);
    if (el) {
      el.value = String(val);
      changed.push(id);
    }
  };
  if (params.optimizer) {
    const radio = doc.getElementById(`optimizer_${params.optimizer}`);
    if (radio) {
      radio.checked = true;
      changed.push(`optimizer_${params.optimizer}`);
    }
  }
  setChecked("id_simplify_constant_fold", params.constantFold);
  setChecked("id_simplify_shape_inference", params.shapeInference);
  setChecked("id_inline_functions", params.inlineFunctions);
  setValue("id_simplify_tensor_size_threshold", params.tensorSizeThreshold);
  setValue("id_simplify_target_opset", params.targetOpset);
  return changed;
}

// Unit tests for query_params.mjs, the pure parser + control-prefiller behind
// the converter page's URL-driven input (e.g. ?model=…). No DOM/network needed;
// applyOptionParams is driven through a tiny fake document.
//
// Usage:
//   node test/query_params.test.mjs

import assert from "node:assert/strict";
import {
  parseInputParams,
  applyOptionParams,
  setInputParam,
  setUrlParam,
  buildShareSearch,
} from "../query_params.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("parses a model URL and defaults autoload to true", () => {
  const c = parseInputParams("?model=https://example.com/foo.onnx");
  assert.equal(c.model, "https://example.com/foo.onnx");
  assert.equal(c.autoload, true);
  assert.equal(c.optimizer, null);
  assert.equal(c.constantFold, null);
});

check("accepts model aliases (hf, url, input)", () => {
  assert.equal(parseInputParams("?hf=onnxmodelzoo/resnet18d_Opset18").model, "onnxmodelzoo/resnet18d_Opset18");
  assert.equal(parseInputParams("?url=https://x/y.onnx").model, "https://x/y.onnx");
  assert.equal(parseInputParams("?input=owner/repo").model, "owner/repo");
});

check("returns null model when absent", () => {
  const c = parseInputParams("?optimizer=optimize");
  assert.equal(c.model, null);
  assert.equal(c.optimizer, "optimize");
});

check("parses booleans, empty flag, and aliases", () => {
  assert.equal(parseInputParams("?cf=0").constantFold, false);
  assert.equal(parseInputParams("?constant_fold=true").constantFold, true);
  assert.equal(parseInputParams("?cf").constantFold, true); // bare flag = on
  assert.equal(parseInputParams("?si=off").shapeInference, false);
  assert.equal(parseInputParams("?shape_inference=1").shapeInference, true);
});

check("autoload can be turned off", () => {
  assert.equal(parseInputParams("?model=x&autoload=0").autoload, false);
  assert.equal(parseInputParams("?model=x&run=false").autoload, false);
  assert.equal(parseInputParams("?model=x").autoload, true);
});

check("parses integer options and ignores garbage", () => {
  assert.equal(parseInputParams("?tst=500000").tensorSizeThreshold, 500000);
  assert.equal(parseInputParams("?tensor_size_threshold=42").tensorSizeThreshold, 42);
  assert.equal(parseInputParams("?opset=17").targetOpset, 17);
  assert.equal(parseInputParams("?opset=notanumber").targetOpset, null);
});

check("only accepts known optimizer values, incl. inline and single-pass modes", () => {
  assert.equal(parseInputParams("?optimizer=optimize_fixed").optimizer, "optimize_fixed");
  assert.equal(parseInputParams("?optimizer=inline").optimizer, "inline");
  assert.equal(parseInputParams("?optimizer=fold_constant").optimizer, "fold_constant");
  assert.equal(parseInputParams("?optimizer=infer_shapes").optimizer, "infer_shapes");
  assert.equal(parseInputParams("?optimizer=data_propagation").optimizer, "data_propagation");
  assert.equal(parseInputParams("?optimizer=bogus").optimizer, null);
});

check("parses the inline_functions flag and its alias", () => {
  // Absent → null so callers can tell "not set" from off.
  assert.equal(parseInputParams("?model=x").inlineFunctions, null);
  assert.equal(parseInputParams("?inline_functions=1").inlineFunctions, true);
  assert.equal(parseInputParams("?inline_functions=0").inlineFunctions, false);
  assert.equal(parseInputParams("?inline").inlineFunctions, true); // bare alias = on
  assert.equal(parseInputParams("?inline=off").inlineFunctions, false);
});

check("parses the autorun (auto-run inference) flag and its aliases", () => {
  // Absent → null so callers can tell "not set" from off.
  assert.equal(parseInputParams("?model=x").autoRunInference, null);
  assert.equal(parseInputParams("?autorun=1").autoRunInference, true);
  assert.equal(parseInputParams("?autorun").autoRunInference, true); // bare flag = on
  assert.equal(parseInputParams("?autorun=0").autoRunInference, false);
  assert.equal(parseInputParams("?autoinfer=on").autoRunInference, true);
  assert.equal(parseInputParams("?autorun_inference=false").autoRunInference, false);
});

check("optimizer accepts mode/processor/opt aliases", () => {
  assert.equal(parseInputParams("?mode=simplify").optimizer, "simplify");
  assert.equal(parseInputParams("?processor=fold_constant").optimizer, "fold_constant");
  assert.equal(parseInputParams("?opt=optimize").optimizer, "optimize");
});

check("parses a text graph param (graph/text), preserving interior whitespace", () => {
  const graph = "agraph (float[N] X) => (float[N] Y) {\n  Y = Relu(X)\n}";
  const c = parseInputParams("?graph=" + encodeURIComponent(graph));
  assert.equal(c.graph, graph);
  assert.equal(parseInputParams("?text=" + encodeURIComponent(graph)).graph, graph);
  assert.equal(parseInputParams("?model=x").graph, null);
});

check("parses a backend test-case URL", () => {
  const c = parseInputParams("?backend=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_relu");
  assert.ok(c.backend.endsWith("test_relu"));
});

// A minimal document stand-in whose getElementById returns recording elements.
function fakeDoc(ids) {
  const els = {};
  for (const id of ids) els[id] = { checked: false, value: "" };
  return { els, getElementById: (id) => els[id] || null };
}

check("applyOptionParams prefills only the given controls", () => {
  const doc = fakeDoc([
    "optimizer_optimize",
    "id_simplify_constant_fold",
    "id_simplify_shape_inference",
    "id_inline_functions",
    "id_simplify_tensor_size_threshold",
    "id_simplify_target_opset",
    "inference-autorun",
  ]);
  const params = parseInputParams("?optimizer=optimize&cf=0&si=1&inline=1&tst=1000&opset=18&autorun=1");
  const changed = applyOptionParams(params, doc);
  assert.equal(doc.els.optimizer_optimize.checked, true);
  assert.equal(doc.els.id_simplify_constant_fold.checked, false);
  assert.equal(doc.els.id_simplify_shape_inference.checked, true);
  assert.equal(doc.els.id_inline_functions.checked, true);
  assert.equal(doc.els.id_simplify_tensor_size_threshold.value, "1000");
  assert.equal(doc.els.id_simplify_target_opset.value, "18");
  assert.equal(doc.els["inference-autorun"].checked, true);
  assert.ok(changed.includes("optimizer_optimize"));
  assert.ok(changed.includes("id_inline_functions"));
  assert.ok(changed.includes("inference-autorun"));
});

check("applyOptionParams leaves untouched controls alone", () => {
  const doc = fakeDoc(["id_simplify_constant_fold"]);
  doc.els.id_simplify_constant_fold.checked = true;
  const changed = applyOptionParams(parseInputParams("?model=x"), doc);
  // No option params present, so nothing is changed.
  assert.equal(changed.length, 0);
  assert.equal(doc.els.id_simplify_constant_fold.checked, true);
});

check("setInputParam sets model and drops other input keys", () => {
  const s = setInputParam("?hf=old/repo&optimizer=optimize", "model", "owner/new");
  const p = new URLSearchParams(s);
  assert.equal(p.get("model"), "owner/new");
  assert.equal(p.get("hf"), null); // old input key dropped
  assert.equal(p.get("optimizer"), "optimize"); // option param preserved
});

check("setInputParam sets backend and clears model", () => {
  const s = setInputParam("?model=owner/repo", "backend", "https://github.com/onnx/onnx/tree/main/x");
  const p = new URLSearchParams(s);
  assert.equal(p.get("backend"), "https://github.com/onnx/onnx/tree/main/x");
  assert.equal(p.get("model"), null);
});

check("setInputParam round-trips through parseInputParams", () => {
  const s = setInputParam("", "model", "https://x/y.onnx");
  assert.ok(s.startsWith("?"));
  assert.equal(parseInputParams(s).model, "https://x/y.onnx");
});

check("setInputParam sets graph and clears other inputs", () => {
  const s = setInputParam("?model=owner/repo&optimizer=fold_constant", "graph", "agraph () => () {}");
  const p = new URLSearchParams(s);
  assert.equal(p.get("graph"), "agraph () => () {}");
  assert.equal(p.get("model"), null);
  assert.equal(p.get("optimizer"), "fold_constant"); // option preserved
});

check("setUrlParam sets/deletes one param, preserving inputs", () => {
  assert.equal(new URLSearchParams(setUrlParam("?model=x", "optimizer", "optimize")).get("optimizer"), "optimize");
  // Existing input param is preserved.
  assert.equal(new URLSearchParams(setUrlParam("?model=x", "optimizer", "optimize")).get("model"), "x");
  // Empty value deletes the key.
  assert.equal(new URLSearchParams(setUrlParam("?model=x&optimizer=optimize", "optimizer", "")).get("optimizer"), null);
});

// A document stand-in for buildShareSearch: the option controls addressed by id
// plus a querySelector that resolves the single supported selector, the checked
// optimizer radio. `opts` overrides the defaults (all at their index.html state).
function shareDoc(opts = {}) {
  const optimizer = opts.optimizer || "simplify";
  const els = {
    id_simplify_constant_fold: { checked: opts.constantFold ?? true },
    id_simplify_shape_inference: { checked: opts.shapeInference ?? true },
    id_simplify_tensor_size_threshold: { value: String(opts.tst ?? 1000000000) },
    id_simplify_target_opset: { value: String(opts.opset ?? 0) },
    "inference-autorun": { checked: opts.autoRunInference ?? false },
  };
  return {
    getElementById: (id) => els[id] || null,
    querySelector: (sel) =>
      sel === 'input[name="optimizer"]:checked' ? { value: optimizer } : null,
  };
}

check("buildShareSearch keeps a compact URL when all options are default", () => {
  const s = buildShareSearch("?model=owner/repo", shareDoc());
  const p = new URLSearchParams(s);
  assert.equal(p.get("model"), "owner/repo"); // input preserved
  assert.equal(p.get("optimizer"), "simplify"); // processor always spelled out
  // Defaults are dropped to keep the link short.
  assert.equal(p.get("cf"), null);
  assert.equal(p.get("si"), null);
  assert.equal(p.get("tst"), null);
  assert.equal(p.get("opset"), null);
  assert.equal(p.get("autorun"), null); // unchecked (default) → dropped
});

check("buildShareSearch encodes every changed option", () => {
  const doc = shareDoc({
    optimizer: "optimize",
    constantFold: false,
    shapeInference: false,
    tst: 500000,
    opset: 18,
    autoRunInference: true,
  });
  const s = buildShareSearch("?model=owner/repo", doc);
  const p = new URLSearchParams(s);
  assert.equal(p.get("model"), "owner/repo");
  assert.equal(p.get("optimizer"), "optimize");
  assert.equal(p.get("cf"), "0");
  assert.equal(p.get("si"), "0");
  assert.equal(p.get("tst"), "500000");
  assert.equal(p.get("opset"), "18");
  assert.equal(p.get("autorun"), "1"); // checked → emitted
});

check("buildShareSearch round-trips through parseInputParams", () => {
  const doc = shareDoc({ optimizer: "optimize", constantFold: false, opset: 17 });
  const s = buildShareSearch("?backend=https://x/tree/y", doc);
  const c = parseInputParams(s);
  assert.equal(c.backend, "https://x/tree/y");
  assert.equal(c.optimizer, "optimize");
  assert.equal(c.constantFold, false);
  assert.equal(c.shapeInference, null); // left at default → absent → default on load
  assert.equal(c.targetOpset, 17);
});

check("buildShareSearch overwrites a stale option already in the query", () => {
  // A previously shared cf=0 must be cleared once the control is back to default.
  const s = buildShareSearch("?model=x&cf=0&opset=18", shareDoc({ opset: 0 }));
  const p = new URLSearchParams(s);
  assert.equal(p.get("cf"), null); // re-checked → default → dropped
  assert.equal(p.get("opset"), null); // back to 0 (keep) → dropped
  assert.equal(p.get("model"), "x");
});

console.log(`PASS: ${passed} checks`);

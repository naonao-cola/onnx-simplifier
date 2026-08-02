// Unit tests for query_params.mjs, the pure parser + control-prefiller behind
// the converter page's URL-driven input (e.g. ?model=…). No DOM/network needed;
// applyOptionParams is driven through a tiny fake document.
//
// Usage:
//   node test/query_params.test.mjs

import assert from "node:assert/strict";
import { parseInputParams, applyOptionParams } from "../query_params.mjs";

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

check("only accepts known optimizer values", () => {
  assert.equal(parseInputParams("?optimizer=optimize_fixed").optimizer, "optimize_fixed");
  assert.equal(parseInputParams("?optimizer=bogus").optimizer, null);
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
    "id_simplify_tensor_size_threshold",
    "id_simplify_target_opset",
  ]);
  const params = parseInputParams("?optimizer=optimize&cf=0&si=1&tst=1000&opset=18");
  const changed = applyOptionParams(params, doc);
  assert.equal(doc.els.optimizer_optimize.checked, true);
  assert.equal(doc.els.id_simplify_constant_fold.checked, false);
  assert.equal(doc.els.id_simplify_shape_inference.checked, true);
  assert.equal(doc.els.id_simplify_tensor_size_threshold.value, "1000");
  assert.equal(doc.els.id_simplify_target_opset.value, "18");
  assert.ok(changed.includes("optimizer_optimize"));
});

check("applyOptionParams leaves untouched controls alone", () => {
  const doc = fakeDoc(["id_simplify_constant_fold"]);
  doc.els.id_simplify_constant_fold.checked = true;
  const changed = applyOptionParams(parseInputParams("?model=x"), doc);
  // No option params present, so nothing is changed.
  assert.equal(changed.length, 0);
  assert.equal(doc.els.id_simplify_constant_fold.checked, true);
});

console.log(`PASS: ${passed} checks`);

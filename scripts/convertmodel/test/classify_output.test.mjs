// Unit tests for classify_output.mjs — the pure output-summarizing helpers
// behind the inference panel's sample-data output preview.
//
// Usage:
//   node test/classify_output.test.mjs

import assert from "node:assert/strict";
import { isImagenetLogits, softmax, topK, summarizeOutput } from "../classify_output.mjs";
import { IMAGENET_CLASSES } from "../imagenet_classes.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("IMAGENET_CLASSES has exactly 1000 entries", () => {
  assert.equal(IMAGENET_CLASSES.length, 1000);
  assert.equal(IMAGENET_CLASSES[0], "tench");
});

check("isImagenetLogits accepts [1, 1000] and [1000]", () => {
  assert.equal(isImagenetLogits([1, 1000]), true);
  assert.equal(isImagenetLogits([1000]), true);
});

check("isImagenetLogits rejects other shapes", () => {
  assert.equal(isImagenetLogits([1, 999]), false);
  assert.equal(isImagenetLogits([2, 1000]), false); // batch > 1, not single-sample
  assert.equal(isImagenetLogits([1, 3, 1000]), false); // extra non-1 leading axis
  assert.equal(isImagenetLogits([]), false);
  assert.equal(isImagenetLogits(null), false);
});

check("softmax sums to ~1 and is monotonic with input order", () => {
  const probs = softmax([1, 2, 3]);
  assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);
  assert.ok(probs[0] < probs[1] && probs[1] < probs[2]);
});

check("softmax is numerically stable for large values", () => {
  const probs = softmax([1000, 1001, 1002]);
  assert.ok(probs.every((p) => Number.isFinite(p)));
  assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);
});

check("topK returns the highest-probability entries, sorted descending", () => {
  const probs = [0.1, 0.5, 0.2, 0.05, 0.15];
  const top = topK(probs, 3);
  assert.deepEqual(top.map((t) => t.index), [1, 2, 4]);
  assert.ok(top[0].prob >= top[1].prob && top[1].prob >= top[2].prob);
});

check("summarizeOutput decodes a 1000-way output through IMAGENET_CLASSES", () => {
  const values = new Array(1000).fill(0);
  values[281] = 10; // "tabby" in the standard mapping
  const out = summarizeOutput(values, [1, 1000], 3);
  assert.equal(out.kind, "imagenet");
  assert.equal(out.top.length, 3);
  assert.equal(out.top[0].label, IMAGENET_CLASSES[281]);
  assert.ok(out.top[0].prob > out.top[1].prob);
});

check("summarizeOutput falls back to a raw preview for other shapes", () => {
  const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const out = summarizeOutput(values, [1, 10], 5);
  assert.equal(out.kind, "raw");
  assert.deepEqual(out.preview, [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(out.total, 10);
});

check("summarizeOutput works on a typed array input", () => {
  const values = new Float32Array(1000);
  values[5] = 9;
  const out = summarizeOutput(values, [1, 1000], 1);
  assert.equal(out.kind, "imagenet");
  assert.equal(out.top[0].label, IMAGENET_CLASSES[5]);
});

console.log(`PASS: ${passed} checks`);

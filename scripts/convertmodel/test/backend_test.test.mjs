// Unit tests for the pure helpers behind the "Run an ONNX backend test case"
// panel: GitHub URL parsing, tensor ordering, and tensor comparison. No DOM or
// network needed (the DOM/onnxruntime-web glue is browser-only).
//
// Usage:
//   node test/backend_test.test.mjs

import assert from "node:assert/strict";
import {
  parseGithubDirUrl,
  tensorIndex,
  compareTensors,
  ONNX_DTYPE,
} from "../backend_test.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("parses a github tree URL", () => {
  const loc = parseGithubDirUrl(
    "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_relu",
  );
  assert.deepEqual(loc, {
    owner: "onnx",
    repo: "onnx",
    ref: "main",
    path: "onnx/backend/test/data/node/test_relu",
  });
});

check("parses a github blob URL", () => {
  const loc = parseGithubDirUrl(
    "https://github.com/onnx/onnx/blob/rel-1.16.0/onnx/backend/test/data/node/test_abs",
  );
  assert.equal(loc.ref, "rel-1.16.0");
  assert.equal(loc.path, "onnx/backend/test/data/node/test_abs");
});

check("parses a raw.githubusercontent URL", () => {
  const loc = parseGithubDirUrl(
    "https://raw.githubusercontent.com/onnx/onnx/main/onnx/backend/test/data/node/test_relu",
  );
  assert.deepEqual(loc, {
    owner: "onnx",
    repo: "onnx",
    ref: "main",
    path: "onnx/backend/test/data/node/test_relu",
  });
});

check("parses an api.github.com contents URL", () => {
  const loc = parseGithubDirUrl(
    "https://api.github.com/repos/onnx/onnx/contents/onnx/backend/test/data/node/test_relu?ref=main",
  );
  assert.equal(loc.owner, "onnx");
  assert.equal(loc.path, "onnx/backend/test/data/node/test_relu");
  assert.equal(loc.ref, "main");
});

check("rejects non-GitHub / malformed URLs", () => {
  assert.equal(parseGithubDirUrl("https://example.com/foo/bar"), null);
  assert.equal(parseGithubDirUrl("not a url"), null);
  assert.equal(parseGithubDirUrl("https://github.com/onnx/onnx"), null);
});

check("orders tensor files numerically, not lexically", () => {
  const names = ["input_10.pb", "input_2.pb", "input_1.pb"];
  names.sort((a, b) => tensorIndex(a) - tensorIndex(b));
  assert.deepEqual(names, ["input_1.pb", "input_2.pb", "input_10.pb"]);
});

// Build an expected-tensor object (as parseTensor would produce) from a typed
// array: raw little-endian bytes + dtype + dims.
function expectedTensor(dtype, dims, typed) {
  const bytes = new Uint8Array(typed.buffer.slice(0));
  return { dtype, dims, bytes };
}

check("float outputs pass within tolerance", () => {
  const expected = expectedTensor(1, [2], new Float32Array([1.0, 2.0]));
  const actual = new Float32Array([1.00001, 2.00005]);
  const cmp = compareTensors(actual, [2], expected, { atol: 1e-4, rtol: 1e-3 });
  assert.ok(cmp.ok, `expected pass, got ${JSON.stringify(cmp)}`);
});

check("float outputs fail outside tolerance", () => {
  const expected = expectedTensor(1, [2], new Float32Array([1.0, 2.0]));
  const actual = new Float32Array([1.0, 2.5]);
  const cmp = compareTensors(actual, [2], expected, { atol: 1e-4, rtol: 1e-3 });
  assert.equal(cmp.ok, false);
  assert.ok(cmp.maxDiff > 0.4);
});

check("shape mismatch fails even when values line up", () => {
  const expected = expectedTensor(1, [1, 2], new Float32Array([1.0, 2.0]));
  const actual = new Float32Array([1.0, 2.0]);
  const cmp = compareTensors(actual, [2], expected, {});
  assert.equal(cmp.ok, false);
  assert.equal(cmp.shapeOk, false);
});

check("int64 (bigint) outputs compare exactly", () => {
  const expected = expectedTensor(7, [3], new BigInt64Array([1n, 2n, 3n]));
  const good = new BigInt64Array([1n, 2n, 3n]);
  const bad = new BigInt64Array([1n, 2n, 4n]);
  assert.ok(compareTensors(good, [3], expected, {}).ok);
  assert.equal(compareTensors(bad, [3], expected, {}).ok, false);
});

check("dtype table maps the common ONNX types", () => {
  assert.equal(ONNX_DTYPE[1].ort, "float32");
  assert.equal(ONNX_DTYPE[7].ort, "int64");
  assert.equal(ONNX_DTYPE[9].ort, "bool");
  assert.equal(ONNX_DTYPE[11].ort, "float64");
});

console.log(`PASS: ${passed} checks`);

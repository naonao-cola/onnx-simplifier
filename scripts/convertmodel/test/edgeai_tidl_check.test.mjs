// Checks edgeai_tidl_check.mjs's hand-rolled protobuf reader + blocker
// analysis against real .onnx fixtures (make_edgeai_tidl_fixture.py), the
// same four cases scripts/edgeai/tests/test_edgeai_tidl_compat.py already
// covers on the Python side (clean / control-flow / dynamic-shape /
// QOperator-format) -- so both sides of this port agree on the same inputs.
//
// Usage:
//   python3 make_edgeai_tidl_fixture.py   # regenerate fixtures if needed
//   node test/edgeai_tidl_check.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { analyzeTidlCompat } from "../edgeai_tidl_check.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const load = (name) => new Uint8Array(readFileSync(join(HERE, name)));

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("a clean Conv->Relu model has full coverage and no blockers", () => {
  const result = analyzeTidlCompat(load("clean.onnx"));
  assert.equal(result.coverage, "full");
  assert.deepEqual(result.blockers, []);
  assert.equal(result.hasDynamicShape, false);
  assert.equal(result.opTypeCounts.get("Conv"), 1);
  assert.equal(result.opTypeCounts.get("Relu"), 1);
});

check("a top-level If node is flagged as a control-flow blocker", () => {
  const result = analyzeTidlCompat(load("control_flow.onnx"));
  assert.equal(result.coverage, "partial");
  assert.equal(result.blockers.length, 1);
  assert.equal(result.blockers[0].opType, "If");
  assert.match(result.blockers[0].reason, /control flow/);
});

check("a symbolic batch dimension is flagged as a dynamic-shape risk", () => {
  const result = analyzeTidlCompat(load("dynamic_shape.onnx"));
  assert.equal(result.coverage, "partial");
  assert.equal(result.hasDynamicShape, true);
  assert.deepEqual(result.blockers, []); // Conv/Relu themselves are clean
});

check("QLinearConv (QOperator format) is flagged, not just any quantized op", () => {
  const result = analyzeTidlCompat(load("qoperator.onnx"));
  assert.equal(result.coverage, "partial");
  assert.equal(result.blockers.length, 1);
  assert.equal(result.blockers[0].opType, "QLinearConv");
  assert.match(result.blockers[0].reason, /QDQ/);
});

console.log(`${passed} check(s) passed`);

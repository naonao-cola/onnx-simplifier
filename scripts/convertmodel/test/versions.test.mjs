// Unit tests for the pure helpers behind the converter page's "Versions" panel.
// The initVersions() DOM/runtime glue is browser-only (guarded by a window
// check in the module), so we test only normalizeVersions and renderVersions
// here, driving the latter through a tiny fake DOM.
//
// Usage:
//   node test/versions.test.mjs

import assert from "node:assert/strict";
import { normalizeVersions, renderVersions } from "../versions.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("normalizes a full versions object", () => {
  const v = normalizeVersions({
    onnxsim: "0.7.0",
    onnx_optimizer: "0.4.2",
    onnx: "IR v10, opset 22",
    protobuf: "4.25.1",
  });
  assert.deepEqual(v, {
    onnxsim: "0.7.0",
    onnx_optimizer: "0.4.2",
    onnx: "IR v10, opset 22",
    protobuf: "4.25.1",
  });
});

check("fills missing fields with 'unknown' and handles null", () => {
  assert.equal(normalizeVersions(null), null);
  const v = normalizeVersions({ onnxsim: "0.7.0" });
  assert.equal(v.onnxsim, "0.7.0");
  assert.equal(v.onnx_optimizer, "unknown");
  assert.equal(v.onnx, "unknown");
  assert.equal(v.protobuf, "unknown");
});

// A minimal element stand-in recording appended children and their text.
function fakeElement() {
  return {
    className: "",
    textContent: "",
    children: [],
    appendChild(c) {
      this.children.push(c);
    },
  };
}

check("renders one badge per library with label and value", () => {
  // renderVersions calls document.createElement; stub a minimal document.
  globalThis.document = { createElement: () => fakeElement() };
  const panel = fakeElement();
  renderVersions(panel, {
    onnxsim: "0.7.0",
    onnx_optimizer: "0.4.2",
    onnx: "IR v10, opset 22",
    protobuf: "4.25.1",
  });
  // Four badges appended.
  assert.equal(panel.children.length, 4);
  // Flatten all the text that ended up in the panel.
  const text = JSON.stringify(collectText(panel));
  assert.ok(text.includes("onnxsim"));
  assert.ok(text.includes("0.7.0"));
  assert.ok(text.includes("onnx-optimizer"));
  assert.ok(text.includes("0.4.2"));
  assert.ok(text.includes("protobuf"));
  delete globalThis.document;
});

function collectText(el) {
  const out = [];
  if (el.textContent) out.push(el.textContent);
  for (const c of el.children || []) out.push(...collectText(c));
  return out;
}

console.log(`PASS: ${passed} checks`);

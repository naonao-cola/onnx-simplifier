// Unit test for onnx_conv_node_reader.mjs's listConvNodeNames -- the "what
// could I offer to tune" list webgpu_kernel_annotations_view.mjs's panel
// uses to offer a "Tune this kernel" button on every Conv node a model has,
// not just ones already carrying an attached kernel spec (see that view's
// own setSide comment for why that distinction matters -- almost no real
// upload has any pre-attached kernel at all). No browser, no GPU, no
// Python -- pure-JS, against real committed fixtures.
//
// Usage:
//   node test/onnx_conv_node_reader.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { listConvNodeNames, readConvNodeInfo } from "../onnx_conv_node_reader.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

function main() {
  check("finds the named Conv node in a fixture with an attached kernel spec", () => {
    const bytes = new Uint8Array(readFileSync(join(HERE, "webgpu_tinygrad_conv3d.onnx")));
    assert.deepEqual(listConvNodeNames(bytes), ["conv3d_node"]);
  });

  check("finds the named Conv node in a plain model with no attached kernel spec", () => {
    const bytes = new Uint8Array(readFileSync(join(HERE, "webgpu_kernel_tuning_fixture.onnx")));
    assert.deepEqual(listConvNodeNames(bytes), ["conv_node"]);
  });

  check("returns an empty list for a model with no Conv node at all", () => {
    const bytes = new Uint8Array(readFileSync(join(HERE, "webgpu_tinygrad_resize.onnx")));
    assert.deepEqual(listConvNodeNames(bytes), []);
  });

  check("every name it returns is actually usable with readConvNodeInfo", () => {
    const bytes = new Uint8Array(readFileSync(join(HERE, "webgpu_kernel_tuning_fixture.onnx")));
    for (const name of listConvNodeNames(bytes)) {
      const info = readConvNodeInfo(bytes, name);
      assert.ok(info.xName && info.wName && info.outputName);
    }
  });

  console.log(`\nonnx conv node reader: ${passed} checks passed`);
}

main();

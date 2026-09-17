// Unit test for onnx_conv_node_reader.mjs's listConvNodeNames/
// listAllNodeNames -- the "what could I offer to tune" lists
// webgpu_kernel_annotations_view.mjs's panel uses to offer a "Tune this
// kernel" button on every node a model has (any op type, via
// listAllNodeNames -- see webgpu_kernel_tuner.mjs's own docstring), not just
// ones already carrying an attached kernel spec (see that view's own
// setSide comment for why that distinction matters -- almost no real upload
// has any pre-attached kernel at all). No browser, no GPU, no Python --
// pure-JS, against real committed fixtures.
//
// Usage:
//   node test/onnx_conv_node_reader.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { listConvNodeNames, readConvNodeInfo, listAllNodeNames } from "../onnx_conv_node_reader.mjs";

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

  check("listAllNodeNames returns every node, any op type, with its own graph position", () => {
    const bytes = new Uint8Array(readFileSync(join(HERE, "webgpu_onnxrunner_conv_relu.onnx")));
    const entries = listAllNodeNames(bytes);
    assert.deepEqual(
      entries.map((e) => [e.name, e.opType, e.index]),
      [
        ["conv3d_node", "Conv", 0],
        ["relu_node", "Relu", 1],
      ],
    );
  });

  check("listAllNodeNames's index agrees with listConvNodeNames for the same node", () => {
    const bytes = new Uint8Array(readFileSync(join(HERE, "webgpu_kernel_tuning_fixture.onnx")));
    const [convName] = listConvNodeNames(bytes);
    const entry = listAllNodeNames(bytes).find((e) => e.name === convName);
    assert.ok(entry);
    assert.equal(entry.opType, "Conv");
  });

  console.log(`\nonnx conv node reader: ${passed} checks passed`);
}

main();

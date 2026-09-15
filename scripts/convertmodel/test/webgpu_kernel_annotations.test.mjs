// Unit test for webgpu_kernel_annotations.mjs -- the pure reshaping logic
// behind the converter page's "Custom WebGPU kernels" panel. No network,
// browser, or GPU needed; drives the exact same code the panel uses, against
// a real .onnx fixture (test/webgpu_kernel_add.onnx, see
// make_webgpu_kernel_fixture.py) plus plain in-memory specs for the
// intermediate/constant binding cases that fixture doesn't happen to cover.
//
//   node test/webgpu_kernel_annotations.test.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";
import { describeBinding, summarizeKernelSpec, listWebgpuKernelAnnotations } from "../webgpu_kernel_annotations.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("describeBinding formats a tensor binding", () => {
  assert.equal(describeBinding({ group: 0, binding: 0, access: "read", tensor: "a" }), "tensor:a (read)");
});

check("describeBinding formats an intermediate binding", () => {
  assert.equal(
    describeBinding({ group: 0, binding: 1, access: "read_write", intermediate: "scratch" }),
    "intermediate:scratch (read_write)",
  );
});

check("describeBinding formats a constant binding, including non-finite values", () => {
  assert.equal(
    describeBinding({ group: 0, binding: 0, access: "uniform", constant: ["Infinity", 2.5] }),
    "constant:[Infinity, 2.5] (uniform)",
  );
});

check("describeBinding defaults access to 'read' when absent", () => {
  assert.equal(describeBinding({ group: 0, binding: 0, tensor: "a" }), "tensor:a (read)");
});

check("summarizeKernelSpec reshapes a multi-step spec with an intermediate", () => {
  const spec = {
    steps: [
      {
        wgsl: "/* step1 */",
        entry_point: "step1",
        dispatch: [1, 1, 1],
        bindings: [
          { group: 0, binding: 0, access: "read", tensor: "a" },
          { group: 0, binding: 1, access: "read_write", intermediate: "scratch" },
        ],
      },
      {
        wgsl: "/* step2 */",
        entry_point: "step2",
        dispatch: [2, 3, 4],
        bindings: [{ group: 0, binding: 0, access: "read_write", tensor: "c" }],
      },
    ],
    intermediates: { scratch: 256 },
  };
  const summary = summarizeKernelSpec("softmax_node", spec);
  assert.equal(summary.nodeName, "softmax_node");
  assert.equal(summary.stepCount, 2);
  assert.equal(summary.intermediateCount, 1);
  assert.deepEqual(summary.intermediates, { scratch: 256 });
  assert.equal(summary.steps[0].entryPoint, "step1");
  assert.deepEqual(summary.steps[1].dispatch, [2, 3, 4]);
  assert.deepEqual(summary.steps[0].bindings, ["tensor:a (read)", "intermediate:scratch (read_write)"]);
});

check("summarizeKernelSpec handles a spec with no intermediates", () => {
  const summary = summarizeKernelSpec("n", { steps: [], intermediates: {} });
  assert.equal(summary.stepCount, 0);
  assert.equal(summary.intermediateCount, 0);
});

const addModel = new Uint8Array(readFileSync(join(HERE, "webgpu_kernel_add.onnx")));

check("listWebgpuKernelAnnotations finds the one annotated node in a real .onnx fixture", () => {
  const annotations = listWebgpuKernelAnnotations(addModel);
  assert.equal(annotations.length, 1);
  const [add] = annotations;
  assert.equal(add.nodeName, "add_node");
  assert.equal(add.stepCount, 1);
  assert.equal(add.steps[0].entryPoint, "main");
  assert.deepEqual(add.steps[0].dispatch, [4, 1, 1]);
  assert.deepEqual(add.steps[0].bindings, [
    "tensor:a (read)",
    "tensor:b (read)",
    "tensor:c (read_write)",
  ]);
  assert.ok(add.steps[0].wgsl.includes("@compute"));
});

check("listWebgpuKernelAnnotations returns an empty list for a model with no annotations", () => {
  // webgpu_custom_kernel_runtime_pre.onnx (see
  // make_webgpu_custom_kernel_runtime_fixture.py) is a plain Relu -- no
  // node in it has a webgpu_kernel attachment.
  const plainModel = new Uint8Array(readFileSync(join(HERE, "webgpu_custom_kernel_runtime_pre.onnx")));
  assert.deepEqual(listWebgpuKernelAnnotations(plainModel), []);
});

console.log(`\nwebgpu kernel annotations: ${passed} checks passed`);

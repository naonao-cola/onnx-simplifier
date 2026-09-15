// Pure logic for the "Custom WebGPU kernels" panel: reshapes the raw
// WebgpuKernelSpec JSON onnx_node_metadata.mjs's readWebgpuKernelSpecs
// returns (see onnxsim/webgpu_kernel_metadata.py's own docstring for that
// schema) into flat, display-ready summaries, kept DOM-free so it's testable
// directly in Node (see test/webgpu_kernel_annotations.test.mjs) -- the same
// pure-logic/DOM-glue split shapes.mjs/shapes_view.mjs uses for the
// "Dynamic dimensions" panel.
//
// This only *displays* whatever a model already carries -- it does not run
// onnxsim.webgpu_target's gap-detection (Conv3D/Resize/Attention) itself,
// generate a kernel, or dispatch one; see this repo's own conversation
// history / docs/webgpu-kernel-dispatch.md for why those are bigger,
// separate pieces (gap-detection has no JS port yet; generation needs
// tinygrad, which the WASM UI can't run in-browser today).

import { readWebgpuKernelSpecs } from "./onnx_node_metadata.mjs";

/**
 * One `WebgpuKernelBinding`, as a short human-readable string -- e.g.
 * `"tensor:x (read)"`, `"intermediate:scratch (read_write)"`,
 * `"constant:[Infinity] (uniform)"`.
 */
export function describeBinding(binding) {
  const access = binding.access || "read";
  if (binding.tensor !== undefined) return `tensor:${binding.tensor} (${access})`;
  if (binding.intermediate !== undefined) return `intermediate:${binding.intermediate} (${access})`;
  if (binding.constant !== undefined) return `constant:[${binding.constant.join(", ")}] (${access})`;
  return `(unrecognized binding at group=${binding.group} binding=${binding.binding})`;
}

/**
 * Reshapes one `WebgpuKernelSpec` (the JSON a single node's
 * `"onnxsim.webgpu_kernel"` metadata entry decodes to) into a flat summary
 * for display.
 *
 * @param {string} nodeName
 * @param {object} spec - `{steps: [...], intermediates: {...}}`
 */
export function summarizeKernelSpec(nodeName, spec) {
  const steps = spec.steps || [];
  const intermediates = spec.intermediates || {};
  return {
    nodeName,
    stepCount: steps.length,
    intermediateCount: Object.keys(intermediates).length,
    intermediates,
    steps: steps.map((step, index) => ({
      index,
      entryPoint: step.entry_point,
      dispatch: step.dispatch,
      bindings: (step.bindings || []).map(describeBinding),
      wgsl: step.wgsl,
    })),
  };
}

/**
 * Every node in `modelBytes` carrying a custom WebGPU kernel/program
 * attachment, as flat summaries in graph node order (the order
 * onnx_node_metadata.mjs's readWebgpuKernelSpecs -- and, upstream of it,
 * onnxsim's own protobuf field iteration -- returns them in).
 *
 * @param {Uint8Array} modelBytes
 * @returns {Array<ReturnType<typeof summarizeKernelSpec>>}
 */
export function listWebgpuKernelAnnotations(modelBytes) {
  const specs = readWebgpuKernelSpecs(modelBytes);
  return [...specs.entries()].map(([nodeName, spec]) => summarizeKernelSpec(nodeName, spec));
}

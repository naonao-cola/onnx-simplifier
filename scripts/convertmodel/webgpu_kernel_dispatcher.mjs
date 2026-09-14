// Compiles and dispatches one custom WebGPU compute kernel, described by a
// spec shaped like onnxsim.webgpu_kernel_metadata's JSON schema (see that
// module's docstring, and onnx_node_metadata.mjs's readWebgpuKernelSpecs for
// how such a spec is read back out of a model's own metadata_props), against
// caller-supplied GPUBuffers.
//
// This is a standalone execution primitive: "run this WGSL kernel against
// these buffers", nothing more. It does not know about onnxruntime-web, ONNX
// graphs, or tensor shapes/dtypes beyond raw byte buffers -- splicing a
// dispatch like this into an actual onnxruntime-web session (so a flagged
// node runs through here instead of ORT-web's own WebGPU kernel, most likely
// via ORT-web's GPU-buffer IO binding to avoid a CPU round-trip) is future
// work; see onnxsim/webgpu_kernel_metadata.py's docstring for the current
// state of that boundary.
//
// webgpu_kernel_dispatcher.test.mjs exercises this against a real WebGPU
// device (via Playwright/Chromium, the same way the other webgpu_*.test.mjs
// files in this directory do) with a real elementwise-add WGSL kernel,
// checking the GPU-computed output against a plain JS reference.

/**
 * Compiles ``spec.wgsl`` and dispatches ``spec.entry_point`` against
 * ``buffersByTensor``, once.
 *
 * @param {GPUDevice} device
 * @param {{wgsl: string, entry_point: string, dispatch: [number, number, number],
 *          bindings: Array<{tensor: string, group: number, binding: number, access: "read"|"read_write"}>}} spec
 *        Same shape as onnxsim.webgpu_kernel_metadata.WebgpuKernelSpec.to_json().
 * @param {Map<string, GPUBuffer>} buffersByTensor - tensor name (as named in
 *        ``spec.bindings``) -> a GPUBuffer already created with ``STORAGE``
 *        usage (plus whatever ``COPY_SRC``/``COPY_DST`` the caller needs for
 *        upload/readback) and large enough for the kernel's access pattern.
 *        This function does not create, size, or validate these buffers --
 *        see ``createStorageBuffer``/``readBackFloat32Buffer`` below for
 *        convenience wrappers a caller (or a test) can use to do so.
 * @param {{awaitCompletion?: boolean}} [opts] - awaitCompletion (default
 *        true) waits for ``device.queue.onSubmittedWorkDone()`` before
 *        resolving; a caller chaining several dispatches back-to-back on the
 *        GPU without reading intermediate results back to the CPU can pass
 *        ``false`` to skip that wait and let the queue pipeline them.
 * @returns {Promise<void>}
 */
export async function dispatchWebgpuKernel(device, spec, buffersByTensor, opts = {}) {
  const { awaitCompletion = true } = opts;

  const shaderModule = device.createShaderModule({ code: spec.wgsl });

  const bindingsByGroup = new Map();
  for (const binding of spec.bindings) {
    if (!bindingsByGroup.has(binding.group)) {
      bindingsByGroup.set(binding.group, []);
    }
    bindingsByGroup.get(binding.group).push(binding);
  }
  const groupIndices = [...bindingsByGroup.keys()].sort((a, b) => a - b);
  if (groupIndices.length > 0 && groupIndices[groupIndices.length - 1] !== groupIndices.length - 1) {
    throw new Error(
      `bind groups must be contiguous starting at 0, got groups [${groupIndices.join(", ")}]`,
    );
  }

  const bindGroupLayouts = [];
  const bindGroups = [];
  for (const groupIndex of groupIndices) {
    const entries = bindingsByGroup.get(groupIndex);
    const layout = device.createBindGroupLayout({
      entries: entries.map((b) => ({
        binding: b.binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: b.access === "read_write" ? "storage" : "read-only-storage" },
      })),
    });
    bindGroupLayouts[groupIndex] = layout;

    bindGroups[groupIndex] = device.createBindGroup({
      layout,
      entries: entries.map((b) => {
        const buffer = buffersByTensor.get(b.tensor);
        if (!buffer) {
          throw new Error(`no GPU buffer supplied for tensor ${JSON.stringify(b.tensor)}`);
        }
        return { binding: b.binding, resource: { buffer } };
      }),
    });
  }

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts }),
    compute: { module: shaderModule, entryPoint: spec.entry_point },
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  bindGroups.forEach((bindGroup, groupIndex) => pass.setBindGroup(groupIndex, bindGroup));
  const [x, y, z] = spec.dispatch;
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
  device.queue.submit([encoder.finish()]);

  if (awaitCompletion) {
    await device.queue.onSubmittedWorkDone();
  }
}

/**
 * Creates a GPU storage buffer initialized from a ``Float32Array``, usable
 * both as a kernel's storage-buffer binding and as the source of a
 * ``copyBufferToBuffer`` readback (see ``readBackFloat32Buffer``).
 *
 * @param {GPUDevice} device
 * @param {Float32Array} data
 * @returns {GPUBuffer}
 */
export function createStorageBuffer(device, data) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Reads a GPU storage buffer (created with ``COPY_SRC`` usage, as
 * ``createStorageBuffer`` does) back to the CPU as a ``Float32Array``, via a
 * staging buffer -- a storage buffer itself generally can't be
 * ``mapAsync``'d directly.
 *
 * @param {GPUDevice} device
 * @param {GPUBuffer} buffer
 * @param {number} floatCount
 * @returns {Promise<Float32Array>}
 */
export async function readBackFloat32Buffer(device, buffer, floatCount) {
  const byteLength = floatCount * Float32Array.BYTES_PER_ELEMENT;
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

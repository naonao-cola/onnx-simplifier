// Compiles and dispatches a custom WebGPU *program* (one or more kernel
// steps run in sequence), described by a spec shaped like
// onnxsim.webgpu_kernel_metadata's JSON schema (see that module's
// docstring, and onnx_node_metadata.mjs's readWebgpuKernelSpecs for how such
// a spec is read back out of a model's own metadata_props), against
// caller-supplied GPUBuffers.
//
// This is a standalone execution primitive: "run this WGSL program against
// these buffers", nothing more. It does not know about onnxruntime-web, ONNX
// graphs, or tensor shapes/dtypes beyond raw byte buffers -- splicing a
// dispatch like this into an actual onnxruntime-web session (so a flagged
// node runs through here instead of ORT-web's own WebGPU kernel, most likely
// via ORT-web's GPU-buffer IO binding to avoid a CPU round-trip) is future
// work; see onnxsim/webgpu_kernel_metadata.py's docstring for the current
// state of that boundary.
//
// Why *program* (steps, plural) rather than one kernel: onnxsim.webgpu_tinygrad_codegen
// generates specs from real tinygrad Tensor graphs, and tinygrad's own
// scheduler doesn't always fuse a node's computation into a single kernel --
// a softmax-bearing op like Attention schedules as several separate kernels
// that share scratch ("intermediate") buffers between them. See
// onnxsim/webgpu_kernel_metadata.py's docstring for the full schema this
// module consumes.
//
// webgpu_kernel_dispatcher.test.mjs exercises this against a real WebGPU
// device (via Playwright/Chromium, the same way the other webgpu_*.test.mjs
// files in this directory do) with a real elementwise-add WGSL kernel,
// checking the GPU-computed output against a plain JS reference.

/**
 * Resolves one binding's WebGPU resource: the caller-supplied buffer for a
 * ``tensor`` binding, this program run's own scratch buffer for an
 * ``intermediate`` one, or a freshly created (and cached, so an identical
 * constant used by two bindings shares one buffer) uniform buffer for a
 * ``constant`` one.
 */
function resolveBindingBuffer(device, binding, buffersByTensor, intermediateBuffersByName, constantBufferCache) {
  if (binding.tensor !== undefined) {
    const buffer = buffersByTensor.get(binding.tensor);
    if (!buffer) {
      throw new Error(`no GPU buffer supplied for tensor ${JSON.stringify(binding.tensor)}`);
    }
    return buffer;
  }
  if (binding.intermediate !== undefined) {
    const buffer = intermediateBuffersByName.get(binding.intermediate);
    if (!buffer) {
      throw new Error(
        `intermediate ${JSON.stringify(binding.intermediate)} was not declared in spec.intermediates`,
      );
    }
    return buffer;
  }
  if (binding.constant !== undefined) {
    const key = JSON.stringify(binding.constant);
    let buffer = constantBufferCache.get(key);
    if (!buffer) {
      // Number(v) recognizes the "Infinity"/"-Infinity"/"NaN" string
      // sentinels onnxsim.webgpu_kernel_metadata encodes non-finite values
      // as (standard JSON has no literal for them) -- see that module's
      // own _encode_float for why.
      const data = Float32Array.from(binding.constant, (v) => Number(v));
      buffer = device.createBuffer({
        size: Math.max(data.byteLength, 4),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(buffer.getMappedRange()).set(data);
      buffer.unmap();
      constantBufferCache.set(key, buffer);
    }
    return buffer;
  }
  throw new Error(
    `binding at group=${binding.group} binding=${binding.binding} has none of tensor/intermediate/constant set`,
  );
}

function bindGroupLayoutEntryType(access) {
  if (access === "uniform") return "uniform";
  return access === "read_write" ? "storage" : "read-only-storage";
}

/** Builds the bind group layouts/groups for one step's bindings. */
function buildBindGroups(device, bindings, buffersByTensor, intermediateBuffersByName, constantBufferCache) {
  const bindingsByGroup = new Map();
  for (const binding of bindings) {
    if (!bindingsByGroup.has(binding.group)) bindingsByGroup.set(binding.group, []);
    bindingsByGroup.get(binding.group).push(binding);
  }
  const groupIndices = [...bindingsByGroup.keys()].sort((a, b) => a - b);
  if (groupIndices.length > 0 && groupIndices[groupIndices.length - 1] !== groupIndices.length - 1) {
    throw new Error(`bind groups must be contiguous starting at 0, got groups [${groupIndices.join(", ")}]`);
  }

  const bindGroupLayouts = [];
  const bindGroups = [];
  for (const groupIndex of groupIndices) {
    const entries = bindingsByGroup.get(groupIndex);
    const layout = device.createBindGroupLayout({
      entries: entries.map((b) => ({
        binding: b.binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: bindGroupLayoutEntryType(b.access) },
      })),
    });
    bindGroupLayouts[groupIndex] = layout;
    bindGroups[groupIndex] = device.createBindGroup({
      layout,
      entries: entries.map((b) => ({
        binding: b.binding,
        resource: {
          buffer: resolveBindingBuffer(device, b, buffersByTensor, intermediateBuffersByName, constantBufferCache),
        },
      })),
    });
  }
  return { bindGroupLayouts, bindGroups };
}

/**
 * Runs every step of ``spec`` in order, on one command encoder, against
 * ``buffersByTensor``.
 *
 * @param {GPUDevice} device
 * @param {{steps: Array<{wgsl: string, entry_point: string, dispatch: [number, number, number],
 *          bindings: Array<{group: number, binding: number, access: "read"|"read_write"|"uniform",
 *          tensor?: string, intermediate?: string, constant?: number[]}>}>,
 *          intermediates?: Record<string, number>}} spec
 *        Same shape as onnxsim.webgpu_kernel_metadata.WebgpuKernelSpec.to_json().
 * @param {Map<string, GPUBuffer>} buffersByTensor - tensor name (as named in
 *        a binding's ``tensor`` field) -> a GPUBuffer already created with
 *        ``STORAGE`` usage (plus whatever ``COPY_SRC``/``COPY_DST`` the
 *        caller needs for upload/readback) and large enough for the
 *        program's access pattern. This function does not create, size, or
 *        validate these buffers -- see ``createStorageBuffer``/
 *        ``readBackFloat32Buffer`` below for convenience wrappers a caller
 *        (or a test) can use to do so. Buffers for ``intermediate`` and
 *        ``constant`` bindings are created and destroyed internally.
 * @returns {Promise<void>} resolves once every step has run and this
 *        program's own intermediate/constant buffers have been freed --
 *        always awaited internally (unlike a single dispatch, a program
 *        owns temporary buffers it must not destroy before the GPU is done
 *        with them, so there is no "skip the wait" option here).
 */
export async function dispatchWebgpuProgram(device, spec, buffersByTensor) {
  const intermediateBuffersByName = new Map();
  for (const [name, byteLength] of Object.entries(spec.intermediates || {})) {
    intermediateBuffersByName.set(name, device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE }));
  }
  const constantBufferCache = new Map();

  const encoder = device.createCommandEncoder();
  for (const step of spec.steps) {
    const shaderModule = device.createShaderModule({ code: step.wgsl });
    const { bindGroupLayouts, bindGroups } = buildBindGroups(
      device,
      step.bindings,
      buffersByTensor,
      intermediateBuffersByName,
      constantBufferCache,
    );

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts }),
      compute: { module: shaderModule, entryPoint: step.entry_point },
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    bindGroups.forEach((bindGroup, groupIndex) => pass.setBindGroup(groupIndex, bindGroup));
    const [x, y, z] = step.dispatch;
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  for (const buffer of intermediateBuffersByName.values()) buffer.destroy();
  for (const buffer of constantBufferCache.values()) buffer.destroy();
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

// Compiles and dispatches a custom WebGPU *program* (one or more kernel
// steps run in sequence), described by a spec shaped like
// onnxsim.webgpu_kernel_metadata's JSON schema (see that module's
// docstring, and onnx_node_metadata.mjs's readWebgpuKernelSpecs for how such
// a spec is read back out of a model's own metadata_props), against
// caller-supplied GPUBuffers.
//
// This is a standalone execution primitive: "run this WGSL program against
// these buffers", nothing more. It does not know about onnxruntime-web or
// ONNX graphs beyond raw byte buffers -- webgpu_custom_kernel_runtime.mjs is
// what splices a dispatch like this into an actual onnxruntime-web session
// (so a flagged node runs through here instead of ORT-web's own WebGPU
// kernel), via ORT-web's own GPU-buffer IO binding to avoid a CPU
// round-trip; see that module's own docstring.
//
// Why *program* (steps, plural) rather than one kernel: onnxsim.webgpu_tinygrad_codegen
// generates specs from real tinygrad Tensor graphs, and tinygrad's own
// scheduler doesn't always fuse a node's computation into a single kernel --
// a softmax-bearing op like Attention schedules as several separate kernels
// that share scratch ("intermediate") buffers between them. See
// onnxsim/webgpu_kernel_metadata.py's docstring for the full schema this
// module consumes.
//
// dispatchWebgpuProgram's own `profile: true` option times each step on the
// GPU itself (see its own docstring for the two WebGPU timestamp-query
// mechanisms it picks between, matching what onnxruntime-web's own WebGPU EP
// profiling does) -- useful for comparing a generated kernel's own cost
// against onnxruntime-web's per-kernel profiling numbers for the node it
// replaced.
//
// webgpu_kernel_dispatcher.test.mjs exercises this against a real WebGPU
// device (via Playwright/Chromium, the same way the other webgpu_*.test.mjs
// files in this directory do) with a real elementwise-add WGSL kernel,
// checking the GPU-computed output against a plain JS reference, and (when
// the device supports it) that profiling reports a real, non-negative
// duration.

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

// The standard WebGPU timing feature (times a whole beginComputePass via its
// own timestampWrites descriptor) and Chromium's own experimental one (times
// arbitrary points *inside* a pass via pass.writeTimestamp(querySet, index),
// which onnxruntime-web's WebGPU backend prefers when the adapter offers
// both -- verified directly against the installed onnxruntime-web bundle:
// it tries this one first and only falls back to the standard feature if
// the adapter doesn't have it). Supporting only the standard feature would
// leave profiling silently unavailable on exactly the device
// webgpu_custom_kernel_runtime.mjs actually shares with onnxruntime-web in
// that (common, Chromium) case.
const STANDARD_TIMESTAMP_FEATURE = "timestamp-query";
const CHROMIUM_INSIDE_PASSES_TIMESTAMP_FEATURE = "chromium-experimental-timestamp-query-inside-passes";

/**
 * Whether ``device`` can report per-step GPU timings for
 * ``dispatchWebgpuProgram``'s ``profile: true`` option -- i.e. whether it was
 * created with the standard ``"timestamp-query"`` feature or Chromium's own
 * ``"chromium-experimental-timestamp-query-inside-passes"`` one (see the
 * constants above). A device's own feature set is fixed at
 * ``requestDevice()`` time and can't be added afterward, so this is a
 * read-only check, not something a caller can turn on later.
 *
 * @param {GPUDevice} device
 * @returns {boolean}
 */
export function supportsWebgpuProfiling(device) {
  return device.features.has(STANDARD_TIMESTAMP_FEATURE) || device.features.has(CHROMIUM_INSIDE_PASSES_TIMESTAMP_FEATURE);
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
 * @param {{profile?: boolean}} [options] - pass ``profile: true`` to time
 *        each step on the GPU itself, via whichever of the two timestamp
 *        query mechanisms ``device`` was created with (see
 *        ``supportsWebgpuProfiling``'s own docstring): the standard
 *        ``"timestamp-query"`` feature (``beginComputePass({timestampWrites})``)
 *        or Chromium's ``"...-inside-passes"`` one
 *        (``pass.writeTimestamp(querySet, index)``), the same two
 *        primitives onnxruntime-web's own WebGPU EP profiling picks between
 *        (it prefers the Chromium one when the adapter has it), so the
 *        reported durations are directly comparable to onnxruntime-web's
 *        own per-kernel profiling numbers either way. Silently produces no
 *        timings (rather than throwing) when the device has neither --
 *        check ``supportsWebgpuProfiling(device)`` first if the caller
 *        needs to know why, or just check ``result.timings === null``.
 * @returns {Promise<{timings: Array<{index: number, entryPoint: string, durationNs: number}> | null}>}
 *        resolves once every step has run and this program's own
 *        intermediate/constant buffers have been freed -- always awaited
 *        internally (unlike a single dispatch, a program owns temporary
 *        buffers it must not destroy before the GPU is done with them, so
 *        there is no "skip the wait" option here). ``timings`` is ``null``
 *        unless ``options.profile`` was true *and* the device supports it;
 *        otherwise one entry per step, in step order, each a real
 *        GPU-measured wall-clock duration in nanoseconds (per the WebGPU
 *        spec's ``timestamp-query`` semantics -- browsers may coarsen the
 *        actual resolution for timing-attack mitigation, but the unit is
 *        always nanoseconds).
 */
export async function dispatchWebgpuProgram(device, spec, buffersByTensor, options = {}) {
  const useStandardTimestamps = !!options.profile && device.features.has(STANDARD_TIMESTAMP_FEATURE);
  const useChromiumInsidePassesTimestamps =
    !!options.profile && !useStandardTimestamps && device.features.has(CHROMIUM_INSIDE_PASSES_TIMESTAMP_FEATURE);
  const profile = useStandardTimestamps || useChromiumInsidePassesTimestamps;

  const intermediateBuffersByName = new Map();
  for (const [name, byteLength] of Object.entries(spec.intermediates || {})) {
    intermediateBuffersByName.set(name, device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE }));
  }
  const constantBufferCache = new Map();

  // One "timestamp" query pair (begin, end) per step -- see
  // https://www.w3.org/TR/webgpu/#timestamp-query for why every count here
  // is doubled (each pass writes two u64 timestamps, not one). Both
  // timestamp mechanisms below share this same query set/layout; they only
  // differ in *how* each pair gets written.
  const querySet = profile
    ? device.createQuerySet({ type: "timestamp", count: spec.steps.length * 2 })
    : null;

  const encoder = device.createCommandEncoder();
  spec.steps.forEach((step, index) => {
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

    // The standard feature times the pass via its own timestampWrites
    // descriptor, fixed at beginComputePass() time; the Chromium one has no
    // such descriptor at all -- it times arbitrary points *inside* the pass
    // via explicit writeTimestamp() calls, so those go immediately after
    // the pass starts and immediately before it ends instead.
    const passDescriptor = useStandardTimestamps
      ? { timestampWrites: { querySet, beginningOfPassWriteIndex: index * 2, endOfPassWriteIndex: index * 2 + 1 } }
      : {};
    const pass = encoder.beginComputePass(passDescriptor);
    if (useChromiumInsidePassesTimestamps) pass.writeTimestamp(querySet, index * 2);
    pass.setPipeline(pipeline);
    bindGroups.forEach((bindGroup, groupIndex) => pass.setBindGroup(groupIndex, bindGroup));
    const [x, y, z] = step.dispatch;
    pass.dispatchWorkgroups(x, y, z);
    if (useChromiumInsidePassesTimestamps) pass.writeTimestamp(querySet, index * 2 + 1);
    pass.end();
  });

  let queryReadback = null;
  let resolveBuffer = null;
  if (profile) {
    resolveBuffer = device.createBuffer({
      size: querySet.count * 8, // one u64 (BigInt64Array element) per timestamp
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    queryReadback = device.createBuffer({
      size: resolveBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.resolveQuerySet(querySet, 0, querySet.count, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, queryReadback, 0, resolveBuffer.size);
  }

  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  let timings = null;
  if (profile) {
    await queryReadback.mapAsync(GPUMapMode.READ);
    const raw = new BigInt64Array(queryReadback.getMappedRange().slice(0));
    queryReadback.unmap();
    queryReadback.destroy();
    resolveBuffer.destroy();
    querySet.destroy();
    timings = spec.steps.map((step, index) => ({
      index,
      entryPoint: step.entry_point,
      durationNs: Number(raw[index * 2 + 1] - raw[index * 2]),
    }));
  }

  for (const buffer of intermediateBuffersByName.values()) buffer.destroy();
  for (const buffer of constantBufferCache.values()) buffer.destroy();

  return { timings };
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

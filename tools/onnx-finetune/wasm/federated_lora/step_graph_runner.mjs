// Runs a federated-round LoRA training-step graph from
// ../../scripts/generate_federated_lora_step_graph.py entirely in the
// browser, via the *official* onnxruntime-web package's plain
// ort.InferenceSession -- the client half of onnxsim.federated's FedAvg loop
// (see ../../../../onnxsim/federated.py and ../../../../tests/test_federated.py
// for the server-side aggregation this runner's output feeds into).
//
// By default the session runs on the wasm (CPU) execution provider, which is
// the only one onnxruntime-web offers everywhere. On a desktop browser with a
// Vulkan driver you can also train on WebGPU -- the browser's own GPU API,
// which Chromium serves over that driver -- by passing
// `{ executionProviders: ["webgpu", "wasm"] }` to `StepGraphSession.create`
// (see its own comment below): the same step graph then runs its GPU-able ops
// on the GPU and falls back to wasm for the rest. WebGPU needs a real browser
// page (`navigator.gpu`), so the wasm default is what keeps this module
// runnable under plain Node (the step_graph_runner.test.mjs path).
//
// Structurally this is ../distill_step_graph/step_graph_runner.mjs's own
// pattern, ported from a classification/distillation step graph to a
// regression/LoRA one: same "no custom WASM build, no onnxruntime.training"
// approach, same manifest-driven feed construction, same
// {loss, state}-returning step() loop. The two differ in what a step's
// per-call inputs *are* (a fixed-shape local batch/target pair here, vs. a
// dynamic-batch classification triple there -- see
// generate_federated_lora_step_graph.py's own docstring for why this one's
// batch size is fixed at generation time rather than a dim_param) and in
// what leaves the browser at the end: only the trained LoRA A/B values
// (exportTrainedState), never this client's own data and never the base
// model's frozen weights.

/** Parses a `<step_graph>.manifest.txt` written by
 * generate_federated_lora_step_graph.py. `weights` lists the LoRA `A`/`B`
 * tensors in the exact order `initial_state.bin` concatenates them in --
 * `exportTrainedState` below writes its own output in that same order, so a
 * server-side aggregator never needs to know this runner's internal tensor
 * names, only the manifest it already has. */
export function parseManifest(text) {
  const manifest = { state: [], weights: [] };
  for (const rawLine of text.split("\n")) {
    const parts = rawLine.trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) continue;
    const [tag, ...rest] = parts;
    if (tag === "input_name") manifest.inputName = rest[0];
    else if (tag === "input_shape") manifest.inputShape = rest.map(Number);
    else if (tag === "teacher_name") manifest.teacherName = rest[0];
    else if (tag === "teacher_shape") manifest.teacherShape = rest.map(Number);
    else if (tag === "lr_name") manifest.lrName = rest[0];
    else if (tag === "loss_name") manifest.lossName = rest[0];
    else if (tag === "state") {
      manifest.state.push({ input: rest[0], output: rest[1], shape: rest.slice(2).map(Number) });
    } else if (tag === "weight") {
      manifest.weights.push({ name: rest[0], shape: rest.slice(1).map(Number) });
    }
  }
  return manifest;
}

function prod(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

/** The initial `{state input name: Float32Array}` map for round zero, or for
 * any later round's broadcast -- every weight from
 * `<step_graph>.initial_state.bin` (or a server's freshly-aggregated
 * replacement for it, same byte layout), every Adam `__m`/`__v` moment at
 * zero. A federated client always starts local training with a *fresh*
 * optimizer state every round (see `onnxsim.federated`'s own module
 * docstring on why FedAvg never carries a client's moments between rounds),
 * so there is nothing to load for those beyond zero-filling them here. */
export function loadInitialState(manifest, initialStateBuffer) {
  const state = {};
  const view = new Float32Array(initialStateBuffer);
  let offset = 0;
  for (const { name, shape } of manifest.weights) {
    const count = prod(shape);
    state[name] = view.slice(offset, offset + count);
    offset += count;
  }
  for (const { input, shape } of manifest.state) {
    if (!(input in state)) state[input] = new Float32Array(prod(shape));
  }
  return state;
}

/** The trained LoRA state, serialized back to the exact byte layout
 * `initial_state.bin`/`loadInitialState` use -- the one thing this client
 * submits to the federated round's aggregator
 * (`../../scripts/federated_lora_aggregate.py`). Deliberately only the
 * `manifest.weights` entries (the `A`/`B` adapters themselves): every
 * `__m`/`__v` Adam moment is this client's own local optimizer state, never
 * part of what FedAvg averages (see `onnxsim.federated.fedavg`'s own scope),
 * so there is nothing to gain -- and privacy to lose, in a scheme that ever
 * wanted to hide which clients trained how much -- by shipping it back. */
export function exportTrainedState(manifest, state) {
  const total = manifest.weights.reduce((n, { shape }) => n + prod(shape), 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const { name, shape } of manifest.weights) {
    out.set(state[name], offset);
    offset += prod(shape);
  }
  return out;
}

/** Adam's two bias-correction factors at (0-based) local step `t`, matching
 * `onnxsim.qat_graph.adam_bias_corrections` exactly -- a federated client's
 * own local step counter, reset to 0 every round since its optimizer state
 * is reset every round too (see `loadInitialState`'s own docstring). */
export function adamBiasCorrections(t) {
  return {
    mCorrection: 1 / (1 - Math.pow(0.9, t + 1)),
    vCorrection: 1 / (1 - Math.pow(0.999, t + 1)),
  };
}

export class StepGraphSession {
  /**
   * @param {object} [options]
   * @param {string[]} [options.executionProviders=["wasm"]] which
   *   onnxruntime-web execution providers the session runs on, in priority
   *   order. The default is wasm-only because the wasm EP is the only one
   *   that exists everywhere (plain Node included); to train on WebGPU --
   *   which in a desktop browser with a Vulkan driver is that driver, i.e.
   *   Vulkan-based training -- pass
   *   `{ executionProviders: ["webgpu", "wasm"] }`, and onnxruntime-web
   *   keeps every op WebGPU supports on the GPU while falling back to the
   *   wasm CPU kernels for the rest. The webgpu EP requires a real browser
   *   (`navigator.gpu`); under plain Node it answers "[webgpu] backend not
   *   found", so the default must stay wasm. The op set these step graphs
   *   are built from is the `EP_FRIENDLY_OPS` allowlist
   *   (`onnxsim/qat_graph.py`), the same allowlist the repo's own real-browser
   *   WebGPU training demos (`scripts/convertmodel/test/webgpu_*.test.mjs`)
   *   prove trains on WebGPU.
   */
  static async create(ort, stepGraphBytes, options = {}) {
    const executionProviders = options.executionProviders ?? ["wasm"];
    const session = await ort.InferenceSession.create(stepGraphBytes, {
      executionProviders,
    });
    return new StepGraphSession(ort, session);
  }

  constructor(ort, session) {
    this.ort = ort;
    this.session = session;
  }

  /** Runs one local training step on this client's own data:
   * `batchInput`/`target` are Float32Arrays shaped per
   * `manifest.inputShape`/`manifest.teacherShape` (both fixed at generation
   * time -- unlike the distillation step graph, this one has no dynamic
   * batch dimension, see generate_federated_lora_step_graph.py's own
   * docstring for why). Returns `{ loss, state }`, `state` ready to feed as
   * the next local step's own state inputs, and (once local training for
   * this round is done) to hand to `exportTrainedState`. */
  async step(manifest, state, batchInput, target, lr, t) {
    const { mCorrection, vCorrection } = adamBiasCorrections(t);
    const T = this.ort.Tensor;
    const feeds = {
      [manifest.inputName]: new T("float32", batchInput, manifest.inputShape),
      [manifest.teacherName]: new T("float32", target, manifest.teacherShape),
      [manifest.lrName]: new T("float32", new Float32Array([lr]), []),
      m_correction: new T("float32", new Float32Array([mCorrection]), []),
      v_correction: new T("float32", new Float32Array([vCorrection]), []),
    };
    for (const { input, shape } of manifest.state) {
      feeds[input] = new T("float32", state[input], shape);
    }

    const results = await this.session.run(feeds);

    const nextState = {};
    for (const { input, output } of manifest.state) {
      nextState[input] = results[output].data;
    }
    return { loss: results[manifest.lossName].data[0], state: nextState };
  }
}

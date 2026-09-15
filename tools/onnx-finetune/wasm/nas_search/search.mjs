// A tiny neural-architecture-search loop: given several candidate
// architectures' own training-step graphs (../../scripts/
// generate_nas_step_graphs.py), train each for the same handful of steps on
// the same synthetic batch and rank them by final loss.
//
// Reuses ../federated_lora/step_graph_runner.mjs's StepGraphSession/
// parseManifest/loadInitialState *unchanged* rather than duplicating them:
// generate_nas_step_graphs.py deliberately writes the exact same manifest
// shape (input_name/input_shape/teacher_name/teacher_shape/lr_name/
// loss_name/state/weight lines, fixed batch size) that generator writes for
// federated LoRA, since from the runner's point of view a NAS candidate's
// step graph is not a new kind of thing -- it is another fixed-batch
// forward+loss+backward+Adam step graph, just with a different node count.
// See that file's own module comment for what onnxruntime-web execution
// providers a StepGraphSession can run on, in particular
// `{ executionProviders: ["webgpu", "wasm"] }` for training on the browser's
// own GPU API. This module adds nothing EP-specific of its own: whatever
// providers a StepGraphSession is created with, every candidate here trains
// on.
import { StepGraphSession, loadInitialState, parseManifest } from "../federated_lora/step_graph_runner.mjs";

/**
 * Trains `candidate` (see `loadCandidates`'s own return shape) for
 * `numSteps` steps against one fixed `{ batchInput, target }` batch, and
 * returns its own loss trajectory plus how long each step actually took on
 * whatever execution provider this ran on. A fixed batch across steps
 * (rather than fresh noise each step) is deliberate: a search loop
 * comparing several architectures wants each one scored on the same
 * problem, the same way k-fold cross-validation holds its folds fixed
 * across the models it compares.
 *
 * `meanStepMs` is a real, measured per-step wall-clock latency on the
 * device this actually ran on -- not a predicted or looked-up latency the
 * way hardware-aware NAS (MnasNet, FBNet, ProxylessNAS) usually scores
 * candidates, since here every candidate already runs its real step graph
 * on a real execution provider. `performance.now()` is available both as a
 * browser global and, since Node 16, as a Node global (no import needed),
 * so this works unmodified in both of this module's own callers (a browser
 * page, and search.test.mjs under plain Node).
 */
export async function trainCandidate(ort, candidate, batchInput, target, { lr, numSteps, executionProviders }) {
  const manifest = parseManifest(candidate.manifestText);
  let state = loadInitialState(manifest, candidate.initialStateBuffer);
  const session = await StepGraphSession.create(ort, candidate.stepGraphBytes, { executionProviders });

  const losses = [];
  const start = performance.now();
  for (let t = 0; t < numSteps; ++t) {
    const { loss, state: nextState } = await session.step(manifest, state, batchInput, target, lr, t);
    losses.push(loss);
    state = nextState;
  }
  const totalMs = performance.now() - start;
  await session.release?.();
  return {
    name: candidate.name,
    paramCount: candidate.paramCount,
    losses,
    finalLoss: losses[losses.length - 1],
    totalMs,
    meanStepMs: totalMs / numSteps,
  };
}

/**
 * `result`'s own loss/latency trade-off as one number to rank by: the loss
 * plus `latencyWeight` times the mean per-step latency, so a slower
 * candidate needs a proportionally better loss to still win. `latencyWeight
 * = 0` (the default `searchArchitectures` uses) recovers plain loss-only
 * ranking. There is no universally "right" weight -- it depends on how much
 * a millisecond of step latency is worth against a unit of loss for
 * whatever this search is actually for -- so this takes it as a parameter
 * rather than picking one.
 */
export function combinedScore(result, latencyWeight = 0) {
  return result.finalLoss + latencyWeight * result.meanStepMs;
}

/**
 * The search loop itself: trains every candidate on the same batch, then
 * ranks them by `combinedScore` (lowest first) -- a truncated-training
 * proxy for "which architecture fits this problem best", the same shape as
 * any few-shot NAS scoring loop, just with the actual scoring signal being
 * `numSteps` real Adam steps of the real loss (and, if `latencyWeight` is
 * nonzero, real measured step latency too) rather than a zero-cost proxy.
 * `candidates` is an array of
 * `{ name, paramCount, stepGraphBytes, manifestText, initialStateBuffer }`
 * (one entry per `<name>.step_graph.onnx` generate_nas_step_graphs.py
 * wrote -- search.test.mjs's `loadCandidates` reads them off disk under
 * Node; a browser page would `fetch()` the same three files per candidate
 * instead). `options.latencyWeight` (default 0, i.e. loss-only ranking) is
 * consumed here; every other option in `options` passes through to
 * `trainCandidate` unchanged. Returns `{ ranked, best }`, `ranked` sorted
 * best-to-worst.
 */
export async function searchArchitectures(ort, candidates, batchInput, target, options) {
  const { latencyWeight = 0, ...trainOptions } = options;
  const results = [];
  for (const candidate of candidates) {
    results.push(await trainCandidate(ort, candidate, batchInput, target, trainOptions));
  }
  const ranked = [...results].sort(
    (a, b) => combinedScore(a, latencyWeight) - combinedScore(b, latencyWeight),
  );
  return { ranked, best: ranked[0] };
}

/**
 * A tiny xorshift PRNG so a synthetic batch is reproducible without pulling
 * in a dependency just for this loop -- same one
 * ../federated_lora/step_graph_runner.test.mjs already uses for its own
 * per-client synthetic data, plus one fix that file's own callers happen to
 * never trigger: xorshift's all-zero state is a fixed point (every shift of
 * 0 is still 0), so a caller passing the very natural default `seed = 0`
 * would otherwise get the same "random" value out of every single call --
 * which is exactly what silently produced a degenerate, all-identical
 * synthetic batch (and from there an immediate Infinity loss) before this
 * guard was added. Any other seed is passed through unchanged.
 */
export function makeRng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

/** A fixed `{ batchInput, target }` regression batch: `target` is a random
 * linear function of `batchInput` plus a small amount of noise, the
 * simplest problem where "more capacity fits it better, up to a point" is
 * still observable in a handful of Adam steps. `dim`/`out`/`batchSize` must
 * match the candidates' own `manifest.inputShape`/`teacherShape`.
 *
 * `inputScale` (default 1) multiplies every `batchInput` entry, and
 * `normalizeByDim` (default true) divides the target by `sqrt(dim)` the
 * way the discrete NAS search space's own problem does; darts_supernet.
 * test.mjs passes `{ inputScale: 5, normalizeByDim: false }` instead, since
 * a DARTS mixture over parameter-free candidate ops (see
 * generate_darts_supernet_step_graph.py) needs a wide-dynamic-range target
 * to make its architecture weights actually prefer the unbounded Identity
 * branch over the two Sigmoid branches (which saturate to a bounded range
 * no matter their input) -- see that test file's own comment for the
 * numeric trade-off this was tuned against.
 */
export function makeSyntheticBatch(batchSize, dim, out, seed = 0, options = {}) {
  const { inputScale = 1, normalizeByDim = true } = options;
  const rng = makeRng(seed);
  const gauss = () => {
    // Box-Muller, using two of the PRNG's own uniforms -- plain noise, not
    // cryptographic, so reusing this generator (rather than pulling in a
    // stats library) is enough.
    const u1 = Math.max(rng(), 1e-9);
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  const wTrue = Array.from({ length: dim * out }, gauss);
  const batchInput = Float32Array.from({ length: batchSize * dim }, () => inputScale * gauss());
  const target = new Float32Array(batchSize * out);
  for (let row = 0; row < batchSize; ++row) {
    for (let o = 0; o < out; ++o) {
      let acc = 0;
      for (let i = 0; i < dim; ++i) acc += batchInput[row * dim + i] * wTrue[i * out + o];
      target[row * out + o] = (normalizeByDim ? acc / Math.sqrt(dim) : acc) + 0.01 * gauss();
    }
  }
  return { batchInput, target };
}

/** The ordinary softmax, for reading an architecture-weight vector like
 * darts_supernet.test.mjs's trained `alpha` state back out as normalized
 * mixture weights -- the same computation
 * generate_darts_supernet_step_graph.py's own step graph does on-device
 * (`Exp`/`ReduceSum`/`Div`), just off-device here since there's no need to
 * differentiate through this particular call. */
export function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

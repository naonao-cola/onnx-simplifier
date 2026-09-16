// Runs onnxsim.webgpu_kernel_tuning's whole point -- generate several
// alternative WebGPU kernels for the same ONNX node, dispatch every one on a
// real device, keep the fastest -- live, inside the actual converter page,
// via Pyodide running tinygrad in-browser. This is the opt-in "Tune this
// kernel" feature behind the Custom WebGPU kernels panel
// (webgpu_kernel_annotations_view.mjs), not something that runs
// automatically: both Pyodide itself (~10MB+ wasm runtime) and tinygrad's
// wheel (fetched fresh from PyPI, no caching layer here yet) are real
// multi-second, multi-megabyte downloads a visitor should choose to start,
// not something loaded for every model.
//
// Scope: ANY op type tinygrad's own generic ONNX importer
// (tinygrad.nn.onnx.OnnxRunner) supports, not just Conv -- see
// generateNodeCandidates/NODE_TUNING_PY below for how. The one structural
// restriction is the same one Conv-only tuning always had: tinygrad must
// schedule the node's own isolated computation to exactly one kernel call.
// A node that schedules to *zero* kernel calls (a pure view op in isolation
// -- a contiguous Reshape/Transpose/Squeeze that doesn't need to move any
// data on its own) isn't an error, it just has nothing to tune -- see
// tuneNodeKernel's own `noKernel` result. A node that schedules to *more*
// than one kernel call, an unsupported op, a non-static shape, or a
// control-flow op needing subgraph handling (If/Loop/Scan/Gradient) raises a
// clear error instead of silently doing nothing, same as the old Conv-only
// reader did.
//
// Why tinygrad's own OnnxRunner instead of a hand-rolled per-op-type
// translation (the way the old Conv-only CONV_TUNING_PY called
// `x.conv2d(...)` directly): OnnxRunner already has its own hand-rolled
// protobuf parser (no onnx-Python-package dependency, matching this
// pipeline's own "no heavy deps inside Pyodide" preference) and a large,
// actively-maintained op table (100+ ops) that already implements the exact
// same op tinygrad's OnnxRunner.__call__ itself would run for that node --
// so generalizing to "any op" means driving that same dispatch
// (`OnnxRunner._select_op`), not re-implementing every op onnxsim doesn't
// otherwise have codegen for. See
// scripts/convertmodel/test/pyodide_webgpu_onnxrunner_codegen.test.mjs for
// the earlier, whole-graph-only proof this runs inside Pyodide at all.
//
// The "python const" problem: some ONNX ops take a *structural* argument as
// a second tensor input rather than an attribute (Reshape's target shape,
// Slice's starts/ends/axes, Squeeze/Unsqueeze's axes, ...) -- tinygrad's own
// OnnxRunner resolves these the same way, via
// `tinygrad.nn.onnx.required_input_python_consts` plus `Tensor.tolist()`.
// But `.tolist()` (like `.realize()`, `.data()`, or any other Tensor method
// that actually runs something) always tries to load tinygrad's *WEBGPU*
// device backend the moment it's asked to evaluate a WEBGPU-tagged tensor --
// even for a tensor that's just a literal constant, needing no real compute
// at all -- and that backend is a native (ctypes-loaded) wgpu library that
// doesn't exist inside Pyodide (or, for that matter, most native builds
// without it installed either). So this can never run *any* WEBGPU-tagged
// tensor through `.tolist()`/`.realize()`/`.data()`, for any reason, ever.
//
// The fix: NODE_TUNING_PY resolves every node in two separate passes.
//   1. A throwaway OnnxRunner instance tagged for tinygrad's "PYTHON"
//      device -- a pure-Python UOp interpreter that needs no native library
//      at all, so it's the one tinygrad device that actually *can* run
//      inside Pyodide -- runs the *whole* graph once, for real, with random
//      (but concrete) data for the model's own declared inputs. This gives
//      real, concrete values for every python-const-required input
//      anywhere in the graph (they're almost always static -- initializers,
//      or Shape/Constant-derived -- so random *graph*-input data doesn't
//      change them), and the real static shape/dtype of every other tensor
//      the target node touches.
//   2. A second, separate OnnxRunner instance tagged "WEBGPU" rebuilds
//      *only* the target node's own computation, from fresh random leaf
//      Tensors (shapes/dtypes from pass 1) for its ordinary tensor inputs
//      and the real resolved values from pass 1 for its python-const ones,
//      calling the exact same op function OnnxRunner._select_op would (so
//      Split/Gather/etc.'s own special-cased extra opts still get applied
//      the same way __call__ itself applies them). This tensor graph is
//      *never* realized -- only `to_program`'d (pure codegen, no execution)
//      -- so the WEBGPU device's own native library is never touched, same
//      as the original Conv-only implementation already relied on.
//   Rebuilding from fresh leaves (rather than reusing pass 1's own chained,
//   whole-graph tensors) is deliberate, not just a side effect of avoiding
//   device crossing: it's what guarantees tinygrad's scheduler can't fuse
//   this node's kernel with a neighbor's, which is what lets a caller swap
//   in exactly one node's own kernel via onnx_metadata_writer.mjs without
//   touching the rest of the graph -- see this file's own top-level
//   docstring in the previous (Conv-only) version of this module for the
//   same correctness argument, which still applies verbatim.
//
// Correctness note: candidates only need to compute the SAME answer as each
// other, not match any particular real input data, since picking a winner
// is purely about speed -- so every leaf tensor here (in pass 2) is filled
// with random data of the right shape, never real pipeline data. tinygrad's
// kernel generation only depends on shapes/attributes/python-consts, never
// on the concrete VALUES of its ordinary tensor inputs, so this is exactly
// as valid a correctness bar as using the model's own real inputs would be.

import { listAllNodeNames } from "./onnx_conv_node_reader.mjs";
import { dispatchWebgpuProgram, supportsWebgpuProfiling, readBackFloat32Buffer } from "./webgpu_kernel_dispatcher.mjs";

export const TINYGRAD_VERSION = "0.14.0";
// Matches the exact version scripts/convertmodel/package.json pins for the
// npm "pyodide" package (used by the Node+Playwright tests in test/) -- so
// what this in-page feature runs on a real visitor's browser is the same
// Pyodide build this repo's own tests already exercise, not just "some
// recent version". jsdelivr's URL shape is Pyodide's own documented CDN
// hosting convention (see scripts/pyodide_demo/index.html's own use of it
// for a different Pyodide version).
export const PYODIDE_VERSION = "314.0.7";
export const PYODIDE_CDN_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`;

const FETCH_TIMEOUT_MS = 60_000;
function fetchWithTimeout(url) {
  return fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
}

function loadPyodideScript() {
  if (globalThis.loadPyodide) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = PYODIDE_CDN_URL;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`failed to load Pyodide from ${PYODIDE_CDN_URL}`));
    document.head.appendChild(script);
  });
}

async function fetchTinygradWheel() {
  const meta = await fetchWithTimeout(`https://pypi.org/pypi/tinygrad/${TINYGRAD_VERSION}/json`).then((r) => r.json());
  const wheelInfo = meta.urls.find((u) => u.packagetype === "bdist_wheel");
  if (!wheelInfo) throw new Error(`no wheel (bdist_wheel) found for tinygrad==${TINYGRAD_VERSION} on PyPI`);
  return fetchWithTimeout(wheelInfo.url).then((r) => r.arrayBuffer());
}

// Cached across calls (module-level, not per-tune) so tuning a second node
// -- or re-tuning the same one -- doesn't reload Pyodide or refetch
// tinygrad's wheel. Cleared on failure so a transient network error doesn't
// permanently wedge the panel.
let pyodideReadyPromise = null;
function ensurePyodideWithTinygrad(report) {
  if (!pyodideReadyPromise) {
    pyodideReadyPromise = (async () => {
      report?.(`loading Pyodide ${PYODIDE_VERSION} from ${new URL(PYODIDE_CDN_URL, location.href).host}...`);
      await loadPyodideScript();
      const pyodide = await globalThis.loadPyodide();
      report?.(`Pyodide ${pyodide.version} ready -- fetching tinygrad ${TINYGRAD_VERSION} wheel from PyPI...`);
      const wheelBuffer = await fetchTinygradWheel();
      pyodide.unpackArchive(wheelBuffer, "zip", { extractDir: "/tinygrad_pkg" });
      await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/tinygrad_pkg")');
      report?.("tinygrad ready");
      return pyodide;
    })().catch((err) => {
      pyodideReadyPromise = null;
      throw err;
    });
  }
  return pyodideReadyPromise;
}

// A numpy-free, onnx-free (Python-package-wise -- it uses tinygrad's own
// OnnxRunner, not the "onnx" package) kernel-candidate generator for ANY
// node in the graph -- see this file's own header comment for the two-pass
// (PYTHON device for real shapes/consts, WEBGPU device for isolated
// codegen) design this implements.
const NODE_TUNING_PY = `
import json
import random
import re
from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.search import get_kernel_actions
from tinygrad.helpers import Context, Target
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.uop.ops import Ops
# tinygrad.nn.onnx itself is NOT imported at module top level: importing it
# evaluates a stray Tensor(0) default argument (for ConvInteger) at *import*
# time, which resolves tinygrad's ambient default device the same way any
# other tensor construction would -- and Pyodide has no usable device at all
# (no GPU, no native compiler for a CPU backend) outside of an explicit
# Context(DEV=...), so a bare top-level import crashes with "no usable
# devices" the moment this whole script is first run. Every place below that
# needs it imports it locally, inside its own Context block -- see
# make_webgpu_onnxrunner_fixture.py's own docstring for where this was first
# found (this file's own single-op predecessor, CONV_TUNING_PY, never hit it
# because it only ever imported tinygrad.nn.onnx's *sibling* modules).

_ANSI_RE = re.compile(r"\\x1b\\[[0-9;]*m")
_UNSUPPORTED_CONTROL_FLOW_OPS = ("Gradient", "If", "Loop", "Scan")


def _rand(*shape):
    if len(shape) == 1:
        return [random.gauss(0, 1) for _ in range(shape[0])]
    return [_rand(*shape[1:]) for _ in range(shape[0])]


def _base_buffer(t):
    for u in t.uop.toposort():
        if u.op is Ops.BUFFER:
            return u
    return None


def _encode_float(v):
    # json.dumps happily emits the bare (non-standard) tokens Infinity/NaN
    # for a real float("inf")/float("nan") -- valid Python, not valid JSON,
    # so JS's own JSON.parse rejects it. webgpu_kernel_dispatcher.mjs's own
    # resolveBindingBuffer already expects these as the string sentinels
    # below (via a plain Number(v)), matching onnxsim.webgpu_kernel_metadata's
    # own _encode_float for the exact same reason.
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "Infinity"
    if v == float("-inf"):
        return "-Infinity"
    return v


def _render_step(ast, buffer_uops, uop_to_name, renderer):
    prg = to_program(ast, renderer)
    info = prg.arg
    source_uop = next(s for s in prg.src if s.op is Ops.SOURCE)
    wgsl = source_uop.arg
    entry_point = _ANSI_RE.sub("", info.function_name)

    bindings = [{"group": 0, "binding": 0, "access": "uniform", "constant": [_encode_float(float("inf"))]}]
    binding_bytes = {}
    for slot, buf_uop in enumerate(buffer_uops):
        name = uop_to_name.get(buf_uop)
        if name is None:
            raise RuntimeError(
                "node tuning candidate references a buffer with no known tensor name -- "
                "not supported (this can only happen for a node with more than one kernel "
                "call, which is already rejected before reaching here)"
            )
        bindings.append({"group": 0, "binding": slot + 1, "access": "read_write", "tensor": name})
        binding_bytes[name] = buf_uop.size()[0] * buf_uop.dtype.itemsize

    padded_size = [int(x) for x in info.global_size] + [1, 1, 1]
    return {"wgsl": wgsl, "entry_point": entry_point, "dispatch": padded_size[:3], "bindings": bindings}, binding_bytes


def _resolve_node(model_path, node_index):
    """Pass 1 -- see this file's own module docstring. Runs the *whole*
    graph, for real, on tinygrad's "PYTHON" device (the only one that needs
    no native library, so the only one usable inside Pyodide at all for
    actually running something), purely to learn the target node's own real
    input shapes/dtypes and any python-const values it structurally needs.
    """
    random.seed(0)
    with Context(DEV="PYTHON"):
        from tinygrad.nn.onnx import OnnxRunner, required_input_python_consts

        run_py = OnnxRunner(model_path)
        feed = {}
        for name, spec in run_py.graph_inputs.items():
            shape = [d if isinstance(d, int) else 1 for d in spec.shape]
            feed[name] = Tensor(_rand(*shape) if shape else 0.0, dtype=spec.dtype)
        run_py(feed)

        if node_index >= len(run_py.graph_nodes):
            raise RuntimeError(f"node index {node_index} out of range ({len(run_py.graph_nodes)} node(s) in graph)")
        node = run_py.graph_nodes[node_index]

        real_inputs = {}
        for i, name in enumerate(node.inputs):
            if not name:
                continue
            t = run_py.graph_values.get(name)
            if t is None:
                raise RuntimeError(f"no value available for input {name!r} of node {node_index} ({node.op})")
            if i in required_input_python_consts.get(node.op, ()):
                real_inputs[name] = {"kind": "const", "value": t.tolist() if isinstance(t, Tensor) else t}
            elif isinstance(t, Tensor):
                real_inputs[name] = {"kind": "shape", "shape": [int(d) for d in t.shape], "dtype": t.dtype}
            else:
                raise RuntimeError(f"input {name!r} of node {node_index} ({node.op}) isn't a tensor -- not supported")
        return node.op, real_inputs, len(node.outputs)


def generate_node_tuning_candidates(model_path, node_index, search):
    node_op, real_inputs, num_outputs = _resolve_node(model_path, node_index)
    if node_op in _UNSUPPORTED_CONTROL_FLOW_OPS:
        raise RuntimeError(f"{node_op} is not supported by node tuning (needs subgraph handling)")

    random.seed(1)
    with Context(DEV="WEBGPU"):
        from tinygrad.nn.onnx import OnnxRunner

        # Pass 2 -- see this file's own module docstring. A fresh OnnxRunner
        # instance, never realized: only used for its op table
        # (_select_op)/per-node opset resolution, which is device-agnostic.
        run_wg = OnnxRunner(model_path)
        node = run_wg.graph_nodes[node_index]
        if node.op != node_op:
            raise RuntimeError("internal error: node index resolved to a different op between passes")

        inps = []
        named = {}
        for i, name in enumerate(node.inputs):
            if not name:
                inps.append(None)
                continue
            info = real_inputs[name]
            if info["kind"] == "const":
                inps.append(info["value"])
            else:
                shape = info["shape"]
                leaf = Tensor(_rand(*shape) if shape else 0.0, dtype=info["dtype"], device="WEBGPU")
                named[name] = leaf
                inps.append(leaf)

        opts = dict(node.opts)
        if node.op == "Split" and "num_outputs" not in opts:
            opts["num_outputs"] = num_outputs

        fn = run_wg._select_op(node.op, node.opset_id)
        ret = fn(*inps, **opts)
        ret = ret if isinstance(ret, tuple) else (ret,)
        if not all(isinstance(r, Tensor) for r in ret):
            raise RuntimeError(f"{node_op} does not produce plain tensor output(s) -- not supported by node tuning")
        output_names = []
        for out_name, out_t in zip(node.outputs, ret):
            if out_name:
                named[out_name] = out_t
                output_names.append(out_name)

        linear = ret[0].schedule_linear(*ret[1:])
        kernel_calls = [u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK]

        output_shapes = [list(t.shape) for t in ret]
        output_is_float = bool(dtypes.is_float(ret[0].dtype)) if ret else False
        if len(kernel_calls) == 0:
            return {
                "opType": node_op,
                "outputShapes": output_shapes,
                "outputNames": output_names,
                "outputIsFloat": output_is_float,
                "kernelCallCount": 0,
                "candidates": [],
                "bindingByteLengths": {},
            }
        if len(kernel_calls) != 1:
            raise RuntimeError(
                f"node tuning only supports a node tinygrad schedules to exactly one kernel call, "
                f"got {len(kernel_calls)} for {node_op}"
            )

        call = kernel_calls[0]
        ast = call.src[0]
        buffer_uops = list(call.src[1:])

        uop_to_name = {}
        for name, t in named.items():
            buf = _base_buffer(t)
            if buf is not None:
                uop_to_name[buf] = name

        renderer = WGSLRenderer(Target())
        scheduler = Scheduler(ast, renderer)
        actions = get_kernel_actions(scheduler, include_0=True)
        if not search:
            # Profiling fast path: only render the baseline (unoptimized)
            # kernel -- key 0 is always the original, un-acted-on scheduler
            # when include_0=True -- rather than searching every action.
            actions = {0: actions[0]}

        candidates = []
        binding_bytes = {}
        for _, candidate in actions.items():
            candidate_ast = candidate.get_optimized_ast()
            step, step_bytes = _render_step(candidate_ast, buffer_uops, uop_to_name, renderer)
            binding_bytes.update(step_bytes)
            candidates.append({"appliedOpts": repr(candidate.applied_opts), "step": step})

        return {
            "opType": node_op,
            "outputShapes": output_shapes,
            "outputNames": output_names,
            "outputIsFloat": output_is_float,
            "kernelCallCount": 1,
            "candidates": candidates,
            "bindingByteLengths": binding_bytes,
        }


json.dumps(generate_node_tuning_candidates(MODEL_PATH, NODE_INDEX, SEARCH))
`;

async function generateNodeCandidates(pyodide, modelBytes, nodeIndex, search) {
  pyodide.FS.writeFile("/model.onnx", modelBytes);
  pyodide.globals.set("MODEL_PATH", "/model.onnx");
  pyodide.globals.set("NODE_INDEX", nodeIndex);
  pyodide.globals.set("SEARCH", search);
  const raw = await pyodide.runPythonAsync(NODE_TUNING_PY);
  return JSON.parse(raw);
}

function findNodeIndex(modelBytes, nodeName) {
  const entry = listAllNodeNames(modelBytes).find((n) => n.name === nodeName);
  if (!entry) {
    throw new Error(`no node named ${JSON.stringify(nodeName)} in the graph`);
  }
  return entry.index;
}

// A dedicated, zero-filled storage buffer -- unlike
// webgpu_kernel_dispatcher.mjs's own createStorageBuffer (Float32Array-only,
// used where a caller has real or random *float* pipeline data to seed),
// node tuning candidates can bind tensors of any dtype tinygrad's own op
// table produces (e.g. an int-typed Gather index), so this only needs the
// right *byte length* (reported by NODE_TUNING_PY, computed from tinygrad's
// own real dtype/itemsize for that buffer) -- the exact bytes don't matter
// for a timing-only dispatch (WebGPU's own storage-buffer bounds guarantees
// mean stray/out-of-range values can't fault the GPU, only produce
// meaningless output, which is fine since candidates are never checked for
// correctness here -- see this file's own module docstring).
function zeroStorageBuffer(device, byteLength) {
  return device.createBuffer({
    size: Math.max(byteLength, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

async function dispatchAndTime(device, step, bindingByteLengths, { warmupRuns, timedRuns, profile, sampleOutputName }) {
  const buffersByTensor = new Map();
  for (const [name, byteLength] of Object.entries(bindingByteLengths)) {
    buffersByTensor.set(name, zeroStorageBuffer(device, byteLength));
  }
  const spec = { steps: [step], intermediates: {} };
  try {
    for (let i = 0; i < warmupRuns; i++) {
      await dispatchWebgpuProgram(device, spec, buffersByTensor, {});
    }
    const durationsMs = [];
    let lastTimings = null;
    for (let i = 0; i < timedRuns; i++) {
      const t0 = performance.now();
      const { timings } = await dispatchWebgpuProgram(device, spec, buffersByTensor, { profile });
      durationsMs.push(performance.now() - t0);
      lastTimings = timings;
    }
    // A soft correctness sanity check, not a real one (candidates are ranked
    // purely on speed against zero-filled/garbage inputs -- see this file's
    // own module docstring): a candidate that produced NaN/Infinity likely
    // has a broken binding, worth flagging even though it isn't compared
    // against any reference here. Only meaningful for a float-dtype output
    // -- an int-typed one reinterpreted as float32 would trip this on
    // perfectly good output, so callers only pass sampleOutputName for a
    // float output (see tuneNodeKernel/NODE_TUNING_PY's own outputIsFloat).
    let sampleFinite = null;
    if (sampleOutputName) {
      const buffer = buffersByTensor.get(sampleOutputName);
      const byteLength = bindingByteLengths[sampleOutputName];
      const sample = await readBackFloat32Buffer(device, buffer, Math.min(Math.floor(byteLength / 4), 16));
      sampleFinite = Array.from(sample).every(Number.isFinite);
    }
    return {
      medianMs: median(durationsMs),
      gpuDurationNs: lastTimings ? lastTimings.reduce((sum, t) => sum + t.durationNs, 0) : null,
      sampleFinite,
    };
  } finally {
    for (const buffer of buffersByTensor.values()) buffer.destroy();
  }
}

/**
 * Tunes the node named ``nodeName`` in ``modelBytes``, whatever its op type:
 * generates every alternative WebGPU kernel tinygrad's own kernel-opt search
 * finds for it (live, via Pyodide+tinygrad), dispatches every one on
 * ``device`` with zero-filled buffers of the node's own real shapes, times
 * each, and returns them ranked fastest first. No cap on how many
 * candidates get tried -- every one ``get_kernel_actions`` finds is
 * dispatched.
 *
 * @param {GPUDevice} device
 * @param {Uint8Array} modelBytes
 * @param {string} nodeName
 * @param {{warmupRuns?: number, timedRuns?: number,
 *          onProgress?: (msg: string) => void}} [options]
 * @returns {Promise<{nodeName: string, opType: string, noKernel: boolean,
 *          results: Array<{appliedOpts: string, spec: object, medianMs: number,
 *          gpuDurationNs: number|null}>, winner: object|null}>}
 *          ``results`` is sorted fastest (lowest ``medianMs``) first;
 *          ``winner`` is ``results[0]``. ``noKernel: true`` (with empty
 *          ``results``/null ``winner``) means tinygrad scheduled this node's
 *          own isolated computation to zero kernel calls -- a pure view op
 *          that needs no dedicated kernel on its own -- which is a normal
 *          outcome, not an error.
 * @throws if ``nodeName`` doesn't exist, is a control-flow op needing
 *         subgraph handling, isn't supported by tinygrad's own OnnxRunner op
 *         table, has a non-static shape, or schedules to more than one
 *         kernel call in isolation.
 */
export async function tuneNodeKernel(device, modelBytes, nodeName, options = {}) {
  const { warmupRuns = 2, timedRuns = 5, onProgress } = options;
  const nodeIndex = findNodeIndex(modelBytes, nodeName);

  const pyodide = await ensurePyodideWithTinygrad(onProgress);

  onProgress?.("generating kernel candidate(s)...");
  const { opType, kernelCallCount, candidates, bindingByteLengths, outputNames, outputIsFloat } =
    await generateNodeCandidates(pyodide, modelBytes, nodeIndex, true);

  if (kernelCallCount === 0) {
    onProgress?.(`${opType} needs no dedicated WebGPU kernel in isolation (a pure view op) -- nothing to tune`);
    return { nodeName, opType, noKernel: true, results: [], winner: null };
  }
  if (candidates.length === 0) {
    throw new Error("tinygrad produced no tuning candidates for this node");
  }

  onProgress?.(`dispatching ${candidates.length} candidate(s) on the real WebGPU device...`);
  const canProfile = supportsWebgpuProfiling(device);
  const sampleOutputName = outputIsFloat && outputNames.length ? outputNames[0] : null;

  const results = [];
  for (const candidate of candidates) {
    const timing = await dispatchAndTime(device, candidate.step, bindingByteLengths, {
      warmupRuns,
      timedRuns,
      profile: canProfile,
      sampleOutputName,
    });
    results.push({
      appliedOpts: candidate.appliedOpts,
      spec: { steps: [candidate.step], intermediates: {} },
      medianMs: timing.medianMs,
      gpuDurationNs: timing.gpuDurationNs,
      sampleFinite: timing.sampleFinite,
    });
  }

  results.sort((a, b) => a.medianMs - b.medianMs);
  onProgress?.(`done -- fastest candidate: ${results[0].appliedOpts} (${results[0].medianMs.toFixed(3)}ms)`);
  return { nodeName, opType, noKernel: false, results, winner: results[0] };
}

/**
 * The "profile first" half of the workflow: renders and dispatches just the
 * *baseline* (unoptimized) kernel for every node in ``nodeNames`` -- no
 * candidate search -- timing each once each, cheaply, so a caller can find
 * out which nodes actually dominate a model's real latency before spending
 * a full tuning search on any of them (see
 * webgpu_kernel_profile.mjs's selectDominantNodes for turning this into a
 * "tune only these" subset).
 *
 * Never throws for a single node's own failure (unsupported op, no kernel
 * in isolation, ...) -- that node is reported with ``skipped: true`` and an
 * ``error`` message instead, so one node that can't be profiled doesn't
 * abort profiling the rest of the graph (same philosophy as
 * webgpu_kernel_annotations_view.mjs's "Tune full graph").
 *
 * @param {GPUDevice} device
 * @param {Uint8Array} modelBytes
 * @param {string[]} nodeNames
 * @param {{onProgress?: (msg: string) => void}} [options]
 * @returns {Promise<Array<{nodeName: string, opType: string|null,
 *          medianMs: number, skipped: boolean, error?: string}>>}
 */
export async function profileNodeLatency(device, modelBytes, nodeNames, options = {}) {
  const { onProgress } = options;
  const pyodide = await ensurePyodideWithTinygrad(onProgress);
  const canProfile = supportsWebgpuProfiling(device);

  const results = [];
  for (const nodeName of nodeNames) {
    onProgress?.(`profiling ${nodeName}...`);
    try {
      const nodeIndex = findNodeIndex(modelBytes, nodeName);
      const { opType, kernelCallCount, candidates, bindingByteLengths } = await generateNodeCandidates(
        pyodide,
        modelBytes,
        nodeIndex,
        false,
      );
      if (kernelCallCount === 0 || candidates.length === 0) {
        results.push({ nodeName, opType, medianMs: 0, skipped: true });
        continue;
      }
      const timing = await dispatchAndTime(device, candidates[0].step, bindingByteLengths, {
        warmupRuns: 1,
        timedRuns: 3,
        profile: canProfile,
      });
      results.push({ nodeName, opType, medianMs: timing.medianMs, skipped: false });
    } catch (err) {
      console.error(`webgpu kernel profiler (${nodeName}):`, err);
      results.push({ nodeName, opType: null, medianMs: 0, skipped: true, error: (err && err.message) || String(err) });
    }
  }
  onProgress?.("profiling done");
  return results;
}

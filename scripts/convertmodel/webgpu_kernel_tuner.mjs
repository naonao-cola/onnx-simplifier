// Runs onnxsim.webgpu_kernel_tuning's whole point -- generate several
// alternative WebGPU kernels for the same Conv node, dispatch every one on a
// real device, keep the fastest -- live, inside the actual converter page,
// via Pyodide running tinygrad in-browser. This is the opt-in "Tune this
// kernel" feature behind the Custom WebGPU kernels panel
// (webgpu_kernel_annotations_view.mjs), not something that runs
// automatically: both Pyodide itself (~10MB+ wasm runtime) and tinygrad's
// wheel (fetched fresh from PyPI, no caching layer here yet) are real
// multi-second, multi-megabyte downloads a visitor should choose to start,
// not something loaded for every model.
//
// Scope: Conv nodes only (whatever shape/rank onnx_conv_node_reader.mjs
// itself supports -- includes what this repo calls "Conv3D", which is just
// an ordinary ONNX Conv node with 3 spatial dims), and only when tinygrad
// schedules the node to exactly one kernel call -- true for every Conv
// shape this repo's own fixtures use, but not guaranteed in general (a
// fused Conv+activation, for instance, could in principle schedule
// differently). A node that isn't a plain Conv, or that isn't a Conv this
// module can regenerate the kernel for, raises with a clear error rather
// than silently doing nothing -- see readConvNodeInfo's own scope notes.
//
// Correctness note: candidates only need to compute the SAME answer as each
// other, not match any particular real input data, since picking a winner
// is purely about speed -- so every buffer here is filled with random data
// of the right shape (matching onnxsim.webgpu_kernel_tuning's own offline
// fixtures), never real pipeline data. tinygrad's kernel generation only
// depends on shapes/attributes, never on concrete tensor values (see
// onnxsim.webgpu_tinygrad_codegen's own module docstring, and
// pyodide_webgpu_single_node_codegen.test.mjs's byte-identical-WGSL check),
// so this is exactly as valid a correctness bar as using the model's own
// real inputs would be, without needing to run the rest of the graph first
// to obtain them.

import { readConvNodeInfo } from "./onnx_conv_node_reader.mjs";
import { dispatchWebgpuProgram, createStorageBuffer, readBackFloat32Buffer, supportsWebgpuProfiling } from "./webgpu_kernel_dispatcher.mjs";

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

// A numpy-free, onnx-free kernel-candidate generator -- see this file's own
// header comment for why (never imports numpy/onnx, plain nested Python
// lists as Tensor leaves), translated line-for-line from
// onnxsim.webgpu_kernel_tuning.generate_kernel_candidates +
// onnxsim.webgpu_tinygrad_codegen's own per-call rendering, the same
// translation pyodide_webgpu_single_node_codegen.test.mjs's CONV_CODEGEN_PY
// already did for the non-tuning (single kernel) case -- see that file for
// the original this reuses almost verbatim below the schedule/get_kernel_actions
// split point.
const CONV_TUNING_PY = `
import json
import random
import re
from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.search import get_kernel_actions
from tinygrad.helpers import Target
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.uop.ops import Ops

_ANSI_RE = re.compile(r"\\x1b\\[[0-9;]*m")


def _rand(*shape):
    if len(shape) == 1:
        return [random.gauss(0, 1) for _ in range(shape[0])]
    return [_rand(*shape[1:]) for _ in range(shape[0])]


def _encode_float(v):
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "Infinity"
    if v == float("-inf"):
        return "-Infinity"
    return v


def _base_buffer(t):
    for u in t.uop.toposort():
        if u.op is Ops.BUFFER:
            return u
    return None


def _render_step(ast, buffer_uops, uop_to_name, renderer):
    prg = to_program(ast, renderer)
    info = prg.arg
    source_uop = next(s for s in prg.src if s.op is Ops.SOURCE)
    wgsl = source_uop.arg
    entry_point = _ANSI_RE.sub("", info.function_name)

    bindings = [{"group": 0, "binding": 0, "access": "uniform", "constant": [_encode_float(float("inf"))]}]
    for slot, buf_uop in enumerate(buffer_uops):
        name = uop_to_name.get(buf_uop)
        if name is None:
            raise RuntimeError("tuning candidate references an intermediate buffer -- not supported here (single-kernel Conv only)")
        bindings.append({"group": 0, "binding": slot + 1, "access": "read_write", "tensor": name})

    padded_size = [int(x) for x in info.global_size] + [1, 1, 1]
    return {"wgsl": wgsl, "entry_point": entry_point, "dispatch": padded_size[:3], "bindings": bindings}


def _rename_tensor_bindings(spec, rename):
    for step in spec["steps"]:
        for b in step["bindings"]:
            if "tensor" in b and b["tensor"] in rename:
                b["tensor"] = rename[b["tensor"]]
    return spec


def generate_conv_tuning_candidates(node_info, max_candidates):
    random.seed(0)
    x_shape = node_info["xShape"]
    w_shape = node_info["wShape"]
    spatial_rank = len(w_shape) - 2
    strides = node_info["strides"]
    dilations = node_info["dilations"]
    group = node_info["group"]
    pads = node_info["pads"]
    pads_begin, pads_end = pads[:spatial_rank], pads[spatial_rank:]

    x = Tensor(_rand(*x_shape), device="WEBGPU")
    w = Tensor(_rand(*w_shape), device="WEBGPU")
    named = {"__x": x, "__w": w}
    if pads_begin != [0] * spatial_rank or pads_end != [0] * spatial_rank:
        pad_pairs = [None, None] + [(b, e) for b, e in zip(pads_begin, pads_end)]
        x = x.pad(pad_pairs)

    b_shape = node_info.get("bShape")
    if b_shape:
        b = Tensor(_rand(*b_shape), device="WEBGPU")
        named["__b"] = b
    else:
        b = None

    y = x.conv2d(w, bias=b, groups=group, stride=strides, dilation=dilations, padding=0)
    named["__y"] = y

    linear = y.schedule_linear()
    kernel_calls = [u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK]
    if not kernel_calls:
        raise RuntimeError("tinygrad scheduled no compute kernel for this node -- it may have constant-folded away")
    if len(kernel_calls) != 1:
        raise RuntimeError(
            f"tuning only supports a node tinygrad schedules to exactly one kernel call, got {len(kernel_calls)}"
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

    rename = {"__x": node_info["xName"], "__w": node_info["wName"], "__y": node_info["outputName"]}
    if node_info.get("bName"):
        rename["__b"] = node_info["bName"]

    candidates = []
    for _, candidate in list(actions.items())[:max_candidates]:
        candidate_ast = candidate.get_optimized_ast()
        step = _render_step(candidate_ast, buffer_uops, uop_to_name, renderer)
        spec = {"steps": [step], "intermediates": {}}
        _rename_tensor_bindings(spec, rename)
        candidates.append({"appliedOpts": repr(candidate.applied_opts), "spec": spec})

    return {"outputShape": list(y.shape), "candidates": candidates}


json.dumps(generate_conv_tuning_candidates(json.loads(NODE_INFO_JSON), MAX_CANDIDATES))
`;

async function generateCandidates(pyodide, nodeInfo, maxCandidates) {
  pyodide.globals.set("NODE_INFO_JSON", JSON.stringify(nodeInfo));
  pyodide.globals.set("MAX_CANDIDATES", maxCandidates);
  const raw = await pyodide.runPythonAsync(CONV_TUNING_PY);
  return JSON.parse(raw);
}

function product(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function randomFloat32(n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.random() * 2 - 1;
  return out;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Tunes the Conv node named ``nodeName`` in ``modelBytes``: generates several
 * alternative WebGPU kernels (live, via Pyodide+tinygrad), dispatches every
 * one on ``device`` with random data of the node's own shapes, times each,
 * and returns them ranked fastest first.
 *
 * @param {GPUDevice} device
 * @param {Uint8Array} modelBytes
 * @param {string} nodeName
 * @param {{maxCandidates?: number, warmupRuns?: number, timedRuns?: number,
 *          onProgress?: (msg: string) => void}} [options]
 * @returns {Promise<{nodeInfo: object, results: Array<{appliedOpts: string,
 *          spec: object, medianMs: number, gpuDurationNs: number|null}>,
 *          winner: object}>}
 *          ``results`` is sorted fastest (lowest ``medianMs``) first;
 *          ``winner`` is ``results[0]``.
 * @throws if ``nodeName`` isn't a Conv node this module can regenerate (see
 *         ``onnx_conv_node_reader.mjs``'s own scope notes), or if tinygrad
 *         schedules it to anything other than exactly one kernel call.
 */
export async function tuneConvKernel(device, modelBytes, nodeName, options = {}) {
  const { maxCandidates = 8, warmupRuns = 2, timedRuns = 5, onProgress } = options;

  // Fails fast, before touching Pyodide at all, for a node this can't
  // handle -- the same validation/error messages
  // pyodide_webgpu_single_node_codegen.test.mjs's own live-codegen path uses.
  const nodeInfo = readConvNodeInfo(modelBytes, nodeName);

  const pyodide = await ensurePyodideWithTinygrad(onProgress);

  onProgress?.(`generating up to ${maxCandidates} kernel candidate(s)...`);
  const { outputShape, candidates } = await generateCandidates(pyodide, nodeInfo, maxCandidates);
  if (candidates.length === 0) {
    throw new Error("tinygrad produced no tuning candidates for this node");
  }

  onProgress?.(`dispatching ${candidates.length} candidate(s) on the real WebGPU device...`);
  const outputLength = product(outputShape);
  const canProfile = supportsWebgpuProfiling(device);

  const results = [];
  for (const candidate of candidates) {
    const buffersByTensor = new Map();
    buffersByTensor.set(nodeInfo.xName, createStorageBuffer(device, randomFloat32(product(nodeInfo.xShape))));
    buffersByTensor.set(nodeInfo.wName, createStorageBuffer(device, randomFloat32(product(nodeInfo.wShape))));
    if (nodeInfo.bName) {
      buffersByTensor.set(nodeInfo.bName, createStorageBuffer(device, randomFloat32(product(nodeInfo.bShape))));
    }
    buffersByTensor.set(nodeInfo.outputName, createStorageBuffer(device, new Float32Array(outputLength)));

    for (let i = 0; i < warmupRuns; i++) {
      await dispatchWebgpuProgram(device, candidate.spec, buffersByTensor, {});
    }
    const durationsMs = [];
    let lastTimings = null;
    for (let i = 0; i < timedRuns; i++) {
      const t0 = performance.now();
      const { timings } = await dispatchWebgpuProgram(device, candidate.spec, buffersByTensor, { profile: canProfile });
      durationsMs.push(performance.now() - t0);
      lastTimings = timings;
    }
    // Read back once per candidate, only to prove it actually computed
    // something (not NaN/all-zero from a broken binding) -- not compared
    // against a reference, since random input has no reference to compare
    // against (see this file's own header comment on why that's fine here).
    const sample = await readBackFloat32Buffer(device, buffersByTensor.get(nodeInfo.outputName), Math.min(outputLength, 16));

    results.push({
      appliedOpts: candidate.appliedOpts,
      spec: candidate.spec,
      medianMs: median(durationsMs),
      gpuDurationNs: lastTimings ? lastTimings.reduce((sum, t) => sum + t.durationNs, 0) : null,
      sampleFinite: Array.from(sample).every(Number.isFinite),
    });
  }

  results.sort((a, b) => a.medianMs - b.medianMs);
  onProgress?.(`done -- fastest candidate: ${results[0].appliedOpts} (${results[0].medianMs.toFixed(3)}ms)`);
  return { nodeInfo, results, winner: results[0] };
}

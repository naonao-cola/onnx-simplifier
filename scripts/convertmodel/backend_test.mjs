// Browser glue for the "Run an ONNX backend test case" panel.
//
// The ONNX repo ships backend test cases under onnx/backend/test/data/... Each
// case directory holds a `model.onnx` plus one or more `test_data_set_N/`
// directories of serialized TensorProtos: `input_*.pb` feed the model's graph
// inputs (positionally) and `output_*.pb` are the expected results. Given a URL
// to such a directory, this panel fetches the model and the test data, runs the
// model through onnxruntime-web, and checks each output against the expected
// tensor within tolerance.
//
// GitHub `tree`/`blob` URLs, `raw.githubusercontent.com` URLs, and the contents
// API URL are all accepted (see parseGithubDirUrl). Directory listing and raw
// file bytes come from the GitHub REST contents API (CORS-enabled, but
// unauthenticated and rate-limited — fine for a demo). TensorProtos are decoded
// by the module's `onnxsim_parse_tensor` (this page's runtime), which needs no
// onnxruntime-web, so decoding works even if inference can't run.

import { downloadText } from "./download.mjs";
import { syncInputUrl } from "./query_params.mjs";
import { ORT_VERSION, ORT_BASE } from "./cdn.mjs";

// A curated set of common ONNX backend *node* test cases for the picker, like
// the Hugging Face model dropdown. Each entry's value is the GitHub tree URL of
// the test-case directory; the free-text box accepts any other URL. The base is
// onnx/onnx@main's node-test data directory.
const BACKEND_NODE_BASE =
  "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/";
export const BACKEND_PRESETS = [
  "test_relu",
  "test_sigmoid",
  "test_tanh",
  "test_abs",
  "test_exp",
  "test_log",
  "test_softmax_example",
  "test_add",
  "test_sub",
  "test_mul",
  "test_div",
  "test_matmul_2d",
  "test_gemm_default_zero_bias",
  "test_transpose_default",
  "test_concat_1d_axis_0",
  "test_conv_with_strides_no_padding",
  "test_basic_conv_with_padding",
  "test_maxpool_2d_default",
  "test_averagepool_2d_default",
  "test_reduce_mean_default_axes_keepdims_example",
].map((name) => ({ name, url: BACKEND_NODE_BASE + name }));

// ONNX TensorProto.DataType -> onnxruntime-web tensor type + the typed-array
// constructor over the raw little-endian bytes. STRING/COMPLEX are unbridged
// (the module's parser rejects them before we get here).
export const ONNX_DTYPE = {
  1: { ort: "float32", Ctor: Float32Array },
  2: { ort: "uint8", Ctor: Uint8Array },
  3: { ort: "int8", Ctor: Int8Array },
  4: { ort: "uint16", Ctor: Uint16Array },
  5: { ort: "int16", Ctor: Int16Array },
  6: { ort: "int32", Ctor: Int32Array },
  7: { ort: "int64", Ctor: typeof BigInt64Array !== "undefined" ? BigInt64Array : null },
  9: { ort: "bool", Ctor: Uint8Array },
  10: { ort: "float16", Ctor: Uint16Array },
  11: { ort: "float64", Ctor: Float64Array },
  12: { ort: "uint32", Ctor: Uint32Array },
  13: { ort: "uint64", Ctor: typeof BigUint64Array !== "undefined" ? BigUint64Array : null },
  16: { ort: "float16", Ctor: Uint16Array }, // BFLOAT16 has no ort type; view raw.
};

// Parse a URL that points at a directory in a GitHub repo into
// { owner, repo, ref, path }. Accepts:
//   https://github.com/<owner>/<repo>/tree/<ref>/<path...>
//   https://github.com/<owner>/<repo>/blob/<ref>/<path...>
//   https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path...>
//   https://api.github.com/repos/<owner>/<repo>/contents/<path...>?ref=<ref>
// Returns null if it is not a recognizable GitHub URL.
export function parseGithubDirUrl(url) {
  let u;
  try {
    u = new URL(String(url).trim());
  } catch {
    return null;
  }
  const seg = u.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const host = u.hostname.toLowerCase();

  if (host === "github.com") {
    // <owner>/<repo>/(tree|blob)/<ref>/<path...>
    if (seg.length >= 4 && (seg[2] === "tree" || seg[2] === "blob")) {
      return {
        owner: seg[0],
        repo: seg[1],
        ref: seg[3],
        path: seg.slice(4).join("/"),
      };
    }
    return null;
  }
  if (host === "raw.githubusercontent.com") {
    // <owner>/<repo>/<ref>/<path...>
    if (seg.length >= 3) {
      return {
        owner: seg[0],
        repo: seg[1],
        ref: seg[2],
        path: seg.slice(3).join("/"),
      };
    }
    return null;
  }
  if (host === "api.github.com") {
    // repos/<owner>/<repo>/contents/<path...>?ref=<ref>
    if (seg.length >= 4 && seg[0] === "repos" && seg[3] === "contents") {
      return {
        owner: seg[1],
        repo: seg[2],
        ref: u.searchParams.get("ref") || "main",
        path: seg.slice(4).join("/"),
      };
    }
    return null;
  }
  return null;
}

// The numeric suffix of an "input_12.pb" / "output_3.pb" name, for ordering the
// tensors by graph position (so input_10 sorts after input_2, not before).
export function tensorIndex(name) {
  const m = /_(\d+)\.pb$/.exec(name);
  return m ? parseInt(m[1], 10) : Number.MAX_SAFE_INTEGER;
}

// Build the GitHub contents-API URL for a repo path.
function contentsApiUrl({ owner, repo, ref, path }) {
  const base = `https://api.github.com/repos/${owner}/${repo}/contents/`;
  const p = path ? path.split("/").map(encodeURIComponent).join("/") : "";
  return `${base}${p}?ref=${encodeURIComponent(ref)}`;
}

async function listDir(loc, onLog) {
  const url = contentsApiUrl(loc);
  const resp = await fetch(url, { headers: { Accept: "application/vnd.github+json" } });
  if (!resp.ok) {
    const hint =
      resp.status === 403
        ? " (GitHub's unauthenticated API rate limit may be exhausted — try again later)"
        : resp.status === 404
          ? " (not found — check the URL points at a test-case directory)"
          : "";
    throw new Error(`listing ${loc.path || "/"} failed: HTTP ${resp.status}${hint}`);
  }
  const json = await resp.json();
  if (!Array.isArray(json)) {
    throw new Error(`${loc.path || "/"} is not a directory`);
  }
  return json; // [{ name, type: "file"|"dir", download_url, ... }]
}

async function fetchBytes(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} failed: HTTP ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
}

// Decode a serialized TensorProto via the page runtime into
// { dtype, name, dims, bytes } (bytes copied out of the wasm heap).
function parseTensor(runtime, pbBytes) {
  // onnxsim_parse_tensor takes the raw bytes as a string arg (embind std::string
  // accepts a JS string or a typed array of bytes); pass the Uint8Array.
  const res = runtime.onnxsim_parse_tensor(pbBytes);
  if (!res || res.error) {
    throw new Error(res ? res.error : "failed to parse tensor");
  }
  return {
    dtype: res.dtype,
    name: res.name,
    dims: Array.from(res.dims, (d) => Number(d)),
    bytes: new Uint8Array(res.data), // copy out of wasm heap
  };
}

// Build a typed array of the tensor's elements from its raw little-endian bytes.
function typedFromTensor(t) {
  const info = ONNX_DTYPE[t.dtype];
  if (!info || !info.Ctor) {
    throw new Error(`unsupported tensor dtype ${t.dtype}`);
  }
  const per = info.Ctor.BYTES_PER_ELEMENT;
  if (t.bytes.byteLength % per !== 0) {
    throw new Error(`tensor byte length ${t.bytes.byteLength} not a multiple of ${per}`);
  }
  return new info.Ctor(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength / per);
}

// Compare an actual onnxruntime output (typed array + dims) against an expected
// parsed tensor within tolerance. Integer/bool/bigint data must match exactly;
// floats use |a-b| <= atol + rtol*|expected|. Returns
// { ok, maxDiff, shapeOk, n, firstBad }.
export function compareTensors(actualData, actualDims, expected, { atol = 1e-4, rtol = 1e-3 } = {}) {
  const exp = typedFromTensor(expected);
  const shapeOk = arraysEqual(actualDims.map(Number), expected.dims.map(Number));
  const n = Math.min(actualData.length, exp.length);
  let maxDiff = 0;
  let firstBad = -1;
  const isBig = typeof actualData[0] === "bigint" || typeof exp[0] === "bigint";
  for (let i = 0; i < n; i++) {
    if (isBig) {
      if (BigInt(actualData[i]) !== BigInt(exp[i])) {
        if (firstBad < 0) firstBad = i;
        maxDiff = Infinity;
      }
      continue;
    }
    const a = Number(actualData[i]);
    const b = Number(exp[i]);
    const diff = Math.abs(a - b);
    if (diff > maxDiff) maxDiff = diff;
    if (diff > atol + rtol * Math.abs(b) && firstBad < 0) firstBad = i;
  }
  const lenOk = actualData.length === exp.length;
  const ok = shapeOk && lenOk && firstBad < 0;
  return { ok, maxDiff, shapeOk: shapeOk && lenOk, n, firstBad };
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

let ortPromise = null;
function loadOrt() {
  if (!ortPromise) {
    ortPromise = import(/* @vite-ignore */ `${ORT_BASE}ort.min.mjs`).then((m) => {
      const ort = m.default ?? m;
      ort.env.wasm.wasmPaths = ORT_BASE;
      return ort;
    });
  }
  return ortPromise;
}

// Group a test-case directory's entries into { modelUrl, dataSets: [{name,url}] }.
function classifyCaseDir(entries) {
  let modelUrl = null;
  const dataSets = [];
  for (const e of entries) {
    if (e.type === "file" && e.name === "model.onnx") modelUrl = e.download_url;
    else if (e.type === "dir" && /^test_data_set_\d+$/.test(e.name)) dataSets.push(e);
  }
  dataSets.sort((a, b) => tensorIndex(a.name + ".pb") - tensorIndex(b.name + ".pb"));
  return { modelUrl, dataSets };
}

// Run one test_data_set: fetch/parse its input_*.pb and output_*.pb, run the
// session, and compare. Returns { name, passed, lines }.
async function runDataSet(ort, runtime, session, loc, setEntry, tol, log) {
  const setLoc = { ...loc, path: `${loc.path}/${setEntry.name}` };
  const entries = await listDir(setLoc, log);
  const inputs = entries.filter((e) => e.type === "file" && /^input_\d+\.pb$/.test(e.name));
  const outputs = entries.filter((e) => e.type === "file" && /^output_\d+\.pb$/.test(e.name));
  inputs.sort((a, b) => tensorIndex(a.name) - tensorIndex(b.name));
  outputs.sort((a, b) => tensorIndex(a.name) - tensorIndex(b.name));

  const feeds = {};
  for (let i = 0; i < inputs.length; i++) {
    const t = parseTensor(runtime, await fetchBytes(inputs[i].download_url));
    const name = session.inputNames[i] ?? t.name;
    feeds[name] = new ort.Tensor(ONNX_DTYPE[t.dtype].ort, typedFromTensor(t), t.dims);
  }
  const results = await session.run(feeds);

  const lines = [];
  let passed = true;
  for (let i = 0; i < outputs.length; i++) {
    const expected = parseTensor(runtime, await fetchBytes(outputs[i].download_url));
    const outName = session.outputNames[i];
    const got = results[outName];
    if (!got) {
      passed = false;
      lines.push(`  output ${i} '${outName}': MISSING from run results`);
      continue;
    }
    const cmp = compareTensors(got.data, got.dims, expected, tol);
    if (cmp.ok) {
      lines.push(`  output ${i} '${outName}': OK (maxDiff ${cmp.maxDiff.toExponential(2)})`);
    } else {
      passed = false;
      const why = !cmp.shapeOk
        ? `shape/length mismatch (got [${Array.from(got.dims).join(",")}], expected [${expected.dims.join(",")}])`
        : `maxDiff ${cmp.maxDiff.toExponential(2)} exceeds tol (atol ${tol.atol}, rtol ${tol.rtol}), first at index ${cmp.firstBad}`;
      lines.push(`  output ${i} '${outName}': FAIL — ${why}`);
    }
  }
  return { name: setEntry.name, passed, lines };
}

async function runBackendTest(url, tol, epPref, convert, log) {
  const loc = parseGithubDirUrl(url);
  if (!loc) {
    throw new Error(
      "could not parse the URL — paste a GitHub directory URL like " +
        "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_relu",
    );
  }
  log(`case: ${loc.owner}/${loc.repo}@${loc.ref} :: ${loc.path}`);
  // Reflect the test-case URL in the address bar so the link is shareable.
  syncInputUrl("backend", url);
  const runtime = await window.__onnxsimRuntimePromise;

  log("listing the test-case directory…");
  const entries = await listDir(loc, log);
  const { modelUrl, dataSets } = classifyCaseDir(entries);
  if (!modelUrl) throw new Error("no model.onnx found in that directory");
  if (!dataSets.length) throw new Error("no test_data_set_* directories found");
  log(`found model.onnx and ${dataSets.length} test data set(s)`);

  log(`fetching model.onnx…`);
  const modelBytes = await fetchBytes(modelUrl);
  const caseName = (loc.path.split("/").filter(Boolean).pop() || "model") + ".onnx";

  // Make the fetched model a first-class converter input, like the text-graph
  // parser: preview it in the "Before" Netron pane and publish it as the
  // inference panel's "original" source.
  if (typeof window.netronShowBefore === "function") {
    try {
      window.netronShowBefore(modelBytes, caseName);
    } catch {
      // Best-effort preview: a Netron hiccup must not fail the test run.
    }
  }
  window.__onnxsimOriginal = { bytes: modelBytes, name: caseName };
  // Optionally run it straight through the Simplify / Optimize path (a copy,
  // since startConversion transfers and detaches the buffer it is given, and we
  // still need modelBytes for the inference session below).
  if (convert && typeof window.__onnxsimStartConversion === "function") {
    log(`sending ${caseName} through the converter…`);
    try {
      window.__onnxsimStartConversion(caseName, modelBytes.slice().buffer);
    } catch (e) {
      log(`could not start conversion: ${e && e.message ? e.message : e}`);
    }
  }

  log(`loading onnxruntime-web ${ORT_VERSION}…`);
  const ort = await loadOrt();
  const providers = epPref === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  let session = null;
  let usedEp = null;
  for (const ep of providers) {
    try {
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      usedEp = ep;
      break;
    } catch (e) {
      log(`execution provider '${ep}' unavailable: ${e && e.message ? e.message : e}`);
    }
  }
  if (!session) throw new Error("could not create an inference session on any provider");
  log(`session ready on '${usedEp}' (inputs: ${session.inputNames.join(", ")}; outputs: ${session.outputNames.join(", ")})`);

  let allPassed = true;
  for (const setEntry of dataSets) {
    log(`running ${setEntry.name}…`);
    const r = await runDataSet(ort, runtime, session, loc, setEntry, tol, log);
    for (const line of r.lines) log(line);
    log(`${setEntry.name}: ${r.passed ? "PASS" : "FAIL"}`);
    allPassed = allPassed && r.passed;
  }
  log(allPassed ? "RESULT: PASS ✅" : "RESULT: FAIL ❌");
  return allPassed;
}

function initBackendPanel() {
  const btn = document.getElementById("backend-run");
  const select = document.getElementById("backend-select");
  const urlInput = document.getElementById("backend-url");
  const epSelect = document.getElementById("backend-ep");
  const atolInput = document.getElementById("backend-atol");
  const rtolInput = document.getElementById("backend-rtol");
  const out = document.getElementById("backend-output");
  const dlBtn = document.getElementById("backend-download-log");
  const convertChk = document.getElementById("backend-convert");
  const statusEl = document.getElementById("backend-status");
  if (!btn) return;

  // Populate the preset picker and mirror a pick into the free-text box, so the
  // two never disagree about what "Run test case" will fetch (like the Hugging
  // Face model dropdown / text box pair).
  if (select) {
    for (const { name, url } of BACKEND_PRESETS) {
      const opt = document.createElement("option");
      opt.value = url;
      opt.textContent = name.replace(/^test_/, "");
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      if (select.value && urlInput) urlInput.value = select.value;
    });
  }

  const log = (msg) => {
    if (!out) return;
    out.value += msg + "\n";
    out.scrollTop = out.scrollHeight;
  };
  if (dlBtn) {
    dlBtn.addEventListener("click", () => downloadText((out && out.value) || "", "onnxsim-backend-test.log"));
  }

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    if (out) out.value = "";
    if (statusEl) statusEl.textContent = "running…";
    try {
      const tol = {
        atol: parseFloat(atolInput ? atolInput.value : "") || 1e-4,
        rtol: parseFloat(rtolInput ? rtolInput.value : "") || 1e-3,
      };
      const epPref = epSelect ? epSelect.value : "wasm";
      const convert = convertChk ? convertChk.checked : false;
      const passed = await runBackendTest(urlInput ? urlInput.value : "", tol, epPref, convert, log);
      if (statusEl) statusEl.textContent = passed ? "PASS ✅" : "FAIL ❌";
    } catch (e) {
      log("ERROR: " + (e && e.message ? e.message : String(e)));
      if (statusEl) statusEl.textContent = "error — see output";
    } finally {
      btn.disabled = false;
    }
  });
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  initBackendPanel();
}

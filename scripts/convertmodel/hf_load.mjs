// Wires the "Load from Hugging Face" panel on the converter page.
//
// It populates the dropdown from the curated regression set, resolves whatever
// the user picked or typed to model bytes (hf_models.mjs), and then hands those
// bytes to the same conversion path an uploaded file uses. index.html publishes
// that entry point as `window.__onnxsimStartConversion` and enables the controls
// below (via `maybeEnableInput`) only once both WASM runtimes are ready.

import {
  loadModelList,
  fetchModelBytes,
  fetchModelBytesXet,
  probeModelSize,
  listOnnxFiles,
  parseRef,
  fileUrl,
  humanBytes,
} from "./hf_models.mjs";
import { openXetCache, makeCachingFetch } from "./xet_cache.mjs";
import { parseInputParams, applyOptionParams, syncInputUrl, syncUrlParam } from "./query_params.mjs";

const select = document.getElementById("hf-model-select");
const refInput = document.getElementById("hf-model-input");
const loadBtn = document.getElementById("hf-load-button");
const statusEl = document.getElementById("hf-status");
const fileInput = document.getElementById("file-input");
const logOutput = document.getElementById("log-output");
const xetToggle = document.getElementById("hf-use-xet");
const xetConcurrencyInput = document.getElementById("hf-xet-concurrency");
const clearCacheBtn = document.getElementById("hf-clear-cache");
const cacheSizeEl = document.getElementById("hf-cache-size");
const fileRow = document.getElementById("hf-file-row");
const fileSelect = document.getElementById("hf-file-select");
const fileNote = document.getElementById("hf-file-note");

// parseRef throws on empty/garbage; this never does, for use in event handlers.
function safeParse(ref) {
  try {
    return parseRef((ref || "").trim());
  } catch {
    return null;
  }
}

// True for a plain Hugging Face repo reference (no explicit file, not a direct
// URL) — the only case where a file picker makes sense.
function isPlainRepo(parsed) {
  return !!(parsed && parsed.repo && !parsed.file && !parsed.url);
}

// Ceiling on parallel connections XetBlob may open for a Xet download (it tunes
// the actual count adaptively from measured throughput). Clamped to a sane
// range; past ~10 the returns fade and the Hub may throttle.
const XET_CONCURRENCY_MAX = 16;
function xetConcurrency() {
  const n = parseInt(xetConcurrencyInput && xetConcurrencyInput.value, 10);
  if (!Number.isFinite(n) || n < 1) return 1;
  return Math.min(n, XET_CONCURRENCY_MAX);
}

// The persistent chunk cache is created lazily on first Xet use.
let xetCache = null;
function getXetCache() {
  if (!xetCache) xetCache = openXetCache();
  return xetCache;
}

// Show how much the Xet chunk cache currently holds, next to the clear button.
// Only touches (and thus lazily opens) the cache when there's a place to show
// it, so a user who never uses Xet doesn't open the IndexedDB store.
async function refreshCacheSize() {
  if (!cacheSizeEl) return;
  try {
    const { bytes, count } = await getXetCache().size();
    cacheSizeEl.textContent = count
      ? `chunk cache: ${humanBytes(bytes)} in ${count} chunk${count === 1 ? "" : "s"}`
      : "chunk cache: empty";
  } catch {
    cacheSizeEl.textContent = "";
  }
}

const log = (msg) => {
  if (!logOutput) return;
  logOutput.value += msg + "\n";
  logOutput.scrollTop = logOutput.scrollHeight;
};

// Curated model id -> download size (bytes), from models.json. Lets a dropdown
// pick show its size instantly, with no network probe.
const curatedSizes = new Map();

// Fill the dropdown with the curated onnxmodelzoo set, showing each model's
// download size (from models.json) right in the option so sizes are visible
// before picking. The list is advisory — the free-text box accepts any repo id
// or .onnx URL regardless.
loadModelList().then((models) => {
  if (!select) return;
  if (!models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(model list unavailable — type a repo id or URL)";
    select.appendChild(opt);
    return;
  }
  for (const { id, size } of models) {
    if (size) curatedSizes.set(id, size);
    const opt = document.createElement("option");
    opt.value = id;
    const shortName = id.replace(/^onnxmodelzoo\//, "");
    opt.textContent = size ? `${shortName} (${humanBytes(size)})` : shortName;
    select.appendChild(opt);
  }
});

// Show the download size for the currently-referenced model *before* fetching,
// so a user can see how big it is before committing to the download. A curated
// model's size is known up front (from models.json) and shown with no network
// call; otherwise the size comes from the Hub API listing (for a repo id) or a
// HEAD request (for an exact file / direct URL). A monotonic token guards
// against a slow probe overwriting the status line for a reference the user has
// since changed.
let probeToken = 0;
let loading = false; // a download is in flight; don't let a probe touch the status
function renderSize(size, name) {
  if (!statusEl) return;
  const label = name ? `${name} — ` : "";
  statusEl.textContent = size
    ? `${label}≈ ${humanBytes(size)} to download`
    : `${label}size unavailable`;
}
async function showSizeFor(ref, knownSize) {
  if (loading) return;
  ref = (ref || "").trim();
  const token = ++probeToken;
  if (!ref) {
    if (statusEl) statusEl.textContent = "";
    return;
  }
  // A curated pick already knows its size — show it without touching the network.
  if (knownSize) {
    renderSize(knownSize, ref.replace(/^onnxmodelzoo\//, ""));
    return;
  }
  if (statusEl) statusEl.textContent = "checking size…";
  try {
    const { size, name } = await probeModelSize(ref);
    if (token !== probeToken) return; // a newer reference superseded this probe
    renderSize(size, name);
  } catch (e) {
    if (token !== probeToken) return;
    if (statusEl) statusEl.textContent = "size unavailable — see console output";
    log(`could not determine model size for "${ref}": ${e && e.message ? e.message : e}`);
  }
}

// --- File selection within a repo -----------------------------------------
// When the reference is a plain repo, list its .onnx files so the user can pick
// which one to convert (the largest is auto-detected and pre-selected). A
// separate token guards against a slow listing landing after the ref changed.
let fileListToken = 0;

function hideFileRow() {
  if (fileRow) fileRow.style.display = "none";
  if (fileSelect) fileSelect.innerHTML = "";
  if (fileNote) fileNote.textContent = "";
}

// Reflect the currently-selected file's size in the status line.
function renderSizeForSelectedFile() {
  if (loading || !fileSelect) return;
  const opt = fileSelect.options[fileSelect.selectedIndex];
  if (!opt) return;
  const size = Number(opt.dataset.size) || 0;
  renderSize(size || null, opt.value.split("/").pop());
}

// List a repo's .onnx files and populate the picker (largest first / selected).
// `knownSize` (from models.json for a curated pick) is shown immediately for
// snappiness while the listing loads.
async function refreshFileList(repo, knownSize) {
  const token = ++fileListToken;
  if (knownSize) renderSize(knownSize, repo.replace(/^onnxmodelzoo\//, ""));
  else if (statusEl) statusEl.textContent = "checking size…";
  if (fileNote) fileNote.textContent = "listing files…";
  if (fileRow) fileRow.style.display = "";
  try {
    const files = await listOnnxFiles(repo);
    if (token !== fileListToken || loading) return;
    if (!files.length) {
      hideFileRow();
      if (statusEl) statusEl.textContent = "no .onnx file found in repo";
      return;
    }
    fileSelect.innerHTML = "";
    files.forEach((f, i) => {
      const opt = document.createElement("option");
      opt.value = f.file;
      opt.dataset.size = f.size || "";
      const sz = f.size ? ` (${humanBytes(f.size)})` : "";
      opt.textContent = `${f.file}${sz}${i === 0 ? " — auto (largest)" : ""}`;
      fileSelect.appendChild(opt);
    });
    fileSelect.selectedIndex = 0; // largest, i.e. the auto-detected file
    if (fileNote) {
      fileNote.textContent =
        files.length > 1
          ? `${files.length} .onnx files; largest auto-detected`
          : "1 .onnx file (auto-detected)";
    }
    if (fileRow) fileRow.style.display = "";
    renderSizeForSelectedFile();
  } catch (e) {
    if (token !== fileListToken) return;
    // Listing failed (CORS, offline, …) — hide the picker and fall back to a
    // plain size probe so the user still sees something.
    hideFileRow();
    showSizeFor(repo);
  }
}

// Handle a committed reference: repos get a file picker + size; a specific file
// or direct URL just gets a size probe (no picker, the file is fixed).
function onRefCommitted(ref, knownSize) {
  const parsed = safeParse(ref);
  if (isPlainRepo(parsed)) {
    refreshFileList(parsed.repo, knownSize);
  } else {
    hideFileRow();
    showSizeFor(ref);
  }
}

// The effective reference to load: when a specific file is chosen for a plain
// repo, resolve it to that file's URL so the download uses the picked file
// instead of the auto-detected largest.
function effectiveRef() {
  const raw = ((refInput && refInput.value) || (select && select.value) || "").trim();
  if (fileRow && fileRow.style.display !== "none" && fileSelect && fileSelect.value) {
    const parsed = safeParse(raw);
    if (isPlainRepo(parsed)) {
      return fileUrl(parsed.repo, fileSelect.value);
    }
  }
  return raw;
}

// Picking from the dropdown mirrors into the text box so the two never disagree
// about what "Load" will fetch, and the user can tweak the id before loading.
// Both a dropdown pick and a committed edit of the text box preview the size and
// (for a repo) the file picker.
if (select && refInput) {
  select.addEventListener("change", () => {
    if (select.value) {
      refInput.value = select.value;
      onRefCommitted(select.value, curatedSizes.get(select.value));
    }
  });
}
// `change` fires on commit (blur / Enter), so partial keystrokes don't each
// trigger a network call; the text box is mirrored from the dropdown
// programmatically, which does not fire `change`, so there's no double request.
if (refInput) {
  refInput.addEventListener("change", () => onRefCommitted(refInput.value));
}
// Choosing a different file updates the previewed size; effectiveRef() picks it
// up at load time.
if (fileSelect) {
  fileSelect.addEventListener("change", renderSizeForSelectedFile);
}

async function doLoad() {
  // Resolve to the picked file when one is chosen, else the raw repo id / URL.
  const ref = effectiveRef();
  if (!ref) {
    log("enter a Hugging Face repo id or .onnx URL, or pick one from the list");
    return;
  }
  if (typeof window.__onnxsimStartConversion !== "function") {
    log("WebAssembly runtime not ready yet — try again in a moment");
    return;
  }

  // Take over the status line from any size preview: mark a load in flight and
  // invalidate outstanding probes so a late size result can't overwrite the
  // download progress below.
  loading = true;
  probeToken += 1;
  loadBtn.disabled = true;
  if (fileInput) fileInput.disabled = true;
  if (statusEl) statusEl.textContent = "loading…";
  // Reflect this input in the address bar so the link is shareable/reproducible
  // (a Hugging Face repo id or a direct .onnx URL). Uploaded local files have no
  // URL and are handled by the file-input path, which does not call this.
  syncInputUrl("model", ref);
  try {
    log(`loading model from Hugging Face: ${ref}`);
    // Live download progress + speed in the status line. Speed is smoothed with
    // an exponential moving average so it doesn't jitter per chunk. The DOM is
    // only rewritten when the whole-percent (or, without a Content-Length, the
    // rounded MB) changes, but timing is sampled on every chunk for accuracy.
    let lastShown = -1;
    let startedAt = 0;
    let lastAt = 0;
    let lastLoaded = 0;
    let emaRate = 0; // bytes/sec
    // The displayed speed defaults to the streamed-bytes rate, which equals real
    // network throughput on the direct-download path. The Xet path overrides it
    // (see below) with a rate measured from actual network bytes, because its
    // streamed bytes include cached/deduplicated chunks that never hit the wire.
    let speedFn = () => emaRate;
    const resetProgress = () => {
      lastShown = -1;
      startedAt = lastAt = performance.now();
      lastLoaded = 0;
      emaRate = 0;
    };
    resetProgress();
    const onProgress = ({ loaded, total }) => {
      if (!statusEl) return;
      // Update the smoothed rate every call, before the display throttle.
      const now = performance.now();
      const dt = (now - lastAt) / 1000;
      if (dt > 0 && loaded > lastLoaded) {
        const inst = (loaded - lastLoaded) / dt;
        emaRate = emaRate === 0 ? inst : emaRate * 0.7 + inst * 0.3;
        lastAt = now;
        lastLoaded = loaded;
      }
      const rate = speedFn();
      const speed = rate > 0 ? ` — ${humanBytes(rate)}/s` : "";

      if (total > 0) {
        const pct = Math.floor((loaded / total) * 100);
        if (pct === lastShown) return;
        lastShown = pct;
        statusEl.textContent =
          `downloading… ${pct}% (${humanBytes(loaded)} / ${humanBytes(total)})${speed}`;
      } else {
        const mb = Math.floor(loaded / (1024 * 1024));
        if (mb === lastShown) return;
        lastShown = mb;
        statusEl.textContent = `downloading… ${humanBytes(loaded)}${speed}`;
      }
    };
    let bytes;
    let name;
    let usedXet = false;
    let netBytes = 0; // bytes actually pulled over the network (Xet path)
    let hitBytes = 0; // bytes served from the chunk cache (Xet path)
    let settleWrites = null; // await the Xet cache's background writes before reading its size
    if (xetToggle && xetToggle.checked) {
      usedXet = true;
      // Experimental Xet path, with the persistent chunk cache. Any failure
      // (non-Xet repo, CORS on the CAS endpoints, …) falls back to the direct
      // download so the toggle can never break loading.
      let hits = 0;
      // Smoothed rate over *network* bytes only, so cache hits / dedup don't
      // inflate the reported speed (the reconstructed-bytes rate would).
      let netLastAt = performance.now();
      let netLastBytes = 0;
      let netEma = 0;
      const onNetwork = (n) => {
        netBytes += n;
        const now = performance.now();
        const dt = (now - netLastAt) / 1000;
        if (dt > 0 && netBytes > netLastBytes) {
          const inst = (netBytes - netLastBytes) / dt;
          netEma = netEma === 0 ? inst : netEma * 0.7 + inst * 0.3;
          netLastAt = now;
          netLastBytes = netBytes;
        }
      };
      speedFn = () => netEma;
      const cachingFetch = makeCachingFetch(
        globalThis.fetch.bind(globalThis),
        getXetCache(),
        { onHit: (_k, n) => { hits += 1; hitBytes += n; }, onNetwork },
      );
      settleWrites = cachingFetch.settled;
      try {
        ({ bytes, name } = await fetchModelBytesXet(ref, {
          onLog: log,
          onProgress,
          fetchImpl: cachingFetch,
          concurrency: xetConcurrency(),
        }));
        if (hits > 0) log(`reused ${hits} cached chunk(s) (${humanBytes(hitBytes)})`);
      } catch (e) {
        log(`Xet path unavailable (${e && e.message ? e.message : e}); falling back to direct download`);
        // Back to the direct path: its streamed bytes are the network bytes, so
        // the default streamed-rate speed is accurate again.
        usedXet = false;
        speedFn = () => emaRate;
        resetProgress();
        ({ bytes, name } = await fetchModelBytes(ref, log, onProgress));
      }
    } else {
      ({ bytes, name } = await fetchModelBytes(ref, log, onProgress));
    }
    // Report the average download speed over the whole transfer. For Xet, the
    // honest figure is over the bytes that actually crossed the network — cached
    // / deduplicated chunks are reconstructed locally, not downloaded.
    const elapsed = (performance.now() - startedAt) / 1000;
    if (usedXet) {
      if (netBytes > 0 && elapsed > 0) {
        const cached = hitBytes > 0 ? ` (+${humanBytes(hitBytes)} from cache)` : "";
        log(`downloaded ${humanBytes(netBytes)} over the network${cached} — ` +
            `average ${humanBytes(netBytes / elapsed)}/s over ${elapsed.toFixed(1)}s`);
      } else if (elapsed > 0) {
        log(`reconstructed entirely from the chunk cache (${humanBytes(hitBytes)}) ` +
            `in ${elapsed.toFixed(1)}s — no network transfer`);
      }
      // The cache writes are backgrounded during download; wait for them so the
      // readout reflects the chunks this download just added.
      if (settleWrites) {
        try {
          await settleWrites();
        } catch {
          // ignore — refresh anyway with whatever landed
        }
      }
      refreshCacheSize();
    } else if (elapsed > 0 && bytes && bytes.length) {
      log(`download average ${humanBytes(bytes.length / elapsed)}/s over ${elapsed.toFixed(1)}s`);
    }
    if (statusEl) statusEl.textContent = "converting…";
    // Publish as the "original" model so the Run inference panel can run it —
    // there is no file-input entry for a Hugging Face download.
    window.__onnxsimOriginal = { bytes, name };
    // Render the "before" Netron pane too: it normally watches the file input,
    // which a Hugging Face download bypasses, so drive it explicitly here.
    // (toArrayBuffer copies the bytes, leaving our Uint8Array intact.)
    if (typeof window.netronShowBefore === "function") {
      window.netronShowBefore(bytes, name);
    }
    // Likewise list the "before" model's dim_params, which also normally come
    // from the file input a Hugging Face download bypasses.
    if (typeof window.dimParamsShowBefore === "function") {
      window.dimParamsShowBefore(bytes, name);
    }
    // Hand a *copy* to the worker: the buffer is transferred (detached) on
    // postMessage, so copying keeps `bytes` intact for the inference panel.
    const copy = bytes.slice();
    window.__onnxsimStartConversion(name, copy.buffer);
    // Leave "converting…" showing; it is cleared when the worker signals the
    // conversion is done (the onnxsim:converted listener below).
  } catch (e) {
    log("Hugging Face load failed: " + (e && e.message ? e.message : String(e)));
    if (statusEl) statusEl.textContent = "failed — see console output";
    // The conversion never started, so re-enable the file picker here (normally
    // the worker's convert-done message does that).
    if (fileInput) fileInput.disabled = false;
  } finally {
    loading = false;
    loadBtn.disabled = false;
  }
}

if (loadBtn) loadBtn.addEventListener("click", doLoad);
if (refInput) {
  refInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      doLoad();
    }
  });
}

if (clearCacheBtn) {
  clearCacheBtn.addEventListener("click", async () => {
    try {
      await getXetCache().clear();
      log("Xet chunk cache cleared");
    } catch (e) {
      log("could not clear Xet cache: " + (e && e.message ? e.message : e));
    }
    refreshCacheSize();
  });
}

// Show the current cache size once the user opts into Xet (checking the box),
// so the readout appears without forcing the IndexedDB store open on every page
// load. If the box starts checked (browser-restored state), show it up front.
if (xetToggle) {
  xetToggle.addEventListener("change", () => {
    if (xetToggle.checked) refreshCacheSize();
  });
  if (xetToggle.checked) refreshCacheSize();
}

// index.html fires this once a conversion finishes; clear the "converting…"
// status so the panel returns to idle (harmless for file-upload conversions,
// where the status line is already empty).
window.addEventListener("onnxsim:converted", () => {
  if (statusEl) statusEl.textContent = "";
});

// --- Query-parameter input -------------------------------------------------
// Let a shareable link drive the input and conversion options, e.g.
//   ?model=onnxmodelzoo/resnet18d_Opset18
//   ?model=https://…/foo.onnx&optimizer=optimize&cf=0
//   ?graph=<onnx text>&optimizer=fold_constant   (parse a text graph, then run)
//   ?backend=<onnx backend test-case URL>        (prefill the backend panel)
//   ?…&autoload=0                                 (prefill but don't auto-run)
// The model reference reuses the Hugging Face loader's repo-id / .onnx-URL path,
// so nothing is uploaded and the conversion is the same one an upload drives.
(function initFromQueryParams() {
  let params;
  try {
    params = parseInputParams(window.location.search);
  } catch {
    return;
  }
  // Prefill the conversion option controls (including the processor / mode radio
  // from ?optimizer=) so both an auto-run and a later manual run honor the link.
  try {
    const changed = applyOptionParams(params, document);
    if (changed.length) log(`applied options from URL: ${changed.join(", ")}`);
  } catch {
    // Best-effort: a missing control just leaves that option at its default.
  }
  // Reflect the conversion processor in the URL whenever the mode radio changes,
  // so the current processor is always part of the shareable link.
  for (const radio of document.querySelectorAll('input[name="optimizer"]')) {
    radio.addEventListener("change", () => {
      if (radio.checked) syncUrlParam("optimizer", radio.value);
    });
  }
  // Prefill the backend-test panel's URL (it reads the input when Run is clicked).
  if (params.backend) {
    const el = document.getElementById("backend-url");
    if (el) el.value = params.backend;
  }
  // Run one of the input sources once both WASM runtimes are ready (index.html
  // dispatches "onnxsim:ready" from maybeEnableInput), so the conversion entry
  // point and the page runtime both exist; if it already fired, go now.
  const whenReady = (fn) => {
    if (window.__onnxsimReady) fn();
    else window.addEventListener("onnxsim:ready", fn, { once: true });
  };

  // A text graph: prefill the parse box and, on autoload, click Parse (which
  // parses and — with "convert after parsing" on — runs the selected mode).
  if (params.graph) {
    const ta = document.getElementById("parse-graph-text");
    if (ta) ta.value = params.graph;
    if (params.autoload) {
      whenReady(() => {
        const b = document.getElementById("parse-graph-button");
        if (b) b.click();
      });
    }
    return;
  }

  if (!params.model) return;
  if (refInput) refInput.value = params.model;
  whenReady(() => {
    if (params.autoload) {
      log(`loading input model from URL parameter: ${params.model}`);
      doLoad();
    } else {
      // Just preview size / list files so the user can review before loading.
      onRefCommitted(params.model);
    }
  });
})();

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
  humanBytes,
} from "./hf_models.mjs";
import { openXetCache, makeCachingFetch } from "./xet_cache.mjs";

const select = document.getElementById("hf-model-select");
const refInput = document.getElementById("hf-model-input");
const loadBtn = document.getElementById("hf-load-button");
const statusEl = document.getElementById("hf-status");
const fileInput = document.getElementById("file-input");
const logOutput = document.getElementById("log-output");
const xetToggle = document.getElementById("hf-use-xet");
const clearCacheBtn = document.getElementById("hf-clear-cache");
const cacheSizeEl = document.getElementById("hf-cache-size");

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

// Picking from the dropdown mirrors into the text box so the two never disagree
// about what "Load" will fetch, and the user can tweak the id before loading.
// Both a dropdown pick and a committed edit of the text box preview the size.
if (select && refInput) {
  select.addEventListener("change", () => {
    if (select.value) {
      refInput.value = select.value;
      showSizeFor(select.value, curatedSizes.get(select.value));
    }
  });
}
// `change` fires on commit (blur / Enter), so partial keystrokes don't each
// trigger a network probe; the text box is mirrored from the dropdown
// programmatically, which does not fire `change`, so there's no double probe.
if (refInput) {
  refInput.addEventListener("change", () => showSizeFor(refInput.value));
}

async function doLoad() {
  const ref = ((refInput && refInput.value) || (select && select.value) || "").trim();
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
      try {
        ({ bytes, name } = await fetchModelBytesXet(ref, {
          onLog: log,
          onProgress,
          fetchImpl: cachingFetch,
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

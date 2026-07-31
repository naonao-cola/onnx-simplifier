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

// The persistent chunk cache is created lazily on first Xet use.
let xetCache = null;
function getXetCache() {
  if (!xetCache) xetCache = openXetCache();
  return xetCache;
}

const log = (msg) => {
  if (!logOutput) return;
  logOutput.value += msg + "\n";
  logOutput.scrollTop = logOutput.scrollHeight;
};

// Fill the dropdown with the curated onnxmodelzoo set. The list is advisory —
// the free-text box accepts any repo id or .onnx URL regardless.
loadModelList().then((ids) => {
  if (!select) return;
  if (!ids.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(model list unavailable — type a repo id or URL)";
    select.appendChild(opt);
    return;
  }
  for (const id of ids) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id.replace(/^onnxmodelzoo\//, "");
    select.appendChild(opt);
  }
});

// Picking from the dropdown mirrors into the text box so the two never disagree
// about what "Load" will fetch, and the user can tweak the id before loading.
if (select && refInput) {
  select.addEventListener("change", () => {
    if (select.value) refInput.value = select.value;
  });
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
      const speed = emaRate > 0 ? ` — ${humanBytes(emaRate)}/s` : "";

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
    if (xetToggle && xetToggle.checked) {
      // Experimental Xet path, with the persistent chunk cache. Any failure
      // (non-Xet repo, CORS on the CAS endpoints, …) falls back to the direct
      // download so the toggle can never break loading.
      let hits = 0;
      let hitBytes = 0;
      const cachingFetch = makeCachingFetch(
        globalThis.fetch.bind(globalThis),
        getXetCache(),
        { onHit: (_k, n) => { hits += 1; hitBytes += n; } },
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
        resetProgress();
        ({ bytes, name } = await fetchModelBytes(ref, log, onProgress));
      }
    } else {
      ({ bytes, name } = await fetchModelBytes(ref, log, onProgress));
    }
    // Report the average download speed over the whole transfer.
    const elapsed = (performance.now() - startedAt) / 1000;
    if (elapsed > 0 && bytes && bytes.length) {
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
  });
}

// index.html fires this once a conversion finishes; clear the "converting…"
// status so the panel returns to idle (harmless for file-upload conversions,
// where the status line is already empty).
window.addEventListener("onnxsim:converted", () => {
  if (statusEl) statusEl.textContent = "";
});

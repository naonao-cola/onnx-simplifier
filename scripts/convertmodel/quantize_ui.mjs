// Browser glue for the "Quantize a model" panel on the converter page.
//
// Runs one of onnxsim's quantization methods on the currently loaded model
// (see resolveOriginalModelBytes in inference_browser.mjs), entirely in the
// browser -- nothing is uploaded:
//   - Dynamic / Ternary / Weight-only / Weight-only INT4 need no calibration
//     data, so it's a single WASM call.
//   - Static (QDQ) / QOperator (QLinearMatMul) are calibration-based: the
//     candidate tensor names come from the WASM module
//     (onnxsim_list_quantizable_activations / _list_qoperator_quantizable_outputs),
//     and their (min, max) ranges are measured by actually running the model
//     over synthetic random inputs through onnxruntime-web
//     (quantize_calibration.mjs), mirroring onnxsim.calibration.calibrate().
//
// Like debug_tools.mjs's "parse a text graph" panel, the quantization
// rewrites themselves need no model executor, so they run on this page's own
// WASM runtime directly (no worker round trip, unlike Simplify/Optimize) --
// only the Static/QOperator calibration step needs onnxruntime-web, loaded on
// demand the same way the "Run inference" panel does.

import { downloadBytes } from "./download.mjs";
import { resolveOriginalModelBytes } from "./inference_browser.mjs";
import { calibrateRanges } from "./quantize_calibration.mjs";
import { computeQuantizationQuality, renderQuantizationQuality } from "./quantize_metrics.mjs";

// Resolve this page's WASM runtime (published by index.html's inline loader).
function getRuntime() {
  return window.__onnxsimRuntimePromise;
}

// Methods that quantize straight from a single WASM call -- the weight is
// quantized from its own static values and the activation either isn't
// touched (weight-only) or is quantized dynamically at inference time
// (dynamic/ternary), so no calibration data is needed.
const NO_CALIBRATION_METHODS = {
  dynamic: "onnxsim_quantize_dynamic",
  ternary: "onnxsim_quantize_ternary",
  weight_only: "onnxsim_quantize_weight_only",
  weight_only_int4: "onnxsim_quantize_weight_only_int4",
};

function toArray(vec) {
  const out = new Array(vec.size());
  for (let i = 0; i < out.length; i++) out[i] = vec.get(i);
  return out;
}

function initQuantizePanel() {
  const btn = document.getElementById("quantize-button");
  if (!btn) return;
  const fileInput = document.getElementById("file-input");
  const statusEl = document.getElementById("quantize-status");
  const dlBtn = document.getElementById("quantize-download");
  const samplesEl = document.getElementById("quantize-samples");
  const metricsToggleEl = document.getElementById("quantize-metrics-toggle");
  const metricsEl = document.getElementById("quantize-metrics");

  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };

  let lastBytes = null;

  btn.addEventListener("click", async () => {
    const methodEl = document.querySelector('input[name="quantize-method"]:checked');
    const method = methodEl ? methodEl.value : "dynamic";
    btn.disabled = true;
    if (dlBtn) dlBtn.style.display = "none";
    if (metricsEl) {
      metricsEl.style.display = "none";
      metricsEl.innerHTML = "";
    }
    try {
      setStatus("loading model…");
      const { bytes, name } = await resolveOriginalModelBytes(fileInput);
      // A fresh ArrayBuffer copy: the WASM calls below read it (never
      // transferred/detached), and it's reused across several calls when
      // calibrating.
      const modelBuf = bytes.slice().buffer;
      const runtime = await getRuntime();

      let resultBuf;
      if (method in NO_CALIBRATION_METHODS) {
        setStatus(`quantizing (${method})…`);
        resultBuf = runtime[NO_CALIBRATION_METHODS[method]](modelBuf);
      } else if (method === "static" || method === "qoperator") {
        setStatus("finding quantizable tensors…");
        let names = toArray(runtime.onnxsim_list_quantizable_activations(modelBuf));
        if (method === "qoperator") {
          const outNames = toArray(runtime.onnxsim_list_qoperator_quantizable_outputs(modelBuf));
          names = Array.from(new Set([...names, ...outNames]));
        }
        if (names.length === 0) {
          setStatus(
            "no quantizable MatMul/Gemm" + (method === "static" ? "/Conv" : "") +
              " (with a constant float32 weight) found in this model.",
          );
          return;
        }
        const numSamples = Math.max(1, parseInt((samplesEl && samplesEl.value) || "8", 10) || 8);
        setStatus(`calibrating ${names.length} tensor(s) over ${numSamples} random sample(s)…`);
        const { names: calNames, flat } = await calibrateRanges(runtime, modelBuf, names, numSamples, setStatus);
        if (calNames.length === 0) {
          setStatus("calibration produced no ranges (the model may have no runnable inputs).");
          return;
        }
        setStatus(`quantizing (${method})…`);
        const fn = method === "static" ? "onnxsim_quantize_static" : "onnxsim_quantize_qoperator";
        resultBuf = runtime[fn](modelBuf, calNames, flat);
      } else {
        setStatus(`unknown quantize method: ${method}`);
        return;
      }

      if (!resultBuf) {
        setStatus("quantization failed (see console for details).");
        return;
      }
      // Copy out of the wasm heap and grab a base64 copy for Netron/inference
      // before any other wasm call can invalidate the view.
      const outBytes = new Uint8Array(resultBuf).slice();
      const dataUrl = "data:application/octet-stream;base64," + resultBuf.toBase64();
      lastBytes = outBytes;
      const outName = name.replace(/\.onnx$/i, "") + `.quant_${method}.onnx`;
      setStatus(`done (${outBytes.length.toLocaleString()} bytes).`);
      if (dlBtn) {
        dlBtn.style.display = "";
        dlBtn.onclick = () => downloadBytes(lastBytes, outName);
      }
      // Show the result in the "After" Netron pane and publish it as the
      // inference panel's "converted" source, same as a Simplify/Optimize run.
      if (window.netronShowAfter) window.netronShowAfter(dataUrl, outName);
      window.__onnxsimConverted = { bytes: outBytes, name: outName };

      // Best-effort result-quality report: runs float vs. quantized on the
      // same random input through onnxruntime-web. A failure here (e.g. no
      // usable execution provider, or an input shape onnxruntime-web can't
      // auto-fill) shouldn't take away the quantized model itself, which is
      // already valid and downloadable at this point -- so it only appends a
      // note to the status line rather than replacing it or throwing.
      if (!metricsToggleEl || metricsToggleEl.checked) {
        try {
          setStatus(`done (${outBytes.length.toLocaleString()} bytes). measuring result quality…`);
          const quality = await computeQuantizationQuality(bytes, outBytes, setStatus);
          if (metricsEl) {
            metricsEl.innerHTML = renderQuantizationQuality(quality);
            metricsEl.style.display = "";
          }
          setStatus(`done (${outBytes.length.toLocaleString()} bytes).`);
        } catch (metricsErr) {
          setStatus(
            `done (${outBytes.length.toLocaleString()} bytes). ` +
              "result-quality check failed: " +
              (metricsErr && metricsErr.message ? metricsErr.message : metricsErr),
          );
        }
      }
    } catch (err) {
      setStatus("quantize error: " + (err && err.message ? err.message : err));
    } finally {
      btn.disabled = false;
    }
  });
}

initQuantizePanel();

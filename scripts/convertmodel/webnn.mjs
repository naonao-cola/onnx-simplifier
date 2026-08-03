// WebNN helpers for the converter page's "Run inference" panel.
//
// WebNN (the W3C Web Neural Network API, https://www.w3.org/TR/webnn/) lets a
// page run a neural network on the platform's own ML stack — DirectML on
// Windows, Core ML on macOS, and NN APIs elsewhere — reaching the GPU and,
// where present, a dedicated NPU. onnxruntime-web exposes it as an execution
// provider (EP), so the inference panel can offer it alongside WebGPU and WASM.
//
// As of onnxruntime-web 1.27 / early 2026 the WebNN backend is still an
// *experimental* feature and the browser API ships behind a flag (Chrome/Edge:
// chrome://flags/#web-machine-learning-neural-network), with the best coverage
// on Windows. So we never assume it is there: the panel probes navigator.ml at
// load time to show a live status, always lists WASM as a fallback provider,
// and lets onnxruntime-web transparently fall back when a WebNN context or a
// given device type cannot be created.
//
// This module holds the pure, browser-free pieces (provider construction,
// labelling, and the navigator.ml probe accepting an injected navigator) so
// they can be unit-tested under Node; the DOM wiring lives in
// inference_browser.mjs.

// Device types WebNN's MLContext accepts, most-capable first. "gpu" and "npu"
// also take a powerPreference; "cpu" does not.
export const WEBNN_DEVICE_TYPES = ["gpu", "npu", "cpu"];

// Build the onnxruntime-web execution-provider entry for a WebNN device. ORT
// accepts EPs either as a bare name string or as an options object; WebNN needs
// the object form to carry deviceType (and, for accelerator devices, a
// powerPreference).
export function webnnProvider(deviceType) {
  if (!WEBNN_DEVICE_TYPES.includes(deviceType)) {
    throw new Error(`unknown WebNN device type '${deviceType}'`);
  }
  const ep = { name: "webnn", deviceType };
  // powerPreference is meaningful for the accelerator devices; CPU ignores it.
  if (deviceType === "gpu" || deviceType === "npu") {
    ep.powerPreference = "default";
  }
  return ep;
}

// A short human-readable name for a provider entry (string or options object),
// used in the panel log and inference-core's returned `ep` field. Strings pass
// through unchanged so the existing "wasm"/"webgpu" labels are preserved; a
// WebNN object becomes e.g. "webnn:gpu".
export function providerLabel(ep) {
  if (typeof ep === "string") return ep;
  if (ep && ep.name === "webnn") {
    return ep.deviceType ? `webnn:${ep.deviceType}` : "webnn";
  }
  if (ep && ep.name) return ep.name;
  return String(ep);
}

// Map an inference-panel EP dropdown value to the ordered provider list to try
// and whether the WebNN-capable onnxruntime-web bundle is required. Every WebNN
// (and WebGPU) choice ends in "wasm" so a session still gets created when the
// accelerated provider is unavailable.
export function providersForEp(epValue) {
  switch (epValue) {
    case "webgpu":
      return { providers: ["webgpu", "wasm"], needWebnn: false };
    case "webnn-gpu":
      return { providers: [webnnProvider("gpu"), "wasm"], needWebnn: true };
    case "webnn-npu":
      return { providers: [webnnProvider("npu"), "wasm"], needWebnn: true };
    case "webnn-cpu":
      return { providers: [webnnProvider("cpu"), "wasm"], needWebnn: true };
    case "wasm":
    default:
      return { providers: ["wasm"], needWebnn: false };
  }
}

// True when the EP dropdown value names a WebNN provider.
export function isWebnnEp(epValue) {
  return typeof epValue === "string" && epValue.startsWith("webnn-");
}

// Probe the WebNN API on an injected navigator (defaults to the page's).
// Returns { apiPresent, supported, devices: {gpu,npu,cpu}, errors: {…}, message }:
// apiPresent reflects navigator.ml being exposed at all; `devices[type]` is true
// when an MLContext for that device type could actually be created; `errors[type]`
// carries the failure message for a device type that couldn't be created; and
// `supported` is true when at least one device type worked. Never throws — any
// failure is reported as unsupported so a caller can render it directly.
//
// The per-device MLContext failures are also logged to `con` (defaults to the
// page's console) so the actual WebNN error — which the one-line panel status
// can only summarize — is visible in the browser console for debugging. `con` is
// injectable so the probe stays unit-testable under Node without touching the
// global console.
export async function detectWebnn(
  nav = typeof navigator !== "undefined" ? navigator : undefined,
  con = typeof console !== "undefined" ? console : undefined,
) {
  const ml = nav && nav.ml;
  if (!ml || typeof ml.createContext !== "function") {
    return {
      apiPresent: false,
      supported: false,
      devices: {},
      errors: {},
      message:
        "WebNN API (navigator.ml) is not available in this browser. Try a " +
        "recent Chrome/Edge with the WebNN flag enabled " +
        "(chrome://flags/#web-machine-learning-neural-network).",
    };
  }

  const devices = {};
  const errors = {};
  for (const deviceType of WEBNN_DEVICE_TYPES) {
    try {
      const ctx = await ml.createContext({ deviceType });
      devices[deviceType] = !!ctx;
      // Best-effort release; not all implementations expose destroy().
      if (ctx && typeof ctx.destroy === "function") {
        try {
          ctx.destroy();
        } catch {
          /* ignore */
        }
      }
    } catch (err) {
      devices[deviceType] = false;
      errors[deviceType] = err && err.message ? err.message : String(err);
      // Surface the real WebNN failure to the console — the panel status only
      // reports that a device couldn't be created, not why.
      if (con && typeof con.error === "function") {
        con.error(
          `WebNN: could not create an MLContext for deviceType '${deviceType}':`,
          err,
        );
      }
    }
  }

  const usable = WEBNN_DEVICE_TYPES.filter((d) => devices[d]);
  return {
    apiPresent: true,
    supported: usable.length > 0,
    devices,
    errors,
    message: usable.length
      ? `WebNN available — device types: ${usable.join(", ")}.`
      : "navigator.ml exists, but no WebNN device (gpu/npu/cpu) could be " +
        "created here. The API may be present without a working backend.",
  };
}

// Format a detectWebnn() report as a one-line status string for the panel.
export function formatWebnnStatus(report) {
  const prefix = report.supported ? "✅" : report.apiPresent ? "⚠️" : "❌";
  return `WebNN status: ${prefix} ${report.message}`;
}

// Mirror onnxruntime-web's own diagnostics into the inference panel's UI log.
//
// Some onnxruntime-web errors never reach the panel's try/catch, so they only
// appear in the browser devtools console — not the on-page "console" textarea:
//
//   * EP-assignment notes ("Some nodes were not assigned to the preferred
//     execution providers …", "VerifyEachNodeIsAssignedToAnEp") are printed
//     straight to console.error/console.warn from inside the wasm module.
//   * WebNN backend failures ("WebNN backend does not support data type:
//     int64") are thrown on an internal ORT promise that isn't chained to the
//     awaited session.run(), so they surface as an *uncaught* promise rejection
//     rather than rejecting the run the panel awaits.
//
// captureOrtDiagnostics() wraps console.warn/console.error and listens for
// unhandledrejection while a run is active, forwarding the onnxruntime/WebNN
// ones to a provided sink (the panel log). The console output is preserved, so
// devtools still shows everything it did before. The pure pieces take injected
// console/target so they can be unit-tested under Node.

// onnxruntime-web tags these messages with the library or backend name; match on
// that so unrelated page logging (or another library's console noise) is not
// mirrored into the inference log.
const ORT_DIAGNOSTIC_RE = /onnxruntime|webnn/i;

// True when a flattened message looks like one of onnxruntime-web's own
// diagnostics we want mirrored into the panel log.
export function isOrtDiagnostic(text) {
  return typeof text === "string" && ORT_DIAGNOSTIC_RE.test(text);
}

// Normalize one console argument / rejection reason to a string. Errors show
// their message, plain strings pass through, everything else is JSON- (or
// String-) stringified so objects still contribute something matchable.
export function valueToText(v) {
  if (v instanceof Error) return v.message || String(v);
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

// Flatten a console argument list to a single space-joined string.
export function formatLogArgs(args) {
  return (args || []).map(valueToText).join(" ");
}

// Install console.warn/console.error wrappers and an unhandledrejection listener
// that forward onnxruntime-web's own diagnostics to `log`. Returns a restore()
// that removes every hook; call it in a finally so the global console/handlers
// are always put back. `con`/`target` are injectable for testing (default to the
// page's console and window).
export function captureOrtDiagnostics({
  log,
  con = typeof console !== "undefined" ? console : undefined,
  target = typeof window !== "undefined" ? window : undefined,
} = {}) {
  const restores = [];

  if (con) {
    for (const level of ["warn", "error"]) {
      const original = con[level];
      if (typeof original !== "function") continue;
      con[level] = (...args) => {
        // Preserve devtools output unchanged.
        original.apply(con, args);
        const text = formatLogArgs(args);
        if (isOrtDiagnostic(text)) log(`onnxruntime-web ${level}: ${text}`);
      };
      restores.push(() => {
        con[level] = original;
      });
    }
  }

  if (target && typeof target.addEventListener === "function") {
    const onRejection = (event) => {
      const text = valueToText(event && event.reason);
      if (isOrtDiagnostic(text)) {
        log(`onnxruntime-web unhandled rejection: ${text}`);
      }
    };
    target.addEventListener("unhandledrejection", onRejection);
    restores.push(() =>
      target.removeEventListener("unhandledrejection", onRejection),
    );
  }

  return () => {
    for (const restore of restores) restore();
    restores.length = 0;
  };
}

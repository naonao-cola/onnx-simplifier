importScripts("./onnxsim.js");

// onnxruntime-web CDN location, kept in sync with inference_browser.mjs. Only
// used by the ORT-web build of the module (see below).
const ORT_VERSION = "1.27.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

// Turn a low-level WASM out-of-memory abort into an actionable explanation.
// When a model needs more heap than the module can address, Emscripten aborts
// with "Cannot enlarge memory, requested N bytes, but the limit is M bytes!"
// (surfaced via printErr) and throws a RuntimeError from the conversion call.
// Both are accurate but say nothing about *why* or *what to do*, so translate
// them into a message the user can act on. The module caps its heap at 4 GiB,
// the hard addressing limit of a wasm32 build (see MAXIMUM_MEMORY in
// CMakeLists.txt); simplification needs several times the model's size on top
// of the model itself, so large models can exhaust it.
function isOutOfMemory(text) {
    return /Cannot enlarge memory|out of memory|Aborted|enlarge|allocat/i.test(
        String(text || ""));
}

function memoryLimitMessage(text) {
    const gib = (n) => (n / (1024 ** 3)).toFixed(2) + " GiB";
    const m = /requested (\d+) bytes, but the limit is (\d+) bytes/.exec(
        String(text || ""));
    const limit = m ? gib(Number(m[2])) : "4 GiB";
    const needed = m ? ` (it needed about ${gib(Number(m[1]))})` : "";
    return [
        `This model is too large to convert in the browser${needed}.`,
        `The WebAssembly build can address at most ${limit} of memory, and`,
        `simplification (shape inference + constant folding) needs several times`,
        `the model's size on top of the model itself.`,
        ``,
        `Things to try:`,
        `  - run onnxsim locally, where it is not limited to ${limit}:`,
        `      pip install onnxsim && onnxsim input.onnx output.onnx`,
        `  - turn off "constant folding" (or lower the tensor-size threshold),`,
        `    then convert again`,
        `  - reduce the model size first (e.g. split it or externalize weights)`,
    ].join("\n");
}

// For the ORT-web build (onnxsim compiled with ONNXSIM_WASM_ORT_WEB): load
// onnxruntime-web and register the runner JsModelExecutor calls. In the default
// built-in-ORT build onnxsim_needs_ort_web() is false and this is a no-op, so
// the worker keeps behaving exactly as before.
async function setupOrtWebIfNeeded(runtime) {
    if (!(typeof runtime.onnxsim_needs_ort_web === "function" &&
          runtime.onnxsim_needs_ort_web())) {
        return;
    }
    const [ortMod, { makeOrtRunner }] = await Promise.all([
        import(/* @vite-ignore */ `${ORT_BASE}ort.min.mjs`),
        import("./ort_executor.mjs"),
    ]);
    const ort = ortMod.default ?? ortMod;
    // Pull the matching wasm binaries from the same CDN directory.
    ort.env.wasm.wasmPaths = ORT_BASE;
    // JsModelExecutor::Run reaches this via val::module_property("onnxsimOrtWebRun").
    runtime.onnxsimOrtWebRun = makeOrtRunner(ort);
}

create_onnxsim({
    preRun: [(runtime) => {
        runtime.ENV.LOG_THRESHOLD = "-1";
    }],
    print: (str) => {
        console.log("stdout:", str);
        postMessage(["stdout", str]);
    },
    printErr: (str) => {
        console.error("stderr:", [str]);
        postMessage(["stderr", str]);
        // Augment the raw Emscripten OOM abort line with a human-readable hint.
        if (typeof str === "string" && str.includes("Cannot enlarge memory")) {
            postMessage(["stderr", memoryLimitMessage(str)]);
        }
    },
}).then(async (runtime) => {
    // Wire up onnxruntime-web before announcing readiness, so the first
    // conversion already has a runner registered (only matters in the ORT-web
    // build; a no-op otherwise).
    try {
        await setupOrtWebIfNeeded(runtime);
    } catch (err) {
        postMessage(["stderr", "failed to load onnxruntime-web: " + err]);
        return;
    }
    // Tell the page the WASM runtime is initialized so it can enable the
    // "Choose file" picker. Registering the message listener below only
    // happens now, so any file posted earlier would be dropped.
    postMessage(["ready"]);

    // The single-feature debugging passes (the "Run a single feature" panel):
    // each takes a model and returns a transformed model, without the full
    // fixed-point Simplify. They post back ["debug-done", type, dataUrl, error]
    // so the debug module can tell its responses apart from a conversion.
    const DEBUG_TYPES = new Set(["infer_shapes", "data_propagation", "fold_constant"]);
    async function handleDebug(e) {
        const type = e.data[0];
        const buf = e.data[1];
        try {
            let model = null;
            if (type === "infer_shapes") {
                model = runtime.onnxsim_infer_shapes(buf);
            } else if (type === "data_propagation") {
                model = runtime.onnxsim_data_propagation(buf);
            } else if (type === "fold_constant") {
                // Constant folding runs through the executor; in the ORT-web
                // build it is Asyncified and returns a Promise to await.
                let r = runtime.onnxsim_fold_constant(buf, e.data[2]);
                if (r && typeof r.then === "function") r = await r;
                model = r;
            }
            if (!model) {
                postMessage(["debug-done", type, null,
                    type + " produced no model (see the log above)"]);
                return;
            }
            const data_url = "data:application/octet-stream;base64," + model.toBase64();
            postMessage(["debug-done", type, data_url, null]);
        } catch (err) {
            const detail = String((err && err.message) || err);
            if (isOutOfMemory(detail)) {
                postMessage(["stderr", memoryLimitMessage(detail)]);
            }
            postMessage(["debug-done", type, null, detail]);
        }
    }

    addEventListener("message", async (e) => {
        console.log(e.data);
        if (DEBUG_TYPES.has(e.data[0])) {
            await handleDebug(e);
            return;
        }
        const buf = e.data[1];
        // `model` is the converted model bytes (a Uint8Array view); `trace` is
        // the onnxsim profiling trace JSON for "simplify" when profiling was
        // requested, otherwise an empty string.
        let model = null;
        let trace = "";
        // A model that outgrows the wasm heap makes Emscripten abort mid-run and
        // throw a RuntimeError out of the conversion call. Catch it here so the
        // user gets an explanation (see memoryLimitMessage) instead of the worker
        // dying with an opaque, unhandled error.
        try {
        switch (e.data[0]) {
            case "simplify": {
                // Simplify returns { model, trace } so the profiling trace can
                // ride back alongside the converted model. In the ORT-web build
                // onnxsimplify_export is Asyncified and returns a Promise, so
                // await when needed; in the built-in-ORT build it returns the
                // object synchronously and the await is a harmless pass-through.
                let result = runtime.onnxsimplify_export(
                    buf,
                    e.data[2], // skip optimizers
                    e.data[3], // constant folding
                    e.data[4], // shape inference
                    e.data[5], // tensor size threshold
                    e.data[6], // target opset version (<= 0 means keep)
                    e.data[7], // profile (emit a Chrome trace)
                    e.data[8], // annotate model info (MACs/FLOPs) into metadata_props
                );
                if (result && typeof result.then === "function") {
                    result = await result;
                }
                if (result) {
                    model = result.model;
                    trace = result.trace || "";
                }
                break;
            }
            case "optimize":
                model = runtime.onnxoptimizer_optimize(
                    buf,
                    e.data[2], // target optimizers
                    e.data[8], // annotate model info (MACs/FLOPs) into metadata_props
                );
                break;
            case "optimize_fixed":
                model = runtime.onnxoptimizer_optimize_fixed(
                    buf,
                    e.data[2], // target optimizers
                    e.data[8], // annotate model info (MACs/FLOPs) into metadata_props
                );
                break;
            default:
                postMessage(["stderr", "unknown conversion type: " + e.data[0]]);
                return;
        }
        } catch (err) {
            const detail = String((err && err.message) || err);
            if (isOutOfMemory(detail)) {
                // The raw "Cannot enlarge memory" line (if any) already went out
                // via printErr; this adds the actionable explanation.
                postMessage(["stderr", memoryLimitMessage(detail)]);
            } else {
                postMessage(["stderr", e.data[0] + " failed: " + detail]);
            }
            return;
        }
        if (!model) {
            postMessage(["stderr", e.data[0] + " failed!"]);
            return;
        }
        console.log("to data url start")
        const data_url = "data:application/octet-stream;base64," + model.toBase64();
        console.log("to data url end")
        // When "annotate model info" is on, also bake the MAC/FLOP metrics into
        // the *original* uploaded model so the "Run inference" panel can report
        // its throughput too — letting the user compare original vs converted
        // inference speed. Annotation only adds metadata_props, so the bytes run
        // identically. Best-effort: a failure here just leaves the original
        // un-annotated (the panel falls back to the raw upload).
        let original_data_url = "";
        if (e.data[8]) {
            try {
                const annotated = runtime.onnxsim_annotate_model_info(buf);
                if (annotated) {
                    original_data_url =
                        "data:application/octet-stream;base64," + annotated.toBase64();
                }
            } catch (err) {
                postMessage(["stderr", "annotate original model failed: " + err]);
            }
        }
        postMessage(["convert-done", data_url, trace, original_data_url]);
    });
});

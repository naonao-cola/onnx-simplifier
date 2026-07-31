importScripts("./onnxsim.js");

// onnxruntime-web CDN location, kept in sync with inference_browser.mjs. Only
// used by the ORT-web build of the module (see below).
const ORT_VERSION = "1.27.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

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
    addEventListener("message", async (e) => {
        console.log(e.data);
        const buf = e.data[1];
        // `model` is the converted model bytes (a Uint8Array view); `trace` is
        // the onnxsim profiling trace JSON for "simplify" when profiling was
        // requested, otherwise an empty string.
        let model = null;
        let trace = "";
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

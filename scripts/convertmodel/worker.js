importScripts("./onnxsim.js");

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
}).then((runtime) => {
    // Tell the page the WASM runtime is initialized so it can enable the
    // "Choose file" picker. Registering the message listener below only
    // happens now, so any file posted earlier would be dropped.
    postMessage(["ready"]);
    addEventListener("message", (e) => {
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
                // ride back alongside the converted model.
                const result = runtime.onnxsimplify_export(
                    buf,
                    e.data[2], // skip optimizers
                    e.data[3], // constant folding
                    e.data[4], // shape inference
                    e.data[5], // tensor size threshold
                    e.data[6], // target opset version (<= 0 means keep)
                    e.data[7], // profile (emit a Chrome trace)
                    e.data[8], // annotate model info (MACs/FLOPs) into metadata_props
                );
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
        postMessage(["convert-done", data_url, trace]);
    });
});

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
    addEventListener("message", (e) => {
        console.log(e.data);
        const buf = e.data[1];
        let result = null;
        switch (e.data[0]) {
            case "simplify":
                result = runtime.onnxsimplify_export(
                    buf,
                    [], // skip optimizers
                    true, // constant folding
                    true, // shape inference
                    1024 * 1024 * 1024 * 1, // tensor size threshold
                );
                break;
            case "optimize":
                result = runtime.onnxoptimizer_optimize(
                    buf,
                    e.data[2], // target optimizers
                );
                break;
            case "optimize_fixed":
                result = runtime.onnxoptimizer_optimize_fixed(
                    buf,
                    e.data[2], // target optimizers
                );
                break;
            default:
                postMessage(["stderr", "unknown conversion type: " + e.data[0]]);
                return;
        }
        if (!result) {
            postMessage(["stderr", e.data[0] + " failed!"]);
            return;
        }
        console.log("to data url start")
        const data_url = "data:application/octet-stream;base64," + result.toBase64();
        console.log("to data url end")
        postMessage(["convert-done", data_url]);
    });
});

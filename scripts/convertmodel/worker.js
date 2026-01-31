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
        const simplify_result = runtime.onnxsimplify_export(
            buf,
            [], // skip optimizers
            true, // constant folding
            true, // shape inference
            1024 * 1024 * 1024 * 1, // tensor size threshold
        );
        if (!simplify_result) {
            postMessage(["stderr", "simplify failed!"]);
            return;
        }
        postMessage(["convert-done", simplify_result]);
    });
});

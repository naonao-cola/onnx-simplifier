//! End-to-end tests that require the native `onnxsim_c` library to be linked.
//!
//! They are `#[ignore]`d so they don't run (or force a native build) during a
//! plain `cargo test`. Once the library is available, run them with:
//!
//! ```sh
//! cargo test -- --ignored
//! ```
//!
//! `roundtrip_simplify_path` additionally needs a real model; point it at one
//! with `ONNXSIM_TEST_MODEL=/path/to/model.onnx`.

#[test]
#[ignore = "requires the native onnxsim_c library"]
fn list_optimizers_is_non_empty() {
    let passes = onnxsim::list_optimizers();
    assert!(
        !passes.is_empty(),
        "expected at least one optimizer pass to be reported"
    );
}

/// The path-based API cannot carry FunctionProto rules; it must reject them
/// before touching the filesystem or the native library, so this runs without
/// a model file.
#[test]
fn simplify_path_rejects_function_rules() {
    let opts = onnxsim::Options::new().function_rewrite_rule(vec![0u8, 1, 2], vec![3u8, 4, 5]);
    let err = onnxsim::simplify_path_with("in.onnx", "out.onnx", &opts)
        .expect_err("path variant must reject function_rewrite_rules");
    assert!(
        matches!(err, onnxsim::Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

/// Exercise the FunctionProto-based rewriter through the C ABI. Point the three
/// env vars at a serialized ModelProto and the (pattern, replacement) pair of
/// serialized FunctionProto files; the rule should rewrite the model so the
/// output is non-empty and differs from the input.
#[test]
#[ignore = "requires the native onnxsim_c library and model + rule files"]
fn simplify_with_function_rules() {
    let (model, pattern, replacement) = match (
        std::env::var("ONNXSIM_TEST_MODEL"),
        std::env::var("ONNXSIM_TEST_PATTERN"),
        std::env::var("ONNXSIM_TEST_REPLACEMENT"),
    ) {
        (Ok(m), Ok(p), Ok(r)) => (m, p, r),
        _ => {
            eprintln!(
                "set ONNXSIM_TEST_MODEL / ONNXSIM_TEST_PATTERN / ONNXSIM_TEST_REPLACEMENT \
                 to run this test"
            );
            return;
        }
    };
    let model_bytes = std::fs::read(&model).expect("model file should be readable");
    let pattern_bytes = std::fs::read(&pattern).expect("pattern file should be readable");
    let replacement_bytes =
        std::fs::read(&replacement).expect("replacement file should be readable");

    let opts = onnxsim::Options::new().function_rewrite_rule(pattern_bytes, replacement_bytes);
    let simplified =
        onnxsim::simplify_with(&model_bytes, &opts).expect("simplify_with should succeed");
    assert!(
        !simplified.is_empty(),
        "simplified model should not be empty"
    );
    assert_ne!(
        simplified, model_bytes,
        "the rule should have rewritten the model"
    );
}

#[test]
#[ignore = "requires the native onnxsim_c library and a model file"]
fn roundtrip_simplify_path() {
    let model = match std::env::var("ONNXSIM_TEST_MODEL") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("set ONNXSIM_TEST_MODEL to run this test");
            return;
        }
    };
    let dir = std::env::temp_dir();
    let out = dir.join("onnxsim_rust_test_out.onnx");
    onnxsim::simplify_path(&model, &out).expect("simplify_path should succeed");
    let simplified = std::fs::read(&out).expect("output model should exist");
    assert!(
        !simplified.is_empty(),
        "simplified model should not be empty"
    );
    let _ = std::fs::remove_file(&out);
}

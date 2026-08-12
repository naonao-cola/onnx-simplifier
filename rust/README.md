# onnxsim — Rust bindings

Safe Rust bindings to the [ONNX Simplifier](https://github.com/onnxsim/onnxsim).
Simplify ONNX models (shape inference + constant folding) directly from Rust,
using the same C++ core as the Python package and the CLI — no need to shell out
to `onnxsim` or embed a Python interpreter.

This addresses [onnxsim/onnxsim#292](https://github.com/onnxsim/onnxsim/issues/292),
which requested a Rust wrapper so importers such as [Burn](https://github.com/tracel-ai/burn),
[tract](https://github.com/sonos/tract) and [wonnx](https://github.com/webonnx/wonnx)
can simplify models as part of their own pipelines.

## Layout

| Crate         | Role                                                             |
| ------------- | --------------------------------------------------------------- |
| `onnxsim`     | Safe, idiomatic API. Depend on this.                            |
| `onnxsim-sys` | Raw FFI declarations + the build script that links the C core.  |

Both wrap `onnxsim/capi/onnxsim_c_api.h`, a small C ABI over the C++ simplifier.

## Usage

```toml
[dependencies]
onnxsim = { git = "https://github.com/onnxsim/onnxsim", subdir = "rust/onnxsim" }
```

In-memory (serialized `ModelProto` bytes, e.g. from the `prost`/`protobuf`
generated ONNX types, or straight from disk):

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let simplified = onnxsim::simplify(&model)?;
    std::fs::write("model.opt.onnx", &simplified)?;
    Ok(())
}
```

File in, file out:

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    onnxsim::simplify_path("model.onnx", "model.opt.onnx")?;
    Ok(())
}
```

With options:

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let opts = onnxsim::Options::new()
        .shape_inference(false)                       // skip if it crashes on your model
        .skip_optimizer("eliminate_nop_transpose")    // keep a specific pass off
        .tensor_size_threshold(512 * 1024 * 1024);
    let simplified = onnxsim::simplify_with(&model, &opts)?;
    Ok(())
}
```

List the optimizer passes you can skip:

```rust
for name in onnxsim::list_optimizers() {
    println!("{name}");
}
```

Print the before/after difference, the same op-count and model-size summary the
Python CLI shows after simplifying:

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let simplified = onnxsim::simplify(&model)?;
    print!("{}", onnxsim::model_info_diff(&model, &simplified)?);
    Ok(())
}
```

For the specific nodes and values that changed rather than just the aggregate
counts, use `graph_diff` instead: which nodes/values were removed, added, or
changed (matched by output tensor name), e.g. a Conv whose bias input got
folded into its weight.

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let simplified = onnxsim::simplify(&model)?;
    print!("{}", onnxsim::graph_diff(&model, &simplified)?);
    Ok(())
}
```

## Custom rewriter

Run your own graph-rewriting logic inside the simplification fixed point — the
Rust equivalent of the Python `custom_rewriter` parameter. The closure is called
each round with the current model as serialized `ModelProto` bytes and returns
`Ok(None)` (nothing changed this round), `Ok(Some(bytes))` (the rewritten
model), or `Err(..)` to abort. Because it is interleaved with the built-in
optimizer, shape inference and constant folding, a rewrite can unlock further
simplification and vice versa.

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let simplified = onnxsim::simplify_with_rewriter(
        &model,
        &onnxsim::Options::new(),
        |bytes: &[u8]| {
            // Decode `bytes`, rewrite the graph, and return the new bytes,
            // or `Ok(None)` to report that nothing changed this round.
            let _ = bytes;
            Ok::<_, onnxsim::Error>(None)
        },
    )?;
    std::fs::write("model.opt.onnx", &simplified)?;
    Ok(())
}
```

## Building the native library

`onnxsim-sys` needs the `onnxsim_c` shared library. Its build script supports
three modes:

1. **From source (default).** Runs CMake to build the full onnxsim stack
   (ONNX Runtime, onnx-optimizer, protobuf). This is heavy the first time. Check
   out the git submodules first (for onnx-optimizer); the ONNX Runtime source is
   not a submodule and is downloaded automatically on the first build:

   ```sh
   git submodule update --init --recursive
   cargo build
   ```

   Set `ONNXSIM_SKIP_ORT_DOWNLOAD=1` to forbid the automatic download (the build
   then requires the ONNX Runtime source to already be present at
   `third_party/onnxruntime-1.28.0`).

   **Fast path — prebuilt ONNX Runtime.** To skip compiling ONNX Runtime from
   source, set `ONNXSIM_PREBUILT_ORT=1`. The build then links an official
   [ONNX Runtime release](https://github.com/microsoft/onnxruntime/releases)
   (downloaded and cached automatically) instead. onnx-optimizer, onnx and
   protobuf are still built from source, but the slowest dependency is skipped:

   ```sh
   ONNXSIM_PREBUILT_ORT=1 cargo build
   # optionally pin a version or reuse an already-extracted release:
   ONNXSIM_PREBUILT_ORT=1 ONNXSIM_ORT_VERSION=1.28.0 cargo build
   ONNXSIM_PREBUILT_ORT=1 ONNXSIM_ORT_HOME=/path/to/onnxruntime-linux-x64-1.28.0 cargo build
   ```

2. **Pre-built library.** If you already have `onnxsim_c` (and its dependencies)
   built, point the build script at the directory (or directories, `:`-separated)
   holding the shared libraries:

   ```sh
   ONNXSIM_LIB_DIR=/path/to/libs cargo build
   ```

   To produce it from this repo:

   ```sh
   cmake -B build -DONNXSIM_C_API=ON -DONNXSIM_BUILTIN_ORT=ON
   cmake --build build --target onnxsim_c
   ```

   Add `-DONNXSIM_PREBUILT_ORT=ON` to link an official ONNX Runtime release
   (downloaded and cached under the build tree) instead of compiling it from
   source. `-DONNXSIM_ORT_VERSION=<ver>` pins the release and
   `-DONNXSIM_ORT_HOME=<dir>` reuses an already-extracted one.

3. **Skip building** (for `cargo check` / docs.rs). Set `ONNXSIM_NO_BUILD=1`
   (docs.rs sets `DOCS_RS` automatically). The crate type-checks but cannot be
   linked into a runnable binary.

### Environment variables

| Variable             | Effect                                                        |
| -------------------- | ------------------------------------------------------------- |
| `ONNXSIM_NO_BUILD`        | Skip the native build entirely (type-check only).        |
| `ONNXSIM_LIB_DIR`         | `:`-separated dirs holding a pre-built `onnxsim_c`.      |
| `ONNXSIM_SOURCE_DIR`      | Override the onnxsim C++ source path (default `../..`).  |
| `ONNXSIM_SKIP_ORT_DOWNLOAD` | Forbid the automatic ONNX Runtime source download.    |
| `ONNXSIM_PREBUILT_ORT`    | Link a prebuilt ONNX Runtime release instead of building it. |
| `ONNXSIM_ORT_VERSION`     | Prebuilt release version to fetch (default `1.28.0`).    |
| `ONNXSIM_ORT_HOME`        | Use an already-extracted prebuilt release (no download). |
| `ONNXSIM_ORT_URL`         | Override the prebuilt release download URL.              |

## Examples & tests

```sh
cargo run --example simplify -- input.onnx output.onnx
cargo test          # builds/links the native lib, then runs the tests
```

Most of the unit tests exercise pure-Rust logic (the options builder, the
rewriter/executor trampolines, the DLPack conversions) and never call into the
native library. They still link against it, though — the crate's other code
references the C ABI — so `cargo test` builds `onnxsim_c` like any other build.
Use the prebuilt-ORT fast path (`ONNXSIM_PREBUILT_ORT=1`) to avoid compiling
ONNX Runtime from source.

### Coverage

Rust line/region coverage uses
[`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov). Because the tests
link the native library, measuring coverage builds it too; the prebuilt-ORT fast
path keeps that quick:

```sh
cargo install cargo-llvm-cov          # once
rustup component add llvm-tools-preview

# Terminal summary of the workspace's Rust coverage.
ONNXSIM_PREBUILT_ORT=1 cargo llvm-cov --workspace

# Accumulate the default run and the native-only integration test, then render
# an HTML report and a Cobertura XML (the format the project's other coverage
# reports use).
ONNXSIM_PREBUILT_ORT=1 cargo llvm-cov --workspace --no-report
ONNXSIM_PREBUILT_ORT=1 cargo llvm-cov --no-report -- --ignored list_optimizers_is_non_empty
cargo llvm-cov report --html          # target/llvm-cov/html/index.html
cargo llvm-cov report --cobertura --output-path rust-coverage.xml
```

Branch coverage needs the **nightly** toolchain — `--branch` sets
`-Zcoverage-options=branch`, which stable rejects. Add it to both the runs and
the report to populate the branch-rate column:

```sh
ONNXSIM_PREBUILT_ORT=1 cargo +nightly llvm-cov --branch --workspace
```

On stable, `cargo-llvm-cov` still reports region/line/function coverage; only the
branch column is blank.

`cargo-llvm-cov` instruments only the wrapper crates (`onnxsim`,
`onnxsim-sys`); the C++ core is measured separately by the C++ coverage job.
In CI this runs as the `rust` job in
[`.github/workflows/coverage.yml`](../.github/workflows/coverage.yml), which uses
the nightly toolchain to collect branch coverage; its Cobertura report is folded
together with the C++, Python and JS reports into a single combined coverage
summary and pull-request comment.

The integration test in `onnxsim/tests/` is ignored by default because it needs
the linked native library and an ONNX model; see the file header to enable it.

## License

Apache-2.0, matching the parent project.

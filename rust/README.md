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
let model = std::fs::read("model.onnx")?;
let simplified = onnxsim::simplify(&model)?;
std::fs::write("model.opt.onnx", &simplified)?;
```

File in, file out:

```rust
onnxsim::simplify_path("model.onnx", "model.opt.onnx")?;
```

With options:

```rust
let opts = onnxsim::Options::new()
    .shape_inference(false)                       // skip if it crashes on your model
    .skip_optimizer("eliminate_nop_transpose")    // keep a specific pass off
    .tensor_size_threshold(512 * 1024 * 1024);
let simplified = onnxsim::simplify_with(&model, &opts)?;
```

List the optimizer passes you can skip:

```rust
for name in onnxsim::list_optimizers() {
    println!("{name}");
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
   `third_party/onnxruntime-1.27.1`).

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

## Examples & tests

```sh
cargo run --example simplify -- input.onnx output.onnx
cargo test          # pure-Rust unit tests run without the native lib
```

The integration test in `onnxsim/tests/` is ignored by default because it needs
the linked native library and an ONNX model; see the file header to enable it.

## License

Apache-2.0, matching the parent project.

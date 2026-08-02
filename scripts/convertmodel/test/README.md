# convertmodel inference test

A small [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) smoke
test for the model converter's inference path. It runs the fixture model
(`model.onnx`) for several iterations and checks that every iteration produces
identical output that matches the reference in `io.json`.

It drives the same `inference_core.mjs` the browser converter page uses via
`inference_browser.mjs`, so a green run means that execution path works.

## Run

```bash
cd scripts/convertmodel
npm ci
npm run test:inference          # wasm EP, 5 iterations, batch 1
ORT_ITERS=20 npm run test:inference
ORT_BATCH=64 npm run test:inference    # feed 64 input rows
ORT_EP=webgpu npm run test:inference   # only where a WebGPU runtime exists
npm run test:netron             # Netron embedding helpers unit test (no GPU/ORT needed)
npm run test:trace              # trace-assembler unit test (no GPU/ORT needed)
```

The fixture model's batch (first) input dimension is symbolic, so `ORT_BATCH`
controls how many input rows to feed (default `1`). Each row runs independently,
so the test replicates the single reference row in `io.json` `ORT_BATCH` times
and checks that every output row still matches it.

## Netron embedding test

`npm run test:netron` unit-tests `netron.mjs`, the pure helpers behind the
converter page's "Visualize with Netron (before / after)" panel. The panel
renders each model with a **self-hosted** [Netron](https://github.com/onnxsim/netron)
(built into the site at `./netron/` by the deploy workflow) and hands it the
model bytes over a postMessage embedding protocol — so nothing is uploaded and
there is no model-size limit (the old URL-based path was capped at ~2 MB by the
browser). Each pane also has an **export SVG** button that asks Netron (over the
same protocol's `export` command) to render the shown graph and hand the SVG
bytes back for download. The test covers buffer normalization, data-URL
decoding, and the model/export message shapes; it needs no dependencies or
network:

Netron is used here as an embeddable **model-preview component** driven entirely
over `postMessage`; that idea and protocol are being discussed upstream in
[lutzroeder/netron#1591](https://github.com/lutzroeder/netron/pull/1591#issuecomment-5148706256).

```bash
npm run test:netron
```

## MAC / FLOP metrics

`npm run test:macs` unit-tests `macs.mjs`, the dependency-free protobuf reader
behind the inference panel's MAC/FLOP display. It reads onnxsim's per-model and
per-node metrics out of a model's `metadata_props` (onnxsim
[PR #527](https://github.com/onnxsim/onnxsim/pull/527)), substitutes dynamic
dimensions with 1, and turns a model's FLOPs plus a measured latency into
GFLOP/s.

With **annotate model info** on (the default), the converter bakes these metrics
into *both* outputs of a conversion: the converted result and — via the WASM
`onnxsim_annotate_model_info` binding — the original upload
(`window.__onnxsimOriginalAnnotated`). The inference panel then reports MACs and
throughput for either **model** source, so the original and converted models can
be compared on inference speed. Annotation only adds `metadata_props`, so the
annotated bytes execute identically to the upload.

## Report-an-issue URL

`npm run test:issue` unit-tests `issue_report.mjs`, the pure builder behind the
converter page's **Report an issue** button. It assembles the pre-filled GitHub
issue body (page URL, user agent, the last conversion's parameters, and the tail
of the console log) and — crucially — trims the log from the front until the
whole URL fits under a conservative length cap, so GitHub never rejects it with
*"Whoa there! Your request URL is too long."*. The test covers the field
contents, the untrimmed short-log case, and that a huge log is trimmed (keeping
its error-bearing tail) so the URL stays under the cap. No DOM or network:

```bash
npm run test:issue
```

## Library versions

The page shows the **versions** of the libraries built into the WebAssembly
module — onnxsim, onnx-optimizer, onnx (its IR version + highest supported
opset), and protobuf — as a row of badges under the title, and includes them in
the **Report an issue** body. onnxsim and onnx-optimizer come from their
`VERSION` files (baked in by CMake); onnx and protobuf are read from the linked
libraries at runtime by the `onnxsim_versions` binding. `npm run test:versions`
unit-tests the pure `versions.mjs` helpers (normalization + badge rendering
against a tiny fake DOM); no DOM or network:

```bash
npm run test:versions
```

## URL-driven input (query parameters)

The input model and conversion options can be set straight from the page URL, so
a link converts a specific model on open. The model reference reuses the Hugging
Face loader's repo-id / `.onnx`-URL path (nothing is uploaded):

```
?model=onnxmodelzoo/resnet18d_Opset18
?model=https://…/foo.onnx&optimizer=optimize&cf=0
?model=…&autoload=0          # prefill the box + options but don't auto-run
?backend=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_relu
```

Keys (aliases in parentheses): `model` (`hf`/`url`/`input`), `optimizer`,
`constant_fold` (`cf`), `shape_inference` (`si`), `tensor_size_threshold`
(`tst`), `target_opset` (`opset`), `autoload` (`run`/`convert`), and `backend`
(prefills the backend-test panel).

Conversely, **setting** an input model updates the address bar (via
`history.replaceState`) to the matching `?model=` / `?backend=` link, so the
current input is always shareable — except an uploaded local file, which has no
URL. `npm run test:query` unit-tests the pure `query_params.mjs` parser, the
option-prefiller (against a fake DOM), and the `setInputParam` URL builder; no
DOM/network:

```bash
npm run test:query
```

## Parse a text graph

The **Parse a text graph** panel takes an ONNX
[textual representation](https://onnx.ai/onnx/repo-docs/Syntax.html) — the form
`onnx.parser.parse_graph` / `parse_model` accept — and parses it into a model
with the `onnxsim_parse_graph` binding (whole-model text is used as-is; a bare
graph is wrapped into a model with a default-domain opset import). The parsed
model is shown in the **Before** Netron pane, becomes the source for the
single-feature passes, and — with *convert after parsing* on — is run straight
through the Simplify path.

## Single-pass debug modes

Alongside **Simplify** / **Optimize** / **Fixed Optimize**, the converter's mode
radios include **Shape inference**, **Data propagation**, and **Constant
folding** — each runs exactly one of the transforms `Simplify` otherwise drives
to a fixed point, once, so its isolated effect can be inspected. They map to
onnxsim's `InferShapesOnce` / `PropagateDataOnce` / `FoldConstantOnce` core
helpers, exposed as the `onnxsim_infer_shapes` / `onnxsim_data_propagation` /
`onnxsim_fold_constant` bindings, and run through the same conversion worker (so
constant folding uses the same model executor `Simplify` does — onnxruntime-web
in the ORT-web build). The result flows through the normal convert path: shown in
the **After** Netron pane, downloadable, and runnable in the inference panel.

## Run an ONNX backend test case

The **ONNX backend test case** picker (next to the Hugging Face model selector)
takes an [ONNX backend test case](https://github.com/onnx/onnx/tree/main/onnx/backend/test/data)
directory — a `model.onnx` plus one or more `test_data_set_N/` of `input_*.pb` /
`output_*.pb` TensorProtos. A dropdown offers a curated set of common node tests
(`test_relu`, `test_matmul_2d`, …); the free-text box accepts any GitHub `tree` /
`raw.githubusercontent.com` URL. It fetches the model and test data from GitHub,
**runs the model through onnxruntime-web with the test inputs** (the `input_*.pb`
tensors, not dummy data), and compares each output against the expected
`output_*.pb` tensor within tolerance. With **convert the model** on, the fetched
model also runs through the selected convert mode (Netron panes + download).
TensorProtos are decoded by the `onnxsim_parse_tensor` binding. `npm run
test:backend` unit-tests the pure `backend_test.mjs` helpers — GitHub URL parsing
(tree / blob / `raw.githubusercontent.com` / contents-API forms), numeric
ordering of `input_N.pb` files, float/int/shape tensor comparison, and that every
curated preset is a parseable node-test URL; no DOM or network:

```bash
npm run test:backend
```

## Profiling traces

Both the browser panels can emit a [Chrome Trace Event][cte] JSON that the page
renders inline as a flame graph (and offers for download / hand-off to
[Perfetto](https://ui.perfetto.dev)):

- **onnxsim** — ticking “profile simplification” makes the WASM core run its
  fixed-point profiler (`ONNXSIM_PROFILE`) with ONNX Runtime's constant-folding
  spans merged in (`ONNXSIM_MERGE_ORT_PROFILE`), and returns the trace with the
  converted model.
- **onnxruntime-web** — the inference panel captures per-`session.run` wall
  spans and, on WebGPU, one span per GPU kernel via
  `ort.env.webgpu.profiling.ondata`. `trace.test.mjs` covers the assembler
  (`trace_build.mjs`) that turns those records into a Chrome trace.

[cte]: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview

## Execution providers

onnxruntime-web exposes the `wasm` execution provider everywhere and the
`webgpu` provider where the host has a WebGPU runtime (a browser, or Node built
with one). A plain CI/Node container has no GPU, so CI runs the `wasm` provider;
the browser page defaults to `webgpu` and falls back to `wasm`.

## Regenerate the fixture

`model.onnx` and `io.json` are produced by `make_fixture.py` (needs `onnx` and
`numpy`):

```bash
python3 make_fixture.py
```

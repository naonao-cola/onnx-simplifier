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
browser). The test covers buffer normalization, data-URL decoding, and the
message shape; it needs no dependencies or network:

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

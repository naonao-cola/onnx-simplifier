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
npm run test:inference          # wasm EP, 5 iterations
ORT_ITERS=20 npm run test:inference
ORT_EP=webgpu npm run test:inference   # only where a WebGPU runtime exists
npm run test:trace              # trace-assembler unit test (no GPU/ORT needed)
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

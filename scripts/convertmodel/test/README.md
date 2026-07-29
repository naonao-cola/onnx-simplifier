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
```

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

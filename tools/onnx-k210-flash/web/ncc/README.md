# ncc.js / ncc.wasm (vendored)

`ncc.js` + `ncc.wasm` are nncase's own ONNX -> kmodel compiler (the real
`ncc` CLI, K210 + cpu targets, no Vulkan/TFLite/Python), cross-compiled to
wasm32-emscripten from
[`onnxsim/nncase`](https://github.com/onnxsim/nncase)'s `wasm-k210-build`
branch (`scripts/build-wasm.sh` there). `../ncc_wasm.mjs` drives it from
this page -- see that file and `../README.md`'s "Convert ONNX -> kmodel" for
how.

## Status

**Built and verified for real this session**, not just compiled: the fork's
own `scripts/build-wasm.sh` had never actually produced a working artifact
before -- three real bugs were found and fixed getting it to build at all
(wrong fmt version for this build's clang, a real missing-include bug in
the fmt version after that, and a missing `-G Ninja` on nncase's own
configure step -- see that script's own comments and its branch's commit
history for the full writeup). Once it built, the result was driven from
Node using **only** the browser-safe interface (`Module.FS.writeFile` /
`Module.callMain` / `Module.FS.readFile` -- no `NODERAWFS`, no host
filesystem access, the same interface a real page uses) for a real ONNX ->
kmodel compile, for both the `cpu` and `k210` targets, producing correctly
shaped, non-empty kmodels each time.

**Not yet verified:** actually loading and running `ncc_ui.mjs`/`ncc_wasm.mjs`
in a real browser tab -- this session's browser-automation tooling runs on
a different network than the sandbox this was built in, so a local dev
server here was never reachable from it. The wasm module's own interface is
environment-agnostic by design (Emscripten's MEMFS + `callMain` behave the
same in Node and a browser, unlike the `NODERAWFS` mode this build
deliberately avoids), so the Node-side verification above is strong
evidence, not a substitute for actually clicking through the page. Try it
for real before trusting it end to end.

## Rebuilding it yourself

```sh
git clone --branch wasm-k210-build https://github.com/onnxsim/nncase.git
cd nncase
bash scripts/build-wasm.sh
# -> build-wasm/bin/ncc.js, build-wasm/bin/ncc.wasm
cp build-wasm/bin/ncc.js build-wasm/bin/ncc.wasm /path/to/onnx-k210-flash/web/ncc/
```

No root/sudo needed (the script fetches a native `protoc` via
`apt-get download` + `dpkg-deb -x`, not `apt-get install`) and no conan --
see the script's own header comment for what it does and why. Takes real
compile time (fmt + protobuf + all of nncase's compiler, cross-compiled);
not instant.

`sha256sum` of the committed files:
```
520873f505e7b6a52289c495b212ea75d6609ecb2306b18c8c9602f4bbf11611  ncc.js
9955b2aefd01ba17910af0d10a04f5298d32ba2bda668af6e8db92f531415a26  ncc.wasm
```

# Big-endian status

onnxsim is **not correct on big-endian hosts today**. Constant folding is
disabled there and fails quietly, so `simplify()` returns a model that is
semantically valid but barely simplified. This document records what was
measured, how to reproduce it, and where the byte-order assumptions live.

## Why byte order matters here

ONNX fixes the byte order of tensor payloads. From `onnx.in.proto`, on
`TensorProto.raw_data`:

> When this raw_data field is used to store tensor value, elements MUST be
> stored in as fixed-width, little-endian order.

So `raw_data` is little-endian on *every* host. Code that `memcpy`s or
`reinterpret_cast`s it into native integers or floats is correct only by
accident of running on a little-endian CPU.

## How it was measured

s390x, under `qemu-s390x-static` on an x86_64 host, against an **amd64 control**
running the same commit, the same test suite and the same package versions — so
byte order is the only variable. See `scripts/cross/README.md` for the
procedure; both runs go through `scripts/cross/run_s390x_tests.sh`.

Environment for both runs: CPython 3.12.3, numpy 1.26.4, onnx 1.23.0 (the
vendored version), onnxsim 0.7.0, no onnxruntime (so the reference-evaluator
fallback is in use).

```
little endian : 3 failed, 99 passed, 21 skipped, 1 xfailed
big endian    : 6 failed, 95 passed, 22 skipped, 1 xfailed
```

The three failures common to both are environmental, not byte-order related
(`test_simplify_with_unavailable_provider_raises`, `test_fuse_conv_bn_into_conv`
and `test_fuse_convtranspose_bn` all need onnxruntime).

Failing **only** on big endian:

| Test | Symptom |
| --- | --- |
| `test_backend.py::test_simplify_without_onnxruntime` | `assert 2 == 1` — the foldable `Add` is still in the graph |
| `test_fusion_patterns.py::test_defer_constant_expand_folds_small_consumer` | `assert 1 == 0` — the `Expand` was never folded |
| `test_fusion_patterns.py::test_defer_constantofshape_folds_small_consumer` | same, for `ConstantOfShape` |

Plus one test that *self-skips* rather than failing:
`test_profiling.py:134: constant folding did not run the ONNX Runtime executor here`.

Every one of them has the same cause, visible on stderr:

```
WARNING: failed to run "Add" op (name is ""), skip... dlpack bridge: only little endian is supported
```

## Finding 1 — constant folding is silently disabled (confirmed)

`onnxsim/dlpack_bridge.h:131`, in `BuildFromProto`:

```cpp
if constexpr (std::endian::native != std::endian::little) {
  delete ctx;
  throw std::invalid_argument("dlpack bridge: only little endian is supported");
}
```

Every constant fold goes through this. `RunOps` catches per-op failures, logs a
warning and moves on, which is the right behaviour for an op the executor cannot
run — but here it turns a total loss of constant folding into a warning on
stderr. `simplify()` still returns `check_ok=True`, because the unfolded model
*is* equivalent to the input. The user gets a model that was not simplified and
no error.

The damage cascades: once one fold is skipped, later folds reference
initializers that were never produced, so the log fills with
`no initializer _v_7`-style follow-on failures and dependent optimizations
(e.g. folding a `Mul` scale into `Conv` weights) also stop happening.

Reproduced directly — same model, same everything but the CPU:

```
little endian:  nodes = ['Conv']                                W folded, scaled correctly
big endian:     nodes = ['Reshape', 'Unsqueeze', 'Mul', 'Conv']  W unscaled, three nodes left over
```

Worth stating explicitly: this is **not** silent numerical corruption. The
big-endian output stays mathematically equivalent to the input model, and
`check_n` (which runs both models and compares) would catch it if it were not.
The bug is a silent loss of optimization, not wrong numbers.

## Finding 2 — `raw_data` read in host order (by inspection)

`onnxsim/onnxsim.cpp:798`, `IntTensorToSymTensor`, which seeds onnxsim's
symbolic shape inference from INT64/INT32 initializers:

```cpp
// Raw data is read little-endian, matching how this
// file already memcpys raw_data into onnxruntime tensors.
...
if (n) std::memcpy(vals.data(), raw.data(), n * sizeof(int64_t));
```

The comment states the intent, but a `memcpy` into a host `int64_t` reads host
order — correct only on little-endian machines. The same applies to
`ToTensorProto` in `dlpack_bridge.h`, which writes native bytes into `raw_data`
with no endianness guard (unlike `BuildFromProto` on the input side); it is
currently unreachable on big endian only because the input side throws first.

This one is a defect by inspection: **no test or targeted probe was found where
it changes onnxsim's output.** Probes over dynamic-shape `Reshape` and `Expand`
models produced identical inferred shapes on both byte orders, most likely
because ONNX's own shape inference already resolves those cases before the
symbolic engine's seeds matter. It is recorded here because the code is wrong as
written and would surface on models that do depend on the symbolic path — and
because fixing Finding 1 without fixing this would turn a loud failure into a
quiet wrong answer.

## Finding 3 — upstream: onnx's C++ `Tensor` (by inspection)

`onnx/common/tensor.h` (vendored, and upstream) reads `raw_data` natively:

```cpp
inline type* Tensor::data<type>() {
  if (is_raw_data_) {
    return reinterpret_cast<type*>(raw_data_.data());
```

and `ir_pb_converter.cc` copies `raw_data` in verbatim without byte-swapping, so
the optimizer passes onnxsim and onnx-optimizer build on top of this inherit the
assumption. Not onnxsim's code to fix, but it bounds how far onnxsim can be made
big-endian correct on its own.

Note also that **onnx < 1.16 actively corrupts models on big endian**:
`numpy_helper.to_array()` called `convert_endian(tensor)`, byte-swapping the
proto *in place*, so merely reading an initializer rewrote its `raw_data` into
spec-violating big-endian order. Fixed upstream by 1.23 (which byte-swaps into a
local instead), but relevant to anyone pairing onnxsim with an old onnx.

## What a fix would involve

The honest summary is that supporting big endian is a real piece of work, not a
one-line change:

1. Byte-swap on both edges of the DLPack bridge (`BuildFromProto` in,
   `ToTensorProto` out) instead of refusing, which costs a copy on big endian
   only.
2. Read `raw_data` little-endian explicitly in `IntTensorToSymTensor`.
3. Decide what to do about onnx's `Tensor::data<T>()` — either upstream a fix or
   normalize tensors before handing them to the optimizer.

Until then, the most valuable small change would be making the failure **loud**:
constant folding collapsing entirely should not be reported to the user as a
successful simplification.

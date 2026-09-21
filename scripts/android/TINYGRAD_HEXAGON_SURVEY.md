# Tinygrad Hexagon DSP codegen survey

Surveyed the locally installed Tinygrad 0.14.0 source on 2026-09-21, focusing
on `tinygrad/runtime/ops_dsp.py`, `tinygrad/renderer/cstyle.py`, and the runtime
compiler and device initialization path.

## What the backend provides

- `DSPRenderer` extends Tinygrad's C-style Clang renderer. It emits C kernels
  and compiles them for `hexagonv65` with 128-byte HVX enabled. The inspected
  renderer does not itself select V73 or provide a V73-specific schedule.
- The DSP runtime talks to FastRPC from a Linux host. It opens `/dev/ion` and
  `/dev/adsprpc-smd`, allocates shared ION buffers, and submits kernels with
  FastRPC ioctls. It uses DSP-side performance timestamps for kernel timing.
- The renderer's C output leaves instruction selection to Clang. The source
  inspected here does not show a Tinygrad Hexagon memory-bandwidth or cache
  cost model. Tinygrad's general scheduling/codegen search may still produce
  useful tiling, but this backend alone does not establish that its schedule
  choices will predict memory limits on the Xiaomi 12S.

## Fit for the Xiaomi 12S experiment

The checked-out TVM harness connects to Android through ADB and TVM's Hexagon
RPC server. Tinygrad's DSP runtime expects Linux FastRPC device nodes instead,
so its runtime cannot be used directly from the current host setup. Running
Tinygrad-generated kernels on the phone would require a V73-capable compiler
configuration and a runtime bridge compatible with the phone's Android
FastRPC environment. Until that exists, Tinygrad can be surveyed or used for
offline codegen experiments, but its DSP timings cannot be compared directly
with the TVM phone measurements.

## Sources

- [Tinygrad DSP runtime and renderer](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_dsp.py)
- [Tinygrad C-style renderer](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/renderer/cstyle.py)
- [Tinygrad compiler and runtime overview](https://github.com/tinygrad/tinygrad/blob/master/docs/developer/developer.md)

# ResNet18 stem Gather `[16,1,3,50176] -> 614656` on AX650

The ResNet18 training step's first Gather (the 7x7 stride-2 stem im2col:
49 taps x 112x112 = 614,656 indices over a 224x224 image) failed in Pulsar2
7.0-lite when compiled alone. This records why and the workaround that gives a
working template for `emit_gather_last_axis_from_template`.

## Why it fails

```
AssertionError: job io size > ocm size, AxGather
    13095936 > 3141632
```

Pulsar2 lowers a Gather to one on-chip-memory (OCM) job made of three tensors:
the transposed source `(W, L)`, the S32 index vector, and the gathered output
`(N, L)`, with `L` the leading dimensions padded to 16 lanes. The OCM budget is
3,141,632 bytes. It tiles the batch (here 16 -> 4) but never splits along N.

| tensor | shape | bytes |
| --- | --- | --- |
| transposed source | `(50176, 16)` | 802,816 |
| S32 indices | `(614656,)` | 2,458,624 |
| gathered output | `(614656, 16)` | 9,834,496 |
| **total** | | **13,095,936** |

Failures elsewhere fit the same accounting: `[4,1,3,50176]` (rows padded to 4)
needs 5,117,952 B, and a 130,000-index chunk needs 3,402,816 B
(`802816 + 4*130000 + 16*130000`). For some shapes the compiler picks a second
lowering whose tensors are `(N, 4, 3)` padded to 16 per index, twice
(`2 * 16 * N`); a 116,000-index chunk needs 3,712,000 B there. Measured on
`[16,1,3,50176]`:

| indices N | result |
| --- | --- |
| 87,808 | builds |
| 116,000 | fails (3,712,000 > 3,141,632) |
| 130,000 | fails (3,402,816 > 3,141,632) |
| 614,656 | fails (13,095,936 > 3,141,632) |
| 614,656 at batch 1, `[1,1,3,50176]` | builds |

So it is a per-node size limit, not something wrong with these indices, the
constant stem input, or the consumer. Every earlier template in this series
built because its job fit in 3.14 MB.

## What works

Split the Gather along N into chunks that fit, then `Concat` along the last
axis. The stem is 7 kernel rows x 87,808 indices (7 taps x 12,544 positions),
each chunk within the limit:

```
Gather(x, idx[0:87808]) ... Gather(x, idx[526848:614656]) -> Concat(axis=-1)
```

Both forms built at batch 16: 7 `Gather` + `Concat` into one `[16,1,3,614656]`
output, and 7 separate outputs. The Concat form keeps the original interface
(input `x`, output `y`), so it is a drop-in template for the
existing last-axis emitter:

- `npu_params` = the seven index chunks back to back (chunk `k` at word
  `k * 87808`), i.e. the full 614,656-word index vector first, then a
  2,380-word tail. That is exactly the "first N words are the indices, keep the
  rest" rule the emitter already uses. The 7-output form has a 3,500-word tail.
- MCode is 56,680 bytes (one Gather is 11,240 for 87,808 indices).
- A batch-1 single Gather (`[1,1,3,50176]`, 4,040 B MCode, 200-word tail) also
  builds, but the step runs at batch 16.

The template is
`scripts/axera/fixtures/gather_16x1x3x50176_axis3_n614656.axmodel.gz`
(18 KB, index words zeroed by `make_gather_fixture`).

## Verification

- **Index independence.** A compiler build of the 7-chunk graph with random
  indices has `npu_params` byte-identical to the emitter's output from the
  real-index template. Its MCode differs in nine bytes, all in the noise region
  below.
- **Noise region.** Rebuilding the same graph with identical indices flips
  bytes 333-357 in three-byte groups (12 bytes differ), and the random-index
  build differs in the same region. So it is compiler noise, not index
  dependence. It is wider than the single-Gather window (301-325), so
  `_GATHER_NOISE_END` gives this one template a window of 301-357; the
  emitter still rejects a change outside it (tests flip bytes 341 and 400).
- **Device** (AX8850 in `axcl-vm`, V3.6.5, runs serialized under a shared
  lock, inputs uniform in +/-0.8 inside the calibration range). A control run of
  the unmodified template matched numpy (max error 0.0035). Emitted variants
  with random, shuffled-real, and reversed-real index vectors each matched
  numpy Gather with maximum error 0.0035 over 29,503,488 outputs, and a health
  run after them was clean.

## Caveats

- Standalone, not in the real step. The real graph has a `Mul` (presumably the
  padding mask) after this Gather and a `Reshape` before it; compiled with its
  neighbours the compiler may tile it differently and produce different MCode.
  This template covers the standalone 7-chunk model only.
- Other shapes that exceed the budget need the same treatment. The rule is that
  one Gather job must fit in about 3.14 MB: roughly `16*W + 20*N` bytes for the
  first lowering.
- Chunking is done by the graph author. Nothing here makes Pulsar2 tile along N.

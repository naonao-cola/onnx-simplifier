# MCC (single-image 3D reconstruction) on the Xiaomi 12S

[Multiview Compressive Coding](https://mcc3d.github.io/) (Wu, Johnson, Malik, Feichtenhofer,
Gkioxari, CVPR 2023) reconstructs an object's full 3D shape and color from one RGB-D view: an
encoder reads the image and the seen points, and a decoder is queried at 3D points for
occupancy and color.

**License: the upstream code and weights are CC BY-NC 4.0 (non-commercial).** Nothing from
upstream is vendored here: `model.py` imports the upstream `mcc_model.py` from a clone
(`MCC_REPO`, default `~/.cache/onnxsim-mcc/MCC`) and loads the released checkpoint
`co3dv2_all_categories.pth` (CO3D v2, all categories) from `dl.fbaipublicfiles.com/MCC/`.

## Model facts (from the code)

- RGB encoder: ViT-B (`get_mcc_model`: width 768, 12 blocks, 12 heads) on 224x224, 197 tokens.
- XYZ encoder: seen points resampled to 112x112, a 1-block transformer per 8x8 window
  (196 windows x 65 tokens), then another ViT-B stack (12 blocks) -> 197 tokens.
- Decoder: 8 blocks, width 512, 16 heads, on `[197 seen tokens ; Q queries]`; output 1 occupancy
  logit + 3x256 color logits (softmax with temperature 0.1 -> expected value) per query.
- Queries: a grid over [-3, 3]^3. The demo's default granularity 0.05 is 120^3 = 1.73M queries;
  the training/eval default 0.1 is 60^3 = 216k.

## Deployment split (`model.py`, exact)

Upstream masks the decoder so seen tokens attend only to seen tokens and each query only to the
seen tokens and itself. So the seen stream is query-independent:

- `Encoder`: both encoders + the decoder's seen-token stream, run once per image, outputs each
  decoder block's seen K/V (8 x 16 heads x 197 x 32, twice).
- `QueryDecoder`: a fixed-size chunk of queries, each attending to `[K_seen ; k_self]`. Cost is
  linear in the chunk (about 27 MMAC per query) instead of upstream's (197+Q)^2 masked attention.
- The XYZ encoder's 8x8 window partition (a rank-6 view upstream) moves to the host; everything
  in the graphs is rank <= 4.

## Results

Checkpoint `co3dv2_all_categories.pth`, sha256
`ca861bee4c2cb27acc6855da34227ce7026cf9eb275171da3c5a33976b3d86bd` (2.4 GB with optimizer state).
Demo input: upstream `demo/quest2` (image + iPhone point cloud + mask).

**Split vs upstream** (`validate.py`, 2000 queries, host fp32): occupancy logit max abs 1.5e-5,
color max abs 1.7e-5 -- the K/V-cache split is exact.

**Host fp32 reference** (split, 8 threads; the reference every phone run is scored against):

| granularity | queries | host fp32 | occupied (p > 0.1 / 0.3 / 0.5) |
|---|---|---|---|
| 0.1 | 216,000 | 29.6 s | 20,684 / 13,604 / 9,725 |
| 0.05 (demo default) | 1,728,000 | 243 s | 165,139 / 108,881 / 78,007 |

## Files

| file | what |
|---|---|
| `model.py` | deployment split on top of the upstream module; demo preprocessing without pytorch3d |
| `validate.py` | split vs upstream forward; fp32 full-grid reference `ref_<demo>_<g>.npz` |
| `mcc.py` | export (onnxsim, ORT check), phone runs (strict all-HTP via `phone.sh`), scoring |
| `phone.sh` | one ONNX piece on the phone (ORT + QNN EP), under the shared phone lock |

## Follow-ups

- The decoder is dense attention + MLP over many queries: a natural target for the HMX GEMM
  work (`codex/android-hmx-gemm`), not used here.

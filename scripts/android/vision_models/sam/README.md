# Segment-Anything variants on the Hexagon HTP (Xiaomi 12S, SM8475 / V69)

Plan item 5 of `../../vision_models_plan.md`: which small SAM runs interactively on this phone.
Each variant is split into an **image encoder** (once per image) and a **prompt encoder + mask
decoder** (once per click/box). The interface is the same for all of them (`variants.py`): the
encoder takes the padded RGB image, uint8 NHWC, with normalization inside the graph; the decoder
takes the 256x64x64 embedding and 2 prompt points (a point + padding, or the 2 box corners) and
returns 4 low-res masks + IoU scores. There is no mask input and no resize to the original size.

## Results

All runs are strict all-HTP (no CPU fallback) through ORT + the QNN EP (QNN 2.50,
`qnn_run_multi`), burst mode. Medians of 10 runs, taken under the shared phone lock.
Accuracy is measured against fp32 ORT on 10 COCO val2017 images x 4 prompts
(3 foreground points, 1 box):
- **emb cos:** cosine of the image embedding vs fp32.
- **mask IoU:** IoU of the phone's 256x256 mask vs the fp32 mask for the same prompt and slot. The
  mask is computed from the phone embedding, decoded by the fp32 decoder.

**fp16, as exported** (every variant is exact vs upstream torch, see "What it took" below):

| variant | encoder input | enc MB | encoder HTP | emb cos min / mask IoU | decoder HTP | decoder CPU x4 | first mask |
|---|---|---|---|---|---|---|---|
| **EdgeSAM** (RepViT, fused) | 1024 | 21.0 | **81.3 ms** | 0.99971 / 0.995 | 10.9 ms | 44.9 ms | **92 ms** |
| MobileSAM (TinyViT-5M) | 1024 | 26.6 | 319.8 ms | 0.99999 / 0.999 | 10.9 ms | 52.9 ms | 331 ms |
| **EfficientViT-SAM-L0** (bicubic neck as polyphase convs) | 512 | 117.3 | **41.8 ms** | 0.99939 / 0.968 | 11.0 ms | 44.5 ms | **53 ms** |
| EfficientViT-SAM-L0, upstream bicubic `Resize` | 512 | 117.3 | 1503 ms | 0.99939 / 0.968 | 11.0 ms | 44.5 ms | 1514 ms |

The decoder is the same SAM mask decoder for all three: **11 ms on the HTP in fp16** (mask IoU
0.998-0.999 vs fp32), 45-53 ms on 4 CPU threads. So every extra click costs ~11 ms.

**Faster encoders** (approximations and quantization; `sam.py quantize --policy/--method`,
`sam.py export --gelu/--upsample`):

| variant | encoder | HTP | emb cos min | mask IoU mean / min | verdict |
|---|---|---|---|---|---|
| EdgeSAM | fp16 | 81.3 ms | 0.9997 | 0.995 / 0.978 | **default** |
| EdgeSAM | W8A16, depthwise convs fp16 (`dw16`) | 40.9 ms | 0.972 | 0.964 / 0.639 | 2x, visible mask loss |
| EdgeSAM | W8A16 (`a16`) | 20.5 ms | 0.944 | 0.947 / 0.508 | 4x, worse |
| EdgeSAM | int8 (percentile) | 7.4 ms | 0.616 | 0.752 / 0.331 | unusable |
| EfficientViT-SAM-L0 | fp16, bicubic as exact polyphase convs | 41.8 ms | 0.9994 | 0.968 / - | **36x, exact (max abs 8e-7 vs upstream)** |
| EfficientViT-SAM-L0 | fp16, bilinear neck upsample | 40.2 ms | 0.995 | 0.923 / 0.039 | superseded by polyphase (not the trained model) |
| EfficientViT-SAM-L0 | W8A16 / int8 (polyphase upsample) | - | 0.37 / 0.24 (host) | 0.22 / 0.24 | unusable: the collapse is not the bicubic |
| MobileSAM | fp16, sigmoid-GELU | 169.6 ms | 0.991 | 0.955 / 0.059 | 1.9x, some masks break |
| MobileSAM | fp16, tanh-GELU (attribute) | 318.8 ms | 1.0000 | 0.999 / 0.989 | QNN ignores `approximate`: same speed |
| MobileSAM | fp16, tanh-GELU as Mul/Tanh ops | 604.1 ms | 1.0000 | 0.999 / 0.992 | slower |
| MobileSAM | W8A16 (`a16`) | 202.9 ms | 0.996 | 0.980 / 0.734 | 1.6x |
| MobileSAM | int8 (percentile) | 90.3 ms | 0.907 | 0.783 / 0.000 | unusable |

On the host the int8 mask decoder is unusable for every variant (mask IoU 0.26-0.33 vs fp32,
with an fp32 embedding in). Keep the decoder fp16; at 11 ms it doesn't need int8.

## Recommendation

**EdgeSAM, fp16 encoder + fp16 decoder, all on the HTP.** The first mask comes 92 ms after a new
image (81 + 11) and each further click takes 11 ms, at fp32-level accuracy (embedding cos 0.9997,
mask IoU 0.995). That is interactive. With a camera it gives ~11 new images/s, and clicks on the
current image are instant. The encoder is the smallest (21 MB). Its RepViT design (convs, no
attention) is what the HTP runs best: the only other variant under 100 ms needs a
non-trained-model swap (EfficientViT-SAM with bilinear upsampling, 40 ms, mask IoU drops to 0.92).

If 81 ms is too slow, the next lever is not int8 activations: EdgeSAM loses too much there
(embedding cos 0.62). Candidates:
- quantization-aware fine-tuning, or
- W8A16 with more layers left in fp16. Keeping the depthwise convs in fp16 alone gets 41 ms at
  embedding cos 0.972.

## Why each model runs the speed it does (per-op QNN profiles, `profiling_level=detailed`)

- **MobileSAM:**
  - GELU is 44% of the encoder. The TinyViT stage-0 MBConv blocks run GELU on
    4x-expanded 256x256 maps. MatMul is 19%, Softmax 8%.
  - A sigmoid GELU halves the time (170 ms) but moves masks.
  - QNN's Gelu op ignores the tanh `approximate` attribute (identical time and output), and
    spelling tanh-GELU out as ops is 2x slower.
- **EfficientViT-SAM-L0:**
  - The three bicubic `Resize` ops of its neck are **97% of the encoder** (each ~2.2 G cycles).
    The HTP has no fast cubic resize. One of them resizes 64x64 to 64x64, which is an identity.
  - Fix, exact: an integer-factor bicubic upsample is a fixed linear map. It becomes an
    edge pad, a (5,1) then a (1,5) depthwise conv producing the s x s output phases, then
    DepthToSpace (CRD); the identity resize is dropped. That runs in **41.8 ms instead of 1503 ms**
    at the same embedding (cos 0.99939). Max abs error vs PyTorch's bicubic is 7e-7
    (`_upsample_polyphase` in `variants.py`).
  - Bilinear is 37x faster overall, but it's a different model than the trained one (mask IoU
    0.92, one prompt breaks).
  - It is also the only variant whose int8/W8A16 quantization collapses (cos 0.24-0.35).
- **EdgeSAM:** after fusing its 109 RepViT reparameterizable modules (exact to 4e-6), it is plain
  convs + GELU + one LayerNorm2d neck, which the HTP runs at ~81 ms in fp16.

## What it took (all exact or near-exact vs upstream; `sam.py export` checks it)

Each fix is a small patch in `variants.py`, applied only for the export. `with
variants.upstream():` swaps the originals back, and `export.json`'s `rewrite_check` records the
max difference to the unpatched torch model (0 to 8e-4 on the masks). Without them the models are
not strict all-HTP, or they crash or give wrong values:

- **Rank <= 4 everywhere:**
  - TinyViT's (MobileSAM) and SAM ViT's window partition/unpartition use rank-6
    view/permute. QNN refuses them (30 Reshape/Transpose nodes). The fix: for batch 1,
    `(nH, ws, nW, ws*C)` with axes 1/2 swapped is the same permutation.
  - SAM ViT's decomposed relative-position add is rank 5; it becomes batched matmuls plus an
    `(N, kh, kw)` broadcast.
  - The mask decoder's `repeat_interleave` exports as a rank-5 Unsqueeze+Tile. QNN's compile
    **segfaults** on it (it doesn't refuse it), so it is dropped (identity for one image).
- **No ops the QNN EP doesn't take:**
  - Export at opset 20 so GELU is the `Gelu` op; the opset-17 `Erf` form puts 23 nodes on the CPU.
  - onnxsim's `fuse_attention` is skipped: its `com.microsoft::Attention` isn't taken either.
- **fp16 overflow:** SAM's `LayerNorm2d` computes mean, `(x-u)^2`, sqrt by hand. The squares of
  the neck's ~1e3 activations overflow fp16, and the embedding comes out garbage. The fix is one
  `LayerNormalization` on the channels-last view.
- **uint8 input as `DequantizeLinear(1, 0)`, not `Cast`:** a uint8->float `Cast` as the first node
  of an fp16 HTP graph gives wrong values (MobileSAM stem cos 0.81, embedding 0.45-0.82). The same
  graph with a DequantizeLinear matches fp32 at every probed tensor (cos 0.99999).
- **EdgeSAM:** RepViT reparameterization (`fuse()`, 109 modules) and bilinear neck upsampling (as
  EdgeSAM's own ONNX export does).

## Reproduce

```sh
# once: repos + weights (MobileSAM weights come with its repo; EdgeSAM/EfficientViT-SAM from HF)
git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM ~/.cache/sam-src/MobileSAM
git clone --depth 1 https://github.com/chongzhou96/EdgeSAM ~/.cache/sam-src/EdgeSAM
git clone --depth 1 https://github.com/mit-han-lab/efficientvit ~/.cache/sam-src/efficientvit
git clone --depth 1 https://github.com/facebookresearch/segment-anything ~/.cache/sam-src/segment-anything
# weights -> ~/.cache/sam-weights/ (sha256):
#   edge_sam_3x.pth        5a3e3261da9e2f4c8154410885689ae57d09cbf4cf5d0c848b9d80b6df336ece  (hf chongzhou/EdgeSAM)
#   efficientvit_sam_l0.pt c4f994b01a16d48bcf2fbbb089448cfbf58fae5811edfa8113c953b8b8cc64b8  (hf han-cai/efficientvit-sam)
#   sam_vit_b_01ec64.pth   ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912  (dl.fbaipublicfiles.com)
#   MobileSAM weights/mobile_sam.pt 6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f
# EdgeSAM imports mmdet/mmengine only for its unused RPN head: stub modules on PYTHONPATH suffice
# (mmdet.models.dense_heads.{RPNHead,CenterNetUpdateHead}, mmdet.models.necks.FPN,
#  mmengine.ConfigDict, projects.EfficientDet.efficientdet); pip: einops loralib yacs omegaconf

S="systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD"
for v in edgesam mobilesam efficientvit_sam_l0; do
  $S python sam.py export $v          # enc.sim / dec.sim / enc.fp16 (uint8 NHWC in), exactness check
  $S python sam.py ref $v             # fp32 embeddings + masks for calibration/eval images
  $S python sam.py quantize $v --policy a16            # also: int8 --method percentile|mse, dw16, ...
  $S python sam.py host $v            # host accuracy of every quantized/approximated encoder
  python sam.py phone $v              # phone runs (takes ~/.cache/android-phone/phone-run's lock)
done
python sam.py export mobilesam --gelu sigmoid; python sam.py export efficientvit_sam_l0 --upsample bilinear
python sam.py report                  # the tables above
```

`phone.sh` pushes into `/data/local/tmp/codex-android-sam-variants/` and compiles an EP-context
model once per model (re-done when the model's md5 changes). Encoder compile takes 3-12 s for
EdgeSAM/EfficientViT and ~60 s for MobileSAM.

## Not done

- **EfficientSAM-Ti/S:** skipped. Its encoder is a plain ViT-Ti/S at 1024 (4096 tokens of global
  attention in every block), which is heavier than SAM ViT-B's windowed attention per block. Its
  decoder is SAM's, so the decoder numbers above apply.
- **SAM2.1-hiera-tiny:** skipped. Its Hiera encoder plus the image-only decoder need the
  sam2 repo's own export path (multi-scale high-res features into the decoder). That is a
  different decoder interface from the one above.
- **The deploy pipeline** (`../../deploy/`) has no two-piece + prompt-input spec yet. What it
  would need:
  - two models per spec, where the second takes the first's output plus prompt tensors;
  - a `prompts:` section in place of COCO-derived inputs;
  - a `mask_iou` accuracy kind.

  `sam.py` does those parts standalone.

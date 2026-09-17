"""PTQ4ViT (Yuan, Xie, Chen, Xu, Suo, Ma, 2022, ECCV 2022, "PTQ4ViT:
Post-Training Quantization for Vision Transformers with Twin Uniform
Quantization", https://arxiv.org/abs/2111.12293) -- the paper's own **twin
uniform quantization** piece specifically. onnxsim ports the *algorithm*,
not any framework's code, per the same rationale as
:mod:`onnxsim.ibert_gelu`/:mod:`onnxsim.bwa_ptq` (the paper's own reference
implementation quantizes live PyTorch modules with no ONNX export path).

**The problem twin uniform quantization targets.** Two activations inside
a Vision Transformer block have a distribution an ordinary single-scale
uniform quantizer represents badly:

- A ``Softmax`` output (the attention probabilities) lies in ``[0, 1]`` by
  construction, but is heavily concentrated near ``0`` with a thin, important
  tail near ``1`` -- the handful of tokens each query actually attends to
  strongly. :mod:`onnxsim.attention_quantization` already quantizes this
  exact tensor, but with a *fixed* ``1/255`` per-tensor scale that spends
  the same resolution everywhere in ``[0, 1]`` regardless of where the real
  mass sits -- see that module's own docstring. This module is a
  complementary, calibration-driven alternative for the same tensor, not a
  replacement for it.
- A ``Gelu``/``Erf``-decomposed-GELU output is concentrated in two separate
  clusters: a small negative dip (GELU's own behavior for slightly-negative
  inputs) and a much wider spread of positive values -- an asymmetric,
  roughly bimodal shape.

A single uniform quantizer covering the whole observed range wastes most of
its levels on the sparse in-between region and under-resolves the two
regions that actually carry the data's mass.

**Twin uniform quantization, the paper's own fix**: split the value range
at one threshold ``t`` into two sub-ranges, ``[lo, t]`` and ``[t, hi]``, and
quantize each with its *own* independent uniform quantizer (own scale, own
zero point) at the same per-side bit width -- doubling the usable
resolution exactly where the real distribution concentrates its mass, at
the cost of one extra bit of selector information per element (which side
of ``t`` it falls on).

This module finds ``t`` (and each side's ``lo``/``hi``) directly from
calibration data by a small grid search minimizing the mean squared
reconstruction error twin quantization introduces, the same
"search a scalar threshold against a directly-measured reconstruction
error" idea :func:`onnxsim.calibration._mse_threshold` already uses for
ordinary single-scale calibration -- **not** the paper's own reported
split-point/percentile constants, which this module does not try to
reproduce (this project has previously shipped a wrong recalled numeric
constant that only direct verification caught -- see
:mod:`onnxsim.ibert_gelu`'s own docstring for the precedent this follows).
See ``tests/test_ptq4vit.py`` for the empirical check that twin
quantization actually beats an equal-per-side-resolution single quantizer
on synthetic post-Softmax/post-GELU-shaped data.

**Where this is applied**: right after a standalone ``Softmax`` node's
output, and right after a standalone ``Gelu`` node's output or the final
``Mul`` of the standard ``0.5 * x * (1 + Erf(x / sqrt(2)))`` GELU export
decomposition (matched structurally: an ``Erf`` node feeding an ``Add``,
feeding a ``Mul``, feeding a second ``Mul`` -- the same decomposition
:mod:`onnxsim.ibert_gelu` targets, though that module rewrites ``Erf``
itself rather than wrapping the whole GELU's output). The twin
quantize/dequantize is inserted as new ``Less``/``Where``/``Sub``/``Div``/
``Round``/``Clip``/``Mul``/``Add`` nodes immediately after the matched
node's own output, exactly like :mod:`onnxsim.attention_quantization`'s own
Softmax-output quantization and :func:`onnxsim.calibration.quantize_static`'s
QDQ insertion -- the rest of the graph is untouched. Everything computes in
ordinary float32; onnxsim has no lower-than-float32 arithmetic ONNX op, so
this represents the *twin-uniform-ness* of the scheme (two independent
scale/zero-point pairs, selected per element) rather than a literal packed
sub-byte storage format, the same simplification :mod:`onnxsim.ibert_gelu`
documents for its own polynomial.

**What this module does not claim to reproduce**: the paper's own separate
Hessian-guided search for ordinary (non-twin) per-channel *weight*
quantization scales elsewhere in the network -- that is a different piece
of the paper's full pipeline and out of scope here; only the twin-uniform
quantization of Softmax/GELU *activations* is ported. Also not reproduced:
the paper's own reported ViT/DeiT/Swin end-task accuracy numbers, and any
literal low-bit hardware storage format -- see above.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors

_EPS = 1e-12


def _twin_quantize_dequantize(
    values: np.ndarray, lo: float, split: float, hi: float, n_levels: int
) -> np.ndarray:
    """Simulates the exact reconstruction the ONNX graph this module
    inserts will compute: independently quantize-dequantize ``values <=
    split`` against ``[lo, split]`` and ``values > split`` against
    ``[split, hi]``, each to ``n_levels`` uniform levels. Used both by the
    calibration search below and directly verifiable against numpy in
    tests (onnxruntime is not bit-exact across CPU architectures, so a
    tight numeric assertion belongs here, not in an onnxruntime round
    trip).
    """
    scale_lo = max(split - lo, _EPS) / (n_levels - 1)
    scale_hi = max(hi - split, _EPS) / (n_levels - 1)
    q_lo = np.clip(np.round((values - lo) / scale_lo), 0, n_levels - 1)
    dq_lo = q_lo * scale_lo + lo
    q_hi = np.clip(np.round((values - split) / scale_hi), 0, n_levels - 1)
    dq_hi = q_hi * scale_hi + split
    return np.where(values <= split, dq_lo, dq_hi)


def _single_uniform_quantize_dequantize(
    values: np.ndarray, lo: float, hi: float, n_levels: int
) -> np.ndarray:
    """An ordinary single-scale uniform quantizer over ``[lo, hi]`` -- the
    baseline twin uniform quantization is compared against, both by
    :func:`_search_twin_split` (to confirm splitting is actually worth it
    on this tensor's data before touching the graph) and by
    ``tests/test_ptq4vit.py`` (to confirm the win empirically rather than
    assuming it by construction).
    """
    scale = max(hi - lo, _EPS) / (n_levels - 1)
    q = np.clip(np.round((values - lo) / scale), 0, n_levels - 1)
    return q * scale + lo


def _search_twin_split(
    values: np.ndarray,
    lo: float,
    hi: float,
    n_levels: int = 256,
    num_candidates: int = 97,
) -> Optional[float]:
    """Grid search over candidate split points ``t`` strictly between
    ``lo`` and ``hi``, minimizing the mean squared error
    :func:`_twin_quantize_dequantize` introduces against ``values`` --
    PTQ4ViT's own "search the split minimizing reconstruction error"
    idea, applied directly (see this module's own docstring) rather than
    via the paper's own reported percentile constants.

    The bar a candidate split has to clear is not an ordinary single
    quantizer at the *same* per-side level count (twin quantization,
    spending one extra selector bit, has roughly double that quantizer's
    raw level count and would then win on almost any data, which would be
    an unfair, not-really-informative comparison) but one at ``2 *
    n_levels`` levels -- the *equal total bit budget* comparison (one
    extra bit spent uniformly on every level, vs. spent on a side
    selector). ``tests/test_ptq4vit.py`` verifies directly that this bar
    is only cleared by a wide margin on distributions with real
    concentration to exploit (a skewed Beta or a bimodal mixture), and is
    roughly a toss-up on a flat distribution with nothing to exploit --
    twin quantization is not a free win "by construction".

    Returns ``None`` when no split clears that bar on this data (e.g.
    ``values`` too small/degenerate to search meaningfully, or a
    distribution flat enough that a split buys nothing beyond what the
    extra bit alone already would) -- the caller then leaves that tensor
    unquantized by this module rather than inserting machinery that would
    only add complexity for no measured benefit.
    """
    v = values.astype(np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 * n_levels or hi - lo <= _EPS:
        return None

    baseline = _single_uniform_quantize_dequantize(v, lo, hi, 2 * n_levels)
    baseline_mse = float(np.mean((v - baseline) ** 2))

    candidates = np.linspace(lo, hi, num_candidates + 2)[1:-1]
    best_t: Optional[float] = None
    best_mse = baseline_mse
    for t in candidates:
        recon = _twin_quantize_dequantize(v, lo, float(t), hi, n_levels)
        mse = float(np.mean((v - recon) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_t = float(t)
    return best_t


def apply_ptq4vit_quantization(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    n_levels: int = 256,
) -> onnx.ModelProto:
    """Applies PTQ4ViT's own twin uniform quantization (see this module's
    docstring) to every standalone ``Softmax`` output and every GELU
    output (a standalone ``Gelu`` node, or the standard ``Erf``-decomposed
    GELU's final ``Mul``) in ``model``.

    For each matched tensor, ``model`` is run over ``calibration_data``
    (falling back to :func:`onnxsim.calibration.generate_random_calibration_data`
    when omitted, the same default every other calibration-based
    ``quantize_*``/``apply_*`` function in this package uses) to observe
    its actual values, then :func:`_search_twin_split` finds the split
    point minimizing reconstruction error directly against those observed
    values. A tensor whose search finds no split that beats a single
    ordinary uniform quantizer (see :func:`_search_twin_split`) is left
    unquantized by this module.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to observe each
            matched tensor's real values from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching the model's graph
            inputs -- see :func:`onnxsim.calibration.generate_random_calibration_data`
            (the default, a quick smoke test) and
            :func:`onnxsim.calibration.load_huggingface_calibration_data`
            (real data, a much better calibration source for real
            deployment).
    :param num_calibration_samples: number of random batches to generate
            when ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param n_levels: number of uniform levels *each* sub-quantizer uses
            (default 256, an 8-bit round trip per side -- the paper's own
            "twin" scheme costs one extra selector bit versus a single
            8-bit quantizer at this setting)
    :returns: ``model`` with every matched tensor's consuming nodes
            rewired to read the twin-uniform quantize-dequantize round
            trip of it instead. A model with no matching ``Softmax``/GELU
            pattern, or an opset older than 11 (the 3-input ``Clip`` form
            this module's quantize-dequantize round trip needs), is
            returned unchanged.

    Delegates to the verified C++ port
    (:func:`onnxsim.apply_ptq4vit_quantization_cpp`), which reimplements
    this function's own candidate matching, per-tensor value capture, and
    :func:`_search_twin_split`-equivalent grid search exactly (a bounded,
    deterministic search with no RNG -- see ``ptq4vit_entry.h`` for the
    full scope). This function's own former pure-Python graph-rewrite
    implementation (``_find_softmax_targets``/``_find_gelu_targets``/
    ``_insert_twin_quantize``) is preserved as-is in this module's own git
    history; :func:`_search_twin_split`/:func:`_twin_quantize_dequantize`/
    :func:`_single_uniform_quantize_dequantize` above remain (still
    directly unit-tested by ``tests/test_ptq4vit.py`` at the pure-math
    level, independent of this function).
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    from onnxsim.onnx_simplifier import apply_ptq4vit_quantization_cpp

    return apply_ptq4vit_quantization_cpp(
        model,
        calibration_data=calibration_data,
        num_calibration_samples=num_calibration_samples,
        seed=seed,
        providers=providers,
        n_levels=n_levels,
    )

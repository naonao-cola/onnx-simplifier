"""BRECQ (Li, Gong, Tan, Yang, Hu, Zhang, Yu, Wang, Gu, 2021, "BRECQ: Pushing
the Limit of Post-Training Quantization by Block Reconstruction",
https://arxiv.org/abs/2102.05426, ICLR 2021).

Read :mod:`onnxsim.adaround` first -- this module extends its own
rectified-sigmoid rounding relaxation directly, in exactly the sense BRECQ's
own paper frames its contribution: not a new relaxation mechanism, but a
different *objective* to optimize it against.

Every reconstruction-based pass already in onnxsim
(:mod:`onnxsim.adaround`, :mod:`onnxsim.adaquant`, :mod:`onnxsim.gptq`,
:mod:`onnxsim.foem`) optimizes rounding **one layer at a time**, minimizing
that single layer's own output reconstruction error
(``||W_float @ x - W_quant @ x||^2``) against calibration activations. BRECQ's
own point: this is only provably layer-optimal-implies-network-optimal under
an *independence* assumption between layers that a real network never
satisfies -- a later layer's own reconstruction error is computed from an
input that already carries the *previous* layer's rounding error, so the two
errors are correlated, and independently minimizing each layer's own error is
not, in general, the choice that minimizes a multi-layer **block**'s own
final output error (the errors can reinforce instead of cancelling, exactly
the same "rounding choices interact" observation AdaRound itself makes one
level down, at the per-weight-element scale). BRECQ's fix: optimize every
layer inside a block *jointly*, by gradient descent through the whole
block's own forward computation, against the *block's own final output*
reconstruction error -- not each internal layer's output independently. The
paper also weights that reconstruction loss by a diagonal Fisher-information
approximation (each output element's own squared task-loss-gradient
magnitude) instead of plain MSE, so elements the downstream task is more
sensitive to get prioritized; a full block-scale Hessian (as
:mod:`onnxsim.gptq` affords for a *single* layer) is prohibitively expensive
here, so a cheap diagonal approximation is the paper's own compromise.

**What this module simplifies, honestly:**

- *Block definition.* The paper auto-detects blocks in real CNN/transformer
  architectures via architecture-specific heuristics (a ResNet "BasicBlock",
  a transformer encoder layer, ...). This module does not attempt that --
  arbitrary-graph block auto-detection is a much larger, architecture-specific
  problem, not a reconstruction-objective one. Instead, the caller identifies
  a block by two tensor names: ``block_input_name`` (the activation entering
  the block, e.g. its own input or a previous block's own output) and
  ``block_output_name`` (the block's own final output, e.g. a residual Add's
  own output). This module then auto-discovers, by walking the float graph
  between those two names, which quantized MatMul/Gemm layers belong to the
  block and in what order -- the caller does not have to enumerate them.
- *Block topology.* Discovery only recognizes a **linear chain** of
  quantized MatMul/Gemm layers -- each layer's own activation input must be
  exactly the previous layer's own output, with no intervening node -- plus
  an *optional* trailing residual ``Add`` whose other input is
  ``block_input_name`` itself. This covers a ResNet "BasicBlock"'s own two
  stacked convolutions plus its skip connection, or a transformer FFN's own
  up/down projection pair plus its residual, exactly the shapes the paper's
  own Section 4 targets -- but not a block with an interleaved
  normalization/activation node (BatchNorm, ReLU, LayerNorm, GELU, ...)
  between its own quantized layers. Extending discovery to walk through a
  supported allowlist of elementwise ops is future work, not attempted here;
  see this module's own tests for exactly the topology exercised.
- *Fisher-information weighting.* The paper's own diagonal Fisher estimate
  is the squared gradient of a real downstream *task* loss (e.g.
  classification cross-entropy) with respect to each block output element,
  which requires a task loss this generic ONNX setting does not have access
  to. This module instead uses each output element's own empirical variance
  across calibration samples (normalized to a mean of 1, so the overall loss
  scale stays comparable to plain MSE) as a cheap, computable proxy for "how
  much this element's own value varies, and therefore plausibly matters" --
  a real approximation of an approximation, not a claim of reproducing the
  paper's own Fisher estimate.
- *Benchmark numbers.* This module does not claim to reproduce the paper's
  own reported ImageNet/BERT accuracy gains -- only the mechanism (joint
  block reconstruction beats independent per-layer reconstruction on the
  block's own final output, on a toy scenario engineered to make the two
  differ), verified empirically in ``tests/test_brecq.py`` the same way
  :mod:`onnxsim.foem`'s own docstring scopes its measured (not assumed)
  improvement. The paper's own claim is modest -- an incremental gain on top
  of AdaRound, not a dramatic one -- and this module's own tests confirm
  only that same modest, honest, measured direction: on a scenario
  engineered so a residual block's two layers' errors interact, jointly
  optimizing the whole block against its own final output measurably beats
  optimizing each layer independently (via :func:`onnxsim.apply_adaround`)
  against that same final output; it is not a claim that joint block
  optimization always wins by a wide margin, or at all on every topology.

Targets the exact same scheme :mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/
:mod:`onnxsim.foem` do (:func:`onnxsim.quantize_weight_only_int4`'s
block-wise symmetric INT4 MatMul/Gemm), reusing
:func:`onnxsim.adaround._find_int4_matmul_candidates` to locate individual
layer candidates and :func:`onnxsim.adaround._h_and_dhdv` for the identical
rectified-sigmoid relaxation. Plain numpy throughout, hand-derived gradients
backpropagated through the block's own chain -- no autodiff framework,
matching every other reconstruction-based pass in this repository.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import onnx

from onnxsim.calibration import Tensors


def apply_brecq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    blocks: Sequence[Tuple[str, str]],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.1,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    fisher_eps: float = 1e-3,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes BRECQ-style joint, Fisher-weighted block reconstruction for
    every ``quantize_weight_only_int4``-quantized MatMul/Gemm chain
    delimited by ``blocks``, using real activations captured from
    ``float_model``. See this module's own docstring for the technique, its
    block-discovery contract, and what it simplifies relative to the paper.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor --
            true of every onnxsim ``quantize_*`` function.
    :param blocks: ``(block_input_name, block_output_name)`` pairs, one per
            residual/linear block to jointly optimize -- see this module's
            own docstring for exactly which topologies between those two
            tensor names are recognized. A pair whose topology isn't
            recognized (or that matches no quantized layer at all) is
            silently skipped, the same "no matching candidate" tolerance
            :func:`onnxsim.apply_adaround` and friends already have.
    :param calibration_data: representative input batches to optimize the
            rounding on. Each batch is a ``{input_name: np.ndarray}`` dict
            matching ``float_model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative optimization target than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: Adam steps to run per block
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation, same role as :func:`onnxsim.apply_adaround`'s
            parameter of the same name
    :param reg_param: weight of the regularization term that pulls each
            element's relaxation toward a hard 0/1 (floor/ceil) decision,
            applied independently per layer inside the block
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the regularization
            term's exponent, linearly annealed across the iterations after
            ``warm_start``
    :param fisher_eps: numerical floor added to the empirical per-element
            variance before normalizing it into the Fisher-diagonal weight
            (see this module's own docstring) -- keeps a near-constant
            output element (near-zero variance) from collapsing to a
            near-zero loss weight and losing all optimization signal
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every discovered block's layers'
            INT4 weight initializers rewritten to their jointly-optimized
            codes (same shape, dtype, and scale -- only which integer each
            element rounds to changes)

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_brecq_cpp`) -- full parameter parity, no
    functionality gap (BRECQ has no :func:`onnxsim.apply_adaround`-style
    ``step_providers`` accelerator path to preserve a pure-Python
    candidate-matching loop for). Same accepted-numerical-scope class as
    :func:`onnxsim.apply_adaround` itself (see that function's own
    docstring): this is an iterative Adam optimization, not a closed-form
    computation, so floating-point summation-order differences between the
    C++ port's own scalar dense-matmul kernels and this module's former
    in-process numpy loop can compound across iterations -- measured
    (tests/test_brecq_cpp.py) to agree exactly in every configuration that
    test file exercises, but (matching AdaRound's own documented
    possibility, since the joint block Adam loop here is the same
    numerical class) not guaranteed to on every input. Imported lazily
    (inside the function body, not at module scope), matching
    :func:`onnxsim.apply_adaround`'s own identical precedent, to avoid a
    module-load-time import cycle with ``onnxsim.onnx_simplifier``.
    """
    from onnxsim.onnx_simplifier import apply_brecq_cpp

    return apply_brecq_cpp(
        float_model,
        quantized_model,
        blocks,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        reg_param=reg_param,
        warm_start=warm_start,
        beta_range=beta_range,
        fisher_eps=fisher_eps,
        providers=providers,
    )

"""Canonicalizes ``MatMul`` into ``Gemm`` so N:M-pruned weights are actually
eligible for ONNX Runtime's TensorRT execution provider sparse math.

The TensorRT execution provider (``ORT_TENSORRT_SPARSITY_ENABLE=1``) dispatches
a weight to NVIDIA's Sparse Tensor Cores (Ampere+, via cuSPARSELt) whenever it
finds a valid 2:4 structured-sparse pattern -- exactly what
:func:`onnxsim.apply_magnitude_pruning`/:func:`onnxsim.apply_wanda_pruning`
produce with ``n=2, m=4``. No further weight-side work is needed to make that
happen. The catch is that TensorRT's N:M sparse math only ever applies to
``Gemm`` nodes, not ``MatMul``
(https://github.com/NVIDIA/TensorRT/issues/2271) -- and most exported
transformer FFN/attention projections are ``MatMul``, not ``Gemm``, because
their activation is 3-D (``[batch, seq, hidden]``) and ONNX's ``Gemm`` only
ever accepts 2-D operands (no batch dimension), unlike ``MatMul``'s
numpy-style batched semantics. So a correctly 2:4-pruned model's weights can
still silently get dense math from TensorRT, simply because they're wired up
through the wrong op.

:func:`convert_matmul_to_gemm` closes that gap: for every ``MatMul(X, W)``
whose ``W`` is a constant 2-D float32 initializer ``[K, N]`` (exactly what
onnxsim's pruning passes already require to match a layer at all), it
rewrites the node into an equivalent ``Gemm``. Because ``W`` never carries a
batch dimension of its own, ``MatMul``'s batched semantics reduce to "flatten
every leading dimension of ``X`` into one, do a single 2-D matmul against
``W``, unflatten the result back" -- a value-preserving rewrite regardless of
``X``'s rank (including a plain 2-D ``X``, where it degenerates to a direct
swap with no reshaping at all, and even a 1-D ``X``, matching MatMul's
"promote to a row vector, then drop it again" convention). Shape inference is
used only to *detect* the already-2-D case cheaply, taking the zero-overhead
``Gemm(X, W)`` swap in the very common case where every attention/FFN
projection's activation is 2-D at the point it's called; when the rank isn't
statically known, or is known to be higher, a small ``Reshape``/``Shape``/
``Slice``/``Concat``/``Reshape`` scaffold flattens and unflattens around the
``Gemm`` -- more nodes, but still exact, and still enough to make the
underlying weight eligible for TensorRT's sparse kernel.

This is a pure graph-shape rewrite: it does not change what the model
computes, does not touch weight values at all (pruned or not), and composes
with (should run after) :func:`onnxsim.apply_magnitude_pruning`/
:func:`onnxsim.apply_wanda_pruning`'s N:M mode in a
prune -> canonicalize -> export -> deploy-with-``ORT_TENSORRT_SPARSITY_ENABLE=1``
pipeline. Folding a following bias ``Add`` into ``Gemm``'s own optional third
input is intentionally left out of scope -- TensorRT's own graph optimizer
already fuses a trailing bias add into the Gemm it compiles, so there is
nothing this pass needs to do for that.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference

from onnxsim.bias_correction import _all_names, _unique_name


def _max_opset(model: onnx.ModelProto) -> int:
    return max(
        (imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")),
        default=0,
    )


def _static_rank(
    value_info_by_name: Dict[str, onnx.ValueInfoProto], name: str
) -> Optional[int]:
    vi = value_info_by_name.get(name)
    if vi is None or not vi.type.HasField("tensor_type"):
        return None
    tensor_type = vi.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None  # known dtype, but shape inference couldn't pin down a rank
    return len(tensor_type.shape.dim)


def convert_matmul_to_gemm(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    """Rewrites every eligible ``MatMul(X, W)`` into an equivalent ``Gemm``,
    so a 2:4-pruned ``W`` becomes eligible for ONNX Runtime's TensorRT
    execution provider sparse math (``ORT_TENSORRT_SPARSITY_ENABLE=1``),
    which only ever applies to ``Gemm``. See this module's own docstring for
    the technique and why it's needed.

    :param model: the original onnx ModelProto or file path
    :returns: ``model`` with every matched ``MatMul`` replaced in place by a
            ``Gemm`` (directly, when ``X`` is provably already 2-D, or
            wrapped in a flatten/unflatten ``Reshape`` scaffold otherwise);
            requires opset >= 13 (the scaffold's ``Slice`` needs its inputs,
            not attributes) -- below that, or for anything not matching (a
            non-constant, non-2-D, or non-float32 weight), the node is left
            completely untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    if _max_opset(out) < 13:
        return out

    initializer_map = {t.name: t for t in graph.initializer}
    matmul_nodes = [
        n for n in graph.node if n.op_type == "MatMul" and len(n.input) == 2
    ]
    if not matmul_nodes:
        return out

    value_info_by_name = {}
    try:
        inferred = onnx.shape_inference.infer_shapes(out)
        for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
            value_info_by_name[vi.name] = vi
    except Exception:
        pass  # rank unknown for every tensor -- every node falls back to the general scaffold

    taken_names = _all_names(graph)

    for node in matmul_nodes:
        x_name, w_name = node.input[0], node.input[1]
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        k, n_channels = w_init.dims[0], w_init.dims[1]
        out_name = node.output[0]
        node_idx = next(i for i, nd in enumerate(graph.node) if nd is node)

        if _static_rank(value_info_by_name, x_name) == 2:
            # X is already the [M, K] Gemm expects -- a direct, zero-overhead swap.
            new_node = onnx.helper.make_node(
                "Gemm",
                [x_name, w_name],
                [out_name],
                name=_unique_name(f"{node.name or out_name}_gemm", taken_names),
            )
            graph.node[node_idx].CopyFrom(new_node)
            continue

        # General case: flatten every leading dim of X into one, Gemm, then
        # unflatten back to X's leading dims + N. Exact for any rank(X) >= 1
        # (including still-unknown rank) because W carries no batch dim of
        # its own for MatMul's batched semantics to broadcast against.
        flat_shape_name = _unique_name(f"{x_name}_flat_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([-1, k], dtype="int64"), name=flat_shape_name
            )
        )
        flat_x_name = _unique_name(f"{x_name}_flat", taken_names)
        reshape_flat = onnx.helper.make_node(
            "Reshape",
            [x_name, flat_shape_name],
            [flat_x_name],
            name=_unique_name(f"{x_name}_flatten", taken_names),
        )

        gemm_out_name = _unique_name(f"{out_name}_flat", taken_names)
        gemm_node = onnx.helper.make_node(
            "Gemm",
            [flat_x_name, w_name],
            [gemm_out_name],
            name=_unique_name(f"{node.name or out_name}_gemm", taken_names),
        )

        x_shape_name = _unique_name(f"{x_name}_shape", taken_names)
        shape_node = onnx.helper.make_node(
            "Shape",
            [x_name],
            [x_shape_name],
            name=_unique_name(f"{x_name}_shape_of", taken_names),
        )

        slice_starts_name = _unique_name(f"{x_name}_leading_starts", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([0], dtype="int64"), name=slice_starts_name
            )
        )
        slice_ends_name = _unique_name(f"{x_name}_leading_ends", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([-1], dtype="int64"), name=slice_ends_name
            )
        )
        leading_shape_name = _unique_name(f"{x_name}_leading_shape", taken_names)
        slice_node = onnx.helper.make_node(
            "Slice",
            [x_shape_name, slice_starts_name, slice_ends_name],
            [leading_shape_name],
            name=_unique_name(f"{x_name}_leading_shape_of", taken_names),
        )

        n_shape_name = _unique_name(f"{out_name}_n_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([n_channels], dtype="int64"), name=n_shape_name
            )
        )
        out_shape_name = _unique_name(f"{out_name}_shape", taken_names)
        concat_node = onnx.helper.make_node(
            "Concat",
            [leading_shape_name, n_shape_name],
            [out_shape_name],
            axis=0,
            name=_unique_name(f"{out_name}_shape_of", taken_names),
        )

        reshape_unflat = onnx.helper.make_node(
            "Reshape",
            [gemm_out_name, out_shape_name],
            [out_name],
            name=_unique_name(f"{x_name}_unflatten", taken_names),
        )

        new_nodes = [
            reshape_flat,
            gemm_node,
            shape_node,
            slice_node,
            concat_node,
            reshape_unflat,
        ]
        del graph.node[node_idx]
        for offset, nn in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, nn)

    return out

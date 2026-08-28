"""Shared LoRA graph-surgery helpers used by inject_lora.py,
apply_lora_adapter.py and prepare_qlora.py.

Grafts a low-rank adapter branch onto a frozen MatMul/Gemm/1x1-Conv weight
without touching the original node or any downstream consumer:

    Y = base_linear(X, W)                      # unchanged, W stays frozen
    Y += (alpha / rank) * (X @ lora_A @ lora_B) # new, small, trainable

lora_A is Kaiming-normal, lora_B is zero, so injecting is a no-op until the
adapter is actually trained -- the standard LoRA initialization. Eligible
layers:

  - MatMul: any 2-D initializer as input[1]
  - Gemm: any 2-D initializer as input[1] (any transA/transB)
  - Conv: a 4-D initializer as input[1] with a 1x1 kernel and group=1
    (a pointwise conv is just a per-pixel linear layer over channels)
"""

import numpy as np
from onnx import helper, numpy_helper


def _attr_value(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _target_info(initializers, node):
    """Return (initializer, kind, extra) for an eligible node, or None.
    kind is "linear" (MatMul/Gemm, extra={"transA", "transB"}) or "conv1x1"
    (extra={})."""
    if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
        init = initializers.get(node.input[1])
        if init is None or len(init.dims) != 2:
            return None
        transA = bool(_attr_value(node, "transA", 0))
        transB = bool(_attr_value(node, "transB", 0))
        return init, "linear", {"transA": transA, "transB": transB}
    if node.op_type == "Conv" and len(node.input) >= 2:
        init = initializers.get(node.input[1])
        if init is None or len(init.dims) != 4:
            return None
        if tuple(init.dims[2:4]) != (1, 1):
            return None
        if _attr_value(node, "group", 1) != 1:
            return None
        return init, "conv1x1", {}
    return None


def find_targets(model, name_contains=None, exact_names=None):
    """Eligible MatMul/Gemm/1x1-Conv nodes (see module docstring). Returns
    [(node, initializer, kind, extra)]. `exact_names` (exact initializer
    name match) takes precedence over `name_contains` (substring match,
    empty/None means "match all")."""
    initializers = {i.name: i for i in model.graph.initializer}
    targets = []
    for node in model.graph.node:
        info = _target_info(initializers, node)
        if info is None:
            continue
        init, kind, extra = info
        if exact_names is not None:
            if init.name not in exact_names:
                continue
        elif name_contains and not any(s in init.name for s in name_contains):
            continue
        targets.append((node, init, kind, extra))
    return targets


def lora_param_names(weight_name):
    return f"{weight_name}.lora_A", f"{weight_name}.lora_B"


def inject(
    model, name_contains, rank, alpha, lora_values=None, seed=0, exact_names=None
):
    """Add a LoRA branch to every targeted node.

    lora_values, when given, maps weight name -> (A, B) numpy arrays to use
    verbatim (grafting a trained adapter); otherwise A/B are freshly
    initialized (preparing a model for training).

    Returns the list of (lora_A_name, lora_B_name) pairs added, one per
    targeted weight.
    """
    rng = np.random.default_rng(seed)
    targets = find_targets(model, name_contains, exact_names)

    added = []
    for node, init, kind, extra in targets:
        w = numpy_helper.to_array(init)
        dtype = w.dtype

        if kind == "linear":
            transB = extra["transB"]
            in_dim, out_dim = (
                (w.shape[1], w.shape[0]) if transB else (w.shape[0], w.shape[1])
            )
            a_shape, b_shape = (in_dim, rank), (rank, out_dim)
        else:  # conv1x1: weight is [out_channels, in_channels, 1, 1]
            out_channels, in_channels = w.shape[0], w.shape[1]
            in_dim = in_channels
            a_shape, b_shape = (rank, in_channels, 1, 1), (out_channels, rank, 1, 1)

        a_name, b_name = lora_param_names(init.name)
        if lora_values is not None:
            a = np.asarray(lora_values[init.name][0], dtype=dtype)
            b = np.asarray(lora_values[init.name][1], dtype=dtype)
        else:
            a = (rng.standard_normal(a_shape) / np.sqrt(in_dim)).astype(dtype)
            b = np.zeros(b_shape, dtype=dtype)

        model.graph.initializer.append(numpy_helper.from_array(a, a_name))
        model.graph.initializer.append(numpy_helper.from_array(b, b_name))

        x_name = node.input[0]
        orig_out = node.output[0]
        base_out_name = f"{init.name}.lora_base_out"
        h_name = f"{init.name}.lora_h"
        delta_name = f"{init.name}.lora_delta"

        # Redirect the original node's output to an intermediate tensor and
        # add the low-rank delta back under the original tensor name, so
        # every downstream consumer sees the same output name as before.
        node.output[0] = base_out_name

        new_nodes = []
        if kind == "linear":
            if extra["transA"]:
                # Gemm's op(A) = A^T when transA is set; feed the branch
                # the same effective input the original node multiplies.
                x_eff = f"{init.name}.lora_x_transposed"
                new_nodes.append(
                    helper.make_node(
                        "Transpose",
                        [x_name],
                        [x_eff],
                        perm=[1, 0],
                        name=f"{init.name}/lora_transpose_x",
                    )
                )
                x_name = x_eff
            new_nodes += [
                helper.make_node(
                    "MatMul",
                    [x_name, a_name],
                    [h_name],
                    name=f"{init.name}/lora_A_matmul",
                ),
                helper.make_node(
                    "MatMul",
                    [h_name, b_name],
                    [delta_name],
                    name=f"{init.name}/lora_B_matmul",
                ),
            ]
        else:  # conv1x1
            # The first conv reproduces the original node's spatial striding
            # (in_channels -> rank); the second is a plain 1x1/stride-1
            # channel mix (rank -> out_channels) since the first conv
            # already applied any subsampling.
            carried = [
                a
                for a in node.attribute
                if a.name in ("strides", "pads", "auto_pad", "dilations")
            ]
            conv_a = helper.make_node(
                "Conv",
                [x_name, a_name],
                [h_name],
                name=f"{init.name}/lora_A_conv",
                kernel_shape=[1, 1],
            )
            conv_a.attribute.extend(carried)
            new_nodes += [
                conv_a,
                helper.make_node(
                    "Conv",
                    [h_name, b_name],
                    [delta_name],
                    name=f"{init.name}/lora_B_conv",
                    kernel_shape=[1, 1],
                ),
            ]

        scale = alpha / rank
        add_input = delta_name
        if scale != 1.0:
            scale_name = f"{init.name}.lora_scale"
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(scale, dtype=dtype), scale_name)
            )
            scaled_name = f"{init.name}.lora_scaled"
            new_nodes.append(
                helper.make_node(
                    "Mul",
                    [delta_name, scale_name],
                    [scaled_name],
                    name=f"{init.name}/lora_scale_mul",
                )
            )
            add_input = scaled_name
        new_nodes.append(
            helper.make_node(
                "Add",
                [base_out_name, add_input],
                [orig_out],
                name=f"{init.name}/lora_add",
            )
        )

        insert_at = list(model.graph.node).index(node) + 1
        for offset, n in enumerate(new_nodes):
            model.graph.node.insert(insert_at + offset, n)

        added.append((a_name, b_name))

    return added

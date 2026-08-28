"""Shared LoRA graph-surgery helpers used by inject_lora.py and
apply_lora_adapter.py.

Grafts a low-rank adapter branch onto a frozen MatMul/Gemm weight without
touching the original node or any downstream consumer:

    Y = base_linear(X, W)                      # unchanged, W stays frozen
    Y += (alpha / rank) * (X @ lora_A @ lora_B) # new, small, trainable

lora_A is Kaiming-normal, lora_B is zero, so injecting is a no-op until the
adapter is actually trained -- the standard LoRA initialization.
"""

import numpy as np
from onnx import helper, numpy_helper


def find_targets(model, name_contains=None, exact_names=None):
    """Eligible MatMul/Gemm nodes: op has a 2-D initializer as input[1], and
    (for Gemm) does not transpose input[0]. Returns [(node, initializer,
    transB)]. `exact_names` (exact initializer name match) takes precedence
    over `name_contains` (substring match, empty/None means "match all")."""
    initializers = {i.name: i for i in model.graph.initializer}
    targets = []
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm") or len(node.input) < 2:
            continue
        init = initializers.get(node.input[1])
        if init is None or len(init.dims) != 2:
            continue
        if exact_names is not None:
            if init.name not in exact_names:
                continue
        elif name_contains and not any(s in init.name for s in name_contains):
            continue
        transA, transB = False, False
        for attr in node.attribute:
            if attr.name == "transA":
                transA = bool(attr.i)
            elif attr.name == "transB":
                transB = bool(attr.i)
        if transA:
            continue  # unusual for a linear layer; skip rather than guess
        targets.append((node, init, transB))
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
    for node, init, transB in targets:
        w = numpy_helper.to_array(init)
        dtype = w.dtype
        in_dim, out_dim = (
            (w.shape[1], w.shape[0]) if transB else (w.shape[0], w.shape[1])
        )
        a_name, b_name = lora_param_names(init.name)

        if lora_values is not None:
            a = np.asarray(lora_values[init.name][0], dtype=dtype)
            b = np.asarray(lora_values[init.name][1], dtype=dtype)
        else:
            a = (rng.standard_normal((in_dim, rank)) / np.sqrt(in_dim)).astype(dtype)
            b = np.zeros((rank, out_dim), dtype=dtype)

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

        new_nodes = [
            helper.make_node(
                "MatMul", [x_name, a_name], [h_name], name=f"{init.name}/lora_A_matmul"
            ),
            helper.make_node(
                "MatMul",
                [h_name, b_name],
                [delta_name],
                name=f"{init.name}/lora_B_matmul",
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

"""Attaches/reads a custom WebGPU compute kernel (WGSL source plus a static
dispatch/bindings description) on a specific node's own ``metadata_props``,
so a downstream browser runtime can execute that node with a hand-written
WebGPU kernel instead of whatever onnxruntime-web's own WebGPU execution
provider would otherwise run for it.

This is the metadata half of "run a custom WebGPU kernel from the model" --
see ``scripts/convertmodel/onnx_node_metadata.mjs`` for the browser-side
reader this schema round-trips through (a hand-rolled protobuf reader, not a
full onnx.js port -- see that file's own docstring for why, and for the
exact field numbers this schema depends on staying stable, which protobuf's
own backward-compatibility rules guarantee), and
``scripts/convertmodel/webgpu_kernel_dispatcher.mjs`` for the code that
actually compiles and dispatches the WGSL against real GPU buffers.

**What this does not do (yet):** splice the custom kernel into an
onnxruntime-web session -- i.e. there is no graph splitter that runs an ORT
session up to the flagged node, executes the WGSL kernel via this metadata,
and resumes another ORT session past it. That "runtime on top of ort-web"
piece needs its own design (most likely handing GPU buffers between an
ORT-web session and a hand-dispatched kernel via onnxruntime-web's
GPU-buffer IO binding, to avoid a CPU round-trip) and is not built here.
What *is* here and works end to end today: a schema to describe a kernel,
and a standalone way to read one back out of a real ``.onnx`` file's bytes
and run it against arbitrary GPU buffers -- directly useful for "run this
WGSL kernel on this data" testing and tuning, and the foundation the
graph-splicing runtime would build on next.

## Schema

One ``metadata_props`` entry per node that has a custom kernel, keyed
``"onnxsim.webgpu_kernel"`` (``onnxsim.model_info.METADATA_PREFIX`` +
``"webgpu_kernel"``), JSON-valued::

    {
      "wgsl": "<WGSL source text>",
      "entry_point": "<the WGSL @compute function name>",
      "dispatch": [x, y, z],          // workgroup counts, static only
      "bindings": [
        {"tensor": "<node input/output name>", "group": 0, "binding": 0,
         "access": "read" | "read_write"},
        ...
      ]
    }

``dispatch`` is a fixed ``[x, y, z]`` triple, not a formula over input
shapes -- a kernel whose dispatch size depends on a dynamic input shape
isn't expressible yet; onnxsim does not always have shape inference
available at the point a kernel would be attached, so
:func:`attach_webgpu_kernel` does not attempt to derive one itself.

``bindings`` conceptually lines up with the WGSL side's own
``@group(g) @binding(b)`` declarations, but ``group``/``binding`` are stored
explicitly per entry rather than inferred from list position, so a caller
can bind tensors out of order or skip slots. Each ``tensor`` name must be
one of the node's own input or output names (:func:`attach_webgpu_kernel`
validates this) -- resolving that name to an actual GPU buffer is the
dispatcher's job, not onnxsim's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import onnx

from onnxsim.model_info import METADATA_PREFIX

_KERNEL_METADATA_KEY = METADATA_PREFIX + "webgpu_kernel"
_VALID_ACCESS = ("read", "read_write")


@dataclass(frozen=True)
class WebgpuKernelBinding:
    """One WGSL ``@group(group) @binding(binding)`` storage-buffer slot,
    bound to a node input/output tensor by name.

    :param tensor: name of the node's own input or output this binds to.
    :param group: WGSL bind group index.
    :param binding: WGSL binding index within that group.
    :param access: ``"read"`` (an input) or ``"read_write"`` (written by the
            kernel -- normally one of the node's own outputs).
    """

    tensor: str
    group: int
    binding: int
    access: str = "read"

    def to_json(self) -> dict:
        return {
            "tensor": self.tensor,
            "group": self.group,
            "binding": self.binding,
            "access": self.access,
        }

    @staticmethod
    def from_json(data: dict) -> "WebgpuKernelBinding":
        return WebgpuKernelBinding(
            tensor=data["tensor"],
            group=int(data["group"]),
            binding=int(data["binding"]),
            access=data.get("access", "read"),
        )


@dataclass(frozen=True)
class WebgpuKernelSpec:
    """A custom WebGPU compute kernel for one node -- see this module's
    docstring for the schema this (de)serializes to/from JSON.
    """

    wgsl: str
    entry_point: str
    dispatch: Tuple[int, int, int]
    bindings: Tuple[WebgpuKernelBinding, ...]

    def to_json(self) -> dict:
        return {
            "wgsl": self.wgsl,
            "entry_point": self.entry_point,
            "dispatch": list(self.dispatch),
            "bindings": [b.to_json() for b in self.bindings],
        }

    @staticmethod
    def from_json(data: dict) -> "WebgpuKernelSpec":
        dispatch = data["dispatch"]
        if len(dispatch) != 3:
            raise ValueError(f"dispatch must have exactly 3 entries, got {dispatch!r}")
        return WebgpuKernelSpec(
            wgsl=data["wgsl"],
            entry_point=data["entry_point"],
            dispatch=(int(dispatch[0]), int(dispatch[1]), int(dispatch[2])),
            bindings=tuple(WebgpuKernelBinding.from_json(b) for b in data["bindings"]),
        )


def _find_node(graph: onnx.GraphProto, node_name: str) -> onnx.NodeProto:
    for node in graph.node:
        if node.name == node_name:
            return node
    raise ValueError(f"no node named {node_name!r} in the graph")


def _set_node_metadata(node: onnx.NodeProto, key: str, value: str) -> None:
    """``node.metadata_props[key] = value``, overwriting any existing entry.

    A local copy of ``model_info._set_metadata``'s three lines (same
    precedent as ``qat_interop._set_metadata``) rather than importing a
    private helper -- ``METADATA_PREFIX`` is imported so all three stay
    consistent about where onnxsim's own metadata goes.
    """
    for entry in node.metadata_props:
        if entry.key == key:
            entry.value = value
            return
    entry = node.metadata_props.add()
    entry.key = key
    entry.value = value


def attach_webgpu_kernel(
    model: onnx.ModelProto,
    node_name: str,
    wgsl: str,
    entry_point: str,
    dispatch: Sequence[int],
    bindings: Sequence[WebgpuKernelBinding],
) -> onnx.ModelProto:
    """Attaches a custom WebGPU kernel to the node named ``node_name`` in
    ``model``, as the JSON schema documented on this module.

    ``model`` is mutated in place and also returned, for chaining.

    :param model: the model to modify; must have a node named ``node_name``.
    :param node_name: the target node's ``NodeProto.name`` -- not an output
            name, an op type, or anything else. Every node onnxsim itself
            produces has a name (see ``model_prep.h``'s node-naming pass); a
            hand-built or third-party-exported graph might not, in which
            case it has no node this can target until one is given.
    :param wgsl: the WGSL source for the kernel's ``@compute`` shader.
    :param entry_point: the WGSL entry point function name.
    :param dispatch: ``(x, y, z)`` workgroup counts -- static only, see this
            module's docstring for why.
    :param bindings: one :class:`WebgpuKernelBinding` per WGSL storage-buffer
            slot the kernel reads/writes. Every ``tensor`` named must be one
            of the node's own input or output names.
    :raises ValueError: ``node_name`` is empty, no node with that name
            exists, ``dispatch`` isn't a 3-tuple, or a binding names a
            tensor that isn't one of the node's own inputs/outputs, or an
            invalid ``access`` value.
    :returns: ``model``, mutated in place.
    """
    if not node_name:
        raise ValueError("node_name must be a non-empty NodeProto.name")
    node = _find_node(model.graph, node_name)
    if len(dispatch) != 3:
        raise ValueError(f"dispatch must have exactly 3 entries, got {dispatch!r}")
    node_tensors = set(node.input) | set(node.output)
    for b in bindings:
        if b.tensor not in node_tensors:
            raise ValueError(
                f"binding names tensor {b.tensor!r}, which is not one of "
                f"node {node_name!r}'s own inputs {list(node.input)!r} or "
                f"outputs {list(node.output)!r}"
            )
        if b.access not in _VALID_ACCESS:
            raise ValueError(
                f"binding for tensor {b.tensor!r} has access={b.access!r}, "
                f"must be one of {_VALID_ACCESS!r}"
            )

    spec = WebgpuKernelSpec(
        wgsl=wgsl,
        entry_point=entry_point,
        dispatch=(int(dispatch[0]), int(dispatch[1]), int(dispatch[2])),
        bindings=tuple(bindings),
    )
    _set_node_metadata(node, _KERNEL_METADATA_KEY, json.dumps(spec.to_json()))
    return model


def read_webgpu_kernel(
    model: onnx.ModelProto, node_name: str
) -> Optional[WebgpuKernelSpec]:
    """Reads back the :class:`WebgpuKernelSpec` :func:`attach_webgpu_kernel`
    attached to the node named ``node_name``, or ``None`` if that node has
    no such metadata (including if the node itself doesn't exist -- this is
    a read, not a validity check; use :func:`list_webgpu_kernels` to see
    what's actually there).

    :param model: the model to read from.
    :param node_name: the target node's ``NodeProto.name``.
    """
    for node in model.graph.node:
        if node.name != node_name:
            continue
        for entry in node.metadata_props:
            if entry.key == _KERNEL_METADATA_KEY:
                return WebgpuKernelSpec.from_json(json.loads(entry.value))
        return None
    return None


def list_webgpu_kernels(model: onnx.ModelProto) -> List[Tuple[str, WebgpuKernelSpec]]:
    """Every node in ``model`` with a custom WebGPU kernel attached, as
    ``(node_name, WebgpuKernelSpec)`` pairs, in graph node order.
    """
    result = []
    for node in model.graph.node:
        for entry in node.metadata_props:
            if entry.key == _KERNEL_METADATA_KEY:
                result.append(
                    (node.name, WebgpuKernelSpec.from_json(json.loads(entry.value)))
                )
                break
    return result

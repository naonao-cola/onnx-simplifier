"""Attaches/reads a custom WebGPU compute *program* (one or more WGSL kernel
steps run in sequence, with static dispatch/bindings descriptions) on a
specific node's own ``metadata_props``, so a downstream browser runtime can
execute that node with hand-written or auto-generated WebGPU kernels instead
of whatever onnxruntime-web's own WebGPU execution provider would otherwise
run for it.

This is the metadata half of "run a custom WebGPU program from the model" --
see ``scripts/convertmodel/onnx_node_metadata.mjs`` for the browser-side
reader this schema round-trips through (a hand-rolled protobuf reader, not a
full onnx.js port -- see that file's own docstring for why, and for the
exact field numbers this schema depends on staying stable, which protobuf's
own backward-compatibility rules guarantee), and
``scripts/convertmodel/webgpu_kernel_dispatcher.mjs`` for the code that
actually compiles and dispatches the WGSL against real GPU buffers.

**Why *steps* (plural), not one kernel:** :mod:`onnxsim.webgpu_tinygrad_codegen`
generates programs from real tinygrad ``Tensor`` graphs, and tinygrad's own
scheduler does not always fuse a node's computation into a single kernel --
a softmax-bearing op like Attention schedules as four separate kernels, and
even a simple ``Resize``-style interpolation schedules as two. A single
``WebgpuKernelSpec`` therefore holds an ordered list of :class:`WebgpuKernelStep`
entries, run one after another on the same ``GPUDevice``; buffers a later
step reads that an earlier one wrote, but that aren't one of the node's own
inputs/outputs, are declared once as named *intermediates* on the spec and
allocated by the dispatcher for the program's lifetime.

**What this does not do (yet):** splice the custom program into an
onnxruntime-web session -- i.e. there is no graph splitter that runs an ORT
session up to the flagged node, executes the WebGPU program via this
metadata, and resumes another ORT session past it. That "runtime on top of
ort-web" piece needs its own design (most likely handing GPU buffers between
an ORT-web session and a hand-dispatched program via onnxruntime-web's
GPU-buffer IO binding, to avoid a CPU round-trip) and is not built here.
What *is* here and works end to end today: a schema to describe a program,
and a standalone way to read one back out of a real ``.onnx`` file's bytes
and run it against arbitrary GPU buffers -- directly useful for "run this
WGSL program on this data" testing and tuning, and the foundation the
graph-splicing runtime would build on next.

## Schema

One ``metadata_props`` entry per node that has a custom program, keyed
``"onnxsim.webgpu_kernel"`` (``onnxsim.model_info.METADATA_PREFIX`` +
``"webgpu_kernel"``), JSON-valued::

    {
      "steps": [
        {
          "wgsl": "<WGSL source text>",
          "entry_point": "<the WGSL @compute function name>",
          "dispatch": [x, y, z],          // workgroup counts, static only
          "bindings": [
            {"group": 0, "binding": 0, "access": "read_write",
             "tensor": "<node input/output name>"},
            {"group": 0, "binding": 1, "access": "read_write",
             "intermediate": "<name declared in the spec's own intermediates>"},
            {"group": 0, "binding": 2, "access": "uniform",
             "constant": [<float>, ...]},
            ...
          ]
        },
        ...
      ],
      "intermediates": {"<name>": <byte length>, ...}
    }

``dispatch`` is a fixed ``[x, y, z]`` triple, not a formula over input
shapes -- a kernel whose dispatch size depends on a dynamic input shape
isn't expressible yet; onnxsim does not always have shape inference
available at the point a program would be attached, so this module does not
attempt to derive one itself.

Each :class:`WebgpuKernelBinding` names exactly one of three mutually
exclusive sources, conceptually lining up with the WGSL side's own
``@group(g) @binding(b)`` declarations (``group``/``binding`` are stored
explicitly per entry rather than inferred from list position, so a caller
can bind slots out of order or skip some):

- ``tensor``: one of the node's own input or output names -- resolving that
  name to an actual GPU buffer is the dispatcher's job, not onnxsim's.
- ``intermediate``: a name declared in the spec's own ``intermediates`` map,
  a scratch buffer that exists only for the program's own steps to pass data
  between each other (e.g. tinygrad's un-fused softmax passes). The
  dispatcher allocates one GPU buffer per declared intermediate, once, for
  the whole program's run, and frees them afterward.
- ``constant``: inline literal data with no backing tensor at all --
  ``access`` is ``"uniform"`` for these (the dispatcher creates a small
  uniform buffer from the literal values rather than looking anything up).
  tinygrad's WGSL renderer, for instance, always emits an ``INFINITY``
  uniform at binding 0 whether or not the kernel actually reads it. Values
  are JSON floats, except non-finite ones: standard JSON has no
  ``Infinity``/``NaN`` literal, so those encode as the strings
  ``"Infinity"``/``"-Infinity"``/``"NaN"`` (see :func:`_encode_float`) --
  the JS dispatcher coerces every element through ``Number(v)`` before use,
  which recognizes the same strings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import onnx

from onnxsim.model_info import METADATA_PREFIX

_KERNEL_METADATA_KEY = METADATA_PREFIX + "webgpu_kernel"
_VALID_ACCESS = ("read", "read_write", "uniform")


def _encode_float(v: float):
    """Standard JSON has no ``Infinity``/``NaN`` literal -- Python's own
    ``json.dumps`` emits the non-standard bare tokens ``Infinity``/``NaN``
    for them by default, which a strict JSON parser (JS's ``JSON.parse``,
    used on the browser side) rejects outright. ``constant`` bindings need
    exactly these values in practice -- tinygrad's WGSL renderer always
    declares an ``INFINITY`` uniform (see
    ``scripts/convertmodel/webgpu_kernel_dispatcher.mjs``'s own note) -- so
    non-finite values are encoded as string sentinels instead, decoded back
    with :func:`_decode_float`. The JS side coerces every ``constant``
    element through ``Number(v)`` before use, which recognizes these same
    strings (``Number("Infinity") === Infinity``).
    """
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


def _decode_float(v) -> float:
    return float(v)


@dataclass(frozen=True)
class WebgpuKernelBinding:
    """One WGSL ``@group(group) @binding(binding)`` slot. Exactly one of
    ``tensor``/``intermediate``/``constant`` must be set -- see this
    module's docstring for what each means.

    :param group: WGSL bind group index.
    :param binding: WGSL binding index within that group.
    :param access: ``"read"``/``"read_write"`` for a storage buffer
            (``tensor`` or ``intermediate``), ``"uniform"`` for ``constant``.
    :param tensor: a node input/output name.
    :param intermediate: a name declared in the spec's ``intermediates`` map.
    :param constant: inline literal float data, no backing tensor.
    """

    group: int
    binding: int
    access: str = "read"
    tensor: Optional[str] = None
    intermediate: Optional[str] = None
    constant: Optional[Tuple[float, ...]] = None

    @staticmethod
    def for_tensor(
        tensor: str, group: int, binding: int, access: str = "read"
    ) -> "WebgpuKernelBinding":
        return WebgpuKernelBinding(
            group=group, binding=binding, access=access, tensor=tensor
        )

    @staticmethod
    def for_intermediate(
        intermediate: str, group: int, binding: int, access: str = "read_write"
    ) -> "WebgpuKernelBinding":
        return WebgpuKernelBinding(
            group=group, binding=binding, access=access, intermediate=intermediate
        )

    @staticmethod
    def for_constant(
        constant: Sequence[float], group: int, binding: int
    ) -> "WebgpuKernelBinding":
        return WebgpuKernelBinding(
            group=group,
            binding=binding,
            access="uniform",
            constant=tuple(float(v) for v in constant),
        )

    def _source_count(self) -> int:
        return sum(
            x is not None for x in (self.tensor, self.intermediate, self.constant)
        )

    def to_json(self) -> dict:
        d = {"group": self.group, "binding": self.binding, "access": self.access}
        if self.tensor is not None:
            d["tensor"] = self.tensor
        if self.intermediate is not None:
            d["intermediate"] = self.intermediate
        if self.constant is not None:
            d["constant"] = [_encode_float(v) for v in self.constant]
        return d

    @staticmethod
    def from_json(data: dict) -> "WebgpuKernelBinding":
        constant = data.get("constant")
        return WebgpuKernelBinding(
            group=int(data["group"]),
            binding=int(data["binding"]),
            access=data.get("access", "read"),
            tensor=data.get("tensor"),
            intermediate=data.get("intermediate"),
            constant=tuple(_decode_float(v) for v in constant)
            if constant is not None
            else None,
        )


@dataclass(frozen=True)
class WebgpuKernelStep:
    """One compiled WGSL kernel within a :class:`WebgpuKernelSpec`'s program,
    run in sequence with the spec's other steps.
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
    def from_json(data: dict) -> "WebgpuKernelStep":
        dispatch = data["dispatch"]
        if len(dispatch) != 3:
            raise ValueError(f"dispatch must have exactly 3 entries, got {dispatch!r}")
        return WebgpuKernelStep(
            wgsl=data["wgsl"],
            entry_point=data["entry_point"],
            dispatch=(int(dispatch[0]), int(dispatch[1]), int(dispatch[2])),
            bindings=tuple(WebgpuKernelBinding.from_json(b) for b in data["bindings"]),
        )


@dataclass(frozen=True)
class WebgpuKernelSpec:
    """A custom WebGPU program (one or more steps) for one node -- see this
    module's docstring for the schema this (de)serializes to/from JSON.

    :param steps: the kernels to run, in order.
    :param intermediates: name -> byte length, for every buffer any step's
            bindings reference via ``intermediate`` -- the dispatcher
            allocates these once per program run.
    """

    steps: Tuple[WebgpuKernelStep, ...]
    intermediates: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def single_step(
        wgsl: str,
        entry_point: str,
        dispatch: Sequence[int],
        bindings: Sequence[WebgpuKernelBinding],
    ) -> "WebgpuKernelSpec":
        """Convenience constructor for the common case of a program with
        exactly one kernel and no intermediates -- e.g. a hand-written
        kernel, or any single-node computation tinygrad happens to fuse into
        one kernel (elementwise ops, and often ``Conv``).
        """
        if len(dispatch) != 3:
            raise ValueError(f"dispatch must have exactly 3 entries, got {dispatch!r}")
        step = WebgpuKernelStep(
            wgsl=wgsl,
            entry_point=entry_point,
            dispatch=(int(dispatch[0]), int(dispatch[1]), int(dispatch[2])),
            bindings=tuple(bindings),
        )
        return WebgpuKernelSpec(steps=(step,))

    def to_json(self) -> dict:
        return {
            "steps": [s.to_json() for s in self.steps],
            "intermediates": dict(self.intermediates),
        }

    @staticmethod
    def from_json(data: dict) -> "WebgpuKernelSpec":
        return WebgpuKernelSpec(
            steps=tuple(WebgpuKernelStep.from_json(s) for s in data["steps"]),
            intermediates={k: int(v) for k, v in data.get("intermediates", {}).items()},
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
    spec: WebgpuKernelSpec,
) -> onnx.ModelProto:
    """Attaches a custom WebGPU program to the node named ``node_name`` in
    ``model``, as the JSON schema documented on this module.

    ``model`` is mutated in place and also returned, for chaining.

    :param model: the model to modify; must have a node named ``node_name``.
    :param node_name: the target node's ``NodeProto.name`` -- not an output
            name, an op type, or anything else. Every node onnxsim itself
            produces has a name (see ``model_prep.h``'s node-naming pass); a
            hand-built or third-party-exported graph might not, in which
            case it has no node this can target until one is given.
    :param spec: the program to attach -- see :class:`WebgpuKernelSpec` and
            :meth:`WebgpuKernelSpec.single_step` for the common one-kernel
            case. Every binding's ``tensor`` must be one of the node's own
            input or output names; every binding's ``intermediate`` must be
            a key in ``spec.intermediates``.
    :raises ValueError: ``node_name`` is empty, no node with that name
            exists, ``spec`` has no steps, a binding names a tensor that
            isn't one of the node's own inputs/outputs, an undeclared
            intermediate, more than one (or none) of
            ``tensor``/``intermediate``/``constant``, or an invalid
            ``access`` value.
    :returns: ``model``, mutated in place.
    """
    if not node_name:
        raise ValueError("node_name must be a non-empty NodeProto.name")
    node = _find_node(model.graph, node_name)
    if not spec.steps:
        raise ValueError("spec must have at least one step")
    node_tensors = set(node.input) | set(node.output)
    for step in spec.steps:
        for b in step.bindings:
            if b._source_count() != 1:
                raise ValueError(
                    f"binding at group={b.group} binding={b.binding} must set exactly one "
                    f"of tensor/intermediate/constant, got {b.to_json()!r}"
                )
            if b.tensor is not None and b.tensor not in node_tensors:
                raise ValueError(
                    f"binding names tensor {b.tensor!r}, which is not one of "
                    f"node {node_name!r}'s own inputs {list(node.input)!r} or "
                    f"outputs {list(node.output)!r}"
                )
            if b.intermediate is not None and b.intermediate not in spec.intermediates:
                raise ValueError(
                    f"binding names intermediate {b.intermediate!r}, which is not "
                    f"declared in spec.intermediates {list(spec.intermediates)!r}"
                )
            if b.access not in _VALID_ACCESS:
                raise ValueError(
                    f"binding {b.to_json()!r} has access={b.access!r}, "
                    f"must be one of {_VALID_ACCESS!r}"
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
    """Every node in ``model`` with a custom WebGPU program attached, as
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

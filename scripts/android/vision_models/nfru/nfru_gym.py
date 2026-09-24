"""Load Arm's NFRU v1 (neural-graphics-model-gym, Apache-2.0) with only torch/torchvision/safetensors.

Like ../nss/nss_gym.py: empty parent packages (so the gym's heavy __init__.py files never run) plus
stand-ins for the framework modules nfru_v1.py imports (config, base model, registry, slang loader),
then NFRUv1Core with the template's test colour pipeline (exposure 2.0, reinhard) and
processing_backend="torch". Uses the same gym clone as NSS (pinned in ../nss/nss_gym.py).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

GYM = Path(os.environ.get("NSS_GYM", Path.home() / ".cache/arm-nss/gym"))
TEST_COLOR = {
    "pipeline": ["reinhard"],
    "exposure": 2.0,
    "auto_exposure_key_value": None,
    "auto_exposure_variance": None,
}
QAT_LAYER_ORDER = (  # forward (graph) order of the ConvBlocks in NFRUAutoEncoder
    "conv1",
    "conv2",
    "conv3",
    "skip1_conv",
    "conv5",
    "conv5a",
    "conv5b",
    "conv5c",
    "conv5c_1",
    "conv5d",
    "conv5d_1",
    "conv5d_2",
    "conv5e",
    "conv6",
    "conv7",
)


def _pkg(name: str) -> None:
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = [str(GYM / "src" / name.replace(".", "/"))]
    sys.modules[name] = m


def _stub(name: str, **attrs) -> None:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m


def _install() -> None:
    if "ng_model_gym.core.model.base_ng_model" in sys.modules and hasattr(
        sys.modules["ng_model_gym.core.model.base_ng_model"], "QATQuantizationProfile"
    ):
        return
    import dataclasses

    import torch

    for p in (
        "ng_model_gym",
        "ng_model_gym.core",
        "ng_model_gym.core.model",
        "ng_model_gym.core.model.layers",
        "ng_model_gym.core.model.shaders",
        "ng_model_gym.core.data",
        "ng_model_gym.core.utils",
        "ng_model_gym.core.config",
        "ng_model_gym.usecases",
        "ng_model_gym.usecases.nfru",
        "ng_model_gym.usecases.nfru.model",
        "ng_model_gym.usecases.nfru.utils",
        "ng_model_gym.usecases.nfru.data",
    ):
        _pkg(p)

    class BaseNGModel(torch.nn.Module):
        def __init__(self, params=None):
            super().__init__()
            self.params = params

    @dataclasses.dataclass
    class QATQuantizationProfile:
        per_channel_weight_quantization: bool = True

    cm = sys.modules.get("ng_model_gym.core.config.config_model")
    if cm is None:
        _stub(
            "ng_model_gym.core.config.config_model",
            ConfigModel=object,
            NFRUV1ModelSettings=object,
        )
    else:
        cm.NFRUV1ModelSettings = object
    _stub(
        "ng_model_gym.core.model.base_ng_model",
        BaseNGModel=BaseNGModel,
        QATQuantizationProfile=QATQuantizationProfile,
    )
    _stub(
        "ng_model_gym.core.model.model_registry",
        register_model=lambda **kw: (lambda c: c),
    )
    _stub(
        "ng_model_gym.core.model.shaders.slang_utils",
        load_slang_module=None,
        SlangOutput=object,
    )
    du = importlib.import_module("ng_model_gym.core.data.data_utils")
    sys.modules["ng_model_gym.core.data"].tonemap_forward = du.tonemap_forward


def build_core(weights: str = "fp32", work: Path | None = None):
    """NFRUv1Core (torch backend, test colour pipeline) with Arm's fp32 or QAT weights, eval mode."""
    import torch

    _install()
    nf = importlib.import_module("ng_model_gym.usecases.nfru.model.nfru_v1")
    core = nf.NFRUv1Core(
        {s: dict(TEST_COLOR) for s in ("train", "validation", "test")},
        processing_backend="torch",
    )
    core.set_color_pipeline("test")
    work = work or Path(os.environ.get("NFRU_WORK", Path.home() / ".cache/arm-nfru"))
    f = torch.load(work / "nfru_v1_fp32.pt", map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    sd = {k[len("network.") :]: v for k, v in f.items() if k.startswith("network.")}
    if (
        weights != "fp32"
    ):  # Arm's QAT checkpoint: the same layers as graph constants, in graph order
        q = torch.load(
            work / "nfru_v1_int8.pt", map_location="cpu", weights_only=False
        )["model_state_dict"]
        p = lambda i: q[f"network.auto_encoder._param_constant{i}"]  # noqa: E731
        t = lambda i: q[f"network.auto_encoder._tensor_constant{i}"]  # noqa: E731
        for i, name in enumerate(QAT_LAYER_ORDER):
            b = f"auto_encoder.{name}"
            sd[f"{b}.conv2d.weight"], sd[f"{b}.conv2d.bias"] = p(4 * i), p(4 * i + 1)
            sd[f"{b}.bn.weight"], sd[f"{b}.bn.bias"] = p(4 * i + 2), p(4 * i + 3)
            sd[f"{b}.bn.running_mean"], sd[f"{b}.bn.running_var"] = (
                t(2 * i),
                t(2 * i + 1),
            )
        (
            sd["auto_encoder.output_conv_mv.weight"],
            sd["auto_encoder.output_conv_mv.bias"],
        ) = p(60), p(61)
    missing, unexpected = core.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "auto_encoder" in k], missing
    return core.eval()

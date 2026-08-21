# Regression tests for GAN-generator export patterns.
#
# Real GAN checkpoints (StyleGAN2/3, CycleGAN, pix2pix, ...) are large external
# downloads, so these tests reimplement the *architectural patterns* those
# frameworks emit -- ConvTranspose+BatchNorm upsampling stacks (DCGAN-style),
# AdaIN-modulated synthesis fed by a separately exported mapping network
# (StyleGAN-style), and ReflectionPad+InstanceNorm residual blocks
# (CycleGAN/pix2pix-style) -- with small random-initialized weights, so the
# tests stay fast and fully offline while still exercising the same graph
# shapes onnxsim has to simplify in practice. Only the generator (the half of
# a GAN that is ever exported for deployment) is under test; the discriminator
# and the adversarial training loop have no ONNX representation.

import onnx
import onnxruntime
import torch

import onnxsim
from onnxsim.test_utils import export_simplify_and_check_by_python_api


def test_dcgan_generator_convtranspose_bn_fuses():
    # DCGAN-style generator: a stack of ConvTranspose2d + BatchNorm2d + ReLU
    # upsampling blocks ending in Tanh. This is the classic GAN generator
    # pattern (also used by many lightweight image-to-image GANs).
    #
    # It doubles as a regression test for fuse_bn_into_conv's ConvTranspose
    # support (onnxsim/passes/fuse_bn_into_conv.h): ConvTranspose's weight
    # layout is (in_channels, out_channels/group, kH, kW) -- out_channels on
    # axis 1, not axis 0 like Conv -- so the fusion must broadcast the folded
    # BatchNorm scale along axis 1 instead of axis 0.
    class DCGANGenerator(torch.nn.Module):
        def __init__(self, nz=16, ngf=8, nc=3):
            super().__init__()
            self.main = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                torch.nn.BatchNorm2d(ngf * 4),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(ngf * 2),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                torch.nn.Tanh(),
            )

        def forward(self, z):
            return self.main(z)

    model = DCGANGenerator().eval()
    z = torch.randn(1, 16, 1, 1)

    opt = export_simplify_and_check_by_python_api(
        model,
        (z,),
        export_kwargs={
            "opset_version": 17,
            "input_names": ["z"],
            "output_names": ["image"],
            "dynamic_axes": {"z": {0: "batch"}, "image": {0: "batch"}},
        },
    )

    op_types = [n.op_type for n in opt.graph.node]
    assert "ConvTranspose" in op_types
    # Every BatchNorm folds into the preceding ConvTranspose's weights.
    assert "BatchNormalization" not in op_types

    sess = onnxruntime.InferenceSession(
        opt.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (out,) = sess.run(None, {"z": z.numpy()})
    assert out.shape == (1, 3, 16, 16)


def test_stylegan_mapping_and_synthesis_exported_separately():
    # StyleGAN-style generator: a mapping network (z -> w) and a synthesis
    # network that modulates convolutions with an AdaIN-like affine transform
    # derived from w. In practice these are exported as two independent ONNX
    # graphs (see the earlier discussion in this conversation) because w is
    # often reused across multiple synthesis calls (truncation, style mixing,
    # per-layer noise) -- there is no single combined "forward" to trace.
    class MappingNetwork(torch.nn.Module):
        def __init__(self, z_dim=32, w_dim=32):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(z_dim, w_dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(w_dim, w_dim),
                torch.nn.LeakyReLU(0.2),
            )

        def forward(self, z):
            return self.net(z)

    class AdaIN(torch.nn.Module):
        # Instance-normalize the feature map, then apply a per-channel
        # scale/shift computed from the style vector w (the core StyleGAN
        # modulation primitive).
        def __init__(self, w_dim, num_features):
            super().__init__()
            self.norm = torch.nn.InstanceNorm2d(num_features, affine=False)
            self.style = torch.nn.Linear(w_dim, num_features * 2)

        def forward(self, x, w):
            style = self.style(w).unsqueeze(-1).unsqueeze(-1)
            gamma, beta = style.chunk(2, dim=1)
            return (1 + gamma) * self.norm(x) + beta

    class SynthesisNetwork(torch.nn.Module):
        def __init__(self, w_dim=32, nf=8, nc=3):
            super().__init__()
            self.const_input = torch.nn.Parameter(torch.randn(1, nf, 4, 4))
            self.conv1 = torch.nn.Conv2d(nf, nf, 3, padding=1)
            self.adain1 = AdaIN(w_dim, nf)
            self.up = torch.nn.Upsample(scale_factor=2, mode="nearest")
            self.conv2 = torch.nn.Conv2d(nf, nf, 3, padding=1)
            self.adain2 = AdaIN(w_dim, nf)
            self.to_rgb = torch.nn.Conv2d(nf, nc, 1)

        def forward(self, w):
            batch = w.shape[0]
            x = self.const_input.expand(batch, -1, -1, -1)
            x = self.adain1(self.conv1(x), w)
            x = self.up(x)
            x = self.adain2(self.conv2(x), w)
            return torch.tanh(self.to_rgb(x))

    mapping = MappingNetwork().eval()
    synthesis = SynthesisNetwork().eval()
    z = torch.randn(1, 32)

    mapping_opt = export_simplify_and_check_by_python_api(
        mapping,
        (z,),
        export_kwargs={
            "opset_version": 17,
            "input_names": ["z"],
            "output_names": ["w"],
        },
    )
    assert any(n.op_type == "Gemm" for n in mapping_opt.graph.node)

    w = mapping(z).detach()
    synthesis_opt = export_simplify_and_check_by_python_api(
        synthesis,
        (w,),
        export_kwargs={
            "opset_version": 17,
            "input_names": ["w"],
            "output_names": ["image"],
        },
    )
    assert any(n.op_type == "Conv" for n in synthesis_opt.graph.node)

    # The two graphs are fully independent: the synthesis graph only takes w
    # as input, never z, and neither graph references the other's nodes.
    assert [i.name for i in mapping_opt.graph.input] == ["z"]
    assert [i.name for i in synthesis_opt.graph.input] == ["w"]

    sess = onnxruntime.InferenceSession(
        synthesis_opt.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (out,) = sess.run(None, {"w": w.numpy()})
    assert out.shape == (1, 3, 8, 8)


def test_cyclegan_resnet_generator_dynamic_hw():
    # CycleGAN/pix2pix-style generator: ReflectionPad2d + InstanceNorm2d
    # residual blocks operating on an image of arbitrary spatial size (these
    # generators are fully convolutional, so H/W are runtime-dynamic).
    class ResnetBlock(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(dim, dim, 3),
                torch.nn.InstanceNorm2d(dim, affine=True),
                torch.nn.ReLU(True),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(dim, dim, 3),
                torch.nn.InstanceNorm2d(dim, affine=True),
            )

        def forward(self, x):
            return x + self.block(x)

    class CycleGANGenerator(torch.nn.Module):
        def __init__(self, nc=3, ngf=8, n_blocks=2):
            super().__init__()
            layers = [
                torch.nn.ReflectionPad2d(3),
                torch.nn.Conv2d(nc, ngf, 7),
                torch.nn.InstanceNorm2d(ngf, affine=True),
                torch.nn.ReLU(True),
            ]
            for _ in range(n_blocks):
                layers.append(ResnetBlock(ngf))
            layers += [
                torch.nn.ReflectionPad2d(3),
                torch.nn.Conv2d(ngf, nc, 7),
                torch.nn.Tanh(),
            ]
            self.model = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    model = CycleGANGenerator().eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    original_nodes = None

    def _capture_node_count(m: onnx.ModelProto) -> bool:
        nonlocal original_nodes
        original_nodes = len(m.graph.node)
        return True

    opt = export_simplify_and_check_by_python_api(
        model,
        (dummy_input,),
        is_model_valid=_capture_node_count,
        export_kwargs={
            "opset_version": 17,
            "input_names": ["input_image"],
            "output_names": ["output_image"],
            "dynamic_axes": {
                "input_image": {2: "h", 3: "w"},
                "output_image": {2: "h", 3: "w"},
            },
        },
    )

    assert len(opt.graph.node) < original_nodes

    # The dynamic H/W axes must survive simplification.
    in_dims = opt.graph.input[0].type.tensor_type.shape.dim
    assert in_dims[2].dim_param and in_dims[3].dim_param

    # A fully convolutional generator must still run at a spatial size that
    # differs from the one used to export/simplify it.
    sess = onnxruntime.InferenceSession(
        opt.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    import numpy as np

    other_size = np.random.RandomState(0).rand(1, 3, 48, 64).astype("float32")
    (out,) = sess.run(None, {"input_image": other_size})
    assert out.shape == (1, 3, 48, 64)

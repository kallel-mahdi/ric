"""ResNet-18/34 encoders (NHWC) and the linear-head classifier used by the supervised baselines.

Attribute names (trunk.stem, layer0..3, blocks, conv0/bn0, ...) are the parameter paths of the
paper's checkpoints; renaming them breaks loading.
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

BLOCKS_PER_STAGE = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}
WIDTHS = (64, 128, 256, 512)
ACTIVATIONS = {"relu": jax.nn.relu, "silu": jax.nn.silu, "gelu": partial(jax.nn.gelu, approximate=False)}
ENCODERS = {  # arch -> (depth, stem)
    "cifarresnet18": (18, "cifar"),
    "cifarresnet34": (34, "cifar"),
    "imagenetresnet18": (18, "imagenet"),
    "imagenetresnet34": (34, "imagenet"),
}


def conv(in_channels, out_channels, kernel, stride, dtype, rngs):
    return nnx.Conv(
        in_channels,
        out_channels,
        kernel_size=(kernel, kernel),
        strides=(stride, stride),
        padding=kernel // 2,
        use_bias=False,
        kernel_init=jax.nn.initializers.he_normal(),
        dtype=dtype,
        rngs=rngs,
    )


def batch_norm(channels, dtype, rngs):
    return nnx.BatchNorm(channels, momentum=0.9, epsilon=1e-5, dtype=dtype, rngs=rngs)


class Downsample(nnx.Module):
    def __init__(self, in_channels, out_channels, stride, *, dtype, rngs):
        self.conv = conv(in_channels, out_channels, 1, stride, dtype, rngs)
        self.bn = batch_norm(out_channels, dtype, rngs)

    def __call__(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, stride, *, activation, dtype, rngs):
        # Downsample first: the order params draw from `rngs` fixes the seeded init of the paper runs.
        projects = stride != 1 or in_channels != out_channels
        self.downsample = (
            Downsample(in_channels, out_channels, stride, dtype=dtype, rngs=rngs) if projects else None
        )
        self.conv0 = conv(in_channels, out_channels, 3, stride, dtype, rngs)
        self.bn0 = batch_norm(out_channels, dtype, rngs)
        self.conv1 = conv(out_channels, out_channels, 3, 1, dtype, rngs)
        self.bn1 = batch_norm(out_channels, dtype, rngs)
        self.activation = activation

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        x = self.activation(self.bn0(self.conv0(x)))
        return self.activation(self.bn1(self.conv1(x)) + shortcut)


class Stage(nnx.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride, *, activation, dtype, rngs):
        kw = {"activation": activation, "dtype": dtype, "rngs": rngs}
        self.blocks = nnx.List([BasicBlock(in_channels, out_channels, stride, **kw)])
        for _ in range(num_blocks - 1):
            self.blocks.append(BasicBlock(out_channels, out_channels, 1, **kw))

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Stem(nnx.Module):
    """'cifar': 3x3 stride-1 conv, no pooling, to keep small images' detail. 'imagenet': 7x7/2 conv + 3x3/2 max-pool."""

    def __init__(self, kind, *, activation, dtype, rngs):
        self.imagenet = kind == "imagenet"
        self.conv = conv(3, 64, 7, 2, dtype, rngs) if self.imagenet else conv(3, 64, 3, 1, dtype, rngs)
        self.bn = batch_norm(64, dtype, rngs)
        self.activation = activation

    def __call__(self, x):
        x = self.activation(self.bn(self.conv(x)))
        if self.imagenet:
            x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        return x


class ResNet(nnx.Module):
    """Stem, four stages, global average pool → 512 features."""

    def __init__(self, depth, *, stem, activation, dtype, rngs):
        n = BLOCKS_PER_STAGE[depth]
        kw = {"activation": activation, "dtype": dtype, "rngs": rngs}
        self.stem = Stem(stem, **kw)
        self.layer0 = Stage(64, WIDTHS[0], n[0], 1, **kw)
        self.layer1 = Stage(WIDTHS[0], WIDTHS[1], n[1], 2, **kw)
        self.layer2 = Stage(WIDTHS[1], WIDTHS[2], n[2], 2, **kw)
        self.layer3 = Stage(WIDTHS[2], WIDTHS[3], n[3], 2, **kw)

    def __call__(self, x):
        x = self.stem(x)
        for stage in (self.layer0, self.layer1, self.layer2, self.layer3):
            x = stage(x)
        return jnp.mean(x, axis=(1, 2))


class ResNetEncoder(nnx.Module):
    """`dtype` is the compute dtype (bfloat16 ≈ 1.6× faster); parameters and the output stay float32."""

    output_dim = WIDTHS[-1]

    def __init__(self, depth, *, stem, activation, dtype, key):
        self.dtype = dtype
        self.trunk = ResNet(
            depth, stem=stem, activation=ACTIVATIONS[activation], dtype=dtype, rngs=nnx.Rngs(key)
        )

    def __call__(self, images):
        return self.trunk(images.astype(self.dtype)).astype(jnp.float32)


def resolve_arch(arch: str, image_size: int) -> str:
    """'auto' → ResNet-34 with the CIFAR stem up to 64px images, the ImageNet stem above."""
    if arch == "auto":
        return "cifarresnet34" if image_size <= 64 else "imagenetresnet34"
    return arch


def build_encoder(arch: str, *, image_size: int, activation: str, dtype=jnp.float32, key) -> ResNetEncoder:
    depth, stem = ENCODERS[resolve_arch(arch, image_size)]
    return ResNetEncoder(depth, stem=stem, activation=activation, dtype=dtype, key=key)


class ResNetClassifier(nnx.Module):
    def __init__(self, num_classes, *, arch, image_size, activation, dtype=jnp.float32, key):
        trunk_key, head_key = jax.random.split(key)
        depth, stem = ENCODERS[resolve_arch(arch, image_size)]
        self.trunk = ResNetEncoder(depth, stem=stem, activation=activation, dtype=dtype, key=trunk_key)
        # As in the paper runs: He-normal head for the CIFAR stem, N(0, 0.01) for the ImageNet stem.
        init = jax.nn.initializers.he_normal() if stem == "cifar" else jax.nn.initializers.normal(0.01)
        self.head = nnx.Linear(self.trunk.output_dim, num_classes, kernel_init=init, rngs=nnx.Rngs(head_key))

    def __call__(self, images):
        return self.head(self.trunk(images))

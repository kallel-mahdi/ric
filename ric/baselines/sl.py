"""Cross-entropy baseline with the paper's ResNet initialization."""

import jax
import jax.numpy as jnp
import optax

from ric.helpers.training import LossOutput, derive_seed_streams
from ric.networks.resnet import ResNetClassifier


def init_model(cfg, *, K, image_size):
    params_key, _, _, _ = derive_seed_streams(cfg.seed)
    return ResNetClassifier(
        K,
        arch=cfg.dataset.encoder_arch,
        image_size=image_size,
        activation=cfg.algorithm.activation,
        key=params_key,
    )


def loss_fn(model, images, labels, cfg, epoch, *, training):
    logits = model(images)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return LossOutput(
        losses=losses,
        probs=jax.nn.softmax(logits),
        metrics={"delta_ce_normalized": 1 - losses / jnp.log(logits.shape[-1])},
    )

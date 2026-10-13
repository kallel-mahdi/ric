"""ACT and PonderNet objectives; predictive NLL is evaluated separately."""

import jax
import jax.numpy as jnp
import optax

from ric.helpers.training import LossOutput
from ric.networks.adaptive import ACTImageClassifier, PonderNetImageClassifier


def init_model(cfg, *, K, image_size):
    model_type = {"act": ACTImageClassifier, "ponder": PonderNetImageClassifier}[cfg.model.name]
    return model_type(
        K, cfg.model, arch=cfg.dataset.encoder_arch, image_size=image_size, key=jax.random.PRNGKey(cfg.seed)
    )


def loss_fn(model, images, labels, cfg, epoch, *, training):
    logits, info = model(images)
    model_cfg = cfg.model
    if model_cfg.name == "act":
        cross_entropy = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        losses = cross_entropy + model_cfg.ponder_weight * info.ponder_cost
        metrics = {"mean_steps": info.n_updates, "ponder_cost": info.ponder_cost}
    else:
        step_labels = jnp.broadcast_to(labels[:, None], info.step_logits.shape[:2])
        step_ce = optax.softmax_cross_entropy_with_integer_labels(info.step_logits, step_labels)
        cross_entropy = jnp.sum(info.step_probs * step_ce, axis=-1)
        steps = jnp.arange(info.step_probs.shape[-1], dtype=jnp.float32)
        prior = (1 - model_cfg.lambda_p) ** steps * model_cfg.lambda_p
        prior = prior / prior.sum()
        # Keep the paper's probability floor on both sides of KL(p_halt || p_prior).
        kl = optax.kl_divergence(jnp.log(jnp.clip(prior, 1e-8, 1.0)), jnp.clip(info.step_probs, 1e-8, 1.0))
        losses = cross_entropy + model_cfg.ponder_beta * kl
        metrics = {"mean_steps": info.expected_steps, "kl_divergence": kl}
    return LossOutput(
        losses=losses,
        probs=jax.nn.softmax(logits),
        metrics=metrics
        | {
            "cross_entropy": cross_entropy,
            "delta_ce_normalized": 1 - cross_entropy / jnp.log(logits.shape[-1]),
        },
    )

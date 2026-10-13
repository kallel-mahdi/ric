"""Evidential classification: expected cross-entropy and annealed Dirichlet KL."""

import jax
import jax.numpy as jnp
import jax.scipy.special as special

from ric.helpers.metrics import log_score
from ric.helpers.training import LossOutput


def kl_to_uniform(alpha):
    strength = alpha.sum(-1)
    log_normalizer = special.gammaln(strength) - special.gammaln(alpha).sum(-1)
    log_normalizer -= special.gammaln(alpha.shape[-1])
    return log_normalizer + jnp.sum(
        (alpha - 1) * (special.digamma(alpha) - special.digamma(strength)[:, None]), axis=-1
    )


def loss_fn(model, images, labels, cfg, epoch, *, training):
    algo = cfg.algorithm
    logits = model(images)
    evidence = jnp.exp(jnp.clip(logits, -algo.evidence_logit_clip, algo.evidence_logit_clip))
    alpha = evidence + algo.alpha_offset + algo.dirichlet_additive_epsilon
    strength = alpha.sum(-1)
    probs = alpha / strength[:, None]
    targets = jax.nn.one_hot(labels, logits.shape[-1])
    cross_entropy = jnp.sum(targets * (special.digamma(strength)[:, None] - special.digamma(alpha)), -1)
    # Suppress the correct class's evidence only in the KL regularizer.
    alpha_hat = targets + (1 - targets) * alpha
    annealing = jnp.minimum(1.0, (epoch - 1) / algo.kl_annealing_epochs) if training else 1.0
    losses = cross_entropy + algo.dirichlet_kl_weight * annealing * kl_to_uniform(alpha_hat)
    nll = -log_score(probs, labels)
    return LossOutput(
        losses=losses,
        probs=probs,
        metrics={
            "strength_mean": strength,
            "concentration_mean": evidence.sum(-1),
            "delta_ce_normalized": 1 - nll / jnp.log(logits.shape[-1]),
        },
    )

"""Classification metrics over an entire split, with optional padding masks."""

import jax
import jax.numpy as jnp
import optax
from jax.scipy.stats import rankdata


def log_score(probs, labels):
    probability = jnp.take_along_axis(probs, labels[..., None], axis=-1)[..., 0]
    return jnp.log(jnp.clip(probability, 1e-8))


def _binary_auc(scores, positive, mask):
    scores = jnp.where(mask, scores, jnp.inf)
    rank = rankdata(scores, method="average")
    pos, neg = mask & positive, mask & ~positive
    # Pooled stochastic evaluation can make pair counts exceed int32.
    n_pos, n_neg = pos.sum(dtype=scores.dtype), neg.sum(dtype=scores.dtype)
    return (jnp.sum(rank * pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def classification_metrics(probs, labels, mask=None):
    mask = jnp.ones(labels.shape, bool) if mask is None else mask
    count = mask.sum()
    confidence = probs.max(-1)
    correct = probs.argmax(-1) == labels
    bins = jnp.clip(jnp.floor(confidence * 20).astype(jnp.int32), 0, 19)
    # Multiplying the bin gap by its population cancels the per-bin denominator.
    gaps = jnp.bincount(bins, weights=jnp.where(mask, confidence - correct, 0), length=20)
    brier = optax.l2_loss(probs, jax.nn.one_hot(labels, probs.shape[-1])).sum(-1)

    def auc(k):
        return _binary_auc(probs[:, k], labels == k, mask)

    return {
        "accuracy": jnp.mean(correct, where=mask),
        "nll": -jnp.mean(log_score(probs, labels), where=mask),
        "ece": jnp.abs(gaps).sum() / count,
        "brier": jnp.mean(brier, where=mask),
        "macro_auc": jnp.nanmean(jax.vmap(auc)(jnp.arange(probs.shape[-1]))),
    }

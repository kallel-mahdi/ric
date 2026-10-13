"""Policy and value heads.

Policy heads share one interface: `__call__(h) -> stats`, `sample(stats, key) -> (action,
log_action, log_prob)`, `log_prob(stats, log_action)`, and `mode(stats)` with the same return as `sample`. Log-probabilities
are always computed from `log_action` so no head ever takes log() of a sampled simplex point.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy.stats import norm

_small = jax.nn.initializers.orthogonal(scale=0.01)


def _anchor(logits):
    """Append the fixed 0 logit of the K-th class."""
    return jnp.concatenate([logits, jnp.zeros_like(logits[..., :1])], axis=-1)


class DirichletHead(nnx.Module):
    """v ~ Dir(c·m + ε) with mean m = softmax(logits) and concentration c ∈ [c_min, c_max]."""

    def __init__(self, in_size, K, *, key, additive_epsilon, c_min, c_max):
        k1, k2 = jax.random.split(key)
        self.eps, self.c_min, self.c_max = additive_epsilon, c_min, c_max
        self.mean_layer = nnx.Linear(in_size, K, kernel_init=_small, rngs=nnx.Rngs(k1))
        self.conc_layer = nnx.Linear(
            in_size, 1, kernel_init=_small, bias_init=nnx.initializers.constant(-4.0), rngs=nnx.Rngs(k2)
        )

    def __call__(self, h):
        mean = jax.nn.softmax(self.mean_layer(h), axis=-1)
        c = self.c_min + jax.nn.sigmoid(self.conc_layer(h)[..., 0]) * (self.c_max - self.c_min)
        return {"alpha": c[..., None] * mean + self.eps, "c_dir": c}

    def log_prob(self, stats, log_action):
        alpha = stats["alpha"]
        log_norm = jnp.sum(jax.lax.lgamma(alpha), -1) - jax.lax.lgamma(jnp.sum(alpha, -1))
        return jnp.sum((alpha - 1) * log_action, -1) - log_norm

    def sample(self, stats, key):
        log_action = jax.nn.log_softmax(jax.random.loggamma(key, stats["alpha"]), axis=-1)
        return jnp.exp(log_action), log_action, self.log_prob(stats, log_action)

    def mode(self, stats):
        """The Dirichlet mean, as (action, log_action, log_prob) like `sample`."""
        log_action = jnp.log(stats["alpha"]) - jnp.log(jnp.sum(stats["alpha"], -1, keepdims=True))
        return jnp.exp(log_action), log_action, self.log_prob(stats, log_action)


class LogisticSoftmaxHead(nnx.Module):
    """Gaussian over K-1 logits anchored at 0, pushed through softmax (logistic-normal policy)."""

    def __init__(self, in_size, K, *, key, log_std_min, log_std_max, log_std_init):
        k1, k2 = jax.random.split(key)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.mu_layer = nnx.Linear(in_size, K - 1, kernel_init=_small, rngs=nnx.Rngs(k1))
        self.log_std_layer = nnx.Linear(
            in_size,
            K - 1,
            kernel_init=_small,
            bias_init=nnx.initializers.constant(log_std_init),
            rngs=nnx.Rngs(k2),
        )

    def __call__(self, h):
        return {
            "mu": self.mu_layer(h),
            "log_std": jnp.clip(self.log_std_layer(h), self.log_std_min, self.log_std_max),
        }

    def log_prob(self, stats, log_action):
        mu, log_std = stats["mu"], stats["log_std"]
        y = log_action[..., :-1] - log_action[..., -1:]
        log_p_y = norm.logpdf(y, loc=mu, scale=jnp.exp(log_std)).sum(-1)
        return log_p_y - jnp.sum(log_action, -1)  # softmax change-of-variables Jacobian

    def sample(self, stats, key):
        y = stats["mu"] + jnp.exp(stats["log_std"]) * jax.random.normal(key, stats["mu"].shape)
        log_action = jax.nn.log_softmax(_anchor(y), axis=-1)
        return jnp.exp(log_action), log_action, self.log_prob(stats, log_action)

    def mode(self, stats):
        log_action = jax.nn.log_softmax(_anchor(stats["mu"]), axis=-1)
        return jnp.exp(log_action), log_action, self.log_prob(stats, log_action)


class ValueHead(nnx.Module):
    def __init__(self, in_size, *, key):
        self.layer = nnx.Linear(in_size, 1, kernel_init=jax.nn.initializers.orthogonal(), rngs=nnx.Rngs(key))

    def __call__(self, h):
        return self.layer(h)[..., 0]

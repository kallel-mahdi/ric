"""RIC policy/value network: image encoder, then a GRU over (embedding, belief p_t), then policy and value heads."""

import jax
import jax.numpy as jnp
from flax import nnx

from ric.networks.heads import DirichletHead, LogisticSoftmaxHead, ValueHead
from ric.networks.resnet import build_encoder

SCAN_UNROLL = 4  # unrolling the T-step scans a little is measurably faster


class EncoderGRU(nnx.Module):
    """Holds the encoder and GRU so parameter paths (rnn_base.encoder, rnn_base.rnn) match released checkpoints."""

    def __init__(self, cfg, *, K, image_size, key):
        encoder_key, _, rnn_key = jax.random.split(key, 3)
        self.encoder = build_encoder(
            cfg.encoder_arch,
            image_size=image_size,
            activation=cfg.activation,
            dtype=jnp.dtype(cfg.compute_dtype),
            key=encoder_key,
        )
        width = self.encoder.output_dim
        hidden = width if cfg.gru_hidden_size == "auto" else cfg.gru_hidden_size
        self.rnn = nnx.GRUCell(width + K, hidden, rngs=nnx.Rngs(rnn_key))


class PolicyValueNetwork(nnx.Module):
    def __init__(self, cfg, *, K: int, image_size: int, key):
        base_key, actor_key, value_key = jax.random.split(key, 3)
        self.K = K
        self.rnn_base = EncoderGRU(cfg, K=K, image_size=image_size, key=base_key)
        hidden = self.rnn_base.rnn.hidden_features
        if cfg.actor_head_type == "dirichlet":
            c_min, c_max = (1.0, float(K)) if cfg.c_bounds == "auto" else cfg.c_bounds
            self.actor_head = DirichletHead(
                hidden,
                K,
                key=actor_key,
                additive_epsilon=cfg.dirichlet_additive_epsilon,
                c_min=c_min,
                c_max=c_max,
            )
        elif cfg.actor_head_type == "gaussian_logit":
            self.actor_head = LogisticSoftmaxHead(
                hidden,
                K,
                key=actor_key,
                log_std_min=cfg.gauss_log_std_min,
                log_std_max=cfg.gauss_log_std_max,
                log_std_init=cfg.gauss_log_std_init,
            )
        else:
            raise ValueError(f"Unknown actor_head_type: {cfg.actor_head_type!r}")
        self.value_head = ValueHead(hidden, key=value_key)

    def encode(self, images):
        return self.rnn_base.encoder(images)

    def initial_carry(self, batch_size: int):
        return jnp.zeros((batch_size, self.rnn_base.rnn.hidden_features))

    def step(self, embedding, belief, carry):
        """One GRU step on (embedding, p_t). Returns (next carry, policy stats, V(s_t))."""
        carry, hidden = self.rnn_base.rnn(carry, jnp.concatenate([embedding, belief], axis=-1))
        return carry, self.actor_head(hidden), self.value_head(hidden)

    def unroll(self, embedding, beliefs):
        """Teacher-force the GRU over beliefs (T, B, K). Returns (final carry, stats (T, B, ...), values (T, B))."""

        def body(carry, belief):
            return self.rnn_base.rnn(carry, jnp.concatenate([embedding, belief], axis=-1))

        carry, hidden = jax.lax.scan(body, self.initial_carry(beliefs.shape[1]), beliefs, unroll=SCAN_UNROLL)
        return carry, self.actor_head(hidden), self.value_head(hidden)

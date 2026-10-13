"""ACT and PonderNet: an image encoder followed by a halting GRU refinement loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx, struct

from ric.networks.resnet import build_encoder


class NormedGRU(nnx.Module):
    """The adaptive checkpoints include LayerNorm on the new hidden state."""

    def __init__(self, input_size: int, hidden_size: int, *, key: jax.Array):
        rngs = nnx.Rngs(key)
        self.cell = nnx.GRUCell(input_size, hidden_size, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden_size, rngs=rngs)

    def __call__(self, inputs: jnp.ndarray, carry: jnp.ndarray) -> jnp.ndarray:
        new_h, _ = self.cell(carry, inputs)
        return self.norm(new_h)


class _HaltingCore(nnx.Module):
    """GRU stepped `max_steps` times on a fixed input, with class and halting heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        *,
        max_steps: int,
        halt_bias: float = 0.0,
        key: jax.Array,
    ):
        k_rnn, k_out, k_halt = jax.random.split(key, 3)
        self.rnn = NormedGRU(input_size, hidden_size, key=k_rnn)
        self.output_head = nnx.Linear(
            hidden_size,
            num_classes,
            kernel_init=nnx.initializers.orthogonal(0.01),
            bias_init=nnx.initializers.zeros,
            rngs=nnx.Rngs(k_out),
        )
        self.halt_head = nnx.Linear(
            hidden_size,
            1,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.constant(halt_bias),
            rngs=nnx.Rngs(k_halt),
        )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.max_steps = int(max_steps)

    def _readout(self, hidden: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Class logits and halting probability for one step."""
        logits = self.output_head(hidden)
        return logits, jax.nn.sigmoid(self.halt_head(hidden).squeeze(-1))


@struct.dataclass
class ACTInfo:
    weights: jnp.ndarray  # (B, max_steps) halting distribution
    n_updates: jnp.ndarray  # (B,) steps taken
    ponder_cost: jnp.ndarray  # (B,) N - 1 + remainder


@struct.dataclass
class ACTCarry:
    halting_sum: jax.Array
    running: jax.Array
    n_updates: jax.Array
    remainder: jax.Array
    logits: jax.Array
    hidden: jax.Array


class ACTCore(_HaltingCore):
    """Graves' ACT: halt once accumulated halting probability reaches 1 - epsilon."""

    def __init__(self, *args, epsilon: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = float(epsilon)

    def __call__(self, inputs: jnp.ndarray) -> tuple[jnp.ndarray, ACTInfo]:
        batch_size = inputs.shape[0]
        threshold = 1.0 - self.epsilon

        def act_step(carry, step_idx):
            hidden = jnp.where(carry.running[:, None], self.rnn(inputs, carry.hidden), carry.hidden)
            logits, p_halt = self._readout(hidden)
            p_t = p_halt * carry.running

            # Halt when the accumulated probability crosses the threshold, and
            # force every lane to halt on the final step so weights sum to one.
            is_last = step_idx == self.max_steps - 1
            will_halt = carry.running & ((carry.halting_sum + p_t >= threshold) | is_last)
            step_weight = jnp.where(will_halt, 1.0 - carry.halting_sum, p_t) * carry.running

            return ACTCarry(
                halting_sum=carry.halting_sum + step_weight,
                running=carry.running & ~will_halt,
                n_updates=carry.n_updates + carry.running,
                remainder=jnp.where(will_halt, step_weight, carry.remainder),
                logits=carry.logits + step_weight[:, None] * logits,
                hidden=hidden,
            ), step_weight

        init = ACTCarry(
            halting_sum=jnp.zeros(batch_size),
            running=jnp.ones(batch_size, bool),
            n_updates=jnp.zeros(batch_size, jnp.int32),
            remainder=jnp.zeros(batch_size),
            logits=jnp.zeros((batch_size, self.num_classes)),
            hidden=jnp.zeros((batch_size, self.hidden_size)),
        )
        final, weight_steps = jax.lax.scan(act_step, init, jnp.arange(self.max_steps))
        ponder_cost = jnp.maximum(final.n_updates - 1.0 + final.remainder, 0.0)
        return final.logits, ACTInfo(
            weights=weight_steps.T, n_updates=final.n_updates, ponder_cost=ponder_cost
        )


@struct.dataclass
class PonderNetInfo:
    step_probs: jnp.ndarray  # (B, max_steps) halting distribution p_n
    expected_steps: jnp.ndarray  # (B,)
    step_logits: jnp.ndarray  # (B, max_steps, K)


class PonderNetCore(_HaltingCore):
    """PonderNet: per-step predictions mixed under the halting distribution p_n."""

    def __call__(self, inputs: jnp.ndarray) -> tuple[jnp.ndarray, PonderNetInfo]:
        batch_size = inputs.shape[0]

        def ponder_step(carry, step_idx):
            unhalted, rnn_carry = carry
            hidden = self.rnn(inputs, rnn_carry)
            logits, lambda_t = self._readout(hidden)
            # The final step halts with probability one so the step distribution sums to one.
            lambda_t = jnp.where(step_idx == self.max_steps - 1, 1.0, lambda_t)
            return (unhalted * (1.0 - lambda_t), hidden), (unhalted * lambda_t, logits)

        init = (jnp.ones((batch_size,), jnp.float32), jnp.zeros((batch_size, self.hidden_size), jnp.float32))
        _, (p_steps, logits_steps) = jax.lax.scan(ponder_step, init, jnp.arange(self.max_steps))
        step_probs = p_steps.T
        step_logits = jnp.swapaxes(logits_steps, 0, 1)
        # Log of the predictive mixture sum_n p_n softmax(z_n): softmax of it is the
        # distribution the expected-NLL loss trains.
        logits = jax.nn.logsumexp(jax.nn.log_softmax(step_logits), axis=1, b=step_probs[:, :, None])
        expected_steps = step_probs @ jnp.arange(1, self.max_steps + 1, dtype=jnp.float32)
        return logits, PonderNetInfo(
            step_probs=step_probs, expected_steps=expected_steps, step_logits=step_logits
        )


class _AdaptiveImageClassifier(nnx.Module):
    """ResNet image encoder feeding a halting GRU core."""

    def _build_encoder(self, cfg, *, arch, image_size, key):
        # Keep the unused middle draw from the paper's removed post-encoder MLP.
        encoder_key, _, core_key = jax.random.split(key, 3)
        self.encoder = build_encoder(arch, image_size=image_size, activation=cfg.activation, key=encoder_key)
        width = self.encoder.output_dim
        hidden = width if cfg.gru_hidden_size == "auto" else cfg.gru_hidden_size
        return width, hidden, core_key


class ACTImageClassifier(_AdaptiveImageClassifier):
    def __init__(self, num_classes, cfg, *, arch, image_size, key):
        input_size, hidden_size, k_core = self._build_encoder(cfg, arch=arch, image_size=image_size, key=key)
        self.act_core = ACTCore(
            input_size,
            hidden_size,
            num_classes,
            max_steps=cfg.max_steps,
            epsilon=cfg.act_epsilon,
            halt_bias=cfg.halt_bias,
            key=k_core,
        )

    def __call__(self, images: jnp.ndarray):
        return self.act_core(self.encoder(images))


class PonderNetImageClassifier(_AdaptiveImageClassifier):
    def __init__(self, num_classes, cfg, *, arch, image_size, key):
        input_size, hidden_size, k_core = self._build_encoder(cfg, arch=arch, image_size=image_size, key=key)
        self.ponder_core = PonderNetCore(
            input_size,
            hidden_size,
            num_classes,
            max_steps=cfg.max_steps,
            halt_bias=cfg.halt_bias,
            key=k_core,
        )

    def __call__(self, images: jnp.ndarray):
        return self.ponder_core(self.encoder(images))

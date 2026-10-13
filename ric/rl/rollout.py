"""The belief-refinement MDP: start from the uniform belief p_0, act a_t ~ π(·|x, p_t), move to p_{t+1} = a_t,
and earn r_t = log p_{t+1}(y) − log p_t(y). Every episode runs the full horizon T."""

import jax
import jax.numpy as jnp
from flax import struct

from ric.helpers.metrics import log_score
from ric.networks.policy import SCAN_UNROLL


@struct.dataclass
class Episode:
    """T steps for a batch of images, time-major."""

    beliefs: jax.Array  # (T, B, K) p_t, the belief the policy acted on
    log_actions: jax.Array  # (T, B, K)
    log_probs: jax.Array  # (T, B) log π(a_t | s_t) under the acting network
    values: jax.Array  # (T, B) V(s_t) under the acting network
    rewards: jax.Array  # (T, B)
    final_belief: jax.Array  # (B, K) p_T


def run_episode(net, embedding, labels, key, horizon: int, *, deterministic: bool = False) -> Episode:
    batch_size = labels.shape[0]

    def body(carry, step_key):
        rnn_carry, belief = carry
        rnn_carry, stats, value = net.step(embedding, belief, rnn_carry)
        if deterministic:
            action, log_action, log_prob = net.actor_head.mode(stats)
        else:
            action, log_action, log_prob = net.actor_head.sample(stats, step_key)
        reward = log_score(action, labels) - log_score(belief, labels)
        return (rnn_carry, action), (belief, log_action, log_prob, value, reward)

    uniform = jnp.full((batch_size, net.K), 1.0 / net.K)
    initial = (net.initial_carry(batch_size), uniform)
    (_, final_belief), steps = jax.lax.scan(body, initial, jax.random.split(key, horizon), unroll=SCAN_UNROLL)
    return Episode(*steps, final_belief=final_belief)

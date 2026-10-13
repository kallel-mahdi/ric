"""RIC-SPO learner.

A frozen snapshot π_old collects every batch (eval mode); each batch gives one gradient step on
π_current (train mode). The critic is re-unrolled under the current parameters, so it learns V^{π_old}.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ric.data.datasets import normalize_images
from ric.helpers.metrics import log_score
from ric.helpers.training import TrainState, batch_chunks, derive_seed_streams
from ric.networks.policy import PolicyValueNetwork
from ric.rl.rollout import Episode, run_episode


class RLTrainState(TrainState):
    reward_var: jax.Array
    key: jax.Array


def weight_matrices(params):
    return jax.tree.map(lambda p: p.ndim >= 2, params)


def make_optimizer(cfg):
    return optax.chain(
        optax.clip_by_global_norm(cfg.policy_grad_clip_norm),
        optax.adamw(cfg.policy_lr, weight_decay=cfg.weight_decay, mask=weight_matrices),
    )


def init_state(cfg, *, K: int, image_size: int):
    params_key, rollout_key, _, _ = derive_seed_streams(cfg.seed)
    model = PolicyValueNetwork(cfg.algorithm.network, K=K, image_size=image_size, key=params_key)
    graphdef, params, batch_stats = nnx.split(model, nnx.Param, ...)
    return RLTrainState.create(
        apply_fn=None,
        graphdef=graphdef,
        params=params,
        batch_stats=batch_stats,
        tx=make_optimizer(cfg.algorithm),
        reward_var=jnp.ones(()),
        key=rollout_key,
    )


def gae(rewards, values, last_value, gamma: float, lam: float):
    """GAE over time-major (T, B) arrays. Episodes are truncated at T, so V(s_T) bootstraps the last step."""
    next_values = jnp.concatenate([values[1:], last_value[None]])
    deltas = rewards + gamma * next_values - values

    def backward(advantage, delta):
        advantage = delta + gamma * lam * advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(backward, jnp.zeros_like(last_value), deltas, reverse=True)
    return advantages, advantages + values


def spo_objective(log_ratio, advantages, cfg):
    """SPO: r·A − |A|/(2ε)·(r−1)². The log-ratio is soft-clipped with c·tanh(x/c) so the square can't overflow."""
    log_ratio = cfg.log_ratio_clip * jnp.tanh(log_ratio / cfg.log_ratio_clip)
    return (
        jnp.exp(log_ratio) * advantages
        - jnp.abs(advantages) / (2 * cfg.clip_coef) * jnp.expm1(log_ratio) ** 2
    )


def ppo_objective(log_ratio, advantages, cfg):
    ratio = jnp.exp(log_ratio)
    return jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * advantages)


OBJECTIVES = {"spo": spo_objective, "ppo": ppo_objective}


def loss_fn(net, images, episode: Episode, reward_var, cfg):
    embedding = net.encode(images)
    carry, stats, values = net.unroll(embedding, episode.beliefs)
    _, _, last_value = net.step(embedding, episode.final_belief, carry)
    advantages, returns = gae(
        episode.rewards / jnp.sqrt(reward_var),
        jax.lax.stop_gradient(values),
        jax.lax.stop_gradient(last_value),
        cfg.gamma,
        cfg.gae_lambda,
    )
    log_prob = net.actor_head.log_prob(stats, episode.log_actions)
    log_ratio = log_prob - episode.log_probs
    policy_loss = -jnp.mean(OBJECTIVES[cfg.policy_loss](log_ratio, advantages, cfg))
    value_loss = jnp.mean(optax.l2_loss(values, returns))
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "approx_kl": -jnp.mean(log_ratio),
        "entropy": -jnp.mean(log_prob),
    }
    return policy_loss + cfg.value_coef * value_loss, metrics


def episode_metrics(episode: Episode, labels, gamma: float):
    horizon, num_classes = episode.rewards.shape[0], episode.final_belief.shape[-1]
    discounts = gamma ** jnp.arange(horizon)
    return {
        "accuracy": jnp.mean(jnp.argmax(episode.final_belief, -1) == labels),
        "nll": -jnp.mean(log_score(episode.final_belief, labels)),
        "discounted_return_normalized": jnp.mean(discounts @ episode.rewards) / jnp.log(num_classes),
        "reward_var": jnp.var(episode.rewards),
    }


def make_train_steps(cfg, norm_stats):
    """Returns jit(state, pi_old, stacked Grain batches) → (state, metrics stacked over updates)."""

    def train_step(state: TrainState, batch):
        pixels, labels, episode = batch
        images = normalize_images(pixels, norm_stats)

        def objective(params):
            net = state.replace(params=params).network(train=True)
            loss, metrics = loss_fn(net, images, episode, state.reward_var, cfg)
            return loss, (metrics, nnx.state(net, nnx.BatchStat))

        (_, (metrics, batch_stats)), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads, batch_stats=batch_stats)
        metrics |= episode_metrics(episode, labels, cfg.gamma) | {"grad_norm": optax.global_norm(grads)}
        return state, metrics

    @jax.jit
    def train_steps(state, pi_old, batches):
        pixels = batches["image"]

        def split_key(key, _):
            next_key, rollout_key = jax.random.split(key)
            return next_key, rollout_key

        final_key, keys = jax.lax.scan(split_key, state.key, None, length=len(pixels))
        acting = pi_old.network(train=False)

        def collect(batch):
            pixels, labels, key = batch
            images = normalize_images(pixels, norm_stats)
            return run_episode(acting, acting.encode(images), labels, key, cfg.T_max)

        # Frozen-policy batches are independent; current-policy updates stay ordered.
        episodes = jax.lax.map(
            collect, (pixels, batches["label"], keys), batch_size=cfg.rollout_batch_group_size
        )
        state, metrics = jax.lax.scan(train_step, state, (pixels, batches["label"], episodes))
        return state.replace(key=final_key), metrics

    return train_steps


def make_train_epoch(cfg, norm_stats, num_batches):
    algorithm = cfg.algorithm
    train_steps = make_train_steps(algorithm, norm_stats)
    updates = num_batches * algorithm.training_epochs_per_policy

    def train_epoch(state, iterator, epoch):
        pi_old = state
        chunks = []
        for batches in batch_chunks(iterator, updates, algorithm.train_scan_chunk_batches):
            state, metrics = train_steps(state, pi_old, batches)
            chunks.append(metrics)
        metrics = jax.tree.map(lambda *xs: jnp.concatenate(xs).mean(), *chunks)
        reward_var = 0.99 * state.reward_var + 0.01 * metrics["reward_var"]
        return state.replace(reward_var=reward_var), metrics

    return train_epoch

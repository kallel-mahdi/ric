"""Evaluation with the critic as a virtual halting rule.

Each episode runs all T steps. Afterwards we stop at the first t > 0 with V(s_t) < threshold and report
the belief p_t held there (no stop: p_T). Stochastic evaluation pools `passes` sampled episodes per image
and also reports the across-pass std of every metric.
"""

import jax
import jax.numpy as jnp

from ric.data.datasets import normalize_images
from ric.helpers.metrics import classification_metrics
from ric.rl.rollout import Episode, run_episode


def halt(episode: Episode, threshold: float, gamma: float):
    """Returns the reported belief (B, K), steps taken (B,), and discounted return up to the stop (B,)."""
    horizon = episode.values.shape[0]
    t = jnp.arange(horizon)[:, None]
    stop = (t > 0) & (episode.values < threshold)
    steps = jnp.where(stop.any(0), jnp.argmax(stop, 0), horizon)
    beliefs = jnp.concatenate([episode.beliefs, episode.final_belief[None]])  # p_0 … p_T
    prediction = jnp.take_along_axis(beliefs, steps[None, :, None], axis=0)[0]
    discounted_return = jnp.sum(jnp.where(t < steps, gamma**t * episode.rewards, 0.0), axis=0)
    return prediction, steps, discounted_return


def dataset_metrics(probs, labels, steps, returns, mask):
    """Metrics over flat per-example arrays; `mask` drops the padding of the last batch."""

    correct = jnp.argmax(probs, -1) == labels
    return classification_metrics(probs, labels, mask) | {
        "avg_steps": jnp.mean(steps, where=mask),
        "avg_steps_correct": jnp.mean(steps, where=mask & correct),
        "avg_steps_incorrect": jnp.mean(steps, where=mask & ~correct),
        "discounted_return_normalized": jnp.mean(returns, where=mask) / jnp.log(probs.shape[-1]),
    }


def make_evaluate(config, norm_stats):
    """Returns jit(state, split, key) → metrics, where `split` is a staged EvalSplit."""

    cfg = config.algorithm
    deterministic = config.evaluation.policy_mode == "deterministic"
    passes = 1 if deterministic else config.evaluation.passes

    def evaluate_batch(net, images, labels, key):
        embedding = net.encode(normalize_images(images, norm_stats))  # once per batch, shared by all passes

        def one_pass(pass_key):
            episode = run_episode(net, embedding, labels, pass_key, cfg.T_max, deterministic=deterministic)
            return halt(episode, cfg.eval_value_stop_threshold, cfg.gamma)

        return jax.vmap(one_pass)(jax.random.split(key, passes))

    def by_pass(x):
        """(num_batches, passes, B, ...) → (passes, num_batches · B, ...)."""
        x = jnp.moveaxis(x, 1, 0)
        return x.reshape(passes, -1, *x.shape[3:])

    @jax.jit
    def evaluate(state, split, key):
        net = state.network(train=False)
        images, labels, mask = split.images, split.labels, split.mask
        keys = jax.random.split(key, len(labels))
        probs, steps, returns = jax.lax.map(lambda batch: evaluate_batch(net, *batch), (images, labels, keys))
        examples = {
            "probs": by_pass(probs),
            "steps": by_pass(steps),
            "returns": by_pass(returns),
            "labels": jnp.broadcast_to(labels.reshape(-1), (passes, labels.size)),
            "mask": jnp.broadcast_to(mask.reshape(-1), (passes, mask.size)),
        }
        pooled = dataset_metrics(**jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), examples))
        if passes == 1:
            return pooled
        per_pass = jax.vmap(lambda ex: dataset_metrics(**ex))(examples)
        return pooled | {f"{name}_sampling_std": jnp.std(v, ddof=1) for name, v in per_pass.items()}

    return evaluate

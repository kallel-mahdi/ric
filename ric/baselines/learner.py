"""Functional NNX updates and staged evaluation shared by the four baselines."""

import math

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ric.baselines import adaptive, edl, sl
from ric.data.datasets import normalize_images
from ric.helpers.metrics import classification_metrics, log_score
from ric.helpers.training import TrainState, batch_chunks

METHODS = {
    "sl": (sl.init_model, sl.loss_fn),
    "edl": (sl.init_model, edl.loss_fn),
    "adaptive": (adaptive.init_model, adaptive.loss_fn),
}
SUBSET_SUMS = ("clean_confidence_sum", "noisy_confidence_sum", "clean_count", "noisy_count")


def make_optimizer(cfg):
    algo = cfg.algorithm
    schedules = {
        "adamw": optax.constant_schedule(algo.learning_rate),
        "sgd_paper": optax.piecewise_constant_schedule(
            algo.learning_rate,
            {
                math.ceil(32_000 * 128 / algo.train_batch_size): 0.1,
                math.ceil(48_000 * 128 / algo.train_batch_size): 0.1,
            },
        ),
    }
    schedule = schedules[algo.optimizer]
    optimizers = {
        "adamw": optax.adamw(schedule, weight_decay=algo.weight_decay),
        "sgd_paper": optax.chain(
            optax.add_decayed_weights(algo.weight_decay), optax.sgd(schedule, momentum=0.9)
        ),
    }
    return optax.chain(optax.clip_by_global_norm(algo.grad_clip_norm), optimizers[algo.optimizer]), schedule


def init_state(cfg, *, K, image_size):
    init_model, _ = METHODS[cfg.method]
    model = init_model(cfg, K=K, image_size=image_size)
    graphdef, params, batch_stats = nnx.split(model, nnx.Param, ...)
    optimizer, _ = make_optimizer(cfg)
    return TrainState.create(
        apply_fn=None, params=params, tx=optimizer, graphdef=graphdef, batch_stats=batch_stats
    )


def make_train_epoch(cfg, norm_stats, num_batches):
    _, loss_fn = METHODS[cfg.method]
    _, schedule = make_optimizer(cfg)
    chunk_size = cfg.algorithm.train_scan_chunk_batches

    def train_step(state, batch, epoch):
        images = normalize_images(batch["image"], norm_stats)
        labels, noisy = batch["label"], batch["is_noisy"]

        def objective(params):
            model = state.replace(params=params).network(train=True)
            output = loss_fn(model, images, labels, cfg, epoch, training=True)
            return output.losses.mean(), (output, nnx.state(model, nnx.BatchStat))

        (loss, (output, batch_stats)), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        probs = output.probs
        confidence = probs.max(-1)
        metrics = {name: values.mean() for name, values in output.metrics.items()}
        metrics |= {
            "loss": loss,
            "accuracy": (probs.argmax(-1) == labels).mean(),
            "nll": -log_score(probs, labels).mean(),
            "p_max_mean": confidence.mean(),
            "grad_norm": optax.global_norm(grads),
            "learning_rate": schedule(state.step),
            "clean_confidence_sum": jnp.sum(confidence * ~noisy),
            "noisy_confidence_sum": jnp.sum(confidence * noisy),
            "clean_count": jnp.sum(~noisy),
            "noisy_count": jnp.sum(noisy),
        }
        return state.apply_gradients(grads=grads, batch_stats=batch_stats), metrics

    @jax.jit
    def train_chunk(state, batches, epoch):
        state, metrics = jax.lax.scan(
            lambda current, batch: train_step(current, batch, epoch), state, batches
        )
        return state, jax.tree.map(lambda values: values.sum(0), metrics)

    def train_epoch(state, iterator, epoch):
        totals = None
        for batches in batch_chunks(iterator, num_batches, chunk_size):
            state, metrics = train_chunk(state, batches, epoch)
            totals = metrics if totals is None else jax.tree.map(jnp.add, totals, metrics)
        metrics = {name: value / num_batches for name, value in totals.items() if name not in SUBSET_SUMS}
        metrics |= {
            "p_max_mean_clean_label_subset": totals["clean_confidence_sum"] / totals["clean_count"],
            "p_max_mean_noisy_label_subset": totals["noisy_confidence_sum"] / totals["noisy_count"],
            "clean_label_subset_count": totals["clean_count"],
            "noisy_label_subset_count": totals["noisy_count"],
        }
        return state, metrics

    return train_epoch


def make_evaluate(cfg, norm_stats):
    _, loss_fn = METHODS[cfg.method]

    @jax.jit
    def evaluate(state, split, key):
        model = state.network(train=False)

        def evaluate_batch(batch):
            images, labels = batch
            return loss_fn(model, normalize_images(images, norm_stats), labels, cfg, 0, training=False)

        output = jax.lax.map(evaluate_batch, (split.images, split.labels))
        probs = output.probs.reshape(-1, output.probs.shape[-1])
        metrics = classification_metrics(probs, split.labels.reshape(-1), split.mask.reshape(-1))
        metrics |= {
            name: jnp.mean(values, where=split.mask)
            for name, values in {"loss": output.losses, **output.metrics}.items()
        }
        return metrics

    return evaluate

"""Epoch checkpoints: latest for resume, independent best epochs for evaluation."""

import re
from datetime import datetime
from operator import itemgetter
from pathlib import Path

import grain
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
from orbax.checkpoint.checkpoint_managers import AnyPreservationPolicy, BestN, LatestN

SELECTORS = {
    "rl": {
        "nll": ("nll", "min"),
        "acc": ("accuracy", "max"),
        "ece": ("ece", "min"),
        "return": ("discounted_return_normalized", "max"),
    },
    "sl": {"nll": ("nll", "min"), "acc": ("accuracy", "max")},
    "edl": {"nll": ("nll", "min"), "acc": ("accuracy", "max")},
    "adaptive": {"loss": ("loss", "min")},
}


def prepare_run(directory, cfg):
    """Keep a portable config with checkpoints and reject incompatible automatic resumes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    if path.exists():
        saved = OmegaConf.load(path)
        sections = (
            "method",
            "seed",
            "split_seed",
            "dataset",
            "algorithm",
            "model",
            "evaluation",
            "data_loader",
        )
        changed = [key for key in sections if saved.get(key) != cfg.get(key)]
        if changed:
            raise ValueError(
                f"Run configuration differs in {', '.join(changed)}. "
                "Use a new run_id for a different experiment."
            )
    else:
        if any(directory.iterdir()):
            raise ValueError(
                f"Existing run {directory} has no config.yaml. "
                "Recover its original resolved training config before resuming."
            )
        with path.open("x") as stream:
            OmegaConf.save(cfg, stream, resolve=True)


def checkpoint_manager(directory, selectors, *, read_only=False):
    policies = [LatestN(1)]
    for metric, direction in selectors.values():
        policies.append(BestN(get_metric_fn=itemgetter(metric), reverse=direction == "min", n=1))
    metric, direction = next(iter(selectors.values()))
    options = ocp.CheckpointManagerOptions(
        preservation_policy=AnyPreservationPolicy(policies),
        # Orbax persists selection metrics only when best_fn is supplied.
        best_fn=itemgetter(metric),
        best_mode=direction,
        read_only=read_only,
    )
    handlers = {"state": ocp.StandardCheckpointHandler(), "data": grain.checkpoint.CheckpointHandler()}
    return ocp.CheckpointManager(Path(directory).resolve(), options=options, item_handlers=handlers)


def best_step(manager, selector, selectors):
    metric, direction = selectors[selector]
    sign = 1 if direction == "max" else -1
    # Native BestN retains the newest checkpoint when scores tie.
    return max(manager.all_steps(), key=lambda step: (sign * manager.metadata(step).metrics[metric], step))


def run_directory(cfg):
    if cfg.run_id is None:
        parts = [cfg.run_tag or datetime.now().strftime("run_%Y%m%d_%H%M%S"), cfg.dataset.name]
        if cfg.dataset.name == "cifar10n":
            parts.append(cfg.dataset.noise_label_key)
        if cfg.method == "rl" and cfg.evaluation.policy_mode == "deterministic":
            parts.append("deterministic_eval")
        parts.append(f"seed{cfg.seed}")
        cfg.run_id = "_".join(parts)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", cfg.run_id).strip("._-")
    return Path(cfg.checkpoint_dir).expanduser().resolve() / name

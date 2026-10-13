"""Hydra commands. Process setup precedes imports that initialize JAX."""

import os

import hydra


def configure():
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/ric/jax"))
    from absl import flags, logging

    flags.FLAGS.mark_as_parsed()
    logging.set_verbosity(logging.ERROR)


def _run(cfg):
    configure()
    from ric.train import train

    train(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="train_rl")
def main_rl(cfg):
    _run(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="sl")
def main_sl(cfg):
    _run(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="edl")
def main_edl(cfg):
    _run(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="adaptive")
def main_adaptive(cfg):
    _run(cfg)


def evaluate_main():
    import argparse
    import json
    import math
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate a validation-selected epoch checkpoint.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--selector")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure()

    import jax
    import orbax.checkpoint as ocp
    from omegaconf import OmegaConf

    from ric.data.pipeline import build_data, stage
    from ric.helpers.checkpointing import SELECTORS, best_step, checkpoint_manager
    from ric.helpers.training import EVAL_SPLIT_IDS, derive_seed_streams
    from ric.train import LEARNERS

    config_path = args.checkpoint_dir / "config.yaml"
    if not config_path.is_file():
        parser.error(f"Missing {config_path}; recover the original resolved training config first.")
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    _, _, eval_key, shuffle_seed = derive_seed_streams(cfg.seed)
    data = build_data(cfg, shuffle_seed=shuffle_seed)
    learner, evaluate_factory = LEARNERS[cfg.method]
    state = learner.init_state(cfg, K=data.num_classes, image_size=cfg.dataset.image_size)
    evaluate = evaluate_factory(cfg, data.norm_stats)
    split = stage({"val": data.val, "test": data.test}[args.split], cfg.algorithm.eval_batch_size)
    key = jax.random.fold_in(eval_key, EVAL_SPLIT_IDS[args.split])
    selectors = SELECTORS[cfg.method]
    selector = args.selector or next(iter(selectors))
    with checkpoint_manager(args.checkpoint_dir, selectors, read_only=True) as manager:
        epoch = best_step(manager, selector, selectors)
        restored = manager.restore(epoch, args=ocp.args.Composite(state=ocp.args.StandardRestore(state)))
    metrics = jax.device_get(evaluate(restored.state, split, key))
    result = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "epoch": epoch,
        "selector": selector,
        "method": cfg.method,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "split": args.split,
        "config": OmegaConf.to_container(cfg, resolve=True),
        # Empty conditional subsets have undefined metrics; represent them as JSON null.
        **{name: float(value) if math.isfinite(value) else None for name, value in metrics.items()},
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)

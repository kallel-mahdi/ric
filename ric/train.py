"""Shared run lifecycle: initialization, epochs, validation, resume, and final test."""

from contextlib import closing
from pathlib import Path

import grain
import jax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from ric.baselines import learner as baselines
from ric.data.pipeline import build_data, stage
from ric.helpers.checkpointing import SELECTORS, best_step, checkpoint_manager, prepare_run, run_directory
from ric.helpers.training import EVAL_SPLIT_IDS, derive_seed_streams
from ric.rl import ppo
from ric.rl.evaluation import make_evaluate

LEARNERS = {
    "rl": (ppo, make_evaluate),
    "sl": (baselines, baselines.make_evaluate),
    "edl": (baselines, baselines.make_evaluate),
    "adaptive": (baselines, baselines.make_evaluate),
}


def train(cfg):
    directory = run_directory(cfg)
    prepare_run(directory, cfg)
    _, _, eval_key, shuffle_seed = derive_seed_streams(cfg.seed)
    data = build_data(cfg, shuffle_seed=shuffle_seed)
    learner, evaluate_factory = LEARNERS[cfg.method]
    state = learner.init_state(cfg, K=data.num_classes, image_size=cfg.dataset.image_size)
    train_epoch = learner.make_train_epoch(
        cfg, data.norm_stats, data.train_size // cfg.algorithm.train_batch_size
    )
    evaluate = evaluate_factory(cfg, data.norm_stats)
    validation = stage(data.val, cfg.algorithm.eval_batch_size)
    val_key = jax.random.fold_in(eval_key, EVAL_SPLIT_IDS["val"])
    test_key = jax.random.fold_in(eval_key, EVAL_SPLIT_IDS["test"])
    selectors = SELECTORS[cfg.method]
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    with (
        wandb.init(
            dir=cfg.output_dir,
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            group=cfg.wandb_group,
            mode=cfg.wandb_mode,
            id=cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
        ) as run,
        checkpoint_manager(directory, selectors) as manager,
        closing(iter(data.train)) as iterator,
    ):

        def save(epoch, metrics):
            manager.save(
                epoch,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    data=grain.checkpoint.CheckpointSave(iterator),
                ),
                metrics=metrics,
            )

        start = manager.latest_step()
        if start is None:
            start = 0
            if cfg.method == "rl":
                val = jax.device_get(evaluate(state, validation, val_key))
                val = {name: float(value) for name, value in val.items()}
                save(0, val)
                run.log({f"val/{name}": value for name, value in val.items()}, step=0)
        else:
            restored = manager.restore(
                start,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(state),
                    data=grain.checkpoint.CheckpointRestore(iterator),
                ),
            )
            state = restored.state
            epoch_kind = "policy epoch" if cfg.method == "rl" else "epoch"
            print(f"Resumed after {epoch_kind} {start}.")

        for epoch in range(start + 1, cfg.total_epochs + 1):
            state, training = train_epoch(state, iterator, epoch)
            val = evaluate(state, validation, val_key)
            training, val = jax.device_get((training, val))
            training = {name: float(value) for name, value in training.items()}
            val = {name: float(value) for name, value in val.items()}
            save(epoch, val)
            run.log(
                {f"train/{name}": value for name, value in training.items()}
                | {f"val/{name}": value for name, value in val.items()},
                step=epoch,
            )
            print(f"epoch {epoch}: val acc {val['accuracy']:.4f} ece {val['ece']:.4f} nll {val['nll']:.4f}")

        manager.wait_until_finished()
        if cfg.run_final_test:
            test_split = stage(data.test, cfg.algorithm.eval_batch_size)
            for selector in selectors:
                step = best_step(manager, selector, selectors)
                restored = manager.restore(
                    step, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
                )
                test = jax.device_get(evaluate(restored.state, test_split, test_key))
                result = {f"test/best_{selector}/{name}": float(value) for name, value in test.items()}
                result[f"test/best_{selector}/epoch"] = step
                run.summary.update(result)
                print(f"best_{selector} (epoch {step}) test acc {float(test['accuracy']):.4f}")

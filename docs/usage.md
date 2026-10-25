# Usage

See the [README](../README.md) for installation and a first training run.

## Data and output

Python ≥ 3.12. Torch is only used for dataset loading and torchvision image operations in the
data workers; `uv sync` installs it from the CPU wheel index. Datasets download to `./other/data` on first use
(`dataset.data_path=...`); ImageWoof comes from the fast.ai release, Galaxy10 needs
`ric-convert-galaxy10` once.

Ruff and JAX cache outside the checkout
(`~/.cache/ruff` and `~/.cache/ric/jax`; JAX honors `JAX_COMPILATION_CACHE_DIR`).
Datasets, checkpoints, and run output live under `other/`.

To suppress Python bytecode files, set this before running commands:

```bash
export PYTHONDONTWRITEBYTECODE=1
```

## Training and sweeps

```bash
ric-train          +experiment=ric_spo dataset=cifar10     # RIC-SPO (ours)
ric-train-sl       +experiment=sl_adamw dataset=cifar10    # cross-entropy
ric-train-edl      +experiment=edl_adamw dataset=cifar10   # evidential deep learning
ric-train-adaptive +experiment=ponder dataset=cifar10      # PonderNet (+experiment=act for ACT)
```

Every entrypoint is a Hydra app: override any key on the command line. `+sweep=paper` runs the five
paper seeds, `+sweep=noise` the two CIFAR-10N label sets, `+sweep=datasets` every dataset, and
`launcher=slurm` submits through submitit. Combine sweep groups with Hydra multi-select,
for example `'+sweep=[noise,paper]' dataset=cifar10n`. Runs log to W&B offline by default (`wandb_mode=online`).

## Evaluate

Every trainer saves epoch checkpoints and resumes automatically when the same run directory exists.
Each run stores its resolved configuration in `config.yaml` beside the checkpoints. Resume rejects
changes to model, algorithm, dataset, seeds, evaluation or loader settings; choose a new `run_id`
for a different experiment. The epoch budget, logging and output settings can change on resume.
RIC retains the best accuracy/ECE/NLL/return epochs; SL/EDL retain accuracy/NLL; ACT/PonderNet retain
full validation loss. Training ends by scoring these selected checkpoints on the test split. To score a
saved checkpoint:

```bash
ric-eval --checkpoint-dir <run> --selector nll
ric-eval --checkpoint-dir <run> --selector nll --override evaluation.passes=20
```

Evaluation loads the saved training configuration. Explicit `--override key=value` options apply
on top, and the output includes the effective configuration. Older runs need their original resolved
training configuration recovered as `<run>/config.yaml`; defaults are never substituted silently.

RIC evaluation runs every episode for `T` steps and then halts virtually at the first step whose
critic value drops below `algorithm.eval_value_stop_threshold`, reporting the belief held at that step.
Stochastic evaluation (`evaluation.policy_mode=stochastic`) pools 10 sampled rollouts per image;
deterministic evaluation follows the Dirichlet mean.

## Precision and throughput

Paper defaults use FP32 and rollout group size 1. To use a BF16 encoder and group
rollout batches:

```bash
ric-train +experiment=ric_spo dataset=cifar10 \
  algorithm.network.compute_dtype=bfloat16 \
  algorithm.rollout_batch_group_size=4 data_loader.train_worker_count=0
```

The GRU, heads, and parameters remain FP32. Larger rollout groups use more GPU
memory. Loader worker count should be tuned for the dataset and hardware.

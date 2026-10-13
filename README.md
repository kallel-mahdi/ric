# Do Not Imitate, Reinforce: Iterative Classification via Belief Refinement

Reference implementation of **Reinforced Iterative Classification (RIC)** and **RIC-SPO** (TMLR).

![RIC unrolled across refinement steps: green critic lights continue to the next RNN step; the red light stops and returns the current belief. All RNN blocks share weights.](.github/assets/ric-refinement.gif)

RIC casts classification as an MDP. A recurrent actor holds a belief `p_t` over the K classes and
refines it for `T` steps; each action is a full probability vector `a_t ~ Dir(c·m)`, the next belief is
`p_{t+1} = a_t`, and the reward is the improvement of the observed label's log score,
`log p_{t+1}(y) − log p_t(y)`. A critic learns how much refinement is still worth taking and doubles as
a stopping rule at inference. Trained with SPO (a PPO variant). JAX + Flax NNX + Optax + Hydra.

## Install

```bash
git clone https://github.com/kallel-mahdi/ric.git
cd ric
export PYTHONDONTWRITEBYTECODE=1  # inherit in Python processes and data workers; no __pycache__
uv sync --extra cuda      # NVIDIA GPU, CUDA 13 (driver >= 580); --extra cuda12 for older drivers; no extra = CPU
source .venv/bin/activate
```

Python ≥ 3.12. Torch is only used for dataset loading and torchvision image operations in the
data workers; `uv sync` installs it from the CPU wheel index. Datasets download to `./other/data` on first use
(`dataset.data_path=...`); ImageWoof comes from the fast.ai release, Galaxy10 needs
`ric-convert-galaxy10` once.

Keep the bytecode setting in each shell running this project. Ruff and JAX cache outside the checkout
(`~/.cache/ruff` and `~/.cache/ric/jax`; JAX honors `JAX_COMPILATION_CACHE_DIR`).
Datasets, checkpoints, and run output live under `other/`.

## Train

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

For faster CIFAR training on an RTX 4090:

```bash
ric-train dataset=cifar10 algorithm.network.compute_dtype=bfloat16 \
  algorithm.rollout_batch_group_size=4 data_loader.train_worker_count=0
```

On an RTX 4090 with ResNet34, batch 256 and horizon 20, this measured about 1.9× the FP32
training-update throughput. A warm CIFAR policy epoch takes about 26 seconds: five training
passes plus validation. Startup and compilation are excluded. The encoder uses BF16; GRU,
heads and parameters remain FP32. Paper defaults retain FP32 and rollout group size 1.
Larger groups use more GPU memory; other datasets may benefit from loader workers.

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

## Results

Test accuracy (%) / ECE, mean of seeds 473075, 742340, 20096, 520801, 199345. Checkpoints selected by
validation NLL (SL, EDL, RIC-SPO) or by full validation loss (ACT, PonderNet).

| Dataset | SL | EDL | ACT | PonderNet | RIC-SPO (det.) | RIC-SPO (stoch.) |
|---|---|---|---|---|---|---|
| CIFAR-10 | 88.6 / 0.049 | 91.8 / 0.047 | 88.9 / 0.050 | 88.9 / 0.049 | 92.6 / 0.030 | 89.0 / 0.030 |
| CIFAR-10N (aggre) | 85.7 / 0.011 | 86.2 / 0.112 | 86.4 / 0.016 | 85.8 / 0.018 | 88.0 / 0.081 | 85.4 / 0.050 |
| CIFAR-10N (worst) | 75.4 / 0.176 | 59.3 / 0.291 | 76.9 / 0.196 | 77.3 / 0.195 | 73.6 / 0.284 | 64.1 / 0.051 |
| SVHN | 95.3 / 0.009 | 95.6 / 0.029 | 95.1 / 0.012 | 95.3 / 0.015 | 95.7 / 0.092 | 94.8 / 0.055 |
| ImageWoof | 82.0 / 0.085 | 84.2 / 0.072 | 81.7 / 0.079 | 82.8 / 0.087 | 85.3 / 0.038 | 82.8 / 0.017 |
| Galaxy10 | 80.4 / 0.063 | 81.4 / 0.100 | 80.3 / 0.054 | 80.5 / 0.068 | 83.1 / 0.068 | 80.4 / 0.037 |
| DermaMNIST | 76.3 / 0.046 | 76.0 / 0.105 | 70.9 / 0.038 | 71.5 / 0.034 | 78.4 / 0.057 | 75.8 / 0.023 |
| OCTMNIST | 86.2 / 0.084 | 85.6 / 0.091 | 73.3 / 0.168 | 75.7 / 0.149 | 85.9 / 0.075 | 85.4 / 0.053 |
| OrganSMNIST | 80.0 / 0.125 | 81.8 / 0.099 | 74.2 / 0.146 | 72.9 / 0.138 | 81.2 / 0.050 | 81.0 / 0.054 |
| PneumoniaMNIST | 88.6 / 0.091 | 89.2 / 0.077 | 85.7 / 0.122 | 85.3 / 0.132 | 89.6 / 0.079 | 88.6 / 0.072 |
| TissueMNIST | 73.9 / 0.049 | 68.8 / 0.090 | 67.0 / 0.046 | 66.9 / 0.034 | 72.3 / 0.069 | 69.1 / 0.041 |

## Layout

```
ric/
  cli.py        Hydra training and checkpoint evaluation commands
  train.py      Shared run lifecycle: epochs, validation, resume, final test
  helpers/      Training state, random streams, checkpoints, classification metrics
  data/         Dataset splits, Galaxy10 conversion, and native Grain pipelines
  networks/     ResNet encoders, policy/value heads, ACT/PonderNet models
  rl/           Belief rollouts, SPO/PPO updates, virtual halting
  baselines/    Shared supervised learner and SL/EDL/ACT/PonderNet objectives
  conf/         Paper configurations and sweeps
other/          Datasets, checkpoints and logs (created at runtime; ignored)
```

## Citation

```bibtex
@article{ric2026,
  author  = {Mahdi Kallel and Johannes Tölle and Ahmed Hendawy and Carlo D'Eramo},
  title   = {Do Not Imitate, Reinforce: Iterative Classification via Belief Refinement},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
}
```

## License

Apache-2.0.

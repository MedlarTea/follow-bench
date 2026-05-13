# Follow-Ahead-Reaction

This package vendors the Follow-Ahead-Reaction runtime, training code, and the
minimum published model assets into the `follow-bench` repository.

## Public evaluation entrypoints

- `socialRPF/example/robot_person_following/follow_ahead_reaction_framework.py`
- `socialRPF/example/robot_person_following/follow_ahead_reaction_fixed_speed.py`

The first entrypoint is the fair benchmark-aligned framework variant. The
second entrypoint is the fixed-speed release variant that keeps the robot speeds
at `1.0 / 1.5`.

Method-name mapping used in papers and result summaries:

- `follow_ahead_reaction1` = `follow_ahead_reaction_framework.py`
- `follow_ahead_reaction2` = `follow_ahead_reaction_fixed_speed.py`

## Environment

```bash
conda activate follow-bench-new
```

## Smoke-test evaluation

```bash
cd /path/to/follow-bench
cd socialRPF/example/robot_person_following/evaluate_scripts
conda activate follow-bench

export PYTHON_BIN=$(which python)
export FOLLOW_BENCH_EXPERIMENTS_ROOT=USER_FOLLOW_BENCH_EXPERIMENTS_PATH

bash ./evaluate.sh config_target_traj_eval.yaml 1.5 \
  --headless \
  --positions back \
  --limit-scenes 1 \
  --limit-trials 1 \
  --baseline-script follow_ahead_reaction_framework.py

bash ./evaluate.sh config_target_traj_eval.yaml 1.5 \
  --headless \
  --positions back \
  --limit-scenes 1 \
  --limit-trials 1 \
  --baseline-script follow_ahead_reaction_fixed_speed.py
```

Both public entrypoints are compatible with the standard `evaluate.sh`
interface and accept the usual evaluation arguments:

- `-c`
- `-s`
- `-p`
- `-d`
- `-m`
- `-i`
- `-l`
- `--headless`

Both public entrypoints default to `save_animation=False`, so batch evaluation
works without extra flags.

## Package layout

- `runtime/`: packaged runtime variants used by the public evaluation scripts
- `mcts/`: MCTS tree search and task utilities
- `models/`: human action prior and RL value model interfaces
- `training/`: training and runtime-regression scripts
- `assets/`: packaged weights needed for evaluation and retraining

See [TRAINING.md](TRAINING.md) for training and retraining commands.

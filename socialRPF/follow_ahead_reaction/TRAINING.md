# Follow-Ahead-Reaction Training

## Environment

```bash
conda activate follow-bench
```

## Train a value model

```bash
cd /path/to/follow-bench

python socialRPF/follow_ahead_reaction/training/train_a2c_follow.py \
  --position back \
  --follow-distance 1.5 \
  --total-timesteps 100000 \
  --obs-mode relative_pose
```

By default:

- models are written to `socialRPF/follow_ahead_reaction/training/outputs/`
- logs are written to `socialRPF/follow_ahead_reaction/training/logs/`

## Evaluate a trained value model

```bash
cd /path/to/follow-bench

python socialRPF/follow_ahead_reaction/training/eval_follow_model.py \
  --model socialRPF/follow_ahead_reaction/training/outputs/follow_task_models/follow_value_back_1p5.zip \
  --position back \
  --follow-distance 1.5
```

## Runtime regression on benchmark scenes

```bash
cd /path/to/follow-bench

python socialRPF/follow_ahead_reaction/training/eval_runtime_follow_scenes.py \
  --position back \
  --follow-distance 1.5 \
  --config-dir socialRPF/dynamic_scenarios/config \
  --scenes NormalCrowdOrcaH10W0 NormalCircularOrcaH10W0 \
  --index 1 \
  --python-bin $(which python)
```

The default runtime script is:

- `socialRPF/example/robot_person_following/follow_ahead_reaction_framework.py`

You can override it with `--runtime-script` if you want to evaluate the fixed
speed variant instead.

## Retrain and promote a better model

```bash
cd /path/to/follow-bench

python socialRPF/follow_ahead_reaction/training/retrain_follow_model.py \
  --position back \
  --follow-distance 1.5 \
  --config-dir socialRPF/dynamic_scenarios/config \
  --scenes NormalCrowdOrcaH10W0 NormalCircularOrcaH10W0 \
  --promote-if-better
```

When promotion is enabled, the candidate model is copied into
`socialRPF/follow_ahead_reaction/assets/` and the previous packaged model is
backed up under `assets/.../backups/`.

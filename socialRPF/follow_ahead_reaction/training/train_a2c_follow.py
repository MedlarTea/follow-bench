import argparse
import importlib.util
import sys
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

SOURCE_ROOT = Path(__file__).resolve().parents[2]
source_root_str = str(SOURCE_ROOT)
if source_root_str not in sys.path:
    sys.path.insert(0, source_root_str)

from follow_ahead_reaction.mcts.follow_task_utils import (
    format_distance_tag,
    model_variant_dir,
)
from follow_ahead_reaction.training.nav_env_follow import FollowTaskEnv



def build_env(args, seed_offset=0):
    env = FollowTaskEnv(
        follow_mode=args.position,
        desired_distance=args.follow_distance,
        max_steps=args.max_steps,
        world_size=args.world_size,
        distance_threshold=args.distance_threshold,
        init_radius_min=args.init_radius_min,
        init_radius_max=args.init_radius_max,
        seed=args.seed + seed_offset,
        obs_mode=args.obs_mode,
    )
    return Monitor(env)



def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train an A2C value model for a specific follow task.")
    parser.add_argument("--position", type=str, required=True, choices=["front", "back", "left_side", "right_side"])
    parser.add_argument("--follow-distance", type=float, required=True)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--world-size", type=float, default=10.0)
    parser.add_argument("--distance-threshold", type=float, default=1e-9)
    parser.add_argument("--init-radius-min", type=float, default=None)
    parser.add_argument("--init-radius-max", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--obs-mode", type=str, default="relative_pose", choices=["relative_pose", "task_error"])
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--reset-num-timesteps", action="store_true")
    args = parser.parse_args()

    dist_tag = format_distance_tag(args.follow_distance)
    model_dir = model_variant_dir(args.obs_mode)
    default_output = script_dir / "outputs" / model_dir / f"follow_value_{args.position}_{dist_tag}"
    default_log_dir = script_dir / "logs" / model_dir / f"{args.position}_{dist_tag}"
    if args.output is None:
        args.output = default_output
    if args.log_dir is None:
        args.log_dir = default_log_dir
    return args



def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.check_env:
        check_env(
            FollowTaskEnv(
                follow_mode=args.position,
                desired_distance=args.follow_distance,
                max_steps=args.max_steps,
                world_size=args.world_size,
                distance_threshold=args.distance_threshold,
                init_radius_min=args.init_radius_min,
                init_radius_max=args.init_radius_max,
                seed=args.seed,
                obs_mode=args.obs_mode,
            ),
            warn=True,
        )

    env = build_env(args, seed_offset=0)
    eval_env = build_env(args, seed_offset=1)

    tensorboard_log = None
    if importlib.util.find_spec("tensorboard") is not None:
        tensorboard_log = str(args.log_dir / "tb")

    if args.init_model is not None:
        model = A2C.load(str(args.init_model), env=env, device='cpu')
        model.verbose = 1
        if tensorboard_log is not None:
            model.tensorboard_log = tensorboard_log
    else:
        model = A2C(
            "MlpPolicy",
            env,
            gamma=args.gamma,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            seed=args.seed,
            verbose=1,
            device='cpu',
            tensorboard_log=tensorboard_log,
            policy_kwargs={"net_arch": [128, 128]},
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq, 1),
        save_path=str(args.log_dir / "checkpoints"),
        name_prefix=f"follow_value_{args.position}_{format_distance_tag(args.follow_distance)}_{args.obs_mode}",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best_model"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=max(args.eval_freq, 1),
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=args.reset_num_timesteps,
    )
    model.save(str(args.output))
    print(f"Saved model to {args.output}.zip")


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C

SOURCE_ROOT = Path(__file__).resolve().parents[2]
source_root_str = str(SOURCE_ROOT)
if source_root_str not in sys.path:
    sys.path.insert(0, source_root_str)

from follow_ahead_reaction.training.nav_env_follow import FollowTaskEnv



def build_env(args):
    return FollowTaskEnv(
        follow_mode=args.position,
        desired_distance=args.follow_distance,
        max_steps=args.max_steps,
        world_size=args.world_size,
        distance_threshold=args.distance_threshold,
        init_radius_min=args.init_radius_min,
        init_radius_max=args.init_radius_max,
        obs_mode=args.obs_mode,
    )



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained follow-task A2C model.")
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--position', type=str, required=True, choices=['front', 'back', 'left_side', 'right_side'])
    parser.add_argument('--follow-distance', type=float, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--world-size', type=float, default=10.0)
    parser.add_argument('--distance-threshold', type=float, default=1e-9)
    parser.add_argument('--init-radius-min', type=float, default=None)
    parser.add_argument('--init-radius-max', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--summary-json', type=Path, default=None)
    parser.add_argument('--obs-mode', type=str, default='relative_pose', choices=['relative_pose', 'task_error'])
    return parser.parse_args()



def episode_summary(metrics, desired_distance):
    distances = np.asarray(metrics['distance'], dtype=np.float64)
    diffs = np.asarray(metrics['diff'], dtype=np.float64)
    lon_errs = np.asarray(metrics['lon_err'], dtype=np.float64)
    lat_errs = np.asarray(metrics['lat_err'], dtype=np.float64)
    yaw_errs = np.asarray(metrics['yaw_err'], dtype=np.float64)
    rewards = np.asarray(metrics['reward'], dtype=np.float64)

    scale = max(float(desired_distance), 1.0)
    good_pose = (
        (np.abs(lon_errs) < 0.35 * scale)
        & (np.abs(lat_errs) < 0.35 * scale)
        & (np.abs(distances - desired_distance) < 0.5)
        & (diffs < 25.0)
    )
    near_collision = distances < max(0.5, float(desired_distance) - 0.7)

    return {
        'steps': int(len(rewards)),
        'episode_reward': float(rewards.sum()),
        'mean_reward': float(rewards.mean()),
        'mean_distance': float(distances.mean()),
        'mean_distance_error': float(np.abs(distances - desired_distance).mean()),
        'mean_diff_deg': float(diffs.mean()),
        'mean_abs_lon_error': float(np.abs(lon_errs).mean()),
        'mean_abs_lat_error': float(np.abs(lat_errs).mean()),
        'mean_abs_yaw_deg': float(np.abs(yaw_errs).mean() * 180.0 / np.pi),
        'good_pose_ratio': float(good_pose.mean()),
        'near_collision_ratio': float(near_collision.mean()),
    }



def aggregate(summaries):
    keys = summaries[0].keys()
    out = {}
    for key in keys:
        vals = np.asarray([s[key] for s in summaries], dtype=np.float64)
        out[key] = {
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'min': float(vals.min()),
            'max': float(vals.max()),
        }
    return out



def main():
    args = parse_args()
    env = build_env(args)
    model = A2C.load(str(args.model), device='cpu')

    episode_summaries = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        metrics = {k: [] for k in ['distance', 'diff', 'lon_err', 'lat_err', 'yaw_err', 'reward']}

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            metrics['distance'].append(info['distance'])
            metrics['diff'].append(info['diff'])
            metrics['lon_err'].append(info['lon_err'])
            metrics['lat_err'].append(info['lat_err'])
            metrics['yaw_err'].append(info['yaw_err'])
            metrics['reward'].append(reward)

        summary = episode_summary(metrics, args.follow_distance)
        episode_summaries.append(summary)
        print(
            f"[EP {ep:02d}] reward={summary['episode_reward']:.3f} "
            f"mean_dist={summary['mean_distance']:.3f} "
            f"mean_dist_err={summary['mean_distance_error']:.3f} "
            f"mean_diff={summary['mean_diff_deg']:.3f} "
            f"mean_abs_lon_err={summary['mean_abs_lon_error']:.3f} "
            f"mean_abs_lat_err={summary['mean_abs_lat_error']:.3f} "
            f"good_pose_ratio={summary['good_pose_ratio']:.3f} "
            f"near_collision_ratio={summary['near_collision_ratio']:.3f}"
        )

    overall = aggregate(episode_summaries)
    print("\n[SUMMARY]")
    for key, stat in overall.items():
        print(f"{key}: mean={stat['mean']:.3f} std={stat['std']:.3f} min={stat['min']:.3f} max={stat['max']:.3f}")

    if args.summary_json is not None:
        payload = {"episodes": episode_summaries, "summary": overall}
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved summary to {args.summary_json}")

if __name__ == '__main__':
    main()

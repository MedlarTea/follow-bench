import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
source_root_str = str(SOURCE_ROOT)
if source_root_str not in sys.path:
    sys.path.insert(0, source_root_str)

from follow_ahead_reaction.mcts.follow_task_utils import desired_local_point


STATE_RE = re.compile(
    r"\[STATE\] step=(\d+).*?dist=([-0-9.]+).*?lon=([-0-9.]+) lat=([-0-9.]+).*?zone=([a-z\-]+) history=(\d+)/(\d+)"
)
COLLISION_RE = re.compile(r"collided with (\S+)")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate runtime follow behavior across multiple scenes using position-based metrics.")
    parser.add_argument('--position', type=str, required=True, choices=['front', 'back', 'left_side', 'right_side'])
    parser.add_argument('--follow-distance', type=float, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--config-dir', type=Path, required=True)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--runtime-script', type=Path, default=(SOURCE_ROOT / 'example' / 'robot_person_following' / 'follow_ahead_reaction_framework.py'))
    parser.add_argument('--python-bin', type=str, default=sys.executable)
    parser.add_argument('--max-steps', type=int, default=180)
    parser.add_argument('--obs-mode', type=str, default='relative_pose', choices=['relative_pose', 'task_error'])
    parser.add_argument('--rl-model-override', type=Path, default=None)
    parser.add_argument('--log-dir', type=Path, default=Path('/tmp/runtime_follow_scene_eval'))
    parser.add_argument('--summary-json', type=Path, default=None)
    return parser.parse_args()


def run_scene(args, scene):
    log_path = args.log_dir / f'{scene}_{args.position}.log'
    mpl_dir = args.log_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        str(args.runtime_script),
        '-c',
        str(args.config_dir),
        '-s',
        scene,
        '-p',
        args.position,
        '-i',
        str(args.index),
        '-d',
        str(args.follow_distance),
        '-m',
        str(args.max_steps),
        '--rl_obs_mode',
        args.obs_mode,
        '--headless',
        '--show_debug',
        '--no_save_animation',
    ]
    if args.rl_model_override is not None:
        cmd.extend(['--rl_model_override', str(args.rl_model_override)])

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_dir)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=env,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)

    desired_lon, desired_lat = desired_local_point(args.position, args.follow_distance)
    states = []
    collisions = []
    for line in proc.stdout.splitlines():
        m = STATE_RE.search(line)
        if m:
            history = int(m.group(6))
            if history < 15:
                continue
            states.append({
                'step': int(m.group(1)),
                'dist': float(m.group(2)),
                'lon': float(m.group(3)),
                'lat': float(m.group(4)),
                'zone': m.group(5),
            })
        mc = COLLISION_RE.search(line)
        if mc:
            collisions.append(mc.group(1))

    if not states:
        return {
            'scene': scene,
            'returncode': proc.returncode,
            'error': 'no_states',
            'collisions': collisions,
            'log': str(log_path),
        }

    dist_err = [abs(s['dist'] - args.follow_distance) for s in states]
    lon_err = [abs(s['lon'] - desired_lon) for s in states]
    lat_err = [abs(s['lat'] - desired_lat) for s in states]
    good = [1.0 if abs(s['lon'] - desired_lon) < 0.5 and abs(s['lat'] - desired_lat) < 0.5 and abs(s['dist'] - args.follow_distance) < 0.5 else 0.0 for s in states]
    front_ratio = [1.0 if s['lon'] > 0.3 else 0.0 for s in states]
    last20 = states[-20:] if len(states) >= 20 else states

    return {
        'scene': scene,
        'returncode': proc.returncode,
        'num_states': len(states),
        'mean_distance_error': statistics.fmean(dist_err),
        'mean_abs_lon_error': statistics.fmean(lon_err),
        'mean_abs_lat_error': statistics.fmean(lat_err),
        'good_pose_ratio': statistics.fmean(good),
        'front_ratio': statistics.fmean(front_ratio),
        'last20_mean_distance_error': statistics.fmean(abs(s['dist'] - args.follow_distance) for s in last20),
        'last20_front_ratio': statistics.fmean(1.0 if s['lon'] > 0.3 else 0.0 for s in last20),
        'collisions': collisions,
        'collision_flag': 1.0 if collisions else 0.0,
        'log': str(log_path),
    }


def aggregate(scene_results):
    numeric_keys = [
        'mean_distance_error',
        'mean_abs_lon_error',
        'mean_abs_lat_error',
        'good_pose_ratio',
        'front_ratio',
        'last20_mean_distance_error',
        'last20_front_ratio',
        'collision_flag',
    ]
    overall = {}
    for key in numeric_keys:
        values = [float(result[key]) for result in scene_results if key in result]
        if not values:
            continue
        overall[key] = {
            'mean': float(statistics.fmean(values)),
            'min': float(min(values)),
            'max': float(max(values)),
        }
    return overall


def main():
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    scene_results = []
    for scene in args.scenes:
        result = run_scene(args, scene)
        scene_results.append(result)
        if 'error' in result:
            print(f"[SCENE] {scene}: error={result['error']} log={result['log']}")
            continue
        print(
            f"[SCENE] {scene}: dist_err={result['mean_distance_error']:.3f} "
            f"lon_err={result['mean_abs_lon_error']:.3f} "
            f"lat_err={result['mean_abs_lat_error']:.3f} "
            f"good_pose_ratio={result['good_pose_ratio']:.3f} "
            f"front_ratio={result['front_ratio']:.3f} "
            f"collisions={','.join(result['collisions']) if result['collisions'] else 'none'}"
        )

    payload = {
        'position': args.position,
        'follow_distance': args.follow_distance,
        'obs_mode': args.obs_mode,
        'rl_model_override': str(args.rl_model_override) if args.rl_model_override else '',
        'scenes': scene_results,
        'overall': aggregate(scene_results),
    }

    distance_tag = str(args.follow_distance).replace('.', 'p')
    summary_json = args.summary_json or (args.log_dir / f"{args.position}_{distance_tag}_summary.json")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved summary to {summary_json}")


if __name__ == '__main__':
    main()

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
source_root_str = str(SOURCE_ROOT)
if source_root_str not in sys.path:
    sys.path.insert(0, source_root_str)

from follow_ahead_reaction.mcts.follow_task_utils import (
    format_distance_tag,
    model_variant_dir,
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Resume training for one follow-task model and compare it with multi-scene runtime metrics.')
    parser.add_argument('--position', type=str, required=True, choices=['front', 'back', 'left_side', 'right_side'])
    parser.add_argument('--follow-distance', type=float, required=True)
    parser.add_argument('--obs-mode', type=str, default='relative_pose', choices=['relative_pose', 'task_error'])
    parser.add_argument('--total-timesteps', type=int, default=40_000)
    parser.add_argument('--eval-freq', type=int, default=10_000)
    parser.add_argument('--save-freq', type=int, default=20_000)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--runtime-max-steps', type=int, default=180)
    parser.add_argument('--python-bin', type=str, default=sys.executable)
    parser.add_argument('--config-dir', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--runtime-script', type=Path, default=(SOURCE_ROOT / 'example' / 'robot_person_following' / 'follow_ahead_reaction_framework.py'))
    parser.add_argument('--work-dir', type=Path, default=Path('/tmp/retrain_follow_model'))
    parser.add_argument('--promote-if-better', action='store_true')
    return parser.parse_args()


def compare_score(summary_payload):
    overall = summary_payload['overall']
    return (
        6.0 * overall['good_pose_ratio']['mean']
        - 2.0 * overall['front_ratio']['mean']
        - overall['mean_abs_lon_error']['mean']
        - overall['mean_abs_lat_error']['mean']
        - 0.5 * overall['mean_distance_error']['mean']
        - 2.0 * overall['collision_flag']['mean']
    )


def run(cmd):
    subprocess.run(cmd, check=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    dist_tag = format_distance_tag(args.follow_distance)
    model_dir = model_variant_dir(args.obs_mode)

    assets_root = script_dir.parent / 'assets' / model_dir
    current_best = assets_root / f'follow_value_{args.position}_{dist_tag}.zip'
    include_model = current_best
    if not current_best.exists():
        raise FileNotFoundError(f'Current best model not found: {current_best}')

    run_root = args.work_dir / f'{args.position}_{dist_tag}_{args.obs_mode}'
    train_dir = run_root / 'train'
    eval_dir = run_root / 'eval'
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_script = script_dir / 'eval_runtime_follow_scenes.py'
    train_script = script_dir / 'train_a2c_follow.py'

    baseline_summary = eval_dir / 'baseline_summary.json'
    run([
        args.python_bin,
        str(eval_script),
        '--position', args.position,
        '--follow-distance', str(args.follow_distance),
        '--obs-mode', args.obs_mode,
        '--config-dir', str(args.config_dir),
        '--runtime-script', str(args.runtime_script),
        '--python-bin', args.python_bin,
        '--max-steps', str(args.runtime_max_steps),
        '--summary-json', str(baseline_summary),
        '--log-dir', str(eval_dir / 'baseline_logs'),
        '--rl-model-override', str(current_best),
        '--scenes', *args.scenes,
    ])

    run([
        args.python_bin,
        str(train_script),
        '--position', args.position,
        '--follow-distance', str(args.follow_distance),
        '--obs-mode', args.obs_mode,
        '--total-timesteps', str(args.total_timesteps),
        '--eval-freq', str(args.eval_freq),
        '--save-freq', str(args.save_freq),
        '--max-steps', str(args.max_steps),
        '--init-model', str(current_best),
        '--log-dir', str(train_dir),
        '--output', str(train_dir / f'follow_value_{args.position}_{dist_tag}_candidate'),
    ])

    candidate_best = train_dir / 'best_model' / 'best_model.zip'
    if not candidate_best.exists():
        raise FileNotFoundError(f'Candidate best model not found: {candidate_best}')

    candidate_summary = eval_dir / 'candidate_summary.json'
    run([
        args.python_bin,
        str(eval_script),
        '--position', args.position,
        '--follow-distance', str(args.follow_distance),
        '--obs-mode', args.obs_mode,
        '--config-dir', str(args.config_dir),
        '--runtime-script', str(args.runtime_script),
        '--python-bin', args.python_bin,
        '--max-steps', str(args.runtime_max_steps),
        '--summary-json', str(candidate_summary),
        '--log-dir', str(eval_dir / 'candidate_logs'),
        '--rl-model-override', str(candidate_best),
        '--scenes', *args.scenes,
    ])

    baseline = load_json(baseline_summary)
    candidate = load_json(candidate_summary)
    baseline_score = compare_score(baseline)
    candidate_score = compare_score(candidate)

    report = {
        'position': args.position,
        'follow_distance': args.follow_distance,
        'obs_mode': args.obs_mode,
        'current_best_model': str(current_best),
        'candidate_best_model': str(candidate_best),
        'baseline_summary': str(baseline_summary),
        'candidate_summary': str(candidate_summary),
        'baseline_score': baseline_score,
        'candidate_score': candidate_score,
        'promoted': False,
    }

    if args.promote_if_better and candidate_score > baseline_score:
        backup_dir = assets_root / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if include_model.exists():
            shutil.copy2(include_model, backup_dir / f'{include_model.stem}_{timestamp}.zip')
        shutil.copy2(candidate_best, include_model)
        report['promoted'] = True
        report['promoted_to'] = str(include_model)

    report_path = run_root / 'report.json'
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f'\nSaved report to {report_path}')


if __name__ == '__main__':
    main()

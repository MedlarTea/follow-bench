import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
source_root_str = str(SOURCE_ROOT)
if source_root_str not in sys.path:
    sys.path.insert(0, source_root_str)

from follow_ahead_reaction.runtime.fixed_speed import (  # noqa: E402
    build_parser,
    main,
    resolve_config_dir,
)


def resolve_world_name(args):
    config_dir = resolve_config_dir(args.config_path)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    indexed_world = config_dir / f"{args.scenario}_{args.position}_{args.index}.yaml"
    plain_world = config_dir / f"{args.scenario}_{args.position}.yaml"

    if args.log_path and indexed_world.exists():
        return indexed_world
    if plain_world.exists():
        return plain_world
    if indexed_world.exists():
        return indexed_world
    raise FileNotFoundError(f"Scenario yaml not found: {indexed_world}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(str(resolve_world_name(args)), args)

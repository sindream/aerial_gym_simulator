import argparse
import distutils.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "aerial_gym" / "rl_training" / "rl_games" / "runner_multimodal.py"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "aerial_gym"
    / "rl_training"
    / "rl_games"
    / "ppo_aigp_racing_segmentation.yaml"
)


def _str_bool(value):
    return "True" if value else "False"


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return bool(distutils.util.strtobool(str(value)))


def main():
    parser = argparse.ArgumentParser(description="Train the AIGP segmentation racing policy.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", action="store_true", help="Disable headless mode.")
    parser.add_argument("--use_warp", type=_parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--experiment_name", default=None)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(RUNNER),
        "--train",
        "--file",
        args.config,
        "--num_envs",
        str(args.num_envs),
        "--headless",
        _str_bool(args.headless and not args.gui),
        "--use_warp",
        _str_bool(args.use_warp),
        "--seed",
        str(args.seed),
        "--task",
        "aigp_racing_segmentation_task",
    ]
    if args.experiment_name is not None:
        command.extend(["--experiment_name", args.experiment_name])

    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()

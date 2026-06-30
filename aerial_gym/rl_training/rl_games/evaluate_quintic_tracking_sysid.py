import argparse
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
    / "ppo_quintic_tracking_sysid.yaml"
)


def _str_bool(value):
    return "True" if value else "False"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a quintic tracking SysID checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", action="store_true", help="Disable headless mode.")
    parser.add_argument("--use_warp", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(RUNNER),
        "--play",
        "--file",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--num_envs",
        str(args.num_envs),
        "--headless",
        _str_bool(args.headless and not args.gui),
        "--use_warp",
        _str_bool(args.use_warp),
        "--seed",
        str(args.seed),
        "--task",
        "quintic_tracking_sysid_task",
    ]
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()

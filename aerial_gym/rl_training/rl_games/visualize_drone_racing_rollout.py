#!/usr/bin/env python3
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Roll out a trained drone racing policy and visualize its trajectory."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="aerial_gym/rl_training/rl_games/ppo_drone_racing_multimodal.yaml",
        help="Path to the rl_games yaml config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load. Required unless --random-policy is used.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task override. Defaults to the task from the yaml config.",
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=True,
        help="Whether to run Isaac Gym headless during rollout.",
    )
    parser.add_argument(
        "--use-warp",
        type=str2bool,
        default=None,
        help="Optional use_warp override. If omitted, the yaml value is used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch/Isaac device string for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional rollout seed override.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of rollout episodes to visualize.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode. Defaults to the task episode length.",
    )
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        default=True,
        help="Use deterministic policy actions.",
    )
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Ignore checkpoints and run with random actions for smoke testing.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the matplotlib window instead of only saving files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rollout_viz",
        help="Directory for saved plots and rollout data.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional output file prefix.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print rollout progress every N steps. Use 0 to disable step progress logs.",
    )
    return parser.parse_args()


def configure_matplotlib(show_plot):
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/mplconfig_aerial_gym"))
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    import matplotlib

    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def extract_policy_observation_space(task):
    if hasattr(task.observation_space, "spaces"):
        vector_key = "state" if "state" in task.observation_space.spaces else "observations"
        policy_spaces = {
            vector_key: task.observation_space.spaces[vector_key],
        }
        if "img_observation" in task.observation_space.spaces:
            policy_spaces["img_observation"] = task.observation_space.spaces["img_observation"]
        from gym import spaces

        return spaces.Dict(policy_spaces)
    return task.observation_space


def build_player(eval_config, observation_space, action_space, checkpoint_path, device_name):
    from aerial_gym.rl_training.rl_games.runner_multimodal import MultimodalRunner

    runner = MultimodalRunner()
    runner.load(eval_config)
    runner.params["config"]["device_name"] = device_name
    runner.params["config"]["env_info"] = {
        "action_space": action_space,
        "observation_space": observation_space,
    }
    runner.params["config"]["vec_env"] = None

    player = runner.create_player()
    player.restore(str(checkpoint_path))
    player.reset()
    return player


def make_eval_config(config_dict, args):
    eval_config = copy.deepcopy(config_dict)
    params = eval_config["params"]
    params["config"]["num_actors"] = 1
    params["config"]["device_name"] = args.device
    params["config"]["player"] = copy.deepcopy(params["config"].get("player", {}))
    params["config"]["player"]["render"] = False
    params["config"]["player"]["deterministic"] = args.deterministic
    params["config"]["player"]["games_num"] = 1
    params["config"]["player"]["use_vecenv"] = False
    return eval_config


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def snapshot_position(task):
    return to_numpy(task.obs_dict["robot_position"][0]).copy()


def snapshot_speed(task):
    return float(np.linalg.norm(to_numpy(task.obs_dict["robot_linvel"][0])))


def extract_info_position(infos, task):
    position_tensor = get_info_tensor(infos, "robot_position", task.obs_dict["robot_position"])
    return to_numpy(position_tensor[0]).copy()


def extract_info_speed(infos, task):
    linvel_tensor = get_info_tensor(infos, "robot_linvel", task.obs_dict["robot_linvel"])
    return float(np.linalg.norm(to_numpy(linvel_tensor[0])))


def get_info_tensor(infos, key, fallback_tensor):
    if isinstance(infos, dict) and key in infos:
        return infos[key]
    return fallback_tensor


def extract_policy_obs(task_obs):
    if not isinstance(task_obs, dict):
        return task_obs
    vector_key = "state" if "state" in task_obs else "observations"
    policy_obs = {vector_key: task_obs[vector_key][0]}
    if "img_observation" in task_obs:
        policy_obs["img_observation"] = task_obs["img_observation"][0]
    return policy_obs


def rollout_episode(task, player, args, episode_index):
    print(
        f"[rollout] episode {episode_index + 1}: reset "
        f"(max_steps={args.max_steps or int(task.task_config.episode_len_steps)})"
    )
    task.task_config.return_state_before_reset = False
    task_obs, *_ = task.reset()
    obs = extract_policy_obs(task_obs)
    if player is not None:
        player.reset()

    max_steps = args.max_steps or int(task.task_config.episode_len_steps)

    rollout = {
        "episode_index": episode_index,
        "positions": [snapshot_position(task)],
        "speeds": [snapshot_speed(task)],
        "actions": [],
        "rewards": [],
        "gate_indices": [int(task.current_gate_index[0].item())],
        "lap_counts": [int(task.completed_laps[0].item())],
        "gate_pass_steps": [],
        "crash_steps": [],
        "done": False,
        "success": False,
        "crash": False,
    }

    action_dim = int(task.task_config.action_space_dim)
    last_info = {}

    for step in range(max_steps):
        if player is None:
            action = 2.0 * np.random.rand(action_dim).astype(np.float32) - 1.0
            action = task.rewards.new_tensor(action)
        else:
            action = player.get_action(obs, is_deterministic=args.deterministic)
            if not hasattr(action, "detach"):
                action = task.rewards.new_tensor(action)
            action = action.to(task.rewards.device).flatten()

        task_obs, rewards, terminated, truncated, infos = task.step(action.unsqueeze(0))
        obs = extract_policy_obs(task_obs)
        last_info = infos

        reward_value = float(rewards[0].item())
        done = bool((terminated[0] | truncated[0]).item())
        gate_passed = bool(
            get_info_tensor(infos, "gate_passed", task.successes.new_zeros(task.successes.shape))[0].item()
        )
        crashed = bool(
            get_info_tensor(infos, "crashes", task.terminations)[0].item()
        )
        succeeded = bool(
            get_info_tensor(infos, "successes", task.successes)[0].item()
        )

        rollout["actions"].append(to_numpy(action))
        rollout["rewards"].append(reward_value)
        rollout["gate_indices"].append(
            int(get_info_tensor(infos, "current_gate_index", task.current_gate_index)[0].item())
        )
        rollout["lap_counts"].append(
            int(get_info_tensor(infos, "lap_count", task.completed_laps)[0].item())
        )

        if gate_passed:
            rollout["gate_pass_steps"].append(step)
        if crashed:
            rollout["crash_steps"].append(step)

        if done:
            rollout["positions"].append(extract_info_position(infos, task))
            rollout["speeds"].append(extract_info_speed(infos, task))
        else:
            rollout["positions"].append(snapshot_position(task))
            rollout["speeds"].append(snapshot_speed(task))

        if args.progress_every > 0 and (
            step == 0 or (step + 1) % args.progress_every == 0 or done
        ):
            print(
                f"[rollout] episode {episode_index + 1} step {step + 1}/"
                f"{max_steps} reward={reward_value:.3f} "
                f"gate={rollout['gate_indices'][-1]} "
                f"lap={rollout['lap_counts'][-1]} "
                f"done={done}"
            )

        if done:
            rollout["done"] = True
            rollout["success"] = succeeded
            rollout["crash"] = crashed
            break

    rollout["steps"] = len(rollout["rewards"])
    rollout["total_reward"] = float(np.sum(rollout["rewards"], dtype=np.float64))
    rollout["final_gate_index"] = int(rollout["gate_indices"][-1])
    rollout["final_lap_count"] = int(rollout["lap_counts"][-1])
    rollout["max_speed"] = float(np.max(rollout["speeds"])) if rollout["speeds"] else 0.0
    rollout["mean_speed"] = float(np.mean(rollout["speeds"])) if rollout["speeds"] else 0.0
    rollout["terminal_info"] = {
        key: to_numpy(value).tolist() if hasattr(value, "shape") else value
        for key, value in last_info.items()
    }
    return rollout


def add_gate_geometry_2d(ax, gate_positions, gate_yaws, gate_half_width):
    for gate_index, (center, yaw) in enumerate(zip(gate_positions, gate_yaws), start=1):
        right = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        left_corner = center[:2] - gate_half_width * right
        right_corner = center[:2] + gate_half_width * right
        ax.plot(
            [left_corner[0], right_corner[0]],
            [left_corner[1], right_corner[1]],
            color="tab:orange",
            linewidth=2.0,
        )
        ax.quiver(
            center[0],
            center[1],
            forward[0],
            forward[1],
            angles="xy",
            scale_units="xy",
            scale=0.4,
            color="tab:red",
            width=0.003,
        )
        ax.text(center[0], center[1], str(gate_index), fontsize=8, color="tab:brown")


def add_gate_geometry_3d(ax, gate_positions, gate_yaws, gate_half_width, gate_half_height):
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for center, yaw in zip(gate_positions, gate_yaws):
        right = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
        corners = np.stack(
            [
                center - gate_half_width * right - gate_half_height * up,
                center + gate_half_width * right - gate_half_height * up,
                center + gate_half_width * right + gate_half_height * up,
                center - gate_half_width * right + gate_half_height * up,
                center - gate_half_width * right - gate_half_height * up,
            ],
            axis=0,
        )
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="tab:orange", linewidth=1.0)


def set_equal_3d_axes(ax, positions, gate_positions, gate_half_width, gate_half_height, obstacle_positions=None):
    all_points = [positions, gate_positions]
    if obstacle_positions is not None and len(obstacle_positions) > 0:
        all_points.append(obstacle_positions)
    all_points = np.concatenate(all_points, axis=0)
    min_xyz = all_points.min(axis=0).astype(np.float32)
    max_xyz = all_points.max(axis=0).astype(np.float32)

    min_xyz[0:2] -= gate_half_width
    max_xyz[0:2] += gate_half_width
    min_xyz[2] -= gate_half_height
    max_xyz[2] += gate_half_height

    center = 0.5 * (min_xyz + max_xyz)
    half_range = 0.5 * np.max(max_xyz - min_xyz)
    half_range = max(float(half_range), 1.0)

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def split_position_segments(positions, jump_threshold=5.0):
    if len(positions) <= 1:
        return [positions]

    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    split_indices = np.where(deltas > jump_threshold)[0] + 1
    return [segment for segment in np.split(positions, split_indices) if len(segment) > 0]


def extract_gate_yaws(task):
    if hasattr(task, "gate_yaws"):
        return to_numpy(task.gate_yaws)
    if hasattr(task, "gate_quats"):
        gate_quats = to_numpy(task.gate_quats)
        qx = gate_quats[:, 0]
        qy = gate_quats[:, 1]
        qz = gate_quats[:, 2]
        qw = gate_quats[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
        return np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)
    raise AttributeError("Task does not expose gate_yaws or gate_quats")


def extract_obstacle_positions(task):
    num_obstacles = int(getattr(task.task_config, "num_random_cylinders", 0))
    if num_obstacles <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if "env_asset_state_tensor" not in task.obs_dict:
        return np.zeros((0, 3), dtype=np.float32)
    return to_numpy(task.obs_dict["env_asset_state_tensor"][0, :num_obstacles, 0:3]).astype(
        np.float32
    )


def plot_rollout_matplotlib(task, rollout, save_path, show_plot):
    plt = configure_matplotlib(show_plot)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    positions = np.asarray(rollout["positions"], dtype=np.float32)
    speeds = np.asarray(rollout["speeds"], dtype=np.float32)
    position_segments = split_position_segments(positions)
    gate_positions = to_numpy(task.gate_positions)
    gate_yaws = extract_gate_yaws(task)
    obstacle_positions = extract_obstacle_positions(task)
    obstacle_radius = float(getattr(task.task_config, "random_cylinder_radius_m", 0.0))
    gate_half_width = float(task.task_config.gate_pass_half_width)
    gate_half_height = float(task.task_config.gate_pass_half_height)

    figure = plt.figure(figsize=(18, 7))
    ax_3d = figure.add_subplot(1, 3, 1, projection="3d")
    ax_top = figure.add_subplot(1, 3, 2)
    ax_speed = figure.add_subplot(1, 3, 3)

    add_gate_geometry_3d(ax_3d, gate_positions, gate_yaws, gate_half_width, gate_half_height)
    for segment in position_segments:
        ax_3d.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="tab:blue", linewidth=2.0)
    ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="tab:green", s=50, label="start")
    ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="tab:red", s=50, label="end")
    ax_3d.scatter(gate_positions[:, 0], gate_positions[:, 1], gate_positions[:, 2], color="tab:orange", s=18)
    if len(obstacle_positions) > 0:
        ax_3d.scatter(
            obstacle_positions[:, 0],
            obstacle_positions[:, 1],
            obstacle_positions[:, 2],
            color="tab:cyan",
            s=30,
            label="obstacles",
        )
    ax_3d.set_title("3D Rollout")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    set_equal_3d_axes(
        ax_3d,
        positions,
        gate_positions,
        gate_half_width,
        gate_half_height,
        obstacle_positions=obstacle_positions,
    )
    ax_3d.legend(loc="upper right")

    add_gate_geometry_2d(ax_top, gate_positions, gate_yaws, gate_half_width)
    for segment in position_segments:
        ax_top.plot(segment[:, 0], segment[:, 1], color="tab:blue", linewidth=2.0, label="trajectory")
    ax_top.scatter(positions[0, 0], positions[0, 1], color="tab:green", s=50, label="start")
    ax_top.scatter(positions[-1, 0], positions[-1, 1], color="tab:red", s=50, label="end")
    if len(obstacle_positions) > 0:
        for obstacle_center in obstacle_positions:
            obstacle_circle = plt.Circle(
                (obstacle_center[0], obstacle_center[1]),
                obstacle_radius,
                color="tab:cyan",
                alpha=0.35,
            )
            ax_top.add_patch(obstacle_circle)
        ax_top.scatter(
            obstacle_positions[:, 0],
            obstacle_positions[:, 1],
            color="tab:cyan",
            s=20,
            label="obstacles",
        )

    if rollout["gate_pass_steps"]:
        pass_indices = np.clip(np.asarray(rollout["gate_pass_steps"]), 0, len(positions) - 1)
        pass_positions = positions[pass_indices]
        ax_top.scatter(
            pass_positions[:, 0],
            pass_positions[:, 1],
            color="gold",
            s=30,
            marker="o",
            label="gate pass",
        )

    if rollout["crash"]:
        ax_top.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color="black",
            s=60,
            marker="x",
            label="crash/end",
        )

    ax_top.set_title(
        f"Top-Down Path | reward={rollout['total_reward']:.2f}, "
        f"steps={rollout['steps']}, final_gate={rollout['final_gate_index']}"
    )
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    ax_top.axis("equal")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best")

    ax_speed.plot(np.arange(len(speeds)), speeds, color="tab:purple", linewidth=2.0)
    if rollout["gate_pass_steps"]:
        for step_index in rollout["gate_pass_steps"]:
            ax_speed.axvline(step_index, color="gold", linestyle="--", alpha=0.6)
    if rollout["crash"]:
        ax_speed.axvline(max(len(speeds) - 1, 0), color="black", linestyle=":", alpha=0.7)
    ax_speed.set_title(
        f"Speed Profile | max={rollout['max_speed']:.2f} m/s, "
        f"mean={rollout['mean_speed']:.2f} m/s"
    )
    ax_speed.set_xlabel("sample")
    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(figure)


def save_rollout_npz(task, rollout, save_path):
    np.savez_compressed(
        save_path,
        positions=np.asarray(rollout["positions"], dtype=np.float32),
        speeds=np.asarray(rollout["speeds"], dtype=np.float32),
        actions=np.asarray(rollout["actions"], dtype=np.float32),
        rewards=np.asarray(rollout["rewards"], dtype=np.float32),
        gate_indices=np.asarray(rollout["gate_indices"], dtype=np.int32),
        lap_counts=np.asarray(rollout["lap_counts"], dtype=np.int32),
        gate_pass_steps=np.asarray(rollout["gate_pass_steps"], dtype=np.int32),
        crash_steps=np.asarray(rollout["crash_steps"], dtype=np.int32),
        gate_positions=to_numpy(task.gate_positions),
        gate_yaws=extract_gate_yaws(task),
        obstacle_positions=extract_obstacle_positions(task),
    )


def main():
    args = get_args()
    if not args.random_policy and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --random-policy is used.")

    # Isaac Gym must be imported before torch.
    import isaacgym  # noqa: F401
    import torch

    from aerial_gym.registry.task_registry import task_registry

    config_path = Path(args.file).resolve()
    checkpoint_path = None if args.checkpoint is None else Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = load_yaml_config(config_path)
    eval_config = make_eval_config(config_dict, args)
    task_name = args.task or eval_config["params"]["config"]["env_name"]
    env_config = eval_config["params"]["config"].get("env_config", {})
    seed = args.seed if args.seed is not None else eval_config["params"].get("seed")
    use_warp = args.use_warp if args.use_warp is not None else env_config.get("use_warp")

    task_config_class = task_registry.get_task_config(task_name)
    previous_device = getattr(task_config_class, "device", "cuda:0")
    previous_return_state = getattr(task_config_class, "return_state_before_reset", False)

    try:
        task_config_class.device = args.device
        task_config_class.return_state_before_reset = True
        task = task_registry.make_task(
            task_name,
            seed=seed,
            num_envs=1,
            headless=args.headless,
            use_warp=use_warp,
        )
        task.task_config.return_state_before_reset = True

        player = None
        if not args.random_policy:
            observation_space = extract_policy_observation_space(task)
            player = build_player(
                eval_config,
                observation_space=observation_space,
                action_space=task.action_space,
                checkpoint_path=checkpoint_path,
                device_name=args.device,
            )
            print(f"[rollout] loaded checkpoint: {checkpoint_path}")
        else:
            print("[rollout] using random policy")

        run_prefix = args.prefix
        if run_prefix is None:
            run_prefix = "random_policy" if args.random_policy else checkpoint_path.stem

        summaries = []
        for episode_index in range(args.episodes):
            rollout = rollout_episode(task, player, args, episode_index)
            image_path = output_dir / f"{run_prefix}_episode_{episode_index:03d}.png"
            data_path = output_dir / f"{run_prefix}_episode_{episode_index:03d}.npz"

            plot_rollout_matplotlib(task, rollout, image_path, args.show)
            save_rollout_npz(task, rollout, data_path)

            summary = {
                "episode_index": episode_index,
                "image_path": str(image_path),
                "data_path": str(data_path),
                "total_reward": rollout["total_reward"],
                "steps": rollout["steps"],
                "done": rollout["done"],
                "success": rollout["success"],
                "crash": rollout["crash"],
                "final_gate_index": rollout["final_gate_index"],
                "final_lap_count": rollout["final_lap_count"],
            }
            summaries.append(summary)
            print(json.dumps(summary, indent=2))

        summary_path = output_dir / f"{run_prefix}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summaries, summary_file, indent=2)
        print(f"Saved summary to {summary_path}")
    finally:
        task_config_class.device = previous_device
        task_config_class.return_state_before_reset = previous_return_state
        if "task" in locals():
            task.close()


if __name__ == "__main__":
    main()

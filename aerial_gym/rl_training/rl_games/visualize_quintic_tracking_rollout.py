#!/usr/bin/env python3
import argparse
import copy
import json
import os
from pathlib import Path

import isaacgym  # noqa: F401
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "aerial_gym"
    / "rl_training"
    / "rl_games"
    / "ppo_quintic_tracking_sysid.yaml"
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def get_args():
    parser = argparse.ArgumentParser(
        description="Roll out and plot the quintic tracking SysID task."
    )
    parser.add_argument("--file", default=str(DEFAULT_CONFIG), help="rl_games yaml config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load.")
    parser.add_argument("--random-policy", action="store_true", help="Use random actions.")
    parser.add_argument("--headless", type=str2bool, default=True)
    parser.add_argument("--use-warp", type=str2bool, default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--output-dir", default="rollout_viz_quintic")
    parser.add_argument("--prefix", default="quintic_tracking")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
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


def make_task(config_dict, args):
    from aerial_gym.registry.task_registry import task_registry

    env_config = config_dict["params"]["config"].get("env_config", {})
    return task_registry.make_task(
        "quintic_tracking_sysid_task",
        num_envs=1,
        headless=args.headless,
        device=args.device,
        use_warp=args.use_warp if args.use_warp is not None else env_config.get("use_warp", False),
        seed=args.seed,
    )


def policy_observation_space(task):
    from gym import spaces

    return spaces.Dict({"state": task.observation_space.spaces["state"]})


def build_player(eval_config, task, checkpoint_path, device_name):
    from aerial_gym.rl_training.rl_games.runner_multimodal import MultimodalRunner

    runner = MultimodalRunner()
    runner.load(eval_config)
    runner.params["config"]["device_name"] = device_name
    runner.params["config"]["env_info"] = {
        "action_space": task.action_space,
        "observation_space": policy_observation_space(task),
    }
    runner.params["config"]["vec_env"] = None

    player = runner.create_player()
    player.restore(str(checkpoint_path))
    player.reset()
    return player


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_policy_obs(task_obs):
    return {"state": task_obs["state"][0]}


def snapshot_initial(task):
    return {
        "robot_position": to_numpy(task.obs_dict["robot_position"][0]).copy(),
        "robot_velocity": to_numpy(task.obs_dict["robot_linvel"][0]).copy(),
        "ref_position": to_numpy(task.ref_position[0]).copy(),
        "ref_velocity": to_numpy(task.ref_velocity[0]).copy(),
        "sysid_target": to_numpy(task.sysid_target[0]).copy(),
    }


def snapshot_from_infos(infos, task):
    if isinstance(infos, dict) and "robot_position" in infos:
        return {
            "robot_position": to_numpy(infos["robot_position"][0]).copy(),
            "robot_velocity": to_numpy(infos["robot_velocity"][0]).copy(),
            "ref_position": to_numpy(infos["ref_position"][0]).copy(),
            "ref_velocity": to_numpy(infos["ref_velocity"][0]).copy(),
            "sysid_target": to_numpy(infos["sysid_target"][0]).copy(),
        }
    return snapshot_initial(task)


def get_estimator_prediction(player):
    if player is None or not hasattr(player, "model"):
        return None
    model = player.model
    network = getattr(model, "a2c_network", None)
    prediction = getattr(network, "last_estimator_prediction", None)
    if prediction is None:
        return None
    return to_numpy(prediction[0]).copy()


def rollout_episode(task, player, args, episode_index):
    task.task_config.return_state_before_reset = False
    task_obs, *_ = task.reset()
    obs = extract_policy_obs(task_obs)
    if player is not None:
        player.reset()

    max_steps = args.max_steps or int(task.task_config.episode_len_steps)
    initial = snapshot_initial(task)
    rollout = {
        "episode_index": episode_index,
        "positions": [initial["robot_position"]],
        "ref_positions": [initial["ref_position"]],
        "velocities": [initial["robot_velocity"]],
        "ref_velocities": [initial["ref_velocity"]],
        "sysid_targets": [initial["sysid_target"]],
        "sysid_predictions": [],
        "actions": [],
        "rewards": [],
        "terminated": False,
        "truncated": False,
    }

    action_dim = int(task.task_config.action_space_dim)

    for step in range(max_steps):
        if player is None:
            low = torch.as_tensor(task.action_space.low, device=task.device)
            high = torch.as_tensor(task.action_space.high, device=task.device)
            action = low + (high - low) * torch.rand(action_dim, device=task.device)
        else:
            action = player.get_action(obs, is_deterministic=args.deterministic)
            if not hasattr(action, "detach"):
                action = torch.as_tensor(action, device=task.device, dtype=torch.float32)
            action = action.to(task.device).flatten()

        prediction = get_estimator_prediction(player)
        task_obs, rewards, terminated, truncated, infos = task.step(action.unsqueeze(0))
        obs = extract_policy_obs(task_obs)
        snap = snapshot_from_infos(infos, task)

        rollout["actions"].append(to_numpy(action).copy())
        rollout["rewards"].append(float(rewards[0].item()))
        rollout["positions"].append(snap["robot_position"])
        rollout["ref_positions"].append(snap["ref_position"])
        rollout["velocities"].append(snap["robot_velocity"])
        rollout["ref_velocities"].append(snap["ref_velocity"])
        rollout["sysid_targets"].append(snap["sysid_target"])
        if prediction is not None:
            rollout["sysid_predictions"].append(prediction)

        done = bool((terminated[0] | truncated[0]).item())
        if args.progress_every > 0 and (
            step == 0 or (step + 1) % args.progress_every == 0 or done
        ):
            pos_err = np.linalg.norm(snap["ref_position"] - snap["robot_position"])
            print(
                "[rollout] episode {} step {}/{} reward={:.3f} pos_err={:.3f} done={}".format(
                    episode_index + 1,
                    step + 1,
                    max_steps,
                    float(rewards[0].item()),
                    float(pos_err),
                    done,
                )
            )

        if done:
            rollout["terminated"] = bool(terminated[0].item())
            rollout["truncated"] = bool(truncated[0].item())
            break

    rollout["positions"] = np.asarray(rollout["positions"], dtype=np.float32)
    rollout["ref_positions"] = np.asarray(rollout["ref_positions"], dtype=np.float32)
    rollout["velocities"] = np.asarray(rollout["velocities"], dtype=np.float32)
    rollout["ref_velocities"] = np.asarray(rollout["ref_velocities"], dtype=np.float32)
    rollout["sysid_targets"] = np.asarray(rollout["sysid_targets"], dtype=np.float32)
    rollout["sysid_predictions"] = np.asarray(
        rollout["sysid_predictions"], dtype=np.float32
    )
    rollout["actions"] = np.asarray(rollout["actions"], dtype=np.float32)
    rollout["rewards"] = np.asarray(rollout["rewards"], dtype=np.float32)
    rollout["steps"] = int(len(rollout["rewards"]))
    rollout["total_reward"] = float(np.sum(rollout["rewards"], dtype=np.float64))
    return rollout


def set_axes_equal_3d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 0.5)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_rollout(plt, rollout, output_path, show_plot):
    positions = rollout["positions"]
    ref_positions = rollout["ref_positions"]
    pos_error = np.linalg.norm(ref_positions - positions, axis=1)
    vel_error = np.linalg.norm(
        rollout["ref_velocities"] - rollout["velocities"], axis=1
    )
    steps = np.arange(len(positions))
    action_steps = np.arange(len(rollout["actions"]))

    fig = plt.figure(figsize=(16, 10))
    ax_3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 3, 2)
    ax_z = fig.add_subplot(2, 3, 3)
    ax_err = fig.add_subplot(2, 3, 4)
    ax_actions = fig.add_subplot(2, 3, 5)
    ax_sysid = fig.add_subplot(2, 3, 6)

    ax_3d.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], label="ref")
    ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="actual")
    ax_3d.scatter(ref_positions[0, 0], ref_positions[0, 1], ref_positions[0, 2], marker="o")
    ax_3d.scatter(ref_positions[-1, 0], ref_positions[-1, 1], ref_positions[-1, 2], marker="x")
    set_axes_equal_3d(ax_3d, np.vstack([positions, ref_positions]))
    ax_3d.set_title("3D trajectory")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.legend()

    ax_xy.plot(ref_positions[:, 0], ref_positions[:, 1], label="ref")
    ax_xy.plot(positions[:, 0], positions[:, 1], label="actual")
    ax_xy.axis("equal")
    ax_xy.set_title("Top view")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.legend()

    ax_z.plot(steps, ref_positions[:, 2], label="ref z")
    ax_z.plot(steps, positions[:, 2], label="actual z")
    ax_z.set_title("Altitude")
    ax_z.set_xlabel("policy step")
    ax_z.set_ylabel("z [m]")
    ax_z.legend()

    ax_err.plot(steps, pos_error, label="position error")
    ax_err.plot(steps, vel_error, label="velocity error")
    ax_err.set_title("Tracking error")
    ax_err.set_xlabel("policy step")
    ax_err.legend()

    if len(rollout["actions"]) > 0:
        labels = ["throttle", "p", "q", "r"]
        for index, label in enumerate(labels):
            ax_actions.plot(action_steps, rollout["actions"][:, index], label=label)
    ax_actions.set_title("Actions")
    ax_actions.set_xlabel("policy step")
    ax_actions.legend()

    target = rollout["sysid_targets"]
    prediction = rollout["sysid_predictions"]
    labels = ["Kp/Ixx", "Kq/Iyy", "Kr/Izz", "hover"]
    if len(prediction) > 0:
        pred_steps = np.arange(len(prediction))
        for index, label in enumerate(labels):
            ax_sysid.plot(pred_steps, prediction[:, index], label="pred {}".format(label))
            ax_sysid.plot(
                steps,
                target[:, index],
                linestyle="--",
                linewidth=0.8,
                label="target {}".format(label),
            )
    else:
        ax_sysid.bar(np.arange(len(labels)), target[-1])
        ax_sysid.set_xticks(np.arange(len(labels)))
        ax_sysid.set_xticklabels(labels)
    ax_sysid.set_title("Normalized SysID")
    ax_sysid.set_ylim(-0.1, 1.1)
    handles, labels = ax_sysid.get_legend_handles_labels()
    if labels:
        ax_sysid.legend(handles, labels, fontsize=8)

    fig.suptitle(
        "steps={} total_reward={:.3f} terminated={} truncated={}".format(
            rollout["steps"],
            rollout["total_reward"],
            rollout["terminated"],
            rollout["truncated"],
        )
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show_plot:
        plt.show()
    plt.close(fig)


def save_summary(rollout, output_path):
    summary = {
        "steps": rollout["steps"],
        "total_reward": rollout["total_reward"],
        "terminated": rollout["terminated"],
        "truncated": rollout["truncated"],
        "final_position": rollout["positions"][-1].tolist(),
        "final_ref_position": rollout["ref_positions"][-1].tolist(),
        "final_position_error": float(
            np.linalg.norm(rollout["ref_positions"][-1] - rollout["positions"][-1])
        ),
        "final_velocity_error": float(
            np.linalg.norm(rollout["ref_velocities"][-1] - rollout["velocities"][-1])
        ),
        "final_sysid_target": rollout["sysid_targets"][-1].tolist(),
    }
    if len(rollout["sysid_predictions"]) > 0:
        summary["final_sysid_prediction"] = rollout["sysid_predictions"][-1].tolist()
    with open(output_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def main():
    args = get_args()
    if args.checkpoint is None and not args.random_policy:
        raise ValueError("Provide --checkpoint or use --random-policy.")

    plt = configure_matplotlib(args.show)
    config = load_yaml_config(args.file)
    config["params"]["config"]["env_config"]["num_envs"] = 1
    config["params"]["config"]["env_config"]["headless"] = args.headless
    config["params"]["config"]["env_config"]["use_warp"] = args.use_warp
    config["params"]["seed"] = args.seed
    config["params"]["config"]["env_config"]["seed"] = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = make_task(config, args)
    player = None
    if not args.random_policy:
        player = build_player(
            make_eval_config(config, args),
            task,
            Path(args.checkpoint),
            args.device,
        )

    try:
        for episode_index in range(args.episodes):
            rollout = rollout_episode(task, player, args, episode_index)
            image_path = output_dir / "{}_episode_{:03d}.png".format(
                args.prefix, episode_index
            )
            summary_path = output_dir / "{}_episode_{:03d}.json".format(
                args.prefix, episode_index
            )
            plot_rollout(plt, rollout, image_path, args.show)
            save_summary(rollout, summary_path)
            print("[rollout] saved {}".format(image_path))
            print("[rollout] saved {}".format(summary_path))
    finally:
        task.close()


if __name__ == "__main__":
    main()

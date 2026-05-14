import os
from pathlib import Path

import numpy as np

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
from aerial_gym.utils.logging import CustomLogger


logger = CustomLogger("velocity_control_racing_diagnostic")


def _build_profile(times, vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd):
    commands = np.zeros((len(times), 4), dtype=np.float32)

    segments = [
        (
            2.0,
            5.0,
            np.array([vx_cmd, 0.0, 0.0, 0.0], dtype=np.float32),
            f"vx {vx_cmd:+.2f}",
        ),
        (
            8.0,
            11.0,
            np.array([0.0, vy_cmd, 0.0, 0.0], dtype=np.float32),
            f"vy {vy_cmd:+.2f}",
        ),
        (
            14.0,
            17.0,
            np.array([0.0, 0.0, vz_cmd, 0.0], dtype=np.float32),
            f"vz {vz_cmd:+.2f}",
        ),
        (
            20.0,
            23.0,
            np.array([0.0, 0.0, 0.0, yaw_rate_cmd], dtype=np.float32),
            f"yaw rate {yaw_rate_cmd:+.2f}",
        ),
    ]

    labels = []
    for start_t, end_t, value, label in segments:
        mask = (times >= start_t) & (times < end_t)
        commands[mask] = value
        labels.append((start_t, end_t, label))
    return commands, labels


def _compute_segment_metrics(times, commanded, observed, labels):
    metrics = []
    channel_names = ["vx", "vy", "vz", "yaw_rate"]
    for start_t, end_t, label in labels:
        mask = (times >= start_t) & (times < end_t)
        if not np.any(mask):
            continue
        cmd_slice = commanded[mask]
        obs_slice = observed[mask]
        metric = {"segment": label, "time_range": [float(start_t), float(end_t)]}
        for idx, name in enumerate(channel_names):
            target = float(np.mean(cmd_slice[:, idx]))
            signal = obs_slice[:, idx]
            error = signal - target
            metric[f"{name}_target"] = target
            metric[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
            metric[f"{name}_steady_state_mean"] = float(np.mean(signal[-max(1, len(signal)//4):]))
            metric[f"{name}_peak_abs"] = float(np.max(np.abs(signal)))
        metrics.append(metric)
    return metrics


def _plot_results(times, commanded, observed, positions, eulers, labels, save_path, show_plot):
    import matplotlib

    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    channel_names = ["vx", "vy", "vz", "yaw_rate"]
    channel_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, (name, color) in enumerate(zip(channel_names, channel_colors)):
        axes[0].plot(times, observed[:, idx], color=color, linewidth=2.0, label=f"{name} obs")
        axes[0].plot(
            times,
            commanded[:, idx],
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label=f"{name} cmd",
        )
    axes[0].set_title("Velocity Controller Tracking")
    axes[0].set_ylabel("m/s or rad/s")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=9)

    axes[1].plot(times, positions[:, 0], label="x", color="tab:blue")
    axes[1].plot(times, positions[:, 1], label="y", color="tab:orange")
    axes[1].plot(times, positions[:, 2], label="z", color="tab:green")
    axes[1].set_title("Position")
    axes[1].set_ylabel("m")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(times, eulers[:, 0], label="roll", color="tab:blue")
    axes[2].plot(times, eulers[:, 1], label="pitch", color="tab:orange")
    axes[2].plot(times, eulers[:, 2], label="yaw", color="tab:green")
    axes[2].set_title("Euler Angles")
    axes[2].set_ylabel("rad")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    for ax in axes:
        for start_t, end_t, label in labels:
            ax.axvspan(start_t, end_t, color="gray", alpha=0.08)
            ax.axvline(start_t, color="gray", alpha=0.2, linestyle=":")

    figure.tight_layout()
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    args = get_args(
        additional_parameters=[
            {
                "name": "--sim_time",
                "type": float,
                "default": 26.0,
                "help": "Total simulated time in seconds.",
            },
            {
                "name": "--device",
                "type": str,
                "default": "cuda:0",
                "help": "Torch / sim device string.",
            },
            {
                "name": "--save_path",
                "type": str,
                "default": "velocity_control_racing_diagnostic.png",
                "help": "Output plot path.",
            },
            {
                "name": "--vx_cmd",
                "type": float,
                "default": 1.5,
                "help": "Forward velocity step command in m/s.",
            },
            {
                "name": "--vy_cmd",
                "type": float,
                "default": 1.0,
                "help": "Lateral velocity step command in m/s.",
            },
            {
                "name": "--vz_cmd",
                "type": float,
                "default": 0.8,
                "help": "Vertical velocity step command in m/s.",
            },
            {
                "name": "--yaw_rate_cmd",
                "type": float,
                "default": 0.8,
                "help": "Yaw-rate step command in rad/s.",
            },
        ]
    )

    save_path = Path(args.save_path).resolve()
    if not args.headless and not os.environ.get("DISPLAY"):
        logger.warning(
            "DISPLAY is not set, so the Isaac Gym viewer cannot open here. "
            "Falling back to headless mode and saving the diagnostic plot only."
        )
        args.headless = True

    logger.warning(f"Diagnostic plot will be saved to: {save_path}")
    logger.warning("Running racing velocity-controller diagnostic with base_quadrotor_racing.")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor_racing",
        controller_name="lee_velocity_control",
        args=None,
        device=args.device,
        num_envs=1,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    env_manager.reset()
    obs_dict = env_manager.get_obs()
    dt = float(obs_dict["dt"])
    num_steps = int(args.sim_time / dt)
    times = np.arange(num_steps, dtype=np.float32) * dt

    controller_cfg = env_manager.robot_manager.robot.controller.cfg
    logger.warning(
        f"Controller gains: K_vel_max={controller_cfg.K_vel_tensor_max}, "
        f"K_vel_min={controller_cfg.K_vel_tensor_min}, "
        f"K_rot_max={controller_cfg.K_rot_tensor_max}, "
        f"K_rot_min={controller_cfg.K_rot_tensor_min}, "
        f"K_angvel_max={controller_cfg.K_angvel_tensor_max}, "
        f"K_angvel_min={controller_cfg.K_angvel_tensor_min}, "
        f"max_yaw_rate={controller_cfg.max_yaw_rate}"
    )

    commanded, labels = _build_profile(
        times,
        args.vx_cmd,
        args.vy_cmd,
        args.vz_cmd,
        args.yaw_rate_cmd,
    )

    import torch

    actions = torch.zeros((1, 4), device=args.device, dtype=torch.float32)
    observed = np.zeros((num_steps, 4), dtype=np.float32)
    positions = np.zeros((num_steps, 3), dtype=np.float32)
    eulers = np.zeros((num_steps, 3), dtype=np.float32)

    for step_idx in range(num_steps):
        actions[0] = torch.tensor(commanded[step_idx], device=args.device)
        env_manager.step(actions=actions)

        observed[step_idx, 0:3] = obs_dict["robot_vehicle_linvel"][0, 0:3].detach().cpu().numpy()
        observed[step_idx, 3] = obs_dict["robot_body_angvel"][0, 2].detach().cpu().item()
        positions[step_idx] = obs_dict["robot_position"][0].detach().cpu().numpy()
        eulers[step_idx] = obs_dict["robot_euler_angles"][0].detach().cpu().numpy()

    metrics = _compute_segment_metrics(times, commanded, observed, labels)
    _plot_results(times, commanded, observed, positions, eulers, labels, save_path, not args.headless)

    logger.warning(f"Saved plot to {save_path}")
    for metric in metrics:
        logger.warning(str(metric))

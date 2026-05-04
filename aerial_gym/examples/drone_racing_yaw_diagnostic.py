import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import get_euler_xyz_tensor, ssa
from aerial_gym.utils.helpers import get_args
import torch


@dataclass
class LoggedSeries:
    time_s: np.ndarray
    yaw_rad: np.ndarray
    yaw_rate_rad_s: np.ndarray
    yaw_cmd: np.ndarray
    yaw_cmd_label: str
    z_m: np.ndarray
    vz_m_s: np.ndarray
    roll_rad: np.ndarray
    pitch_rad: np.ndarray
    thrust_cmd: np.ndarray
    roll_cmd_rad: np.ndarray
    pitch_cmd_rad: np.ndarray
    controller_force_z: np.ndarray
    controller_torque_z: np.ndarray
    action: np.ndarray
    title: str


def _safe_close_env(env):
    try:
        ige_env = getattr(env, "IGE_env", None)
        if ige_env is not None:
            gym = getattr(ige_env, "gym", None)
            viewer_wrapper = getattr(ige_env, "viewer", None)
            sim = getattr(ige_env, "sim", None)
            raw_viewer = getattr(viewer_wrapper, "viewer", viewer_wrapper)
            if gym is not None and raw_viewer is not None:
                try:
                    gym.destroy_viewer(raw_viewer)
                except Exception:
                    pass
                ige_env.viewer = None
            if gym is not None and sim is not None:
                try:
                    gym.destroy_sim(sim)
                except Exception:
                    pass
                ige_env.sim = None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _set_robot_hover_state(env, z_m: float = 1.0, yaw_rad: float = 0.0):
    tensor_dict = env.get_obs()
    robot_state = tensor_dict["robot_state_tensor"]
    robot_state[:, 0:3] = 0.0
    robot_state[:, 2] = z_m
    robot_state[:, 3:7] = 0.0
    robot_state[:, 6] = 1.0
    if abs(yaw_rad) > 1.0e-6:
        half_yaw = 0.5 * yaw_rad
        robot_state[:, 3] = 0.0
        robot_state[:, 4] = 0.0
        robot_state[:, 5] = np.sin(half_yaw)
        robot_state[:, 6] = np.cos(half_yaw)
    robot_state[:, 7:13] = 0.0
    if "robot_actions" in tensor_dict:
        tensor_dict["robot_actions"][:] = 0.0
    if "robot_prev_actions" in tensor_dict:
        tensor_dict["robot_prev_actions"][:] = 0.0
    env.robot_manager.robot.update_states()
    env.IGE_env.write_to_sim()


def _unwrap_angle(angle_rad: np.ndarray) -> np.ndarray:
    return np.unwrap(angle_rad)


def _plot_series(series: LoggedSeries, output_path: Optional[str], show_plot: bool):
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)
    fig.suptitle(series.title, fontsize=14)

    axes[0].plot(series.time_s, series.yaw_rad, label="yaw")
    if "angle" in series.yaw_cmd_label:
        axes[0].plot(series.time_s, series.yaw_cmd, label=series.yaw_cmd_label, linestyle="--")
    axes[0].set_ylabel("yaw [rad]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(series.time_s, series.yaw_rate_rad_s, label="yaw_rate")
    if "rate" in series.yaw_cmd_label:
        axes[1].plot(series.time_s, series.yaw_cmd, label=series.yaw_cmd_label, linestyle="--")
    axes[1].set_ylabel("yaw rate [rad/s]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(series.time_s, series.z_m, label="z")
    axes[2].plot(series.time_s, series.vz_m_s, label="vz")
    axes[2].set_ylabel("z / vz")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(series.time_s, series.roll_rad, label="roll")
    axes[3].plot(series.time_s, series.pitch_rad, label="pitch")
    axes[3].plot(series.time_s, series.roll_cmd_rad, label="roll_cmd", linestyle="--")
    axes[3].plot(series.time_s, series.pitch_cmd_rad, label="pitch_cmd", linestyle="--")
    axes[3].set_ylabel("att [rad]")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(series.time_s, series.thrust_cmd, label="thrust_cmd")
    axes[4].plot(series.time_s, series.controller_force_z, label="controller_fz")
    axes[4].plot(series.time_s, series.controller_torque_z, label="controller_tz")
    axes[4].set_ylabel("ctrl out")
    axes[4].grid(True)
    axes[4].legend()

    for action_index in range(series.action.shape[1]):
        axes[5].plot(series.time_s, series.action[:, action_index], label=f"action[{action_index}]")
    axes[5].set_ylabel("policy action")
    axes[5].set_xlabel("time [s]")
    axes[5].grid(True)
    axes[5].legend()

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _attitude_direct_mode(args) -> LoggedSeries:
    env = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="monorace_paper_camera_quadrotor",
        controller_name="monorace_racing_attitude_control",
        args=None,
        device="cuda:0",
        num_envs=1,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    env.reset()
    tensor_dict = env.get_obs()
    _set_robot_hover_state(env, z_m=args.hover_target_z_m, yaw_rad=args.initial_yaw_rad)
    dt = float(env.sim_config.sim.dt)
    total_steps = int(args.duration_s / dt)

    actions = torch.zeros((1, 4), device="cuda:0", dtype=torch.float32)

    yaw_rate_profile = np.zeros(total_steps, dtype=np.float32)
    start_step = int(args.step_start_s / dt)
    stop_step = int(args.step_stop_s / dt)
    reverse_start_step = int(args.reverse_start_s / dt)
    reverse_stop_step = int(args.reverse_stop_s / dt)
    yaw_rate_profile[start_step:stop_step] = args.yaw_rate_action
    yaw_rate_profile[reverse_start_step:reverse_stop_step] = -args.yaw_rate_action

    yaw_log = np.zeros(total_steps, dtype=np.float32)
    yaw_rate_log = np.zeros(total_steps, dtype=np.float32)
    yaw_rate_cmd_log = np.zeros(total_steps, dtype=np.float32)
    z_log = np.zeros(total_steps, dtype=np.float32)
    vz_log = np.zeros(total_steps, dtype=np.float32)
    roll_log = np.zeros(total_steps, dtype=np.float32)
    pitch_log = np.zeros(total_steps, dtype=np.float32)
    thrust_cmd_log = np.zeros(total_steps, dtype=np.float32)
    roll_cmd_log = np.zeros(total_steps, dtype=np.float32)
    pitch_cmd_log = np.zeros(total_steps, dtype=np.float32)
    controller_force_z_log = np.zeros(total_steps, dtype=np.float32)
    controller_torque_z_log = np.zeros(total_steps, dtype=np.float32)
    action_log = np.zeros((total_steps, 4), dtype=np.float32)
    time_log = np.arange(total_steps, dtype=np.float32) * dt

    for step in range(total_steps):
        actions[:] = 0.0
        z_error = float(args.hover_target_z_m - tensor_dict["robot_position"][0, 2].item())
        z_vel = float(tensor_dict["robot_linvel"][0, 2].item())
        thrust_action = (
            args.hover_thrust_action
            + args.hover_kp * z_error
            - args.hover_kd * z_vel
        )
        actions[:, 0] = float(np.clip(thrust_action, -1.0, 1.0))
        actions[:, 1] = float(
            np.clip(
                -args.level_kp * tensor_dict["robot_euler_angles"][0, 0].item(),
                -args.max_level_angle_rad,
                args.max_level_angle_rad,
            )
        )
        actions[:, 2] = float(
            np.clip(
                -args.level_kp * tensor_dict["robot_euler_angles"][0, 1].item(),
                -args.max_level_angle_rad,
                args.max_level_angle_rad,
            )
        )
        actions[:, 3] = float(yaw_rate_profile[step])
        env.step(actions)
        controller = env.robot_manager.robot.controller

        z_log[step] = tensor_dict["robot_position"][0, 2].item()
        vz_log[step] = tensor_dict["robot_linvel"][0, 2].item()
        roll_log[step] = tensor_dict["robot_euler_angles"][0, 0].item()
        pitch_log[step] = tensor_dict["robot_euler_angles"][0, 1].item()
        yaw_log[step] = tensor_dict["robot_euler_angles"][0, 2].item()
        yaw_rate_log[step] = tensor_dict["robot_body_angvel"][0, 2].item()
        yaw_rate_cmd_log[step] = actions[0, 3].item()
        thrust_cmd_log[step] = actions[0, 0].item()
        roll_cmd_log[step] = actions[0, 1].item()
        pitch_cmd_log[step] = actions[0, 2].item()
        controller_force_z_log[step] = controller.wrench_command[0, 2].item()
        controller_torque_z_log[step] = controller.wrench_command[0, 5].item()
        action_log[step] = actions[0].detach().cpu().numpy()

    _safe_close_env(env)

    return LoggedSeries(
        time_s=time_log,
        yaw_rad=_unwrap_angle(yaw_log),
        yaw_rate_rad_s=yaw_rate_log,
        yaw_cmd=yaw_rate_cmd_log,
        yaw_cmd_label="yaw_rate_cmd",
        z_m=z_log,
        vz_m_s=vz_log,
        roll_rad=roll_log,
        pitch_rad=pitch_log,
        thrust_cmd=thrust_cmd_log,
        roll_cmd_rad=roll_cmd_log,
        pitch_cmd_rad=pitch_cmd_log,
        controller_force_z=controller_force_z_log,
        controller_torque_z=controller_torque_z_log,
        action=action_log,
        title="Yaw Diagnostic: Direct monorace_racing_attitude_control yaw-rate step",
    )


def _position_mode(args) -> LoggedSeries:
    env = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="monorace_paper_camera_quadrotor",
        controller_name="monorace_racing_position_control",
        args=None,
        device="cuda:0",
        num_envs=1,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    env.reset()
    tensor_dict = env.get_obs()
    _set_robot_hover_state(env, z_m=args.hover_target_z_m, yaw_rad=args.initial_yaw_rad)
    dt = float(env.sim_config.sim.dt)
    total_steps = int(args.duration_s / dt)

    actions = torch.zeros((1, 4), device="cuda:0", dtype=torch.float32)
    actions[:, 2] = args.hover_target_z_m

    yaw_profile = np.full(total_steps, args.initial_yaw_rad, dtype=np.float32)
    start_step = int(args.step_start_s / dt)
    stop_step = int(args.step_stop_s / dt)
    reverse_start_step = int(args.reverse_start_s / dt)
    reverse_stop_step = int(args.reverse_stop_s / dt)
    yaw_profile[start_step:stop_step] = args.initial_yaw_rad + args.yaw_angle_step_rad
    yaw_profile[reverse_start_step:reverse_stop_step] = (
        args.initial_yaw_rad - args.yaw_angle_step_rad
    )

    yaw_log = np.zeros(total_steps, dtype=np.float32)
    yaw_rate_log = np.zeros(total_steps, dtype=np.float32)
    yaw_cmd_log = np.zeros(total_steps, dtype=np.float32)
    z_log = np.zeros(total_steps, dtype=np.float32)
    vz_log = np.zeros(total_steps, dtype=np.float32)
    roll_log = np.zeros(total_steps, dtype=np.float32)
    pitch_log = np.zeros(total_steps, dtype=np.float32)
    thrust_cmd_log = np.zeros(total_steps, dtype=np.float32)
    roll_cmd_log = np.zeros(total_steps, dtype=np.float32)
    pitch_cmd_log = np.zeros(total_steps, dtype=np.float32)
    controller_force_z_log = np.zeros(total_steps, dtype=np.float32)
    controller_torque_z_log = np.zeros(total_steps, dtype=np.float32)
    action_log = np.zeros((total_steps, 4), dtype=np.float32)
    time_log = np.arange(total_steps, dtype=np.float32) * dt

    for step in range(total_steps):
        actions[:, 0] = 0.0
        actions[:, 1] = 0.0
        actions[:, 2] = args.hover_target_z_m
        actions[:, 3] = float(yaw_profile[step])
        env.step(actions)
        controller = env.robot_manager.robot.controller
        desired_euler = ssa(get_euler_xyz_tensor(controller.desired_quat))

        z_log[step] = tensor_dict["robot_position"][0, 2].item()
        vz_log[step] = tensor_dict["robot_linvel"][0, 2].item()
        roll_log[step] = tensor_dict["robot_euler_angles"][0, 0].item()
        pitch_log[step] = tensor_dict["robot_euler_angles"][0, 1].item()
        yaw_log[step] = tensor_dict["robot_euler_angles"][0, 2].item()
        yaw_rate_log[step] = tensor_dict["robot_body_angvel"][0, 2].item()
        yaw_cmd_log[step] = actions[0, 3].item()
        thrust_cmd_log[step] = controller.wrench_command[0, 2].item()
        roll_cmd_log[step] = desired_euler[0, 0].item()
        pitch_cmd_log[step] = desired_euler[0, 1].item()
        controller_force_z_log[step] = controller.wrench_command[0, 2].item()
        controller_torque_z_log[step] = controller.wrench_command[0, 5].item()
        action_log[step] = actions[0].detach().cpu().numpy()

    _safe_close_env(env)

    return LoggedSeries(
        time_s=time_log,
        yaw_rad=_unwrap_angle(yaw_log),
        yaw_rate_rad_s=yaw_rate_log,
        yaw_cmd=_unwrap_angle(yaw_cmd_log),
        yaw_cmd_label="yaw_angle_cmd",
        z_m=z_log,
        vz_m_s=vz_log,
        roll_rad=roll_log,
        pitch_rad=pitch_log,
        thrust_cmd=thrust_cmd_log,
        roll_cmd_rad=roll_cmd_log,
        pitch_cmd_rad=pitch_cmd_log,
        controller_force_z=controller_force_z_log,
        controller_torque_z=controller_torque_z_log,
        action=action_log,
        title="Yaw Diagnostic: monorace_racing_position_control yaw-angle step",
    )


def _velocity_mode(args) -> LoggedSeries:
    env = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="monorace_paper_camera_quadrotor",
        controller_name="monorace_racing_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=1,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    env.reset()
    tensor_dict = env.get_obs()
    _set_robot_hover_state(env, z_m=args.hover_target_z_m, yaw_rad=args.initial_yaw_rad)
    dt = float(env.sim_config.sim.dt)
    total_steps = int(args.duration_s / dt)

    actions = torch.zeros((1, 4), device="cuda:0", dtype=torch.float32)

    yaw_rate_profile = np.zeros(total_steps, dtype=np.float32)
    start_step = int(args.step_start_s / dt)
    stop_step = int(args.step_stop_s / dt)
    reverse_start_step = int(args.reverse_start_s / dt)
    reverse_stop_step = int(args.reverse_stop_s / dt)
    yaw_rate_profile[start_step:stop_step] = args.velocity_yaw_rate_cmd_rad_s
    yaw_rate_profile[reverse_start_step:reverse_stop_step] = -args.velocity_yaw_rate_cmd_rad_s

    yaw_log = np.zeros(total_steps, dtype=np.float32)
    yaw_rate_log = np.zeros(total_steps, dtype=np.float32)
    yaw_cmd_log = np.zeros(total_steps, dtype=np.float32)
    z_log = np.zeros(total_steps, dtype=np.float32)
    vz_log = np.zeros(total_steps, dtype=np.float32)
    roll_log = np.zeros(total_steps, dtype=np.float32)
    pitch_log = np.zeros(total_steps, dtype=np.float32)
    thrust_cmd_log = np.zeros(total_steps, dtype=np.float32)
    roll_cmd_log = np.zeros(total_steps, dtype=np.float32)
    pitch_cmd_log = np.zeros(total_steps, dtype=np.float32)
    controller_force_z_log = np.zeros(total_steps, dtype=np.float32)
    controller_torque_z_log = np.zeros(total_steps, dtype=np.float32)
    action_log = np.zeros((total_steps, 4), dtype=np.float32)
    time_log = np.arange(total_steps, dtype=np.float32) * dt

    for step in range(total_steps):
        actions[:, 0] = args.velocity_forward_m_s
        actions[:, 1] = args.velocity_lateral_m_s
        actions[:, 2] = args.velocity_vertical_m_s
        actions[:, 3] = float(yaw_rate_profile[step])
        env.step(actions)
        controller = env.robot_manager.robot.controller
        desired_euler = ssa(get_euler_xyz_tensor(controller.desired_quat))

        z_log[step] = tensor_dict["robot_position"][0, 2].item()
        vz_log[step] = tensor_dict["robot_linvel"][0, 2].item()
        roll_log[step] = tensor_dict["robot_euler_angles"][0, 0].item()
        pitch_log[step] = tensor_dict["robot_euler_angles"][0, 1].item()
        yaw_log[step] = tensor_dict["robot_euler_angles"][0, 2].item()
        yaw_rate_log[step] = tensor_dict["robot_body_angvel"][0, 2].item()
        yaw_cmd_log[step] = actions[0, 3].item()
        thrust_cmd_log[step] = controller.wrench_command[0, 2].item()
        roll_cmd_log[step] = desired_euler[0, 0].item()
        pitch_cmd_log[step] = desired_euler[0, 1].item()
        controller_force_z_log[step] = controller.wrench_command[0, 2].item()
        controller_torque_z_log[step] = controller.wrench_command[0, 5].item()
        action_log[step] = actions[0].detach().cpu().numpy()

    _safe_close_env(env)

    return LoggedSeries(
        time_s=time_log,
        yaw_rad=_unwrap_angle(yaw_log),
        yaw_rate_rad_s=yaw_rate_log,
        yaw_cmd=yaw_cmd_log,
        yaw_cmd_label="yaw_rate_cmd",
        z_m=z_log,
        vz_m_s=vz_log,
        roll_rad=roll_log,
        pitch_rad=pitch_log,
        thrust_cmd=thrust_cmd_log,
        roll_cmd_rad=roll_cmd_log,
        pitch_cmd_rad=pitch_cmd_log,
        controller_force_z=controller_force_z_log,
        controller_torque_z=controller_torque_z_log,
        action=action_log,
        title="Yaw Diagnostic: monorace_racing_velocity_control yaw-rate step",
    )


def _task_mode(args) -> LoggedSeries:
    import aerial_gym.task  # noqa: F401
    from aerial_gym.registry.task_registry import task_registry

    task = task_registry.make_task(
        "drone_racing_body_accel_task",
        seed=0,
        num_envs=1,
        headless=args.headless,
        device="cuda:0",
        use_warp=args.use_warp,
    )

    dt = float(task.dt)
    total_steps = int(args.duration_s / dt)

    action = torch.zeros((1, 3), device="cuda:0", dtype=torch.float32)

    yaw_log = np.zeros(total_steps, dtype=np.float32)
    yaw_rate_log = np.zeros(total_steps, dtype=np.float32)
    yaw_rate_cmd_log = np.zeros(total_steps, dtype=np.float32)
    z_log = np.zeros(total_steps, dtype=np.float32)
    vz_log = np.zeros(total_steps, dtype=np.float32)
    roll_log = np.zeros(total_steps, dtype=np.float32)
    pitch_log = np.zeros(total_steps, dtype=np.float32)
    thrust_cmd_log = np.zeros(total_steps, dtype=np.float32)
    roll_cmd_log = np.zeros(total_steps, dtype=np.float32)
    pitch_cmd_log = np.zeros(total_steps, dtype=np.float32)
    controller_force_z_log = np.zeros(total_steps, dtype=np.float32)
    controller_torque_z_log = np.zeros(total_steps, dtype=np.float32)
    action_log = np.zeros((total_steps, 3), dtype=np.float32)
    time_log = np.arange(total_steps, dtype=np.float32) * dt

    first_phase_end = int(args.task_phase1_end_s / dt)
    second_phase_end = int(args.task_phase2_end_s / dt)

    for step in range(total_steps):
        action[:] = 0.0
        action[:, 2] = args.task_hover_z_action
        if step < first_phase_end:
            action[:, 0] = args.task_phase1_xy_action
        elif step < second_phase_end:
            action[:, 1] = args.task_phase2_xy_action
        else:
            action[:, 0] = -args.task_phase3_xy_action

        task.step(action)
        controller = task.sim_env.robot_manager.robot.controller

        z_log[step] = task.obs_dict["robot_position"][0, 2].item()
        vz_log[step] = task.obs_dict["robot_linvel"][0, 2].item()
        roll_log[step] = task.obs_dict["robot_euler_angles"][0, 0].item()
        pitch_log[step] = task.obs_dict["robot_euler_angles"][0, 1].item()
        yaw_log[step] = task.obs_dict["robot_euler_angles"][0, 2].item()
        yaw_rate_log[step] = task.obs_dict["robot_body_angvel"][0, 2].item()
        yaw_rate_cmd_log[step] = task.attitude_commands[0, 3].item()
        thrust_cmd_log[step] = task.attitude_commands[0, 0].item()
        roll_cmd_log[step] = task.attitude_commands[0, 1].item()
        pitch_cmd_log[step] = task.attitude_commands[0, 2].item()
        controller_force_z_log[step] = controller.wrench_command[0, 2].item()
        controller_torque_z_log[step] = controller.wrench_command[0, 5].item()
        action_log[step] = action[0].detach().cpu().numpy()

    task.close()

    return LoggedSeries(
        time_s=time_log,
        yaw_rad=_unwrap_angle(yaw_log),
        yaw_rate_rad_s=yaw_rate_log,
        yaw_cmd=yaw_rate_cmd_log,
        yaw_cmd_label="task_yaw_rate_cmd",
        z_m=z_log,
        vz_m_s=vz_log,
        roll_rad=roll_log,
        pitch_rad=pitch_log,
        thrust_cmd=thrust_cmd_log,
        roll_cmd_rad=roll_cmd_log,
        pitch_cmd_rad=pitch_cmd_log,
        controller_force_z=controller_force_z_log,
        controller_torque_z=controller_torque_z_log,
        action=action_log,
        title="Yaw Diagnostic: drone_racing_body_accel_task action-to-yaw path",
    )


def main():
    args = get_args(
        [
            {
                "name": "--mode",
                "type": str,
                "default": "position",
                "help": "position, velocity, task, or attitude_direct",
            },
            {
                "name": "--duration_s",
                "type": float,
                "default": 8.0,
                "help": "Total diagnostic duration in seconds.",
            },
            {
                "name": "--step_start_s",
                "type": float,
                "default": 1.0,
                "help": "Controller mode positive yaw step start time.",
            },
            {
                "name": "--step_stop_s",
                "type": float,
                "default": 3.0,
                "help": "Controller mode positive yaw step stop time.",
            },
            {
                "name": "--reverse_start_s",
                "type": float,
                "default": 4.0,
                "help": "Controller mode negative yaw step start time.",
            },
            {
                "name": "--reverse_stop_s",
                "type": float,
                "default": 6.0,
                "help": "Controller mode negative yaw step stop time.",
            },
            {
                "name": "--yaw_rate_action",
                "type": float,
                "default": 0.5,
                "help": "Normalized yaw-rate action for attitude_direct mode.",
            },
            {
                "name": "--yaw_angle_step_rad",
                "type": float,
                "default": 0.7,
                "help": "Yaw angle step used in position mode.",
            },
            {
                "name": "--hover_thrust_action",
                "type": float,
                "default": 0.0,
                "help": "Normalized thrust action for controller mode.",
            },
            {
                "name": "--hover_target_z_m",
                "type": float,
                "default": 1.0,
                "help": "Target height for controller-mode hover stabilization.",
            },
            {
                "name": "--initial_yaw_rad",
                "type": float,
                "default": 0.0,
                "help": "Initial yaw for controller mode reset state.",
            },
            {
                "name": "--hover_kp",
                "type": float,
                "default": 0.20,
                "help": "Altitude hold proportional gain in controller mode.",
            },
            {
                "name": "--hover_kd",
                "type": float,
                "default": 0.10,
                "help": "Altitude hold damping gain in controller mode.",
            },
            {
                "name": "--level_kp",
                "type": float,
                "default": 0.35,
                "help": "Simple roll/pitch leveling gain in controller mode.",
            },
            {
                "name": "--max_level_angle_rad",
                "type": float,
                "default": 0.20,
                "help": "Clamp for controller-mode roll/pitch leveling command in rad.",
            },
            {
                "name": "--task_phase1_end_s",
                "type": float,
                "default": 2.5,
                "help": "Task mode phase 1 end time.",
            },
            {
                "name": "--task_phase2_end_s",
                "type": float,
                "default": 5.0,
                "help": "Task mode phase 2 end time.",
            },
            {
                "name": "--task_phase1_xy_action",
                "type": float,
                "default": 0.5,
                "help": "Task mode phase 1 X action.",
            },
            {
                "name": "--task_phase2_xy_action",
                "type": float,
                "default": 0.5,
                "help": "Task mode phase 2 Y action.",
            },
            {
                "name": "--task_phase3_xy_action",
                "type": float,
                "default": 0.5,
                "help": "Task mode phase 3 negative X action.",
            },
            {
                "name": "--task_hover_z_action",
                "type": float,
                "default": 0.0,
                "help": "Task mode hover-like z action.",
            },
            {
                "name": "--velocity_forward_m_s",
                "type": float,
                "default": 0.0,
                "help": "Velocity controller forward velocity command in body frame.",
            },
            {
                "name": "--velocity_lateral_m_s",
                "type": float,
                "default": 0.0,
                "help": "Velocity controller lateral velocity command in body frame.",
            },
            {
                "name": "--velocity_vertical_m_s",
                "type": float,
                "default": 0.0,
                "help": "Velocity controller vertical velocity command in body frame.",
            },
            {
                "name": "--velocity_yaw_rate_cmd_rad_s",
                "type": float,
                "default": 0.7,
                "help": "Velocity controller yaw-rate command in rad/s.",
            },
            {
                "name": "--output",
                "type": str,
                "default": "yaw_diagnostic.png",
                "help": "Path to save plot image.",
            },
            {
                "name": "--show_plot",
                "type": lambda x: bool(int(x)),
                "default": False,
                "help": "Set 1 to show matplotlib window.",
            },
        ]
    )

    if args.mode == "position":
        series = _position_mode(args)
    elif args.mode == "velocity":
        series = _velocity_mode(args)
    elif args.mode == "task":
        series = _task_mode(args)
    elif args.mode == "attitude_direct":
        series = _attitude_direct_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    _plot_series(series, args.output, bool(args.show_plot))

    peak_yaw_rate = float(np.max(np.abs(series.yaw_rate_rad_s)))
    min_z = float(np.min(series.z_m))
    max_abs_roll = float(np.max(np.abs(series.roll_rad)))
    max_abs_pitch = float(np.max(np.abs(series.pitch_rad)))
    print(f"mode={args.mode}")
    print(f"peak actual yaw rate [rad/s]: {peak_yaw_rate:.4f}")
    print(f"min z [m]: {min_z:.4f}")
    print(f"max |roll| [rad]: {max_abs_roll:.4f}")
    print(f"max |pitch| [rad]: {max_abs_pitch:.4f}")
    peak_cmd = float(np.max(np.abs(series.yaw_cmd)))
    if "rate" in series.yaw_cmd_label:
        print(f"peak commanded yaw rate [rad/s]: {peak_cmd:.4f}")
        if peak_cmd > 1.0e-5:
            print(f"yaw-rate tracking ratio: {peak_yaw_rate / peak_cmd:.4f}")
    else:
        print(f"peak commanded yaw reference [rad]: {peak_cmd:.4f}")


if __name__ == "__main__":
    main()

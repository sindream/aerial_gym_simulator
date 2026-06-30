import torch
import numpy as np

from aerial_gym.task.drone_racing_task.drone_racing_task import DroneRacingTask
from aerial_gym.utils.math import (
    get_euler_xyz_tensor,
    quat_axis,
    quat_rotate_inverse,
    ssa,
)


class DroneRacingNedPrivilegedTask(DroneRacingTask):
    """Optical-flow racing task with gate-aware actor state and gain-estimation aux loss."""

    def _actions_to_attitude_commands(self, raw_actions):
        commands = torch.zeros_like(raw_actions)
        controller = self.sim_env.robot_manager.robot.controller
        gravity_mag = torch.norm(controller.gravity, dim=1)

        thrust_g = torch.clamp(
            raw_actions[:, 0],
            float(getattr(self.task_config, "thrust_command_min", -1.0)),
            float(getattr(self.task_config, "thrust_command_max", 2.0)),
        )
        commands[:, 0] = thrust_g * gravity_mag

        max_rate = (
            float(getattr(self.task_config, "body_rate_command_max_deg_s", 720.0))
            * np.pi
            / 180.0
        )
        commands[:, 1:4] = torch.clamp(raw_actions[:, 1:4], -1.0, 1.0) * max_rate
        return commands

    def _body_up_projection_vehicle(self):
        body_z_axis_world = quat_axis(self.obs_dict["robot_orientation"], 2)
        return quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            body_z_axis_world,
        )

    def _normalized_rate_gain_target(self):
        controller = self.sim_env.robot_manager.robot.controller
        current = getattr(controller, "K_angvel_tensor_current", None)
        minimum = getattr(controller, "K_angvel_tensor_min", None)
        maximum = getattr(controller, "K_angvel_tensor_max", None)
        if current is None or minimum is None or maximum is None:
            return torch.zeros(
                (self.num_envs, self.task_config.rate_gain_target_dim),
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            )

        normalized = (current - minimum) / torch.clamp(maximum - minimum, min=1.0e-6)
        return torch.clamp(normalized, 0.0, 1.0)

    def process_obs_for_task(self):
        first_target_positions = self._active_target_positions()
        second_target_positions = self._second_target_positions()
        current_gate_yaws = self._current_gate_yaws()
        second_gate_yaws = self._second_gate_yaws()
        robot_yaws = get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"])[:, 2]

        relative_first_target_world = first_target_positions - self.obs_dict["robot_position"]
        relative_second_target_world = second_target_positions - self.obs_dict["robot_position"]
        relative_first_target_vehicle = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"], relative_first_target_world
        )
        relative_second_target_vehicle = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"], relative_second_target_world
        )
        relative_first_gate_yaw = ssa(current_gate_yaws - robot_yaws)
        relative_second_gate_yaw = ssa(second_gate_yaws - robot_yaws)

        self.task_obs["state"][:, 0:3] = relative_first_target_vehicle
        self.task_obs["state"][:, 3:6] = relative_second_target_vehicle
        self.task_obs["state"][:, 6:9] = self.obs_dict["robot_vehicle_linvel"]
        self.task_obs["state"][:, 9:12] = self.obs_dict["robot_body_angvel"]
        self.task_obs["state"][:, 12:15] = self._body_up_projection_vehicle()
        self.task_obs["state"][:, 15] = torch.sin(relative_first_gate_yaw)
        self.task_obs["state"][:, 16] = torch.cos(relative_first_gate_yaw)
        self.task_obs["state"][:, 17] = torch.sin(relative_second_gate_yaw)
        self.task_obs["state"][:, 18] = torch.cos(relative_second_gate_yaw)
        self.task_obs["observations"][:] = self.task_obs["state"]

        self.task_obs["critic_state"][:] = self.task_obs["state"]

        self.task_obs["rate_gain_target"][:] = self._normalized_rate_gain_target()

        depth_image = self.obs_dict["depth_range_pixels"][:, 0]
        optical_flow = self._compute_optical_flow(depth_image)
        optical_flow_dual, optical_flow_full, optical_flow_center = self._build_dual_optical_flow_input(
            optical_flow
        )
        self.task_obs["img_observation"][:] = optical_flow_dual
        self._show_env0_optical_flow(optical_flow_full, optical_flow_center)

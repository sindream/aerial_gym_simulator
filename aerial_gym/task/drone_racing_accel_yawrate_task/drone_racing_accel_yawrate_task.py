import numpy as np
import torch

from gym.spaces import Box

from aerial_gym.control.controllers.base_lee_controller import (
    calculate_desired_orientation_from_forces_and_yaw,
)
from aerial_gym.task.drone_racing_task.drone_racing_task import DroneRacingTask
from aerial_gym.utils.math import (
    get_euler_xyz_tensor,
    quat_rotate,
    quat_rotate_inverse,
    quat_to_rotation_matrix,
    ssa,
)


class DroneRacingAccelYawrateTask(DroneRacingTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        super().__init__(
            task_config,
            seed=seed,
            num_envs=num_envs,
            headless=headless,
            device=device,
            use_warp=use_warp,
        )

        self.attitude_commands = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        action_low = np.full((self.task_config.action_space_dim,), -1.0, dtype=np.float32)
        action_high = np.full((self.task_config.action_space_dim,), 1.0, dtype=np.float32)
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)

        controller = self.sim_env.robot_manager.robot.controller
        self.controller_mass = controller.mass
        self.controller_gravity = controller.gravity

        self.world_accel_xy_max = float(self.task_config.world_accel_xy_max_m_s2)
        self.world_accel_z_up_max = float(self.task_config.world_accel_z_up_max_m_s2)
        gravity_mag = torch.norm(self.controller_gravity, dim=1, keepdim=True)
        self.world_accel_z_down_min = (
            -gravity_mag * float(self.task_config.world_accel_z_down_gravity_scale)
        )

    def _actions_to_attitude_commands(self, raw_actions):
        clipped_actions = torch.clamp(raw_actions, -1.0, 1.0)

        accel_vehicle = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=raw_actions.dtype,
            requires_grad=False,
        )
        accel_vehicle[:, 0:2] = clipped_actions[:, 0:2] * self.world_accel_xy_max

        accel_z_alpha = 0.5 * (clipped_actions[:, 2:3] + 1.0)
        accel_vehicle[:, 2:3] = self.world_accel_z_down_min + accel_z_alpha * (
            self.world_accel_z_up_max - self.world_accel_z_down_min
        )
        accel_world = quat_rotate(self.obs_dict["robot_vehicle_orientation"], accel_vehicle)

        forces_world = self.controller_mass * (accel_world - self.controller_gravity)

        current_yaw = self.obs_dict["robot_euler_angles"][:, 2]
        desired_quat = calculate_desired_orientation_from_forces_and_yaw(
            forces_world, current_yaw
        )
        desired_euler = ssa(get_euler_xyz_tensor(desired_quat))

        body_z_world = quat_to_rotation_matrix(self.obs_dict["robot_orientation"])[:, :, 2]
        thrust_world_z = torch.sum(forces_world * body_z_world, dim=1)
        hover_thrust = self.controller_mass.squeeze(1) * torch.norm(
            self.controller_gravity, dim=1
        )
        thrust_command = thrust_world_z / hover_thrust - 1.0

        commands = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
            dtype=raw_actions.dtype,
            requires_grad=False,
        )
        commands[:, 0] = torch.clamp(
            thrust_command,
            float(getattr(self.task_config, "thrust_command_min", -1.0)),
            float(getattr(self.task_config, "thrust_command_max", 1.0)),
        )
        commands[:, 1] = torch.clamp(
            desired_euler[:, 0],
            -self.task_config.attitude_max_inclination_rad,
            self.task_config.attitude_max_inclination_rad,
        )
        commands[:, 2] = torch.clamp(
            desired_euler[:, 1],
            -self.task_config.attitude_max_inclination_rad,
            self.task_config.attitude_max_inclination_rad,
        )
        commands[:, 3] = torch.clamp(
            clipped_actions[:, 3] * self.task_config.attitude_max_yaw_rate_rad_s,
            -self.task_config.attitude_max_yaw_rate_rad_s,
            self.task_config.attitude_max_yaw_rate_rad_s,
        )
        return commands

    def process_obs_for_task(self):
        first_target_positions = self._active_target_positions()
        second_target_positions = self._second_target_positions()
        relative_first_target_world = first_target_positions - self.obs_dict["robot_position"]
        relative_second_target_world = second_target_positions - self.obs_dict["robot_position"]
        relative_first_target_vehicle = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"], relative_first_target_world
        )
        relative_second_target_vehicle = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"], relative_second_target_world
        )

        self.task_obs["state"][:, 0:3] = relative_first_target_vehicle
        self.task_obs["state"][:, 3:6] = relative_second_target_vehicle
        self.task_obs["state"][:, 6:9] = self.obs_dict["robot_vehicle_linvel"]
        self.task_obs["state"][:, 9:13] = self.obs_dict["robot_orientation"]
        self.task_obs["state"][:, 13:16] = self.obs_dict["robot_body_angvel"]

        depth_image = self.obs_dict["depth_range_pixels"][:, 0]
        optical_flow = self._compute_optical_flow(depth_image)
        optical_flow_dual, optical_flow_full, optical_flow_center = self._build_dual_optical_flow_input(
            optical_flow
        )
        self.task_obs["img_observation"][:] = optical_flow_dual
        self._show_env0_optical_flow(optical_flow_full, optical_flow_center)
        self.task_obs["observations"][:] = self.task_obs["state"]

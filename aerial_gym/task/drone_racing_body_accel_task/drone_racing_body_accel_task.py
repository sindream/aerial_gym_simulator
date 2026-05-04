import numpy as np
import torch

from gym.spaces import Box

from aerial_gym.control.controllers.base_lee_controller import (
    calculate_desired_orientation_from_forces_and_yaw,
)
from aerial_gym.task.drone_racing_task.drone_racing_task import DroneRacingTask
from aerial_gym.utils.math import (
    get_euler_xyz_tensor,
    quat_from_euler_xyz_tensor,
    quat_rotate,
    quat_to_rotation_matrix,
    ssa,
)


class DroneRacingBodyAccelTask(DroneRacingTask):
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
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )

        controller = self.sim_env.robot_manager.robot.controller
        self.controller_mass = controller.mass
        self.controller_gravity = controller.gravity
        self.world_accel_xy_max = float(self.task_config.world_accel_xy_max_m_s2)
        self.world_accel_z_max = float(self.task_config.world_accel_z_max_m_s2)
        self.world_accel_scale = torch.tensor(
            [self.world_accel_xy_max, self.world_accel_xy_max, self.world_accel_z_max],
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).view(1, 3)

    def _sample_start_gate_indices(self, env_ids):
        if len(env_ids) == 0:
            return
        if not bool(getattr(self.task_config, "randomize_start_gate", True)):
            self.current_gate_index[env_ids] = 0
            return
        random_gate_index = torch.randint(
            low=0,
            high=self.num_gates,
            size=(len(env_ids),),
            device=self.device,
            dtype=torch.long,
        )
        self.current_gate_index[env_ids] = random_gate_index

    def _set_robot_start_pose(self, env_ids):
        if len(env_ids) == 0:
            return

        gate_positions = self._current_gate_positions(env_ids)
        gate_quats = self._current_gate_quats(env_ids)
        gate_eulers = ssa(get_euler_xyz_tensor(gate_quats))

        distance_center = float(self.task_config.spawn_gate_distance_m)
        distance_jitter = float(self.task_config.spawn_gate_distance_jitter_m)
        lateral_jitter = float(self.task_config.spawn_gate_lateral_jitter_m)
        vertical_jitter = float(self.task_config.spawn_gate_vertical_jitter_m)
        yaw_jitter_rad = np.deg2rad(float(self.task_config.spawn_gate_yaw_jitter_deg))

        local_offset = torch.zeros(
            (len(env_ids), 3),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        local_offset[:, 0] = -distance_center + (
            torch.rand(len(env_ids), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        ) * distance_jitter
        local_offset[:, 1] = (
            torch.rand(len(env_ids), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        ) * lateral_jitter
        local_offset[:, 2] = (
            torch.rand(len(env_ids), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        ) * vertical_jitter

        world_position = gate_positions + quat_rotate(gate_quats, local_offset)
        world_position[:, 2] = torch.clamp(
            world_position[:, 2],
            min=float(self.task_config.spawn_min_height_m),
        )

        spawn_eulers = torch.zeros(
            (len(env_ids), 3),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        spawn_eulers[:, 2] = gate_eulers[:, 2] + (
            torch.rand(len(env_ids), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        ) * yaw_jitter_rad
        spawn_quats = quat_from_euler_xyz_tensor(spawn_eulers)

        robot_state = self.obs_dict["robot_state_tensor"]
        robot_state[env_ids, 0:3] = world_position
        robot_state[env_ids, 3:7] = spawn_quats
        robot_state[env_ids, 7:13] = 0.0
        self.obs_dict["robot_actions"][env_ids] = 0.0
        self.obs_dict["robot_prev_actions"][env_ids] = 0.0
        self.sim_env.robot_manager.robot.update_states()
        self.sim_env.IGE_env.write_to_sim()

    def reset(self):
        self.sim_env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_task_buffers(env_ids)
        self._sample_start_gate_indices(env_ids)
        self._resample_cylinders_away_from_gates(env_ids)
        self._set_robot_start_pose(env_ids)
        self.sim_env.render(render_components="sensors")
        self.prev_robot_position[:] = self.obs_dict["robot_position"]
        self.prev_gate_distance[:] = self._distance_to_active_target()
        self.infos = {}
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_task_buffers(env_ids)
        self._sample_start_gate_indices(env_ids)
        self._resample_cylinders_away_from_gates(env_ids)
        self._set_robot_start_pose(env_ids)
        self.sim_env.render(render_components="sensors")
        self.prev_robot_position[env_ids] = self.obs_dict["robot_position"][env_ids]
        self.prev_gate_distance[env_ids] = self._distance_to_active_target(env_ids)

    def _actions_to_attitude_commands(self, raw_actions):
        clipped_actions = torch.clamp(raw_actions, -1.0, 1.0)
        accel_world = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=raw_actions.dtype,
            requires_grad=False,
        )
        accel_world[:, 0:2] = clipped_actions[:, 0:2] * self.world_accel_scale[:, 0:2]
        accel_world[:, 2] = 0.5 * (clipped_actions[:, 2] + 1.0) * self.world_accel_z_max
        forces_world = self.controller_mass * accel_world

        robot_position = self.obs_dict["robot_position"]
        robot_velocity = self.obs_dict["robot_linvel"]
        current_yaw = self.obs_dict["robot_euler_angles"][:, 2]

        velocity_xy = robot_velocity[:, 0:2]
        speed_xy = torch.norm(velocity_xy, dim=1)
        target_xy = self._active_target_positions()[:, 0:2] - robot_position[:, 0:2]

        velocity_yaw = torch.atan2(velocity_xy[:, 1], velocity_xy[:, 0])
        target_yaw = torch.atan2(target_xy[:, 1], target_xy[:, 0])
        desired_yaw = torch.where(
            speed_xy > self.task_config.heading_velocity_threshold_m_s,
            velocity_yaw,
            target_yaw,
        )

        desired_quat = calculate_desired_orientation_from_forces_and_yaw(
            forces_world, desired_yaw
        )
        desired_euler = ssa(get_euler_xyz_tensor(desired_quat))

        body_z_world = quat_to_rotation_matrix(self.obs_dict["robot_orientation"])[:, :, 2]
        thrust_world_z = torch.sum(forces_world * body_z_world, dim=1)
        hover_thrust = self.controller_mass.squeeze(1) * torch.norm(
            self.controller_gravity, dim=1
        )
        thrust_command = thrust_world_z / hover_thrust - 1.0

        yaw_error = ssa(desired_yaw - current_yaw)
        yaw_rate_command = torch.clamp(
            self.task_config.heading_yaw_rate_gain * yaw_error,
            -self.task_config.attitude_max_yaw_rate_rad_s,
            self.task_config.attitude_max_yaw_rate_rad_s,
        )

        commands = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
            dtype=raw_actions.dtype,
            requires_grad=False,
        )
        commands[:, 0] = torch.clamp(
            thrust_command,
            self.task_config.body_accel_thrust_command_min,
            self.task_config.body_accel_thrust_command_max,
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
        commands[:, 3] = yaw_rate_command
        return commands

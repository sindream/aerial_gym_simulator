import numpy as np
import torch
import torch.nn.functional as F

from aerial_gym.task.drone_racing_task.drone_racing_task import DroneRacingTask
from aerial_gym.utils.math import (
    get_euler_xyz_tensor,
    quat_axis,
    quat_rotate_inverse,
    ssa,
)


class AIGPRacingSegmentationTask(DroneRacingTask):
    """AIGP racing task with spec-limited actor observations.

    Actor input intentionally excludes global pose, attitude quaternion, and
    gate coordinates. It sees only gate segmentation and self-state data that
    can be sourced from HIGHRES_IMU plus previous command history.
    """

    def _actions_to_attitude_commands(self, raw_actions):
        commands = torch.zeros_like(raw_actions)
        controller = self.sim_env.robot_manager.robot.controller
        gravity_mag = torch.norm(controller.gravity, dim=1)

        throttle = torch.clamp(
            raw_actions[:, 0],
            float(getattr(self.task_config, "thrust_command_min", 0.05)),
            float(getattr(self.task_config, "thrust_command_max", 0.90)),
        )
        hover_throttle = max(float(getattr(self.task_config, "hover_throttle", 0.32)), 1.0e-6)
        # LeeRatesController command[0] is desired z acceleration. Convert a
        # public throttle command into total thrust acceleration, then subtract
        # gravity so throttle == hover_throttle produces az == 0.
        commands[:, 0] = (throttle / hover_throttle - 1.0) * gravity_mag

        max_rate = float(getattr(self.task_config, "body_rate_command_max_rad_s", 1.0))
        commands[:, 1:4] = torch.clamp(raw_actions[:, 1:4], -1.0, 1.0) * max_rate
        return commands

    def _body_up_projection_vehicle(self):
        body_z_axis_world = quat_axis(self.obs_dict["robot_orientation"], 2)
        return quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            body_z_axis_world,
        )

    def _build_privileged_critic_state(self):
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

        critic_state = self.task_obs["critic_state"]
        critic_state[:, 0:3] = relative_first_target_vehicle
        critic_state[:, 3:6] = relative_second_target_vehicle
        critic_state[:, 6:9] = self.obs_dict["robot_vehicle_linvel"]
        critic_state[:, 9:12] = self.obs_dict["robot_body_angvel"]
        critic_state[:, 12:15] = self._body_up_projection_vehicle()
        critic_state[:, 15] = torch.sin(relative_first_gate_yaw)
        critic_state[:, 16] = torch.cos(relative_first_gate_yaw)
        critic_state[:, 17] = torch.sin(relative_second_gate_yaw)
        critic_state[:, 18] = torch.cos(relative_second_gate_yaw)
        critic_state[:, 19:24] = self._target_to_next_gate_relation()

    def _target_to_next_gate_relation(self):
        current_gate_positions = self._current_gate_positions()
        current_gate_quats = self._current_gate_quats()
        current_gate_yaws = self._current_gate_yaws()

        next_gate_index = (self.current_gate_index + 1) % self.num_gates
        next_gate_positions = self.gate_positions[next_gate_index]
        next_gate_yaws = self.gate_yaws[next_gate_index]

        if not bool(getattr(self.task_config, "loop_track", False)):
            final_gate_active = self.current_gate_index == (self.num_gates - 1)
            final_goal_positions = self.final_goal_position.unsqueeze(0).expand_as(next_gate_positions)
            next_gate_positions = torch.where(
                final_gate_active.unsqueeze(1),
                final_goal_positions,
                next_gate_positions,
            )
            next_gate_yaws = torch.where(final_gate_active, current_gate_yaws, next_gate_yaws)

        relative_next_in_current_gate = quat_rotate_inverse(
            current_gate_quats,
            next_gate_positions - current_gate_positions,
        )
        position_scale = max(
            float(getattr(self.task_config, "gate_relation_position_normalizer_m", 30.0)),
            1.0e-6,
        )
        relative_next_normalized = torch.clamp(
            relative_next_in_current_gate / position_scale,
            -4.0,
            4.0,
        )
        relative_yaw = ssa(next_gate_yaws - current_gate_yaws)
        relation = torch.zeros((self.num_envs, 5), device=self.device, dtype=torch.float32)
        relation[:, 0:3] = relative_next_normalized
        relation[:, 3] = torch.sin(relative_yaw)
        relation[:, 4] = torch.cos(relative_yaw)

        if not bool(getattr(self.task_config, "loop_track", False)):
            relation[self.goal_active] = 0.0
        return relation

    def _gate_segmentation_mask(self):
        if "segmentation_pixels" not in self.obs_dict:
            return torch.zeros(
                (
                    self.num_envs,
                    1,
                    self.task_config.image_height,
                    self.task_config.image_width,
                ),
                device=self.device,
                dtype=torch.float32,
            )

        segmentation = self.obs_dict["segmentation_pixels"][:, 0]
        if not hasattr(self, "_gate_semantic_id_min"):
            self._gate_semantic_id_min = torch.min(self.gate_semantic_ids)
            self._gate_semantic_id_max = torch.max(self.gate_semantic_ids)
        mask = (
            (segmentation >= self._gate_semantic_id_min)
            & (segmentation <= self._gate_semantic_id_max)
        )

        gate_image = mask.float().unsqueeze(1)
        if (
            gate_image.shape[-2] != self.task_config.image_height
            or gate_image.shape[-1] != self.task_config.image_width
        ):
            gate_image = F.interpolate(
                gate_image,
                size=(self.task_config.image_height, self.task_config.image_width),
                mode="nearest",
            )
        return gate_image

    def _spec_limited_self_state(self):
        if "imu_measurement" in self.obs_dict:
            imu_measurement = self.obs_dict["imu_measurement"]
            accel = imu_measurement[:, 0:3]
            gyro = imu_measurement[:, 3:6]
        else:
            accel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
            gyro = self.obs_dict["robot_body_angvel"]

        accel_scale = max(float(getattr(self.task_config, "imu_accel_normalizer_m_s2", 39.24)), 1.0e-6)
        gyro_scale = max(float(getattr(self.task_config, "imu_gyro_normalizer_rad_s", 4.0)), 1.0e-6)

        self.task_obs["state"][:, 0:3] = torch.clamp(accel / accel_scale, -4.0, 4.0)
        self.task_obs["state"][:, 3:6] = torch.clamp(gyro / gyro_scale, -4.0, 4.0)

        prev_action = self.prev_raw_actions
        throttle_min = float(getattr(self.task_config, "thrust_command_min", 0.05))
        throttle_max = float(getattr(self.task_config, "thrust_command_max", 0.90))
        throttle_mid = 0.5 * (throttle_min + throttle_max)
        throttle_half_range = max(0.5 * (throttle_max - throttle_min), 1.0e-6)
        self.task_obs["state"][:, 6] = torch.clamp(
            (prev_action[:, 0] - throttle_mid) / throttle_half_range,
            -1.0,
            1.0,
        )
        self.task_obs["state"][:, 7:10] = torch.clamp(prev_action[:, 1:4], -1.0, 1.0)
        self.task_obs["state"][:, 10:15] = self._target_to_next_gate_relation()

    def process_obs_for_task(self):
        self._spec_limited_self_state()
        self.task_obs["observations"][:] = self.task_obs["state"]

        if "critic_state" in self.task_obs:
            self._build_privileged_critic_state()

        self.task_obs["img_observation"][:] = self._gate_segmentation_mask()

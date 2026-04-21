from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder

import numpy as np
import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    get_euler_xyz_tensor,
    quat_axis,
    quat_conjugate,
    quat_from_euler_xyz_tensor,
    quat_mul,
    quat_rotate_inverse,
    ssa,
)

from gym.spaces import Box, Dict

logger = CustomLogger("drone_racing_paper_task")


class DroneRacingPaperTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.obs_dict = self.sim_env.get_obs()
        self.num_envs = self.sim_env.num_envs
        self.dt = float(self.sim_env.sim_config.sim.dt)
        self.motor_model = self.sim_env.robot_manager.robot.control_allocator.motor_model

        self._init_track_tensors()

        self.raw_actions = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_raw_actions = torch.zeros_like(self.raw_actions)
        self.motor_thrust_actions = torch.zeros_like(self.raw_actions)

        self.current_gate_index = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.completed_laps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.prev_gate_distance = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.prev_robot_position = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.outside_reposition_claimed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.successes = torch.zeros_like(self.terminations)

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )

        self.task_obs = {
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            )
        }

        self.infos = {}
        self.reset()

    def _init_track_tensors(self):
        gate_positions = []
        gate_yaws = []
        gate_semantic_ids = []
        for gate_definition in self.task_config.gate_layout:
            gate_positions.append(gate_definition["position"])
            gate_yaws.append(np.deg2rad(gate_definition["yaw_deg"]))
            gate_semantic_ids.append(gate_definition["semantic_id"])

        self.num_gates = len(gate_positions)
        self.gate_positions = torch.tensor(
            gate_positions, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.gate_yaws = torch.tensor(
            gate_yaws, device=self.device, dtype=torch.float32, requires_grad=False
        )

        gate_eulers = torch.zeros(
            (self.num_gates, 3), device=self.device, dtype=torch.float32, requires_grad=False
        )
        gate_eulers[:, 2] = self.gate_yaws
        self.gate_quats = quat_from_euler_xyz_tensor(gate_eulers)
        self.gate_semantic_ids = torch.tensor(
            gate_semantic_ids, device=self.device, dtype=torch.int32, requires_grad=False
        )

        self.start_position = torch.tensor(
            self.task_config.start_position,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        start_euler = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
        start_euler[0, 2] = np.deg2rad(self.task_config.start_yaw_deg)
        self.start_quat = quat_from_euler_xyz_tensor(start_euler)[0]

        self.max_body_rate = self.task_config.max_body_rate_deg_s * np.pi / 180.0

        self.reward_params = {}
        for key, value in self.task_config.reward_parameters.items():
            self.reward_params[key] = torch.tensor(
                value, device=self.device, dtype=torch.float32, requires_grad=False
            )

        self.obs_dict["num_obstacles_in_env"] = self.num_gates

    def _current_gate_positions(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return self.gate_positions[gate_index]

    def _current_gate_quats(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return self.gate_quats[gate_index]

    def _current_progress_target_positions(self, env_ids=None):
        gate_positions = self._current_gate_positions(env_ids)
        gate_quats = self._current_gate_quats(env_ids)
        gate_forward = quat_axis(gate_quats, 0)
        return gate_positions + self.task_config.progress_gate_offset_m * gate_forward

    def _distance_to_current_gate(self, env_ids=None):
        if env_ids is None:
            robot_positions = self.obs_dict["robot_position"]
            progress_target_positions = self._current_progress_target_positions()
        else:
            robot_positions = self.obs_dict["robot_position"][env_ids]
            progress_target_positions = self._current_progress_target_positions(env_ids)
        return torch.norm(robot_positions - progress_target_positions, dim=1)

    def _compute_gate_frame_positions(self, robot_positions, gate_positions, gate_quats):
        return quat_rotate_inverse(gate_quats, robot_positions - gate_positions)

    def _set_current_gate_from_pose(self, env_ids):
        if len(env_ids) == 0:
            return

        robot_positions = self.obs_dict["robot_position"][env_ids]
        gate_positions = self.gate_positions.unsqueeze(0).expand(len(env_ids), -1, -1)
        gate_quats = self.gate_quats.unsqueeze(0).expand(len(env_ids), -1, -1)

        rel_positions = (robot_positions.unsqueeze(1) - gate_positions).reshape(-1, 3)
        gate_frame_positions = quat_rotate_inverse(
            gate_quats.reshape(-1, 4), rel_positions
        ).reshape(len(env_ids), self.num_gates, 3)

        distances = torch.norm(robot_positions.unsqueeze(1) - gate_positions, dim=2)
        ahead_mask = gate_frame_positions[:, :, 0] < 0.0
        ahead_distances = torch.where(
            ahead_mask,
            distances,
            torch.full_like(distances, float("inf")),
        )

        nearest_ahead = torch.argmin(ahead_distances, dim=1)
        has_ahead_gate = torch.isfinite(
            ahead_distances.gather(1, nearest_ahead.unsqueeze(1)).squeeze(1)
        )
        nearest_gate = torch.argmin(distances, dim=1)

        self.current_gate_index[env_ids] = torch.where(
            has_ahead_gate, nearest_ahead, nearest_gate
        )

    def _paper_action_to_thrust(self, raw_actions):
        normalized_action = 0.5 * (torch.clamp(raw_actions, -1.0, 1.0) + 1.0)
        k = self.task_config.motor_command_curve_k
        omega = (
            (self.task_config.motor_omega_max - self.task_config.motor_omega_min)
            * torch.sqrt(torch.clamp(k * normalized_action * normalized_action + (1.0 - k) * normalized_action, min=0.0))
            + self.task_config.motor_omega_min
        )
        return self.task_config.motor_thrust_constant * omega * omega

    def _set_initial_motor_state(self, env_ids, initial_raw_actions):
        self.raw_actions[env_ids] = initial_raw_actions
        self.prev_raw_actions[env_ids] = initial_raw_actions
        self.motor_thrust_actions[env_ids] = self._paper_action_to_thrust(initial_raw_actions)
        self.motor_model.current_motor_thrust[env_ids] = self.motor_thrust_actions[env_ids]
        self.motor_model.motor_rate[env_ids] = 0.0
        self.obs_dict["robot_actions"][env_ids] = self.motor_thrust_actions[env_ids]
        self.obs_dict["robot_prev_actions"][env_ids] = self.motor_thrust_actions[env_ids]

    def _align_spawn_to_reset_gate(self, env_ids):
        if len(env_ids) == 0:
            return

        robot_state = self.obs_dict["robot_state_tensor"]
        target_gate_index = int(self.task_config.reset_gate_index)
        target_gate_positions = self.gate_positions[target_gate_index].expand(len(env_ids), -1)
        spawn_positions = robot_state[env_ids, 0:3]

        yaw_to_gate = torch.atan2(
            target_gate_positions[:, 1] - spawn_positions[:, 1],
            target_gate_positions[:, 0] - spawn_positions[:, 0],
        )

        reset_eulers = torch.zeros((len(env_ids), 3), device=self.device)
        roll_pitch_noise = np.deg2rad(self.task_config.reset_roll_pitch_noise_deg)
        yaw_noise = np.deg2rad(self.task_config.reset_yaw_noise_deg)
        if roll_pitch_noise > 0.0:
            reset_eulers[:, 0:2] = (
                2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
            ) * roll_pitch_noise
        if yaw_noise > 0.0:
            yaw_to_gate = yaw_to_gate + (
                2.0 * torch.rand(len(env_ids), device=self.device) - 1.0
            ) * yaw_noise

        reset_eulers[:, 2] = yaw_to_gate
        robot_state[env_ids, 3:7] = quat_from_euler_xyz_tensor(reset_eulers)
        self.current_gate_index[env_ids] = target_gate_index

    def _finalize_resets(self, env_ids):
        if len(env_ids) == 0:
            return

        initial_raw_actions = torch.full(
            (len(env_ids), self.task_config.action_space_dim),
            self.task_config.initial_idle_action,
            device=self.device,
            requires_grad=False,
        )

        self.sim_env.robot_manager.robot.update_states()
        self.sim_env.IGE_env.write_to_sim()
        self.sim_env.IGE_env.refresh_tensors()
        self._align_spawn_to_reset_gate(env_ids)
        self.sim_env.robot_manager.robot.update_states()
        self.sim_env.IGE_env.write_to_sim()
        self.sim_env.IGE_env.refresh_tensors()
        self._set_initial_motor_state(env_ids, initial_raw_actions)

    def _reset_task_buffers(self, env_ids):
        self.completed_laps[env_ids] = 0
        self.outside_reposition_claimed[env_ids] = False
        self.raw_actions[env_ids] = 0.0
        self.prev_raw_actions[env_ids] = 0.0
        self.motor_thrust_actions[env_ids] = 0.0
        self.terminations[env_ids] = False
        self.truncations[env_ids] = False
        self.rewards[env_ids] = 0.0

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.sim_env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_task_buffers(env_ids)
        self._finalize_resets(env_ids)
        self.prev_robot_position[:] = self.obs_dict["robot_position"]
        self.prev_gate_distance[:] = self._distance_to_current_gate()
        self.infos = {}
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_task_buffers(env_ids)
        self._finalize_resets(env_ids)
        self.prev_robot_position[env_ids] = self.obs_dict["robot_position"][env_ids]
        self.prev_gate_distance[env_ids] = self._distance_to_current_gate(env_ids)

    def render(self, mode="human"):
        return self.sim_env.render()

    def _compute_perception_angle(self, gate_positions):
        forward_axis = quat_axis(self.obs_dict["robot_orientation"], 0)
        gate_direction = gate_positions - self.obs_dict["robot_position"]
        gate_direction = gate_direction / torch.clamp(
            torch.norm(gate_direction, dim=1, keepdim=True), min=1.0e-6
        )
        alignment = torch.sum(forward_axis * gate_direction, dim=1).clamp(-1.0, 1.0)
        return torch.arccos(alignment)

    def _update_gate_progress(self, gate_passed):
        previous_gate_index = self.current_gate_index.clone()
        final_gate_pass = gate_passed & (previous_gate_index == (self.num_gates - 1))
        self.completed_laps[final_gate_pass] += 1
        self.current_gate_index[gate_passed] = (previous_gate_index[gate_passed] + 1) % self.num_gates
        success = final_gate_pass & (self.completed_laps >= self.task_config.num_laps)
        return success

    def _compute_reward_terms(self):
        gate_positions = self._current_gate_positions()
        gate_quats = self._current_gate_quats()
        progress_target_positions = self._current_progress_target_positions()

        robot_positions = self.obs_dict["robot_position"]
        current_gate_distance = torch.norm(robot_positions - progress_target_positions, dim=1)
        progress_delta = self.prev_gate_distance - current_gate_distance
        progress_cap = self.reward_params["max_progress_speed"] * self.dt
        progress_reward = self.reward_params["progress_weight"] * torch.minimum(
            progress_delta,
            torch.ones_like(progress_delta, device=self.device) * progress_cap,
        )

        previous_gate_frame_position = self._compute_gate_frame_positions(
            self.prev_robot_position, gate_positions, gate_quats
        )
        current_gate_frame_position = self._compute_gate_frame_positions(
            robot_positions, gate_positions, gate_quats
        )

        gate_half_width = self.task_config.gate_pass_half_width - self.task_config.drone_collision_radius
        gate_half_height = self.task_config.gate_pass_half_height - self.task_config.drone_collision_radius
        gate_half_width = max(gate_half_width, 0.05)
        gate_half_height = max(gate_half_height, 0.05)

        crossed_gate_plane_forward = (previous_gate_frame_position[:, 0] < 0.0) & (
            current_gate_frame_position[:, 0] >= 0.0
        )
        crossed_gate_plane_reverse = (previous_gate_frame_position[:, 0] > 0.0) & (
            current_gate_frame_position[:, 0] <= 0.0
        )
        within_gate_opening = (
            (torch.abs(current_gate_frame_position[:, 1]) <= gate_half_width)
            & (torch.abs(current_gate_frame_position[:, 2]) <= gate_half_height)
        )
        gate_passed = crossed_gate_plane_forward & within_gate_opening

        physical_gate_half_width = 0.5 * self.task_config.physical_gate_inner_width + 0.5
        physical_gate_half_height = 0.5 * self.task_config.physical_gate_inner_height + 0.5
        hit_gate_frame = (
            (torch.abs(current_gate_frame_position[:, 0]) <= 0.5 * self.task_config.gate_collision_depth)
            & (torch.abs(current_gate_frame_position[:, 1]) <= physical_gate_half_width)
            & (torch.abs(current_gate_frame_position[:, 2]) <= physical_gate_half_height)
            & (~within_gate_opening)
        )
        crossed_gate_wrong = crossed_gate_plane_forward & (~within_gate_opening)
        gate_collision = hit_gate_frame | crossed_gate_wrong
        outside_reposition_cross = (
            crossed_gate_plane_reverse
            & (~within_gate_opening)
            & (~hit_gate_frame)
            & (~self.outside_reposition_claimed)
        )

        gate_center_offset = torch.norm(current_gate_frame_position[:, 1:3], dim=1)
        gate_reward = self.reward_params["gate_reward"] * gate_passed.float()
        gate_offset_penalty = self.reward_params["gate_offset_weight"] * gate_center_offset * gate_passed.float()
        outside_reposition_reward = (
            self.reward_params["outside_reposition_reward"] * outside_reposition_cross.float()
        )

        body_rates = self.obs_dict["robot_body_angvel"]
        angular_rate_penalty = self.reward_params["angular_rate_weight"] * torch.sum(
            body_rates * body_rates, dim=1
        )

        next_gate_index = (self.current_gate_index + 1) % self.num_gates
        next_gate_positions = self.gate_positions[next_gate_index]
        camera_angle = self._compute_perception_angle(next_gate_positions)
        perception_penalty = torch.where(
            camera_angle > self.reward_params["perception_angle_threshold_rad"],
            self.reward_params["perception_weight"] * camera_angle,
            torch.zeros_like(camera_angle),
        )

        normalized_action = 0.5 * (torch.clamp(self.raw_actions, -1.0, 1.0) + 1.0)
        previous_normalized_action = 0.5 * (torch.clamp(self.prev_raw_actions, -1.0, 1.0) + 1.0)
        motor_delta = torch.abs(normalized_action - previous_normalized_action)
        motor_delta_penalty = self.reward_params["motor_smoothness_weight"] * torch.sum(
            torch.clamp(
                motor_delta - self.reward_params["motor_smoothness_deadband"],
                min=0.0,
            ),
            dim=1,
        )

        low_action_penalty = self.reward_params["low_action_weight"] * torch.sum(
            torch.clamp(0.5 - normalized_action, min=0.0), dim=1
        )

        speed = torch.norm(self.obs_dict["robot_linvel"], dim=1)
        ground_contact = robot_positions[:, 2] <= self.task_config.ground_collision_height
        if getattr(self.task_config, "ground_collision_requires_speed", False):
            ground_collision = ground_contact & (
                speed > self.task_config.ground_collision_speed
            )
        else:
            ground_collision = ground_contact

        out_of_bounds = (
            (robot_positions[:, 0] < self.task_config.bounds_min[0])
            | (robot_positions[:, 0] > self.task_config.bounds_max[0])
            | (robot_positions[:, 1] < self.task_config.bounds_min[1])
            | (robot_positions[:, 1] > self.task_config.bounds_max[1])
            | (robot_positions[:, 2] < self.task_config.bounds_min[2] - 0.5)
            | (robot_positions[:, 2] > self.task_config.bounds_max[2] + 0.5)
        )
        excessive_body_rate = torch.norm(body_rates, dim=1) > self.max_body_rate

        contact_collision = self.terminations.clone()
        if "robot_contact_force_tensor_all" in self.obs_dict:
            any_link_contact = torch.any(
                torch.norm(self.obs_dict["robot_contact_force_tensor_all"], dim=2)
                > self.sim_env.cfg.env.collision_force_threshold,
                dim=1,
            )
            contact_collision = contact_collision | any_link_contact

        crashes = contact_collision | gate_collision | ground_collision | out_of_bounds | excessive_body_rate

        success = self._update_gate_progress(gate_passed)
        success = success & (~crashes)

        timeouts = self.sim_env.sim_steps >= self.task_config.episode_len_steps
        truncations = success | timeouts

        crash_penalty = self.reward_params["crash_penalty"] * crashes.float()
        success_reward = self.reward_params["success_reward"] * success.float()

        rewards = (
            progress_reward
            + gate_reward
            + outside_reposition_reward
            + success_reward
            - angular_rate_penalty
            - gate_offset_penalty
            - perception_penalty
            - motor_delta_penalty
            - low_action_penalty
            - crash_penalty
        )

        self.outside_reposition_claimed[outside_reposition_cross] = True
        self.outside_reposition_claimed[gate_passed] = False
        self.prev_gate_distance[:] = self._distance_to_current_gate()
        return rewards, crashes, truncations, success, gate_passed

    def step(self, actions):
        self.prev_raw_actions[:] = self.raw_actions
        self.raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self.prev_robot_position[:] = self.obs_dict["robot_position"]

        self.motor_thrust_actions[:] = self._paper_action_to_thrust(self.raw_actions)
        self.sim_env.step(actions=self.motor_thrust_actions)

        (
            self.rewards[:],
            self.terminations[:],
            self.truncations[:],
            self.successes[:],
            gate_passed,
        ) = self._compute_reward_terms()

        self.infos = {
            "successes": self.successes.clone(),
            "crashes": self.terminations.clone(),
            "gate_passed": gate_passed.clone(),
            "lap_count": self.completed_laps.clone(),
            "current_gate_index": self.current_gate_index.clone(),
            "robot_position": self.obs_dict["robot_position"].clone(),
            "robot_linvel": self.obs_dict["robot_linvel"].clone(),
        }

        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()
        else:
            rewards_to_return = self.rewards.clone()
            terminations_to_return = self.terminations.clone()
            truncations_to_return = self.truncations.clone()

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        if not self.task_config.return_state_before_reset:
            return_tuple = (
                self.get_return_tuple()[0],
                rewards_to_return,
                terminations_to_return,
                truncations_to_return,
                self.infos,
            )

        return return_tuple

    def process_obs_for_task(self):
        gate_positions = self._current_gate_positions()
        gate_quats = self._current_gate_quats()
        next_gate_index = (self.current_gate_index + 1) % self.num_gates

        gate_frame_position = self._compute_gate_frame_positions(
            self.obs_dict["robot_position"], gate_positions, gate_quats
        )
        gate_frame_velocity = quat_rotate_inverse(gate_quats, self.obs_dict["robot_linvel"])
        relative_orientation = quat_mul(quat_conjugate(gate_quats), self.obs_dict["robot_orientation"])
        gate_frame_euler = ssa(get_euler_xyz_tensor(relative_orientation))
        body_rates = self.obs_dict["robot_body_angvel"]
        motor_omega = torch.sqrt(
            torch.clamp(
                self.motor_model.current_motor_thrust / self.task_config.motor_thrust_constant,
                min=0.0,
            )
        )
        next_gate_relative_position = quat_rotate_inverse(
            gate_quats,
            self.gate_positions[next_gate_index] - gate_positions,
        )
        next_gate_relative_yaw = ssa(
            self.gate_yaws[next_gate_index] - self.gate_yaws[self.current_gate_index]
        ).unsqueeze(-1)

        self.task_obs["observations"][:, 0:3] = gate_frame_position
        self.task_obs["observations"][:, 3:6] = gate_frame_velocity
        self.task_obs["observations"][:, 6:9] = gate_frame_euler
        self.task_obs["observations"][:, 9:12] = body_rates
        self.task_obs["observations"][:, 12:16] = motor_omega
        self.task_obs["observations"][:, 16:19] = next_gate_relative_position
        self.task_obs["observations"][:, 19:20] = next_gate_relative_yaw

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def reset_done(self):
        return self.get_return_tuple()[0]

    def get_number_of_agents(self):
        return 1

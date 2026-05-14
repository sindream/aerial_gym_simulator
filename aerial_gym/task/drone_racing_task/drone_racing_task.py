from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder

import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
try:
    import cv2
except Exception:
    cv2 = None

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    get_euler_xyz_tensor,
    quat_axis,
    quat_from_euler_xyz_tensor,
    quat_rotate,
    quat_rotate_inverse,
    ssa,
)

from gym.spaces import Box, Dict

logger = CustomLogger("drone_racing_task")


def _display_access_available():
    display = os.environ.get("DISPLAY")
    if not display:
        return False
    try:
        result = subprocess.run(
            ["xdpyinfo"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            env=dict(os.environ, DISPLAY=display),
        )
        return result.returncode == 0
    except Exception:
        return False


class DroneRacingTask(BaseTask):
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
        self.optical_flow_debug_enabled = bool(
            getattr(self.task_config, "show_env0_optical_flow", False)
        ) and (not bool(self.task_config.headless))
        self.optical_flow_debug_available = cv2 is not None and _display_access_available()
        self.optical_flow_debug_env_index = int(
            getattr(self.task_config, "show_env0_optical_flow_index", 0)
        )
        self.optical_flow_debug_scale = int(
            getattr(self.task_config, "show_env0_optical_flow_scale", 2)
        )
        self.optical_flow_debug_wait_ms = int(
            getattr(self.task_config, "show_env0_optical_flow_wait_ms", 1)
        )
        self.optical_flow_debug_window_name = str(
            getattr(self.task_config, "show_env0_optical_flow_window_name", "Env0 Optical Flow")
        )

        self._init_track_tensors()
        self._init_optical_flow_tensors()

        self.raw_actions = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_raw_actions = torch.zeros_like(self.raw_actions)
        self.attitude_commands = torch.zeros_like(self.raw_actions)

        self.current_gate_index = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.last_failure_target_gate_index = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.completed_laps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.goal_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.outside_reposition_claimed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.prev_gate_distance = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.prev_robot_position = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32, requires_grad=False
        )

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.successes = torch.zeros_like(self.terminations)
        self.last_contact_collision = torch.zeros_like(self.terminations)
        self.last_contact_force_norm = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.last_contact_force_peak = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.last_gate_collision = torch.zeros_like(self.terminations)
        self.last_wrong_gate_cross = torch.zeros_like(self.terminations)
        self.last_ground_collision = torch.zeros_like(self.terminations)
        self.last_out_of_bounds = torch.zeros_like(self.terminations)
        self.last_excessive_body_rate = torch.zeros_like(self.terminations)
        self.last_timeouts = torch.zeros_like(self.terminations)
        self.show_env0_reset_reason = bool(
            getattr(self.task_config, "show_env0_reset_reason", False)
        )

        self.observation_space = Dict(
            {
                "state": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.state_observation_dim,),
                    dtype=np.float32,
                ),
                "img_observation": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(
                        self.task_config.image_observation_channels,
                        self.task_config.image_height,
                        self.task_config.image_width,
                    ),
                    dtype=np.float32,
                ),
                "observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
            }
        )
        action_low = np.full((self.task_config.action_space_dim,), -1.0, dtype=np.float32)
        action_high = np.full((self.task_config.action_space_dim,), 1.0, dtype=np.float32)
        action_low[0] = float(getattr(self.task_config, "thrust_command_min", -1.0))
        action_high[0] = float(getattr(self.task_config, "thrust_command_max", 1.0))
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)
        self.action_low_tensor = torch.tensor(
            action_low, device=self.device, dtype=torch.float32, requires_grad=False
        ).unsqueeze(0)
        self.action_high_tensor = torch.tensor(
            action_high, device=self.device, dtype=torch.float32, requires_grad=False
        ).unsqueeze(0)

        self.task_obs = {
            "state": torch.zeros(
                (self.num_envs, self.task_config.state_observation_dim),
                device=self.device,
                requires_grad=False,
            ),
            "img_observation": torch.zeros(
                (
                    self.num_envs,
                    self.task_config.image_observation_channels,
                    self.task_config.image_height,
                    self.task_config.image_width,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
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
        gate_eulers = torch.zeros(
            (self.num_gates, 3), device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.gate_yaws = torch.tensor(
            gate_yaws, device=self.device, dtype=torch.float32, requires_grad=False
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

        self.max_body_rate = (
            self.task_config.max_body_rate_deg_s * np.pi / 180.0
        )

        self.reward_params = {}
        for key, value in self.task_config.reward_parameters.items():
            self.reward_params[key] = torch.tensor(
                value, device=self.device, dtype=torch.float32, requires_grad=False
            )

        self.obs_dict["num_obstacles_in_env"] = self.num_gates + self.task_config.num_random_cylinders

        final_gate_forward_offset = torch.tensor(
            [[self.task_config.post_gate_goal_distance_m, 0.0, 0.0]],
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.final_goal_position = (
            self.gate_positions[-1]
            + quat_rotate(self.gate_quats[-1].view(1, 4), final_gate_forward_offset)[0]
        )

    def _init_optical_flow_tensors(self):
        camera_config = getattr(self.task_config, "camera_config", None)
        if camera_config is None:
            raise ValueError("drone_racing_task requires task_config.camera_config to be set.")

        self.camera_max_range = float(camera_config.max_range)
        self.camera_min_range = float(camera_config.min_range)
        self.camera_normalize_range = bool(camera_config.normalize_range)
        self.flow_clip_value = float(self.task_config.optical_flow_clip_pixels_per_step)
        self.camera_render_height = int(camera_config.height)
        self.camera_render_width = int(camera_config.width)
        self.central_flow_crop_height_ratio = float(
            getattr(self.task_config, "central_flow_crop_height_ratio", 0.5)
        )
        self.central_flow_crop_width_ratio = float(
            getattr(self.task_config, "central_flow_crop_width_ratio", 0.5)
        )

        camera_euler_deg = torch.tensor(
            [camera_config.euler_frame_rot_deg],
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        camera_euler_rad = torch.deg2rad(camera_euler_deg)
        self.camera_frame_quat = quat_from_euler_xyz_tensor(camera_euler_rad).expand(
            self.num_envs, -1
        )
        self.camera_local_position = torch.tensor(
            camera_config.nominal_position,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).expand(self.num_envs, -1)

        image_width = float(self.camera_render_width)
        image_height = float(self.camera_render_height)
        horizontal_fov_rad = np.deg2rad(camera_config.horizontal_fov_deg)
        self.flow_fx = image_width * 0.5 / np.tan(horizontal_fov_rad * 0.5)
        self.flow_fy = self.flow_fx
        self.flow_cx = image_width * 0.5
        self.flow_cy = image_height * 0.5

        pixel_x = torch.arange(
            self.camera_render_width, device=self.device, dtype=torch.float32
        )
        pixel_y = torch.arange(
            self.camera_render_height, device=self.device, dtype=torch.float32
        )
        grid_y, grid_x = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
        self.flow_x_grid = ((grid_x - self.flow_cx) / self.flow_fx).unsqueeze(0)
        self.flow_y_grid = ((grid_y - self.flow_cy) / self.flow_fy).unsqueeze(0)

    def _current_gate_positions(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return self.gate_positions[gate_index]

    def _current_gate_quats(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return self.gate_quats[gate_index]

    def _current_gate_yaws(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return self.gate_yaws[gate_index]

    def _previous_gate_indices(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return (gate_index - 1) % self.num_gates

    def _previous_gate_positions(self, env_ids=None):
        previous_gate_index = self._previous_gate_indices(env_ids)
        return self.gate_positions[previous_gate_index]

    def _previous_gate_quats(self, env_ids=None):
        previous_gate_index = self._previous_gate_indices(env_ids)
        return self.gate_quats[previous_gate_index]

    def _current_gate_semantic_ids(self):
        return self.gate_semantic_ids[self.current_gate_index]

    def _compute_gate_frame_positions(self, robot_positions, gate_positions, gate_quats):
        return quat_rotate_inverse(gate_quats, robot_positions - gate_positions)

    def _distance_to_current_gate(self, env_ids=None):
        if env_ids is None:
            robot_positions = self.obs_dict["robot_position"]
            gate_positions = self._current_gate_positions()
        else:
            robot_positions = self.obs_dict["robot_position"][env_ids]
            gate_positions = self._current_gate_positions(env_ids)
        return torch.norm(robot_positions - gate_positions, dim=1)

    def _active_target_positions(self, env_ids=None):
        if env_ids is None:
            gate_positions = self._current_gate_positions()
            goal_active = self.goal_active
        else:
            gate_positions = self._current_gate_positions(env_ids)
            goal_active = self.goal_active[env_ids]

        if getattr(self.task_config, "loop_track", False):
            return gate_positions

        goal_positions = self.final_goal_position.unsqueeze(0).expand_as(gate_positions)
        return torch.where(goal_active.unsqueeze(1), goal_positions, gate_positions)

    def _second_target_positions(self, env_ids=None):
        if env_ids is None:
            current_gate_index = self.current_gate_index
            goal_active = self.goal_active
        else:
            current_gate_index = self.current_gate_index[env_ids]
            goal_active = self.goal_active[env_ids]

        second_gate_index = (current_gate_index + 1) % self.num_gates
        second_gate_positions = self.gate_positions[second_gate_index]
        if getattr(self.task_config, "loop_track", False):
            return second_gate_positions

        goal_positions = self.final_goal_position.unsqueeze(0).expand_as(second_gate_positions)
        return torch.where(goal_active.unsqueeze(1), goal_positions, second_gate_positions)

    def _second_gate_yaws(self, env_ids=None):
        if env_ids is None:
            current_gate_index = self.current_gate_index
        else:
            current_gate_index = self.current_gate_index[env_ids]
        second_gate_index = (current_gate_index + 1) % self.num_gates
        return self.gate_yaws[second_gate_index]

    def _current_progress_target_positions(self, env_ids=None):
        gate_positions = self._current_gate_positions(env_ids)
        gate_quats = self._current_gate_quats(env_ids)
        offset = torch.tensor(
            [[self.task_config.progress_gate_offset_m, 0.0, 0.0]],
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        return gate_positions + quat_rotate(gate_quats, offset.expand(gate_positions.shape[0], -1))

    def _distance_to_active_target(self, env_ids=None):
        if env_ids is None:
            robot_positions = self.obs_dict["robot_position"]
        else:
            robot_positions = self.obs_dict["robot_position"][env_ids]
        target_positions = self._active_target_positions(env_ids)
        return torch.norm(robot_positions - target_positions, dim=1)

    def _compute_inverse_depth_observation(self, depth_image):
        depth_metric, valid_depth = self._depth_image_to_metric(depth_image)
        depth_safe = torch.clamp(
            depth_metric,
            min=float(self.task_config.inverse_depth_epsilon_m),
        )
        inverse_depth = torch.where(
            valid_depth,
            1.0 / depth_safe,
            torch.zeros_like(depth_safe),
        )
        noise_std = float(getattr(self.task_config, "depth_observation_noise_std", 0.0))
        if noise_std > 0.0:
            inverse_depth = inverse_depth + torch.randn_like(inverse_depth) * noise_std
        inverse_depth = torch.clamp(inverse_depth, min=0.0)

        if (
            inverse_depth.shape[-2] != self.task_config.image_height
            or inverse_depth.shape[-1] != self.task_config.image_width
        ):
            inverse_depth = F.interpolate(
                inverse_depth.unsqueeze(1),
                size=(self.task_config.image_height, self.task_config.image_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return inverse_depth.unsqueeze(1)

    def _compute_target_yaw_alignment_reward(self, target_positions):
        robot_positions = self.obs_dict["robot_position"]
        target_xy = target_positions[:, 0:2] - robot_positions[:, 0:2]
        theta_gate = torch.atan2(target_xy[:, 1], target_xy[:, 0])
        theta_drone = self.obs_dict["robot_euler_angles"][:, 2]
        yaw_error = torch.atan2(
            torch.sin(theta_drone - theta_gate),
            torch.cos(theta_drone - theta_gate),
        )
        return self.reward_params["lambda_2_theta"] * torch.exp(-torch.abs(yaw_error))

    def _nearest_collision_distance_from_depth(self):
        depth_image = self.obs_dict["depth_range_pixels"][:, 0]
        depth_metric, valid_depth = self._depth_image_to_metric(depth_image)
        far_tensor = torch.full_like(depth_metric, self.camera_max_range)
        depth_for_min = torch.where(valid_depth, depth_metric, far_tensor)
        return torch.amin(depth_for_min.view(depth_for_min.shape[0], -1), dim=1)

    def _resample_cylinders_away_from_gates(self, env_ids):
        if len(env_ids) == 0 or self.task_config.num_random_cylinders <= 0:
            return
        if "env_asset_state_tensor" not in self.obs_dict:
            return
        if "asset_min_state_ratio" not in self.obs_dict or "asset_max_state_ratio" not in self.obs_dict:
            return

        num_cylinders = int(self.task_config.num_random_cylinders)
        env_asset_state = self.obs_dict["env_asset_state_tensor"]
        asset_min_ratio = self.obs_dict["asset_min_state_ratio"]
        asset_max_ratio = self.obs_dict["asset_max_state_ratio"]

        cylinder_positions = env_asset_state[env_ids, :num_cylinders, 0:3]
        min_ratio = asset_min_ratio[env_ids, :num_cylinders, 0:3]
        max_ratio = asset_max_ratio[env_ids, :num_cylinders, 0:3]

        bounds_min = torch.tensor(
            self.task_config.bounds_min,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).view(1, 1, 3)
        bounds_max = torch.tensor(
            self.task_config.bounds_max,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).view(1, 1, 3)

        sampled_min = bounds_min + (bounds_max - bounds_min) * min_ratio
        sampled_max = bounds_min + (bounds_max - bounds_min) * max_ratio

        gate_xy = self.gate_positions[:, 0:2].view(1, 1, self.num_gates, 2)
        exclusion_radius = float(self.task_config.cylinder_gate_exclusion_radius_m)

        for _ in range(32):
            cylinder_xy = cylinder_positions[:, :, 0:2].unsqueeze(2)
            gate_distance = torch.norm(cylinder_xy - gate_xy, dim=3)
            invalid_mask = torch.any(gate_distance < exclusion_radius, dim=2)
            if not torch.any(invalid_mask):
                break

            invalid_positions = torch.nonzero(invalid_mask, as_tuple=False)
            sample_count = invalid_positions.shape[0]
            sampled_ratio = torch.rand(
                (sample_count, 3),
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            )

            env_sel = invalid_positions[:, 0]
            cyl_sel = invalid_positions[:, 1]
            cylinder_positions[env_sel, cyl_sel] = (
                sampled_min[env_sel, cyl_sel]
                + (sampled_max[env_sel, cyl_sel] - sampled_min[env_sel, cyl_sel]) * sampled_ratio
            )

        env_asset_state[env_ids, :num_cylinders, 0:3] = cylinder_positions
        self.sim_env.IGE_env.write_to_sim()
        if getattr(self.sim_env, "use_warp", False):
            self.sim_env.warp_env.reset_idx(env_ids)

    def _set_robot_start_pose(self, env_ids):
        if len(env_ids) == 0:
            return

        robot_state = self.obs_dict["robot_state_tensor"]
        if bool(getattr(self.task_config, "randomize_start_gate", False)):
            gate_positions = self._previous_gate_positions(env_ids)
            gate_quats = self._previous_gate_quats(env_ids)
            gate_eulers = get_euler_xyz_tensor(gate_quats)

            distance_center = float(
                getattr(self.task_config, "spawn_after_previous_gate_forward_m", 0.5)
            )
            distance_jitter = float(
                getattr(self.task_config, "spawn_after_previous_gate_forward_jitter_m", 0.25)
            )
            lateral_jitter = float(
                getattr(self.task_config, "spawn_gate_lateral_jitter_m", 0.6)
            )
            vertical_jitter = float(
                getattr(self.task_config, "spawn_gate_vertical_jitter_m", 0.4)
            )
            yaw_jitter_rad = np.deg2rad(
                float(getattr(self.task_config, "spawn_gate_yaw_jitter_deg", 15.0))
            )

            local_offset = torch.zeros(
                (len(env_ids), 3),
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            )
            local_offset[:, 0] = distance_center + (
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
                min=float(getattr(self.task_config, "spawn_min_height_m", 0.25)),
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
            spawn_quat = quat_from_euler_xyz_tensor(spawn_eulers)

            robot_state[env_ids, 0:3] = world_position
            robot_state[env_ids, 3:7] = spawn_quat
        else:
            robot_state[env_ids, 0:3] = self.start_position
            robot_state[env_ids, 3:7] = self.start_quat
        robot_state[env_ids, 7:13] = 0.0
        self.obs_dict["robot_actions"][env_ids] = 0.0
        self.obs_dict["robot_prev_actions"][env_ids] = 0.0
        self.sim_env.robot_manager.robot.update_states()
        self.sim_env.IGE_env.write_to_sim()

    def _sample_start_gate_indices(self, env_ids):
        if len(env_ids) == 0:
            return
        if not bool(getattr(self.task_config, "randomize_start_gate", False)):
            self.current_gate_index[env_ids] = 0
            return
        self.current_gate_index[env_ids] = torch.randint(
            low=0,
            high=self.num_gates,
            size=(len(env_ids),),
            device=self.device,
            dtype=torch.long,
        )

    def _reset_task_buffers(self, env_ids):
        self.current_gate_index[env_ids] = 0
        self.completed_laps[env_ids] = 0
        self.goal_active[env_ids] = False
        self.outside_reposition_claimed[env_ids] = False
        self.raw_actions[env_ids] = 0.0
        self.prev_raw_actions[env_ids] = 0.0
        self.attitude_commands[env_ids] = 0.0
        self.terminations[env_ids] = False
        self.truncations[env_ids] = False
        self.rewards[env_ids] = 0.0
        self.last_contact_collision[env_ids] = False
        self.last_contact_force_norm[env_ids] = 0.0
        self.last_contact_force_peak[env_ids] = 0.0
        self.last_gate_collision[env_ids] = False
        self.last_wrong_gate_cross[env_ids] = False
        self.last_ground_collision[env_ids] = False
        self.last_out_of_bounds[env_ids] = False
        self.last_excessive_body_rate[env_ids] = False
        self.last_timeouts[env_ids] = False
        self.last_failure_target_gate_index[env_ids] = 0

    def close(self):
        self.sim_env.delete_env()

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
        retry_failed_target_gate = bool(
            getattr(self.task_config, "retry_failed_target_gate", False)
        )
        crashed_envs = self.terminations[env_ids].clone()
        retry_gate_indices = self.last_failure_target_gate_index[env_ids].clone()
        self._reset_task_buffers(env_ids)
        if retry_failed_target_gate and torch.any(crashed_envs):
            crash_env_ids = env_ids[crashed_envs]
            self.current_gate_index[crash_env_ids] = retry_gate_indices[crashed_envs]
            non_crash_env_ids = env_ids[~crashed_envs]
            self._sample_start_gate_indices(non_crash_env_ids)
        else:
            self._sample_start_gate_indices(env_ids)
        self._resample_cylinders_away_from_gates(env_ids)
        self._set_robot_start_pose(env_ids)
        self.sim_env.render(render_components="sensors")
        self.prev_robot_position[env_ids] = self.obs_dict["robot_position"][env_ids]
        self.prev_gate_distance[env_ids] = self._distance_to_active_target(env_ids)

    def render(self, mode="human"):
        return self.sim_env.render()

    def _normalize_action_commands(self, raw_actions):
        thrust_min = float(getattr(self.task_config, "thrust_command_min", -1.0))
        thrust_max = float(getattr(self.task_config, "thrust_command_max", 1.0))
        normalized_actions = torch.clamp(raw_actions, -1.0, 1.0)
        normalized_actions[:, 0] = torch.clamp(raw_actions[:, 0], thrust_min, thrust_max)
        return 0.5 * (normalized_actions + 1.0)

    def _actions_to_attitude_commands(self, raw_actions):
        commands = torch.zeros_like(raw_actions)
        commands[:, 0] = torch.clamp(
            raw_actions[:, 0],
            float(getattr(self.task_config, "thrust_command_min", -1.0)),
            float(getattr(self.task_config, "thrust_command_max", 1.0)),
        )
        commands[:, 1] = (
            torch.clamp(raw_actions[:, 1], -1.0, 1.0)
            * self.task_config.attitude_max_inclination_rad
        )
        commands[:, 2] = (
            torch.clamp(raw_actions[:, 2], -1.0, 1.0)
            * self.task_config.attitude_max_inclination_rad
        )
        commands[:, 3] = (
            torch.clamp(raw_actions[:, 3], -1.0, 1.0)
            * self.task_config.attitude_max_yaw_rate_rad_s
        )
        return commands

    def _depth_image_to_metric(self, depth_image):
        if self.camera_normalize_range:
            depth_metric = depth_image * self.camera_max_range
        else:
            depth_metric = depth_image

        valid_depth = (
            torch.isfinite(depth_metric)
            & (depth_metric > self.camera_min_range)
            & (depth_metric < self.camera_max_range)
        )
        return depth_metric, valid_depth

    def _compute_optical_flow(self, depth_image):
        depth_metric, valid_depth = self._depth_image_to_metric(depth_image)
        depth_safe = torch.clamp(depth_metric, min=self.task_config.optical_flow_depth_epsilon_m)

        body_linvel = self.obs_dict["robot_body_linvel"]
        body_angvel = self.obs_dict["robot_body_angvel"]
        camera_origin_vel_body = body_linvel + torch.cross(
            body_angvel, self.camera_local_position, dim=1
        )

        camera_linvel = quat_rotate_inverse(self.camera_frame_quat, camera_origin_vel_body)
        camera_angvel = quat_rotate_inverse(self.camera_frame_quat, body_angvel)

        x = self.flow_x_grid
        y = self.flow_y_grid

        v_x = camera_linvel[:, 0].view(-1, 1, 1)
        v_y = camera_linvel[:, 1].view(-1, 1, 1)
        v_z = camera_linvel[:, 2].view(-1, 1, 1)
        w_x = camera_angvel[:, 0].view(-1, 1, 1)
        w_y = camera_angvel[:, 1].view(-1, 1, 1)
        w_z = camera_angvel[:, 2].view(-1, 1, 1)

        flow_x_dot = ((-v_x + x * v_z) / depth_safe) + (x * y * w_x) - (
            (1.0 + x * x) * w_y
        ) + (y * w_z)
        flow_y_dot = ((-v_y + y * v_z) / depth_safe) + ((1.0 + y * y) * w_x) - (
            x * y * w_y
        ) - (x * w_z)

        flow_u = self.flow_fx * flow_x_dot * self.dt
        flow_v = self.flow_fy * flow_y_dot * self.dt

        zero_flow = torch.zeros_like(flow_u)
        flow_u = torch.where(valid_depth, flow_u, zero_flow)
        flow_v = torch.where(valid_depth, flow_v, zero_flow)

        optical_flow = torch.stack((flow_u, flow_v), dim=1)
        optical_flow = torch.clamp(
            optical_flow / self.flow_clip_value,
            min=-1.0,
            max=1.0,
        )
        return optical_flow

    def _resize_optical_flow_for_policy(self, optical_flow):
        if (
            optical_flow.shape[-2] == self.task_config.image_height
            and optical_flow.shape[-1] == self.task_config.image_width
        ):
            return optical_flow
        return F.interpolate(
            optical_flow,
            size=(self.task_config.image_height, self.task_config.image_width),
            mode="bilinear",
            align_corners=False,
        )

    def _central_flow_crop(self, optical_flow):
        crop_h = max(1, int(round(optical_flow.shape[-2] * self.central_flow_crop_height_ratio)))
        crop_w = max(1, int(round(optical_flow.shape[-1] * self.central_flow_crop_width_ratio)))
        crop_h = min(crop_h, optical_flow.shape[-2])
        crop_w = min(crop_w, optical_flow.shape[-1])

        start_h = max(0, (optical_flow.shape[-2] - crop_h) // 2)
        start_w = max(0, (optical_flow.shape[-1] - crop_w) // 2)
        return optical_flow[:, :, start_h : start_h + crop_h, start_w : start_w + crop_w]

    def _build_dual_optical_flow_input(self, optical_flow):
        full_flow = self._resize_optical_flow_for_policy(optical_flow)
        center_flow = self._resize_optical_flow_for_policy(self._central_flow_crop(optical_flow))
        dual_flow = torch.cat((full_flow, center_flow), dim=1)
        return dual_flow, full_flow, center_flow

    def _flow_to_debug_frame(self, flow):
        flow_u = flow[0]
        flow_v = flow[1]

        magnitude = np.sqrt(flow_u * flow_u + flow_v * flow_v)
        angle = np.arctan2(flow_v, flow_u)
        nonzero = magnitude[magnitude > 1.0e-6]
        if nonzero.size > 0:
            mag_scale = float(np.percentile(nonzero, 95.0))
            mag_scale = max(mag_scale, 1.0e-3)
        else:
            mag_scale = 1.0
        magnitude_vis = np.clip(magnitude / mag_scale, 0.0, 1.0)

        hue = (((angle + np.pi) / (2.0 * np.pi)) * 179.0).astype(np.uint8)
        saturation = np.full_like(hue, 255, dtype=np.uint8)
        value = np.maximum((magnitude_vis * 255.0).astype(np.uint8), 24)
        hsv = np.stack((hue, saturation, value), axis=-1)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame, float(magnitude.max()), float(mag_scale)

    def _show_env0_optical_flow(self, full_flow, center_flow):
        if not self.optical_flow_debug_enabled:
            return
        if not self.optical_flow_debug_available:
            return

        env_index = self.optical_flow_debug_env_index
        if env_index < 0 or env_index >= full_flow.shape[0]:
            return

        full_frame, full_max, full_p95 = self._flow_to_debug_frame(
            full_flow[env_index].detach().cpu().numpy()
        )
        center_frame, center_max, center_p95 = self._flow_to_debug_frame(
            center_flow[env_index].detach().cpu().numpy()
        )
        if center_frame.shape[0] != full_frame.shape[0] or center_frame.shape[1] != full_frame.shape[1]:
            center_frame = cv2.resize(
                center_frame,
                (full_frame.shape[1], full_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        frame = np.concatenate((full_frame, center_frame), axis=1)
        if self.optical_flow_debug_scale > 1:
            frame = cv2.resize(
                frame,
                (
                    frame.shape[1] * self.optical_flow_debug_scale,
                    frame.shape[0] * self.optical_flow_debug_scale,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imshow(self.optical_flow_debug_window_name, frame)
        cv2.waitKey(self.optical_flow_debug_wait_ms)

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
        intermediate_gate_pass = gate_passed & (~final_gate_pass)
        self.current_gate_index[intermediate_gate_pass] = previous_gate_index[
            intermediate_gate_pass
        ] + 1

        if getattr(self.task_config, "loop_track", False):
            self.completed_laps[final_gate_pass] += 1
            self.current_gate_index[final_gate_pass] = 0
            self.goal_active[final_gate_pass] = False
            return torch.zeros_like(final_gate_pass)

        self.completed_laps[final_gate_pass] += 1
        activate_goal = final_gate_pass & (self.completed_laps >= self.task_config.num_laps)
        wrap_to_start = final_gate_pass & (~activate_goal)
        self.current_gate_index[wrap_to_start] = 0
        self.goal_active[activate_goal] = True
        return activate_goal

    def _compute_reward_terms(self):
        gate_positions = self._current_gate_positions()
        gate_quats = self._current_gate_quats()
        active_target_positions = self._active_target_positions()

        robot_positions = self.obs_dict["robot_position"]
        current_target_distance = torch.norm(robot_positions - active_target_positions, dim=1)
        progress_reward = self.reward_params["lambda_1_progress"] * (
            self.prev_gate_distance - current_target_distance
        )

        gate_passed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        gate_collision = torch.zeros_like(gate_passed)
        wrong_gate_cross = torch.zeros_like(gate_passed)
        course_active = ~self.goal_active
        if torch.any(course_active):
            previous_gate_frame_position = self._compute_gate_frame_positions(
                self.prev_robot_position[course_active],
                gate_positions[course_active],
                gate_quats[course_active],
            )
            current_gate_frame_position = self._compute_gate_frame_positions(
                robot_positions[course_active],
                gate_positions[course_active],
                gate_quats[course_active],
            )

            gate_half_width = self.task_config.gate_pass_half_width
            gate_half_height = self.task_config.gate_pass_half_height
            gate_half_width = max(gate_half_width, 0.05)
            gate_half_height = max(gate_half_height, 0.05)

            crossed_gate_plane_forward = (previous_gate_frame_position[:, 0] < 0.0) & (
                current_gate_frame_position[:, 0] >= 0.0
            )
            crossed_gate_plane_reverse = (previous_gate_frame_position[:, 0] > 0.0) & (
                current_gate_frame_position[:, 0] <= 0.0
            )
            crossed_gate_plane_any = crossed_gate_plane_forward | crossed_gate_plane_reverse
            crossing_denom = current_gate_frame_position[:, 0] - previous_gate_frame_position[:, 0]
            crossing_alpha = torch.where(
                torch.abs(crossing_denom) > 1.0e-6,
                -previous_gate_frame_position[:, 0] / crossing_denom,
                torch.zeros_like(crossing_denom),
            )
            crossing_alpha = torch.clamp(crossing_alpha, 0.0, 1.0)
            crossing_gate_frame_position = (
                previous_gate_frame_position
                + crossing_alpha.unsqueeze(1)
                * (current_gate_frame_position - previous_gate_frame_position)
            )
            within_gate_opening = (
                (torch.abs(crossing_gate_frame_position[:, 1]) <= gate_half_width)
                & (torch.abs(crossing_gate_frame_position[:, 2]) <= gate_half_height)
            )
            gate_passed[course_active] = crossed_gate_plane_forward & within_gate_opening

            physical_gate_outer_half_width = 0.5 * self.task_config.physical_gate_outer_width
            physical_gate_outer_half_height = 0.5 * self.task_config.physical_gate_outer_height
            physical_gate_inner_half_width = 0.5 * self.task_config.physical_gate_inner_width
            physical_gate_inner_half_height = 0.5 * self.task_config.physical_gate_inner_height
            within_outer_gate_bounds = (
                (torch.abs(current_gate_frame_position[:, 1]) <= physical_gate_outer_half_width)
                & (torch.abs(current_gate_frame_position[:, 2]) <= physical_gate_outer_half_height)
            )
            inside_inner_void = (
                (torch.abs(current_gate_frame_position[:, 1]) < physical_gate_inner_half_width)
                & (torch.abs(current_gate_frame_position[:, 2]) < physical_gate_inner_half_height)
            )
            hit_gate_frame = (
                (torch.abs(current_gate_frame_position[:, 0]) <= 0.5 * self.task_config.gate_collision_depth)
                & within_outer_gate_bounds
                & (~inside_inner_void)
            )
            crossed_gate_wrong = crossed_gate_plane_any & (~gate_passed[course_active])
            gate_collision[course_active] = hit_gate_frame
            wrong_gate_cross[course_active] = crossed_gate_wrong
        theta_reward = self._compute_target_yaw_alignment_reward(active_target_positions)
        body_rates = self.obs_dict["robot_body_angvel"]
        current_action_norm = torch.norm(self.raw_actions, dim=1)
        delta_action_norm = torch.norm(self.raw_actions - self.prev_raw_actions, dim=1)
        command_reward = (
            self.reward_params["lambda_3_cmd_norm"] * current_action_norm
            + self.reward_params["lambda_4_cmd_delta"] * delta_action_norm
        )
        speed = torch.norm(self.obs_dict["robot_linvel"], dim=1)
        speed_reward = self.reward_params["lambda_5_speed"] * (
            speed - float(self.task_config.desired_speed_m_s)
        )
        overspeed_penalty = self.reward_params["lambda_12_overspeed"] * torch.clamp(
            speed - self.reward_params["overspeed_reference_speed_m_s"],
            min=0.0,
        )
        target_direction = active_target_positions - robot_positions
        target_direction = target_direction / torch.clamp(
            torch.norm(target_direction, dim=1, keepdim=True), min=1.0e-6
        )
        velocity_direction = self.obs_dict["robot_linvel"] / torch.clamp(
            speed.unsqueeze(1), min=1.0e-6
        )
        velocity_target_alignment = torch.sum(velocity_direction * target_direction, dim=1).clamp(
            -1.0, 1.0
        )
        velocity_alignment_active = (
            speed >= float(self.reward_params["velocity_alignment_min_speed_m_s"])
        ).float()
        velocity_target_alignment_reward = (
            self.reward_params["lambda_11_velocity_target_alignment"]
            * (1.0 - velocity_target_alignment)
            * velocity_alignment_active
        )
        altitude_error = torch.abs(active_target_positions[:, 2] - robot_positions[:, 2])
        altitude_reward = self.reward_params["lambda_10_altitude"] * torch.exp(
            -altitude_error / torch.clamp(self.reward_params["altitude_error_scale_m"], min=1.0e-6)
        )
        body_z_axis_world = quat_axis(self.obs_dict["robot_orientation"], 2)
        upright_alignment = body_z_axis_world[:, 2].clamp(-1.0, 1.0)
        upright_penalty = self.reward_params["lambda_9_upright"] * torch.clamp(
            self.reward_params["upright_cos_threshold"] - upright_alignment,
            min=0.0,
        )
        nearest_collision_distance = self._nearest_collision_distance_from_depth()
        avoid_reward = self.reward_params["lambda_6_avoid"] * (
            1.0
            / (
                nearest_collision_distance
                + self.reward_params["avoid_bias_b_omega"]
            )
        )
        gate_reward = self.reward_params["lambda_7_pass"] * gate_passed.float()

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

        contact_force_norm = torch.norm(self.obs_dict["robot_contact_force_tensor"], dim=1)
        if "robot_contact_force_tensor_all" in self.obs_dict:
            contact_force_peak = torch.norm(
                self.obs_dict["robot_contact_force_tensor_all"], dim=2
            ).amax(dim=1)
        else:
            contact_force_peak = contact_force_norm

        contact_collision = self.obs_dict["crashes"].clone()
        crashes = contact_collision | wrong_gate_cross
        self.last_failure_target_gate_index[crashes] = self.current_gate_index[crashes]

        self._update_gate_progress(gate_passed)
        updated_target_distance = self._distance_to_active_target()
        goal_reached = self.goal_active & (
            updated_target_distance <= self.task_config.goal_reach_radius_m
        )
        if getattr(self.task_config, "loop_track", False):
            goal_reached[:] = False
        success = goal_reached & (~crashes)

        timeouts = self.sim_env.sim_steps >= self.task_config.episode_len_steps
        truncations = success | timeouts

        self.last_contact_collision[:] = contact_collision
        self.last_contact_force_norm[:] = contact_force_norm
        self.last_contact_force_peak[:] = contact_force_peak
        self.last_gate_collision[:] = gate_collision
        self.last_wrong_gate_cross[:] = wrong_gate_cross
        self.last_ground_collision[:] = ground_collision
        self.last_out_of_bounds[:] = out_of_bounds
        self.last_excessive_body_rate[:] = excessive_body_rate
        self.last_timeouts[:] = timeouts

        rewards = (
            progress_reward
            + theta_reward
            + command_reward
            + speed_reward
            + overspeed_penalty
            + velocity_target_alignment_reward
            + altitude_reward
            + upright_penalty
            + avoid_reward
            + gate_reward
            + self.reward_params["lambda_8_crash"] * crashes.float()
        )

        self.prev_gate_distance[:] = self._distance_to_active_target()
        return rewards, crashes, truncations, success, gate_passed

    def step(self, actions):
        self.prev_raw_actions[:] = self.raw_actions
        self.raw_actions[:] = torch.minimum(
            torch.maximum(actions, self.action_low_tensor), self.action_high_tensor
        )
        self.prev_robot_position[:] = self.obs_dict["robot_position"]

        self.attitude_commands[:] = self._actions_to_attitude_commands(self.raw_actions)
        self.sim_env.step(actions=self.attitude_commands)

        (
            self.rewards[:],
            self.terminations[:],
            self.truncations[:],
            self.successes[:],
            gate_passed,
        ) = self._compute_reward_terms()

        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()
        else:
            rewards_to_return = self.rewards.clone()
            terminations_to_return = self.terminations.clone()
            truncations_to_return = self.truncations.clone()

        self.infos = {
            "successes": self.successes.clone(),
            "crashes": self.terminations.clone(),
            "gate_passed": gate_passed.clone(),
            "lap_count": self.completed_laps.clone(),
            "current_gate_index": self.current_gate_index.clone(),
            "goal_active": self.goal_active.clone(),
            "goal_position": self.final_goal_position.unsqueeze(0).expand(self.num_envs, -1).clone(),
            "robot_position": self.obs_dict["robot_position"].clone(),
            "robot_linvel": self.obs_dict["robot_linvel"].clone(),
            "contact_collision": self.last_contact_collision.clone(),
            "contact_force_norm": self.last_contact_force_norm.clone(),
            "contact_force_peak": self.last_contact_force_peak.clone(),
            "gate_collision": self.last_gate_collision.clone(),
            "wrong_gate_cross": self.last_wrong_gate_cross.clone(),
            "ground_collision": self.last_ground_collision.clone(),
            "out_of_bounds": self.last_out_of_bounds.clone(),
            "excessive_body_rate": self.last_excessive_body_rate.clone(),
            "timeouts": self.last_timeouts.clone(),
        }

        if self.show_env0_reset_reason and bool((self.terminations[0] | self.truncations[0]).item()):
            env0_position = self.obs_dict["robot_position"][0]
            logger.warning(
                "[env0 reset] "
                f"step={int(self.sim_env.sim_steps[0].item())} "
                f"reward={float(self.rewards[0].item()):.3f} "
                f"gate_passed={int(gate_passed[0].item())} "
                f"success={int(self.successes[0].item())} "
                f"crash={int(self.terminations[0].item())} "
                f"timeout={int(self.last_timeouts[0].item())} "
                f"contact={int(self.last_contact_collision[0].item())} "
                f"contact_force_norm={float(self.last_contact_force_norm[0].item()):.5f} "
                f"contact_force_peak={float(self.last_contact_force_peak[0].item()):.5f} "
                f"gate_collision={int(self.last_gate_collision[0].item())} "
                f"wrong_gate_cross={int(self.last_wrong_gate_cross[0].item())} "
                f"ground={int(self.last_ground_collision[0].item())} "
                f"oob={int(self.last_out_of_bounds[0].item())} "
                f"body_rate={int(self.last_excessive_body_rate[0].item())} "
                f"goal_active={int(self.goal_active[0].item())} "
                f"gate_index={int(self.current_gate_index[0].item())} "
                f"pos=({float(env0_position[0].item()):.2f},"
                f"{float(env0_position[1].item()):.2f},"
                f"{float(env0_position[2].item()):.2f})"
            )

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
        self.task_obs["state"][:, 6:9] = self.obs_dict["robot_body_linvel"]
        self.task_obs["state"][:, 9:13] = self.obs_dict["robot_orientation"]
        self.task_obs["state"][:, 13:16] = self.obs_dict["robot_body_angvel"]
        self.task_obs["state"][:, 16] = torch.sin(relative_first_gate_yaw)
        self.task_obs["state"][:, 17] = torch.cos(relative_first_gate_yaw)
        self.task_obs["state"][:, 18] = torch.sin(relative_second_gate_yaw)
        self.task_obs["state"][:, 19] = torch.cos(relative_second_gate_yaw)
        self.task_obs["observations"][:] = self.task_obs["state"]

        depth_image = self.obs_dict["depth_range_pixels"][:, 0]
        optical_flow = self._compute_optical_flow(depth_image)
        optical_flow_dual, optical_flow_full, optical_flow_center = self._build_dual_optical_flow_input(
            optical_flow
        )
        self.task_obs["img_observation"][:] = optical_flow_dual
        self._show_env0_optical_flow(optical_flow_full, optical_flow_center)

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

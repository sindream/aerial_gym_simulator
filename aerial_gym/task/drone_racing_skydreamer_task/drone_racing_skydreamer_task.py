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
    quat_conjugate,
    quat_from_euler_xyz_tensor,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    ssa,
)

from gym.spaces import Box, Dict

logger = CustomLogger("drone_racing_skydreamer_task")


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


class DroneRacingSkyDreamerTask(BaseTask):
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
        self.motor_model.max_thrust = self.motor_model.max_thrust.clone()
        self.motor_model.min_thrust = self.motor_model.min_thrust.clone()
        self.runtime_mode = "train"
        self.spawn_any_gate_prob = float(self.task_config.train_any_gate_spawn_prob)
        self.action_noise_half_range = float(self.task_config.action_noise_half_range_train)
        self.motor_count = int(self.motor_model.num_motors_per_robot)
        self.segmentation_debug_enabled = bool(
            getattr(self.task_config, "show_env0_segmentation", False)
        ) and (not bool(self.task_config.headless))
        self.segmentation_debug_available = cv2 is not None and _display_access_available()
        self.segmentation_debug_env_index = int(
            getattr(self.task_config, "show_env0_segmentation_index", 0)
        )
        self.segmentation_debug_scale = int(
            getattr(self.task_config, "show_env0_segmentation_scale", 4)
        )
        self.segmentation_debug_wait_ms = int(
            getattr(self.task_config, "show_env0_segmentation_wait_ms", 1)
        )
        self.segmentation_debug_window_name = str(
            getattr(self.task_config, "show_env0_segmentation_window_name", "Env0 Segmentation")
        )

        self._init_track_tensors()

        self.raw_actions = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.prev_raw_actions = torch.zeros_like(self.raw_actions)
        self.executed_raw_actions = torch.zeros_like(self.raw_actions)
        self.motor_thrust_actions = torch.zeros_like(self.raw_actions)
        self.runtime_motor_omega_min = torch.full(
            (self.num_envs, self.motor_count),
            float(self.task_config.motor_omega_min),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.runtime_motor_omega_max = torch.full(
            (self.num_envs, self.motor_count),
            float(self.task_config.motor_omega_max),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.runtime_motor_thrust_constant = torch.full(
            (self.num_envs, self.motor_count),
            float(self.task_config.motor_thrust_constant),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.runtime_motor_curve_k = torch.full(
            (self.num_envs, self.motor_count),
            float(self.task_config.motor_command_curve_k),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self.current_gate_index = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.completed_laps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.prev_pre_gate_distance = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.prev_robot_position = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.gate_trigger_x = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.mask_erode_enabled = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.mask_erode_remaining = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.successes = torch.zeros_like(self.terminations)

        self.observation_space = Dict(
            {
                "image": Box(
                    low=0,
                    high=255,
                    shape=(
                        self.task_config.image_height,
                        self.task_config.image_width,
                        self.task_config.image_channels,
                    ),
                    dtype=np.uint8,
                ),
                "rates": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.body_rate_dim,),
                    dtype=np.float32,
                ),
                "motor_rpm": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.motor_rpm_dim,),
                    dtype=np.float32,
                ),
                "flight_plan": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.flight_plan_dim,),
                    dtype=np.float32,
                ),
                "priv/state": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.privileged_observation_space_dim,),
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
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )

        self.task_obs = {
            "image": torch.zeros(
                (
                    self.num_envs,
                    self.task_config.image_height,
                    self.task_config.image_width,
                    self.task_config.image_channels,
                ),
                device=self.device,
                dtype=torch.uint8,
                requires_grad=False,
            ),
            "rates": torch.zeros(
                (self.num_envs, self.task_config.body_rate_dim),
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            ),
            "motor_rpm": torch.zeros(
                (self.num_envs, self.task_config.motor_rpm_dim),
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            ),
            "flight_plan": torch.zeros(
                (self.num_envs, self.task_config.flight_plan_dim),
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            ),
            "priv/state": torch.zeros(
                (self.num_envs, self.task_config.privileged_observation_space_dim),
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            ),
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                dtype=torch.float32,
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
        self.pre_gate_x = -0.5 * self.task_config.gate_thickness_m
        self.post_gate_x = 0.5 * self.task_config.gate_thickness_m

        self.obs_dict["num_obstacles_in_env"] = self.num_gates

    def _current_gate_positions(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return self.gate_positions[gate_index]

    def _current_gate_quats(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return self.gate_quats[gate_index]

    def _current_gate_yaws(self, env_ids=None):
        gate_index = self.current_gate_index if env_ids is None else self.current_gate_index[env_ids]
        return self.gate_yaws[gate_index]

    def _current_pre_gate_positions(self, env_ids=None):
        gate_positions = self._current_gate_positions(env_ids)
        gate_forward = quat_axis(self._current_gate_quats(env_ids), 0)
        return gate_positions + self.pre_gate_x * gate_forward

    def _distance_to_current_pre_gate(self, env_ids=None):
        if env_ids is None:
            robot_positions = self.obs_dict["robot_position"]
            pre_gate_positions = self._current_pre_gate_positions()
        else:
            robot_positions = self.obs_dict["robot_position"][env_ids]
            pre_gate_positions = self._current_pre_gate_positions(env_ids)
        return torch.norm(robot_positions - pre_gate_positions, dim=1)

    def _compute_gate_frame_positions(self, robot_positions, gate_positions, gate_quats):
        return quat_rotate_inverse(gate_quats, robot_positions - gate_positions)

    def _resample_gate_trigger(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.task_config.gate_index_random_increment:
            self.gate_trigger_x[env_ids] = (
                self.pre_gate_x
                + (self.post_gate_x - self.pre_gate_x)
                * torch.rand(len(env_ids), device=self.device)
            )
        else:
            self.gate_trigger_x[env_ids] = 0.0

    def _paper_action_to_thrust(self, raw_actions, env_ids=None):
        if env_ids is None:
            runtime_curve_k = self.runtime_motor_curve_k
            runtime_omega_min = self.runtime_motor_omega_min
            runtime_omega_max = self.runtime_motor_omega_max
            runtime_thrust_constant = self.runtime_motor_thrust_constant
        else:
            runtime_curve_k = self.runtime_motor_curve_k[env_ids]
            runtime_omega_min = self.runtime_motor_omega_min[env_ids]
            runtime_omega_max = self.runtime_motor_omega_max[env_ids]
            runtime_thrust_constant = self.runtime_motor_thrust_constant[env_ids]
        normalized_action = 0.5 * (torch.clamp(raw_actions, -1.0, 1.0) + 1.0)
        k = runtime_curve_k
        omega = (
            (runtime_omega_max - runtime_omega_min)
            * torch.sqrt(
                torch.clamp(
                    k * normalized_action * normalized_action
                    + (1.0 - k) * normalized_action,
                    min=0.0,
                )
            )
            + runtime_omega_min
        )
        return runtime_thrust_constant * omega * omega

    def _motor_thrust_to_raw_action(self, motor_thrust, env_ids=None):
        if env_ids is None:
            thrust_constant = self.runtime_motor_thrust_constant
            omega_min = self.runtime_motor_omega_min
            omega_max = self.runtime_motor_omega_max
            curve_k = self.runtime_motor_curve_k
        else:
            thrust_constant = self.runtime_motor_thrust_constant[env_ids]
            omega_min = self.runtime_motor_omega_min[env_ids]
            omega_max = self.runtime_motor_omega_max[env_ids]
            curve_k = self.runtime_motor_curve_k[env_ids]

        omega = torch.sqrt(torch.clamp(motor_thrust / thrust_constant, min=0.0))
        speed_ratio = torch.clamp(
            (omega - omega_min) / torch.clamp(omega_max - omega_min, min=1.0e-6),
            0.0,
            1.0,
        )
        target = speed_ratio * speed_ratio
        linear_term = 1.0 - curve_k
        discriminant = torch.clamp(linear_term * linear_term + 4.0 * curve_k * target, min=0.0)
        normalized_action = torch.where(
            torch.abs(curve_k) > 1.0e-6,
            (-linear_term + torch.sqrt(discriminant)) / (2.0 * curve_k),
            target,
        )
        return torch.clamp(2.0 * normalized_action - 1.0, -1.0, 1.0)

    def set_runtime_mode(self, mode, reset_task=False):
        self.runtime_mode = str(mode).lower()
        if self.runtime_mode.startswith("eval") or self.runtime_mode.startswith("test"):
            self.spawn_any_gate_prob = float(self.task_config.eval_any_gate_spawn_prob)
            self.action_noise_half_range = float(self.task_config.action_noise_half_range_eval)
        elif self.runtime_mode.startswith("collect"):
            self.spawn_any_gate_prob = float(self.task_config.collect_any_gate_spawn_prob)
            self.action_noise_half_range = float(self.task_config.action_noise_half_range_collect)
        else:
            self.runtime_mode = "train"
            self.spawn_any_gate_prob = float(self.task_config.train_any_gate_spawn_prob)
            self.action_noise_half_range = float(self.task_config.action_noise_half_range_train)
        if reset_task:
            self.reset()

    def _get_spawn_ranges(self):
        prefix = self.runtime_mode if self.runtime_mode in ("train", "eval", "collect") else "train"
        return (
            getattr(self.task_config, f"{prefix}_spawn_x_g_range"),
            getattr(self.task_config, f"{prefix}_spawn_y_g_range"),
            getattr(self.task_config, f"{prefix}_spawn_z_g_range"),
            getattr(self.task_config, f"{prefix}_spawn_roll_pitch_yaw_g_range"),
            getattr(self.task_config, f"{prefix}_spawn_body_rate_range"),
        )

    def _sample_runtime_dynamics(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.runtime_mode == "train":
            dyn_low, dyn_high = 0.7, 1.3
        else:
            dyn_low, dyn_high = 0.8, 1.2

        env_count = len(env_ids)
        base_k = float(self.task_config.motor_thrust_constant)
        base_omega_min = float(self.task_config.motor_omega_min)
        base_omega_max = float(self.task_config.motor_omega_max)
        base_tau_inc = 0.5 * (
            float(self.motor_model.cfg.motor_time_constant_increasing_min)
            + float(self.motor_model.cfg.motor_time_constant_increasing_max)
        )
        base_tau_dec = 0.5 * (
            float(self.motor_model.cfg.motor_time_constant_decreasing_min)
            + float(self.motor_model.cfg.motor_time_constant_decreasing_max)
        )

        omega_scale_min = 0.8 + 0.4 * torch.rand((env_count, 1), device=self.device)
        omega_scale_max = 0.8 + 0.4 * torch.rand((env_count, 1), device=self.device)
        dyn_scale = dyn_low + (dyn_high - dyn_low) * torch.rand((env_count, 1), device=self.device)
        tau_inc_scale = dyn_low + (dyn_high - dyn_low) * torch.rand((env_count, 1), device=self.device)
        tau_dec_scale = dyn_low + (dyn_high - dyn_low) * torch.rand((env_count, 1), device=self.device)

        runtime_k = dyn_scale * base_k
        runtime_omega_min = omega_scale_min * base_omega_min
        runtime_omega_max = omega_scale_max * base_omega_max
        runtime_tau_inc = tau_inc_scale * base_tau_inc
        runtime_tau_dec = tau_dec_scale * base_tau_dec

        self.runtime_motor_thrust_constant[env_ids] = runtime_k.expand(-1, self.motor_count)
        self.runtime_motor_omega_min[env_ids] = runtime_omega_min.expand(-1, self.motor_count)
        self.runtime_motor_omega_max[env_ids] = runtime_omega_max.expand(-1, self.motor_count)
        self.runtime_motor_curve_k[env_ids] = float(self.task_config.motor_command_curve_k)

        self.motor_model.motor_thrust_constant[env_ids] = self.runtime_motor_thrust_constant[env_ids]
        self.motor_model.motor_time_constants_increasing[env_ids] = runtime_tau_inc.expand(
            -1, self.motor_count
        )
        self.motor_model.motor_time_constants_decreasing[env_ids] = runtime_tau_dec.expand(
            -1, self.motor_count
        )
        self.motor_model.max_thrust[env_ids] = self.runtime_motor_thrust_constant[env_ids] * (
            self.runtime_motor_omega_max[env_ids] ** 2
        )
        self.motor_model.min_thrust[env_ids] = self.runtime_motor_thrust_constant[env_ids] * (
            self.runtime_motor_omega_min[env_ids] ** 2
        )

    def _sample_initial_motor_actions(self, env_ids):
        low, high = self.task_config.initial_motor_speed_ratio_range
        speed_ratio = low + (high - low) * torch.rand(
            (len(env_ids), self.motor_count), device=self.device, dtype=torch.float32
        )
        target_omega = speed_ratio * self.runtime_motor_omega_max[env_ids]
        target_thrust = self.runtime_motor_thrust_constant[env_ids] * target_omega * target_omega
        return self._motor_thrust_to_raw_action(target_thrust, env_ids=env_ids)

    def _set_initial_motor_state(self, env_ids, initial_raw_actions):
        self.raw_actions[env_ids] = initial_raw_actions
        self.prev_raw_actions[env_ids] = initial_raw_actions
        self.executed_raw_actions[env_ids] = initial_raw_actions
        self.motor_thrust_actions[env_ids] = self._paper_action_to_thrust(
            initial_raw_actions, env_ids=env_ids
        )
        self.motor_model.current_motor_thrust[env_ids] = self.motor_thrust_actions[env_ids]
        self.motor_model.motor_rate[env_ids] = 0.0
        self.obs_dict["robot_actions"][env_ids] = self.motor_thrust_actions[env_ids]
        self.obs_dict["robot_prev_actions"][env_ids] = self.motor_thrust_actions[env_ids]

    def _resample_mask_augmentation(self, env_ids):
        if len(env_ids) == 0:
            return
        probability = float(self.task_config.camera_erode_probability)
        if probability <= 0.0:
            self.mask_erode_enabled[env_ids] = False
        else:
            self.mask_erode_enabled[env_ids] = (
                torch.rand(len(env_ids), device=self.device) < probability
            )
        self.mask_erode_remaining[env_ids] = int(self.task_config.camera_erode_hold_steps)

    def _advance_mask_augmentation(self):
        self.mask_erode_remaining -= 1
        env_ids = torch.nonzero(self.mask_erode_remaining <= 0, as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_mask_augmentation(env_ids)

    def _sample_gate_indices(self, env_ids):
        if len(env_ids) == 0:
            return torch.zeros(0, device=self.device, dtype=torch.long)
        use_any_gate = torch.rand(len(env_ids), device=self.device) < self.spawn_any_gate_prob
        gate_indices = torch.zeros(len(env_ids), device=self.device, dtype=torch.long)
        num_any = int(use_any_gate.sum().item())
        if num_any > 0:
            gate_indices[use_any_gate] = torch.randint(
                low=0,
                high=self.num_gates,
                size=(num_any,),
                device=self.device,
            )
        return gate_indices

    def _set_spawn_from_gate_frame(self, env_ids):
        if len(env_ids) == 0:
            return

        gate_indices = self._sample_gate_indices(env_ids)
        gate_positions = self.gate_positions[gate_indices]
        gate_quats = self.gate_quats[gate_indices]
        gate_yaws = self.gate_yaws[gate_indices]
        (
            spawn_x_g_range,
            spawn_y_g_range,
            spawn_z_g_range,
            spawn_roll_pitch_yaw_g_range,
            spawn_body_rate_range,
        ) = self._get_spawn_ranges()

        local_positions = torch.zeros((len(env_ids), 3), device=self.device)
        x_low, x_high = spawn_x_g_range
        y_low, y_high = spawn_y_g_range
        z_low, z_high = spawn_z_g_range
        local_positions[:, 0] = x_low + (x_high - x_low) * torch.rand(len(env_ids), device=self.device)
        local_positions[:, 1] = y_low + (y_high - y_low) * torch.rand(len(env_ids), device=self.device)
        local_positions[:, 2] = z_low + (z_high - z_low) * torch.rand(len(env_ids), device=self.device)

        spawn_positions = gate_positions + quat_rotate(gate_quats, local_positions)

        angle_low, angle_high = spawn_roll_pitch_yaw_g_range
        relative_eulers = torch.zeros((len(env_ids), 3), device=self.device)
        relative_eulers[:] = angle_low + (angle_high - angle_low) * torch.rand(
            (len(env_ids), 3), device=self.device
        )
        world_eulers = relative_eulers.clone()
        world_eulers[:, 2] = gate_yaws + relative_eulers[:, 2]
        spawn_quats = quat_from_euler_xyz_tensor(world_eulers)

        body_rates = torch.zeros((len(env_ids), 3), device=self.device)
        rate_low, rate_high = spawn_body_rate_range
        body_rates[:] = rate_low + (rate_high - rate_low) * torch.rand(
            (len(env_ids), 3), device=self.device
        )
        world_rates = quat_rotate(spawn_quats, body_rates)

        robot_state = self.obs_dict["robot_state_tensor"]
        robot_state[env_ids, 0:3] = spawn_positions
        robot_state[env_ids, 3:7] = spawn_quats
        robot_state[env_ids, 7:10] = 0.0
        robot_state[env_ids, 10:13] = world_rates

        self.current_gate_index[env_ids] = gate_indices
        self.completed_laps[env_ids] = 0

    def _finalize_resets(self, env_ids):
        if len(env_ids) == 0:
            return

        self._sample_runtime_dynamics(env_ids)
        initial_raw_actions = self._sample_initial_motor_actions(env_ids)
        self._set_spawn_from_gate_frame(env_ids)
        self.sim_env.robot_manager.robot.update_states()
        self.sim_env.IGE_env.write_to_sim()
        self.sim_env.IGE_env.refresh_tensors()
        self._set_initial_motor_state(env_ids, initial_raw_actions)
        self.sim_env.render(render_components="sensors")
        self._resample_gate_trigger(env_ids)
        self._resample_mask_augmentation(env_ids)

    def _reset_task_buffers(self, env_ids):
        self.raw_actions[env_ids] = 0.0
        self.prev_raw_actions[env_ids] = 0.0
        self.executed_raw_actions[env_ids] = 0.0
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
        self.prev_pre_gate_distance[:] = self._distance_to_current_pre_gate()
        self.infos = {}
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_task_buffers(env_ids)
        self._finalize_resets(env_ids)
        self.prev_robot_position[env_ids] = self.obs_dict["robot_position"][env_ids]
        self.prev_pre_gate_distance[env_ids] = self._distance_to_current_pre_gate(env_ids)

    def render(self, mode="human"):
        return self.sim_env.render()

    def _compute_flight_plan(self):
        current_idx = self.current_gate_index
        plan_ids = []
        for offset in range(-1, self.task_config.flight_plan_num_gates):
            plan_ids.append((current_idx + offset) % self.num_gates)

        gathered_positions = [self.gate_positions[idx] for idx in plan_ids]
        gathered_yaws = [self.gate_yaws[idx] for idx in plan_ids]

        diffs = []
        absolutes = []
        for offset in range(self.task_config.flight_plan_num_gates):
            diffs.append(gathered_positions[offset + 1] - gathered_positions[offset])
            diffs.append(
                ssa(gathered_yaws[offset + 1] - gathered_yaws[offset]).unsqueeze(-1)
            )
            absolutes.append(gathered_positions[offset + 1])
            absolutes.append(gathered_yaws[offset + 1].unsqueeze(-1))

        return torch.cat(diffs + absolutes, dim=1)

    def _all_gate_segmentation_mask(self):
        segmentation = self.obs_dict["segmentation_pixels"][:, 0].to(torch.int32)
        gate_ids = self.gate_semantic_ids.view(1, 1, 1, -1)
        mask = (segmentation.unsqueeze(-1) == gate_ids).any(dim=-1).float()

        if self.mask_erode_enabled.any():
            eroded = F.avg_pool2d(mask.unsqueeze(1), kernel_size=3, stride=1, padding=1)
            eroded = (eroded >= 0.999).float().squeeze(1)
            erode_mask = self.mask_erode_enabled.view(-1, 1, 1)
            mask = torch.where(erode_mask, eroded, mask)

        return (mask * 255.0).to(torch.uint8).unsqueeze(-1)

    def _show_env0_segmentation(self, image):
        if not self.segmentation_debug_enabled:
            return
        if not self.segmentation_debug_available:
            return
        env_index = self.segmentation_debug_env_index
        if env_index < 0 or env_index >= image.shape[0]:
            return

        frame = image[env_index].detach().cpu().numpy()
        if frame.shape[-1] == 1:
            frame = frame[..., 0]

        if self.segmentation_debug_scale > 1:
            frame = cv2.resize(
                frame,
                (
                    frame.shape[1] * self.segmentation_debug_scale,
                    frame.shape[0] * self.segmentation_debug_scale,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imshow(self.segmentation_debug_window_name, frame)
        cv2.waitKey(self.segmentation_debug_wait_ms)

    def _update_gate_progress(self, gate_passed):
        previous_gate_index = self.current_gate_index.clone()
        final_gate_pass = gate_passed & (previous_gate_index == (self.num_gates - 1))
        self.completed_laps[final_gate_pass] += 1
        self.current_gate_index[gate_passed] = (previous_gate_index[gate_passed] + 1) % self.num_gates
        gate_pass_ids = torch.nonzero(gate_passed, as_tuple=False).flatten()
        if len(gate_pass_ids) > 0:
            self._resample_gate_trigger(gate_pass_ids)
        success = final_gate_pass & (self.completed_laps >= self.task_config.num_laps)
        return success

    def _compute_reward_terms(self):
        gate_positions = self._current_gate_positions()
        gate_quats = self._current_gate_quats()
        pre_gate_positions = self._current_pre_gate_positions()

        robot_positions = self.obs_dict["robot_position"]
        current_pre_gate_distance = torch.norm(robot_positions - pre_gate_positions, dim=1)
        progress_reward = self.task_config.gate_progress_weight * (
            self.prev_pre_gate_distance - current_pre_gate_distance
        )

        previous_gate_frame_position = self._compute_gate_frame_positions(
            self.prev_robot_position, gate_positions, gate_quats
        )
        current_gate_frame_position = self._compute_gate_frame_positions(
            robot_positions, gate_positions, gate_quats
        )

        between_pre_and_post = (
            (current_gate_frame_position[:, 0] >= self.pre_gate_x)
            & (current_gate_frame_position[:, 0] <= self.post_gate_x)
        )
        progress_reward = torch.where(
            between_pre_and_post,
            torch.zeros_like(progress_reward),
            progress_reward,
        )

        max_lateral_offset = torch.maximum(
            torch.abs(current_gate_frame_position[:, 1]),
            torch.abs(current_gate_frame_position[:, 2]),
        )
        within_gate = max_lateral_offset <= self.task_config.gate_effective_size

        crossed_trigger = (previous_gate_frame_position[:, 0] < self.gate_trigger_x) & (
            current_gate_frame_position[:, 0] >= self.gate_trigger_x
        )
        gate_passed = crossed_trigger & within_gate
        gate_reward = self.task_config.gate_reward_weight * (
            1.0 - torch.clamp(max_lateral_offset / self.task_config.gate_effective_size, 0.0, 1.0)
        ) * gate_passed.float()

        crossed_pre_gate = (previous_gate_frame_position[:, 0] < self.pre_gate_x) & (
            current_gate_frame_position[:, 0] >= self.pre_gate_x
        )
        crossed_post_gate = (previous_gate_frame_position[:, 0] < self.post_gate_x) & (
            current_gate_frame_position[:, 0] >= self.post_gate_x
        )
        gate_collision = ((crossed_pre_gate | crossed_post_gate) & (~within_gate)) & (~gate_passed)

        body_rates = self.obs_dict["robot_body_angvel"]
        body_rate_l1 = torch.sum(torch.abs(body_rates), dim=1)
        rate_penalty = (
            0.5
            * self.task_config.control_frequency_hz
            * 1.0e-5
            * (torch.exp(torch.minimum(body_rate_l1, torch.full_like(body_rate_l1, self.task_config.rate_penalty_clip))) - 1.0)
        )

        roll_pitch_yaw_world = ssa(get_euler_xyz_tensor(self.obs_dict["robot_orientation"]))
        robot_linvel = self.obs_dict["robot_linvel"]
        near_ground = robot_positions[:, 2] <= self.task_config.ground_collision_height
        ground_collision = near_ground & (
            (torch.abs(robot_linvel[:, 2]) > self.task_config.ground_collision_speed)
            | (torch.abs(roll_pitch_yaw_world[:, 0]) > self.task_config.ground_collision_angle_rad)
            | (torch.abs(roll_pitch_yaw_world[:, 1]) > self.task_config.ground_collision_angle_rad)
        )

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

        rewards = progress_reward + gate_reward - rate_penalty
        rewards = torch.where(crashes, torch.zeros_like(rewards), rewards)

        self.prev_pre_gate_distance[:] = self._distance_to_current_pre_gate()
        return rewards, crashes, truncations, success, gate_passed

    def step(self, actions):
        self.prev_raw_actions[:] = self.raw_actions
        self.raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self.prev_robot_position[:] = self.obs_dict["robot_position"]

        if self.action_noise_half_range > 0.0:
            command_noise = (2.0 * torch.rand_like(self.raw_actions) - 1.0) * self.action_noise_half_range
            self.executed_raw_actions[:] = torch.clamp(self.raw_actions + command_noise, -1.0, 1.0)
        else:
            self.executed_raw_actions[:] = self.raw_actions

        self.motor_thrust_actions[:] = self._paper_action_to_thrust(self.executed_raw_actions)
        self.sim_env.step(actions=self.motor_thrust_actions)
        self._advance_mask_augmentation()

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
            "normalized_actions": self.raw_actions.clone(),
            "executed_actions": self.executed_raw_actions.clone(),
            "motor_thrust_actions": self.motor_thrust_actions.clone(),
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
        gate_yaws = self._current_gate_yaws()

        gate_frame_position = self._compute_gate_frame_positions(
            self.obs_dict["robot_position"], gate_positions, gate_quats
        )
        gate_frame_velocity = quat_rotate_inverse(gate_quats, self.obs_dict["robot_linvel"])

        world_euler = ssa(get_euler_xyz_tensor(self.obs_dict["robot_orientation"]))
        gate_frame_yaw = ssa(world_euler[:, 2] - gate_yaws).unsqueeze(-1)
        privileged_orientation = torch.cat([world_euler[:, 0:2], world_euler[:, 2:3], gate_frame_yaw], dim=1)

        body_rates = self.obs_dict["robot_body_angvel"]
        motor_rpm = torch.sqrt(
            torch.clamp(
                self.motor_model.current_motor_thrust
                / self.motor_model.motor_thrust_constant,
                min=0.0,
            )
        )
        flight_plan = self._compute_flight_plan()
        image = self._all_gate_segmentation_mask()
        self._show_env0_segmentation(image)

        camera_position = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        camera_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        warp_sensor = getattr(self.sim_env.robot_manager, "warp_sensor", None)
        if warp_sensor is not None:
            camera_position[:] = warp_sensor.sensor_local_position[:, 0]
            camera_euler[:] = ssa(get_euler_xyz_tensor(warp_sensor.sensor_local_orientation[:, 0]))

        dynamics_parameters = torch.cat(
            [
                self.runtime_motor_omega_min[:, 0:1],
                self.runtime_motor_omega_max[:, 0:1],
                torch.mean(self.motor_model.motor_thrust_constant, dim=1, keepdim=True),
                torch.mean(self.motor_model.motor_time_constants_increasing, dim=1, keepdim=True),
                torch.mean(self.motor_model.motor_time_constants_decreasing, dim=1, keepdim=True),
            ],
            dim=1,
        )

        privileged_state = torch.cat(
            [
                self.obs_dict["robot_position"],
                gate_frame_position,
                self.obs_dict["robot_linvel"],
                gate_frame_velocity,
                privileged_orientation,
                body_rates,
                motor_rpm,
                camera_position,
                camera_euler,
                dynamics_parameters,
            ],
            dim=1,
        )

        self.task_obs["image"][:] = image
        self.task_obs["rates"][:] = body_rates
        self.task_obs["motor_rpm"][:] = motor_rpm
        self.task_obs["flight_plan"][:] = flight_plan
        self.task_obs["priv/state"][:] = privileged_state
        self.task_obs["observations"][:, 0:3] = body_rates
        self.task_obs["observations"][:, 3:7] = motor_rpm
        self.task_obs["observations"][:, 7:31] = flight_plan

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

import os
import subprocess

from isaacgym import gymapi, gymtorch
import numpy as np
import torch
try:
    import cv2
except Exception:
    cv2 = None

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    get_euler_xyz_tensor,
    quat_axis,
    quat_from_euler_xyz_tensor,
    quat_rotate,
    quat_rotate_inverse,
)

from gym.spaces import Box, Dict


logger = CustomLogger("quintic_tracking_sysid_task")


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


class QuinticTrackingSysIDTask(BaseTask):
    def __init__(
        self,
        task_config,
        seed=None,
        num_envs=None,
        headless=None,
        device=None,
        use_warp=None,
        show_trajectory_debug=None,
        trajectory_debug_interval=None,
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
        if show_trajectory_debug is not None:
            task_config.show_trajectory_debug = show_trajectory_debug
        if trajectory_debug_interval is not None:
            task_config.trajectory_debug_interval = trajectory_debug_interval

        super().__init__(task_config)
        self.device = self.task_config.device
        self.reward_params = {
            key: torch.tensor(value, device=self.device, dtype=torch.float32)
            for key, value in self.task_config.reward_parameters.items()
        }

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
        self.policy_dt = float(self.task_config.policy_dt)
        self._init_trajectory_debug()

        self.raw_actions = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.prev_raw_actions = torch.zeros_like(self.raw_actions)
        self.rate_commands = torch.zeros_like(self.raw_actions)
        self.body_rate_command_axis_scale = self._make_body_rate_axis_scale()

        self.ref_start = torch.zeros((self.num_envs, 3), device=self.device)
        self.ref_control = torch.zeros_like(self.ref_start)
        self.ref_end = torch.zeros_like(self.ref_start)
        self.ref_start_velocity = torch.zeros_like(self.ref_start)
        self.ref_end_velocity = torch.zeros_like(self.ref_start)
        self.ref_start_acceleration = torch.zeros_like(self.ref_start)
        self.ref_end_acceleration = torch.zeros_like(self.ref_start)
        self.ref_coeffs = torch.zeros((self.num_envs, 6, 3), device=self.device)
        self.ref_duration = torch.zeros(self.num_envs, device=self.device)
        self.ref_segment_start_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self.ref_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.ref_velocity = torch.zeros_like(self.ref_position)
        self.ref_acceleration = torch.zeros_like(self.ref_position)
        self.prev_position_error_norm = torch.zeros(self.num_envs, device=self.device)
        self.sysid_target = torch.zeros(
            (self.num_envs, self.task_config.sysid_target_dim), device=self.device
        )
        self.hover_throttle_current = torch.full(
            (self.num_envs,),
            float(getattr(self.task_config, "hover_throttle", 0.25)),
            device=self.device,
            dtype=torch.float32,
        )
        self.max_obs_delay_steps = self._delay_seconds_to_steps(
            getattr(self.task_config, "obs_delay_max_s", 0.0)
        )
        self.max_action_delay_steps = self._delay_seconds_to_steps(
            getattr(self.task_config, "action_delay_max_s", 0.0)
        )
        self.obs_delay_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.action_delay_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.action_delay_buffer = torch.zeros(
            (
                self.num_envs,
                self.max_action_delay_steps + 1,
                self.task_config.action_space_dim,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.obs_delay_buffer = torch.zeros(
            (
                self.num_envs,
                self.max_obs_delay_steps + 1,
                self.task_config.state_observation_dim,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_start_position = torch.zeros_like(self.ref_start)
        self.reset_vehicle_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_actor_indices = self._get_robot_actor_indices()
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.successes = torch.zeros_like(self.terminations)
        self.segment_done = torch.zeros_like(self.terminations)

        observation_spaces = {
            "state": Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.task_config.state_observation_dim,),
                dtype=np.float32,
            ),
            "observations": Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.task_config.observation_space_dim,),
                dtype=np.float32,
            ),
            "sysid_target": Box(
                low=0.0,
                high=1.0,
                shape=(self.task_config.sysid_target_dim,),
                dtype=np.float32,
            ),
        }
        self.observation_space = Dict(observation_spaces)

        action_low = np.full((self.task_config.action_space_dim,), -1.0, dtype=np.float32)
        action_high = np.full((self.task_config.action_space_dim,), 1.0, dtype=np.float32)
        action_low[0] = float(self.task_config.thrust_command_min)
        action_high[0] = float(self.task_config.thrust_command_max)
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)
        self.action_low_tensor = torch.tensor(action_low, device=self.device).unsqueeze(0)
        self.action_high_tensor = torch.tensor(action_high, device=self.device).unsqueeze(0)

        self.task_obs = {
            "state": torch.zeros(
                (self.num_envs, self.task_config.state_observation_dim),
                device=self.device,
                requires_grad=False,
            ),
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "sysid_target": self.sysid_target,
        }
        self.current_state_obs = torch.zeros_like(self.task_obs["state"])

        self._physics_randomization_failed = False
        self._cache_nominal_physics_properties()
        self.infos = {}
        self._prime_isaacgym_root_state_writes()
        self.reset()

    def _delay_seconds_to_steps(self, delay_seconds):
        return max(0, int(round(float(delay_seconds) / max(self.policy_dt, 1.0e-6))))

    def _sample_delay_range_steps(self, min_seconds, max_seconds, randomize):
        min_steps = self._delay_seconds_to_steps(min_seconds)
        max_steps = self._delay_seconds_to_steps(max_seconds)
        if not bool(randomize):
            max_steps = min_steps
        if max_steps < min_steps:
            min_steps, max_steps = max_steps, min_steps
        return min_steps, max_steps

    def _sample_delay_steps(self, env_ids):
        obs_min, obs_max = self._sample_delay_range_steps(
            getattr(self.task_config, "obs_delay_min_s", 0.0),
            getattr(self.task_config, "obs_delay_max_s", 0.0),
            getattr(self.task_config, "randomize_obs_delay", False),
        )
        action_min, action_max = self._sample_delay_range_steps(
            getattr(self.task_config, "action_delay_min_s", 0.0),
            getattr(self.task_config, "action_delay_max_s", 0.0),
            getattr(self.task_config, "randomize_action_delay", False),
        )
        obs_min = min(obs_min, self.max_obs_delay_steps)
        obs_max = min(obs_max, self.max_obs_delay_steps)
        action_min = min(action_min, self.max_action_delay_steps)
        action_max = min(action_max, self.max_action_delay_steps)
        count = len(env_ids)
        if obs_max > obs_min:
            self.obs_delay_steps[env_ids] = torch.randint(
                obs_min, obs_max + 1, (count,), device=self.device
            )
        else:
            self.obs_delay_steps[env_ids] = obs_min
        if action_max > action_min:
            self.action_delay_steps[env_ids] = torch.randint(
                action_min, action_max + 1, (count,), device=self.device
            )
        else:
            self.action_delay_steps[env_ids] = action_min

    def _make_hover_raw_actions(self, env_ids):
        actions = torch.zeros(
            (len(env_ids), self.task_config.action_space_dim),
            device=self.device,
            dtype=torch.float32,
        )
        actions[:, 0] = self.hover_throttle_current[env_ids]
        return actions

    def _fill_action_delay_buffer(self, env_ids):
        hover_actions = self._make_hover_raw_actions(env_ids)
        self.action_delay_buffer[env_ids] = hover_actions[:, None, :]

    def _push_action_delay_buffer(self, actions):
        if self.max_action_delay_steps > 0:
            self.action_delay_buffer[:, 1:] = self.action_delay_buffer[:, :-1].clone()
        self.action_delay_buffer[:, 0] = actions

    def _get_delayed_raw_actions(self):
        return self.action_delay_buffer[self.all_env_ids, self.action_delay_steps]

    def _init_trajectory_debug(self):
        self.trajectory_debug_enabled = bool(
            getattr(self.task_config, "show_trajectory_debug", False)
        )
        self.trajectory_debug_available = cv2 is not None and _display_access_available()
        self.trajectory_debug_interval = max(
            1, int(getattr(self.task_config, "trajectory_debug_interval", 10))
        )
        self.trajectory_debug_history = max(
            2, int(getattr(self.task_config, "trajectory_debug_history", 500))
        )
        self.trajectory_debug_window_size = max(
            320, int(getattr(self.task_config, "trajectory_debug_window_size", 720))
        )
        self.trajectory_debug_wait_ms = max(
            1, int(getattr(self.task_config, "trajectory_debug_wait_ms", 1))
        )
        self.trajectory_debug_window_name = str(
            getattr(
                self.task_config,
                "trajectory_debug_window_name",
                "Env0 Quintic Tracking",
            )
        )
        self.trajectory_debug_actual_history = []
        self.trajectory_debug_ref_history = []
        self.trajectory_debug_ref_segment_history = []
        if self.trajectory_debug_enabled and not self.trajectory_debug_available:
            logger.warning(
                "Trajectory debug was requested, but OpenCV display access is unavailable."
            )

    def _cache_nominal_physics_properties(self):
        robot_manager = self.sim_env.robot_manager
        self.nominal_mass = self.obs_dict["robot_mass"].clone()
        self.nominal_inertia = self.obs_dict["robot_inertia"].clone()
        self.nominal_inertia_diag = torch.diagonal(
            self.nominal_inertia, dim1=1, dim2=2
        ).clone()
        self.nominal_rigid_body_mass_values = []
        self.nominal_rigid_body_inertia_diag_values = []
        for env_id in range(self.num_envs):
            env_handle = self.sim_env.IGE_env.env_handles[env_id]
            actor_handle = robot_manager.robot_handles[env_id]
            props = robot_manager.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            self.nominal_rigid_body_mass_values.append(
                [float(prop.mass) for prop in props]
            )
            self.nominal_rigid_body_inertia_diag_values.append(
                [
                    (
                        float(prop.inertia.x.x),
                        float(prop.inertia.y.y),
                        float(prop.inertia.z.z),
                    )
                    for prop in props
                ]
            )

    def close(self):
        self.sim_env.delete_env()

    def _prime_isaacgym_root_state_writes(self):
        # Run one safe low-level step before task-controlled resets. The reset
        # code below still handles stale post-reset root writes explicitly.
        self.sim_env.IGE_env.physics_step()
        self.sim_env.IGE_env.post_physics_step()
        self.sim_env.robot_manager.post_physics_step()
        self.sim_env.sim_steps[:] = 0

    def _get_robot_actor_indices(self):
        gym = self.sim_env.IGE_env.gym
        indices = []
        for env_id in range(self.num_envs):
            env_handle = self.sim_env.IGE_env.env_handles[env_id]
            actor_handle = self.sim_env.robot_manager.robot_handles[env_id]
            indices.append(gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM))
        return torch.tensor(indices, device=self.device, dtype=torch.int32)

    def _reset_sim_components_without_root_write(self, env_ids):
        # EnvManager.reset_idx() also mutates the Isaac Gym env bounds and then
        # normally performs a full root-state write. This tracking task owns the
        # robot spawn directly, and calling IGE_env.reset_idx() here can leave
        # the first post-reset actor root write stale in PhysX. Keep bounds
        # fixed and commit only the final robot actor pose below.
        self.sim_env.asset_manager.reset_idx(
            env_ids, self.sim_env.global_tensor_dict["num_obstacles_in_env"]
        )
        if self.sim_env.cfg.env.use_warp:
            self.sim_env.warp_env.reset_idx(env_ids)
        self.sim_env.robot_manager.reset_idx(env_ids)
        self.sim_env.sim_steps[env_ids] = 0
        self.sim_env.collision_tensor[env_ids] = False
        self.sim_env.truncation_tensor[env_ids] = False

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_sim_components_without_root_write(env_ids)
        self._reset_task_buffers(env_ids)
        self._configure_reset_envs(env_ids)
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_sim_components_without_root_write(env_ids)
        self._reset_task_buffers(env_ids)
        self._configure_reset_envs(env_ids)

    def _reset_task_buffers(self, env_ids):
        self.raw_actions[env_ids] = 0.0
        self.prev_raw_actions[env_ids] = 0.0
        self.rate_commands[env_ids] = 0.0
        self.prev_position_error_norm[env_ids] = 0.0
        self.rewards[env_ids] = 0.0
        self.terminations[env_ids] = False
        self.truncations[env_ids] = False
        self.successes[env_ids] = False
        self.segment_done[env_ids] = False
        self.obs_delay_steps[env_ids] = 0
        self.action_delay_steps[env_ids] = 0
        if torch.any(env_ids == 0):
            self.trajectory_debug_actual_history = []
            self.trajectory_debug_ref_history = []
            self.trajectory_debug_ref_segment_history = []

    def _configure_reset_envs(self, env_ids):
        self._apply_domain_randomization(env_ids)
        self._sample_delay_steps(env_ids)
        self.raw_actions[env_ids] = 0.0
        self.prev_raw_actions[env_ids] = 0.0
        self.raw_actions[env_ids, 0] = self.hover_throttle_current[env_ids]
        self.prev_raw_actions[env_ids, 0] = self.hover_throttle_current[env_ids]
        self._fill_action_delay_buffer(env_ids)
        self._set_robot_start_pose(env_ids)
        self._settle_reset_root_state_write(env_ids)
        # Use the refreshed simulator state as the reference start, not just the
        # requested reset pose. This keeps the debug path aligned with the drone
        # if Isaac Gym applies any root-state write correction.
        self._sample_reference_trajectories(env_ids, start_from_robot_state=True)
        self._evaluate_reference()
        self.prev_position_error_norm[env_ids] = torch.norm(
            self.ref_position[env_ids] - self.obs_dict["robot_position"][env_ids],
            dim=1,
        )
        self._fill_obs_delay_buffer(env_ids)
        self._seed_env0_trajectory_debug_at_reset(env_ids)

    def _set_robot_start_pose(self, env_ids):
        robot_state = self.obs_dict["robot_state_tensor"]
        count = len(env_ids)
        start_position = torch.tensor(
            self.task_config.start_position,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        start_jitter = torch.tensor(
            self.task_config.start_position_jitter,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        position_noise = (torch.rand((count, 3), device=self.device) * 2.0 - 1.0) * start_jitter
        positions = start_position + position_noise
        positions[:, 2] = torch.clamp(
            positions[:, 2],
            min=float(self.task_config.min_reference_height),
            max=float(self.task_config.max_reference_height),
        )

        yaw_jitter = np.deg2rad(float(self.task_config.start_yaw_jitter_deg))
        eulers = torch.zeros((count, 3), device=self.device, dtype=torch.float32)
        eulers[:, 2] = (torch.rand(count, device=self.device) * 2.0 - 1.0) * yaw_jitter
        quats = quat_from_euler_xyz_tensor(eulers)

        robot_state[env_ids, 0:3] = positions
        robot_state[env_ids, 3:7] = quats
        robot_state[env_ids, 7:13] = 0.0
        self.reset_start_position[env_ids] = positions
        self.reset_vehicle_orientation[env_ids] = quats
        self.obs_dict["robot_actions"][env_ids] = 0.0
        self.obs_dict["robot_prev_actions"][env_ids] = 0.0
        self.sim_env.robot_manager.robot.update_states()
        self._write_robot_root_states_to_sim(env_ids)

    def _settle_reset_root_state_write(self, env_ids):
        # On this task, the first root-state write after resetting the robot can
        # be consumed as a stale write by PhysX, leaving the actor near origin
        # while the task tensors/reference start at the requested pose. Consume
        # that stale write once, then restore the pre-settle robot states and
        # commit them again. We restore all robot actors so non-reset envs are
        # not advanced by this internal 2 ms settle step.
        desired_robot_states = self.obs_dict["robot_state_tensor"].clone()
        self.sim_env.IGE_env.physics_step()
        self.sim_env.IGE_env.post_physics_step()
        self.sim_env.robot_manager.post_physics_step()

        self.obs_dict["robot_state_tensor"][:] = desired_robot_states
        self.sim_env.robot_manager.robot.update_states()
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._write_robot_root_states_to_sim(all_env_ids)
        self.sim_env.sim_steps[env_ids] = 0
        self.sim_env.collision_tensor[env_ids] = False
        self.sim_env.truncation_tensor[env_ids] = False

    def _write_robot_root_states_to_sim(self, env_ids):
        actor_indices = self.robot_actor_indices[env_ids].contiguous()
        self.sim_env.IGE_env.gym.set_actor_root_state_tensor_indexed(
            self.sim_env.IGE_env.sim,
            gymtorch.unwrap_tensor(self.obs_dict["unfolded_env_asset_state_tensor"]),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )
        # Refresh derived vehicle/body-frame tensors from the tensor we just
        # wrote. Do not call refresh_tensors() here: Isaac Gym root-state writes
        # are consumed by the next simulation step, and an immediate refresh can
        # pull the pre-reset actor pose back from PhysX.
        self.sim_env.robot_manager.robot.update_states()

    def _clamp_vector_norm(self, vectors, max_norm):
        norms = torch.norm(vectors, dim=1, keepdim=True)
        scale = torch.clamp(float(max_norm) / torch.clamp(norms, min=1.0e-6), max=1.0)
        return vectors * scale

    def _sample_bounded_vectors(self, count, max_norm):
        vectors = torch.randn((count, 3), device=self.device, dtype=torch.float32)
        vectors = vectors / torch.clamp(torch.norm(vectors, dim=1, keepdim=True), min=1.0e-6)
        magnitudes = torch.rand((count, 1), device=self.device) * float(max_norm)
        return vectors * magnitudes

    def _compute_quintic_coeffs(self, p0, v0, a0, p1, v1, a1, durations):
        t = torch.clamp(durations, min=1.0e-6).view(-1, 1)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        dp = p1 - (p0 + v0 * t + 0.5 * a0 * t2)
        dv = v1 - (v0 + a0 * t)
        da = a1 - a0
        c3 = 10.0 * dp / t3 - 4.0 * dv / t2 + 0.5 * da / t
        c4 = -15.0 * dp / t4 + 7.0 * dv / t3 - da / t2
        c5 = 6.0 * dp / t5 - 3.0 * dv / t4 + 0.5 * da / t3
        return torch.stack((c0, c1, c2, c3, c4, c5), dim=1)

    def _estimate_quintic_max_speed(self, coeffs, durations, num_samples=32):
        sample_tau = torch.linspace(0.0, 1.0, num_samples, device=self.device)
        t = sample_tau.view(1, -1, 1) * durations.view(-1, 1, 1)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        velocity = (
            coeffs[:, None, 1, :]
            + 2.0 * coeffs[:, None, 2, :] * t
            + 3.0 * coeffs[:, None, 3, :] * t2
            + 4.0 * coeffs[:, None, 4, :] * t3
            + 5.0 * coeffs[:, None, 5, :] * t4
        )
        return torch.max(torch.norm(velocity, dim=2), dim=1).values

    def _evaluate_quintic_position_at(self, coeffs, t):
        t = t.view(-1, 1)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return (
            coeffs[:, 0, :]
            + coeffs[:, 1, :] * t
            + coeffs[:, 2, :] * t2
            + coeffs[:, 3, :] * t3
            + coeffs[:, 4, :] * t4
            + coeffs[:, 5, :] * t5
        )

    def _sample_reference_trajectories(self, env_ids, start_from_robot_state=False):
        count = len(env_ids)
        if start_from_robot_state:
            start = self.obs_dict["robot_position"][env_ids].clone()
            start_velocity = self._clamp_vector_norm(
                self.obs_dict["robot_linvel"][env_ids].clone(),
                float(self.task_config.trajectory_velocity_constraint_m_s),
            )
        else:
            start = self.reset_start_position[env_ids].clone()
            start_velocity = torch.zeros((count, 3), device=self.device)
        start_acceleration = torch.zeros((count, 3), device=self.device)

        duration_min = float(self.task_config.trajectory_duration_min_s)
        duration_max = float(self.task_config.trajectory_duration_max_s)
        durations = duration_min + (duration_max - duration_min) * torch.rand(
            count, device=self.device
        )

        position_limit = float(self.task_config.trajectory_position_constraint_m)
        velocity_limit = float(self.task_config.trajectory_velocity_constraint_m_s)
        acceleration_limit = float(
            self.task_config.trajectory_acceleration_constraint_m_s2
        )
        max_speed = float(self.task_config.trajectory_max_speed_m_s)

        vehicle_orientation = self.obs_dict["robot_vehicle_orientation"][env_ids]
        forward = position_limit * (0.35 + 0.65 * torch.rand((count, 1), device=self.device))
        lateral = position_limit * (torch.rand((count, 1), device=self.device) * 2.0 - 1.0) * 0.5
        vertical = position_limit * (torch.rand((count, 1), device=self.device) * 2.0 - 1.0) * 0.25
        displacement_vehicle = torch.cat((forward, lateral, vertical), dim=1)
        displacement_vehicle = self._clamp_vector_norm(displacement_vehicle, position_limit)
        displacement_world = quat_rotate(vehicle_orientation, displacement_vehicle)
        end = start + displacement_world
        end[:, 2] = torch.clamp(
            end[:, 2],
            min=float(self.task_config.min_reference_height),
            max=float(self.task_config.max_reference_height),
        )
        displacement_world = end - start

        direction = displacement_world / torch.clamp(
            torch.norm(displacement_world, dim=1, keepdim=True), min=1.0e-6
        )
        end_speed = velocity_limit * torch.rand((count, 1), device=self.device)
        end_velocity = direction * end_speed
        end_velocity = end_velocity + self._sample_bounded_vectors(count, 0.25 * velocity_limit)
        end_velocity = self._clamp_vector_norm(end_velocity, velocity_limit)
        end_acceleration = self._sample_bounded_vectors(count, acceleration_limit)

        for _ in range(4):
            coeffs = self._compute_quintic_coeffs(
                start,
                start_velocity,
                start_acceleration,
                end,
                end_velocity,
                end_acceleration,
                durations,
            )
            estimated_max_speed = self._estimate_quintic_max_speed(coeffs, durations)
            speed_scale = torch.clamp(
                estimated_max_speed / max(max_speed, 1.0e-6),
                min=1.0,
            )
            durations = durations * speed_scale

        coeffs = self._compute_quintic_coeffs(
            start,
            start_velocity,
            start_acceleration,
            end,
            end_velocity,
            end_acceleration,
            durations,
        )

        self.ref_duration[env_ids] = durations
        self.ref_segment_start_step[env_ids] = self.sim_env.sim_steps[env_ids]
        self.ref_start[env_ids] = start
        self.ref_end[env_ids] = end
        self.ref_start_velocity[env_ids] = start_velocity
        self.ref_end_velocity[env_ids] = end_velocity
        self.ref_start_acceleration[env_ids] = start_acceleration
        self.ref_end_acceleration[env_ids] = end_acceleration
        self.ref_coeffs[env_ids] = coeffs
        self.ref_control[env_ids] = self._evaluate_quintic_position_at(
            coeffs, 0.5 * durations
        )

    def _evaluate_reference(self):
        elapsed_steps = torch.clamp(
            self.sim_env.sim_steps - self.ref_segment_start_step,
            min=0,
        )
        t = torch.minimum(
            torch.clamp(elapsed_steps.float() * self.policy_dt, min=0.0),
            self.ref_duration,
        )
        t = t.view(-1, 1)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        coeffs = self.ref_coeffs

        self.ref_position[:] = (
            coeffs[:, 0, :]
            + coeffs[:, 1, :] * t
            + coeffs[:, 2, :] * t2
            + coeffs[:, 3, :] * t3
            + coeffs[:, 4, :] * t4
            + coeffs[:, 5, :] * t5
        )
        self.ref_velocity[:] = (
            coeffs[:, 1, :]
            + 2.0 * coeffs[:, 2, :] * t
            + 3.0 * coeffs[:, 3, :] * t2
            + 4.0 * coeffs[:, 4, :] * t3
            + 5.0 * coeffs[:, 5, :] * t4
        )
        self.ref_acceleration[:] = (
            2.0 * coeffs[:, 2, :]
            + 6.0 * coeffs[:, 3, :] * t
            + 12.0 * coeffs[:, 4, :] * t2
            + 20.0 * coeffs[:, 5, :] * t3
        )

    def _compute_effective_rate_gain_bounds(
        self, nominal_inertia_diag, inertia_scale_min, inertia_scale_max, controller, env_ids
    ):
        inertia_min = nominal_inertia_diag * inertia_scale_min
        inertia_max = nominal_inertia_diag * inertia_scale_max
        gain_min = controller.K_angvel_tensor_min[env_ids]
        gain_max = controller.K_angvel_tensor_max[env_ids]
        effective_min = gain_min / torch.clamp(inertia_max, min=1.0e-8)
        effective_max = gain_max / torch.clamp(inertia_min, min=1.0e-8)
        return effective_min, effective_max

    def _apply_domain_randomization(self, env_ids):
        count = len(env_ids)
        mass_scale = float(self.task_config.mass_scale_min) + (
            float(self.task_config.mass_scale_max) - float(self.task_config.mass_scale_min)
        ) * torch.rand(count, device=self.device)
        inertia_min = torch.tensor(
            self.task_config.inertia_scale_min,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        inertia_max = torch.tensor(
            self.task_config.inertia_scale_max,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        inertia_scale = inertia_min + (inertia_max - inertia_min) * torch.rand(
            (count, 3), device=self.device
        )
        nominal_hover_throttle = float(getattr(self.task_config, "hover_throttle", 0.26))
        hover_throttle_min = float(
            getattr(
                self.task_config,
                "hover_throttle_min",
                nominal_hover_throttle * float(self.task_config.mass_scale_min),
            )
        )
        hover_throttle_max = float(
            getattr(
                self.task_config,
                "hover_throttle_max",
                nominal_hover_throttle * float(self.task_config.mass_scale_max),
            )
        )
        sampled_hover_throttle = torch.clamp(
            nominal_hover_throttle * mass_scale,
            min=hover_throttle_min,
            max=hover_throttle_max,
        )

        nominal_mass = self.nominal_mass[env_ids]
        nominal_inertia_diag = self.nominal_inertia_diag[env_ids]
        randomized_mass = nominal_mass * mass_scale
        randomized_inertia_diag = nominal_inertia_diag * inertia_scale
        self.hover_throttle_current[env_ids] = sampled_hover_throttle

        robot_manager = self.sim_env.robot_manager
        if bool(getattr(self.task_config, "randomize_mass_inertia_physics", True)):
            try:
                for local_index, env_id_tensor in enumerate(env_ids):
                    env_id = int(env_id_tensor.item())
                    props = robot_manager.gym.get_actor_rigid_body_properties(
                        self.sim_env.IGE_env.env_handles[env_id],
                        robot_manager.robot_handles[env_id],
                    )
                    for prop_index, prop in enumerate(props):
                        prop.mass = (
                            self.nominal_rigid_body_mass_values[env_id][prop_index]
                            * float(mass_scale[local_index].item())
                        )
                        inertia_diag = self.nominal_rigid_body_inertia_diag_values[env_id][
                            prop_index
                        ]
                        prop.inertia.x.x = inertia_diag[0] * float(
                            inertia_scale[local_index, 0].item()
                        )
                        prop.inertia.y.y = inertia_diag[1] * float(
                            inertia_scale[local_index, 1].item()
                        )
                        prop.inertia.z.z = inertia_diag[2] * float(
                            inertia_scale[local_index, 2].item()
                        )
                    robot_manager.gym.set_actor_rigid_body_properties(
                        self.sim_env.IGE_env.env_handles[env_id],
                        robot_manager.robot_handles[env_id],
                        props,
                        False,
                    )
            except Exception as exc:
                raise RuntimeError(
                    "Runtime mass/inertia randomization failed. "
                    "Stopping because SysID labels would be physically meaningless "
                    "without matching PhysX rigid-body properties."
                ) from exc

        robot_manager.robot_masses[env_ids] = randomized_mass
        robot_manager.robot_inertias[env_ids] = torch.diag_embed(randomized_inertia_diag)

        controller = self.sim_env.robot_manager.robot.controller
        rate_gain = controller.K_angvel_tensor_current[env_ids]
        effective_rate_gain = rate_gain / torch.clamp(randomized_inertia_diag, min=1.0e-8)
        effective_rate_gain_min, effective_rate_gain_max = (
            self._compute_effective_rate_gain_bounds(
                self.nominal_inertia_diag[env_ids],
                inertia_min,
                inertia_max,
                controller,
                env_ids,
            )
        )
        self.sysid_target[env_ids, 0:3] = torch.clamp(
            (effective_rate_gain - effective_rate_gain_min)
            / torch.clamp(effective_rate_gain_max - effective_rate_gain_min, min=1.0e-6),
            0.0,
            1.0,
        )
        self.sysid_target[env_ids, 3] = torch.clamp(
            (sampled_hover_throttle - hover_throttle_min)
            / max(hover_throttle_max - hover_throttle_min, 1.0e-6),
            0.0,
            1.0,
        )

    def _evaluate_reference_numpy(self, env_index, num_points=120):
        coeffs = self.ref_coeffs[env_index].detach().cpu().numpy()
        duration = float(self.ref_duration[env_index].detach().cpu().item())
        t = np.linspace(0.0, max(duration, 1.0e-6), num_points, dtype=np.float32)
        powers = np.stack(
            (
                np.ones_like(t),
                t,
                t**2,
                t**3,
                t**4,
                t**5,
            ),
            axis=1,
        )
        return np.matmul(powers, coeffs).T

    def _append_env0_trajectory_debug_sample(self):
        if not self.trajectory_debug_enabled:
            return
        if self.num_envs <= 0:
            return
        step = int(self.sim_env.sim_steps[0].item())
        if step % self.trajectory_debug_interval != 0:
            return
        self.trajectory_debug_actual_history.append(
            self.obs_dict["robot_position"][0].detach().cpu().numpy().copy()
        )
        self.trajectory_debug_ref_history.append(
            self.ref_position[0].detach().cpu().numpy().copy()
        )
        self.trajectory_debug_ref_segment_history.append(
            int(self.ref_segment_start_step[0].detach().cpu().item())
        )
        if len(self.trajectory_debug_actual_history) > self.trajectory_debug_history:
            self.trajectory_debug_actual_history = self.trajectory_debug_actual_history[
                -self.trajectory_debug_history :
            ]
            self.trajectory_debug_ref_history = self.trajectory_debug_ref_history[
                -self.trajectory_debug_history :
            ]
            self.trajectory_debug_ref_segment_history = self.trajectory_debug_ref_segment_history[
                -self.trajectory_debug_history :
            ]

    def _seed_env0_trajectory_debug_at_reset(self, env_ids):
        if not self.trajectory_debug_enabled:
            return
        if not torch.any(env_ids == 0):
            return
        self.trajectory_debug_actual_history = [
            self.obs_dict["robot_position"][0].detach().cpu().numpy().copy()
        ]
        self.trajectory_debug_ref_history = [
            self.ref_position[0].detach().cpu().numpy().copy()
        ]
        self.trajectory_debug_ref_segment_history = [
            int(self.ref_segment_start_step[0].detach().cpu().item())
        ]

    def _project_debug_points(self, points, bounds_min, scale, offset, axes):
        projected = points[:, axes]
        px = ((projected - bounds_min) * scale + offset).astype(np.int32)
        px[:, 1] = self.trajectory_debug_window_size - px[:, 1]
        return px

    def _draw_debug_polyline(self, image, points, color, thickness=2):
        if len(points) < 2:
            return
        cv2.polylines(
            image,
            [points.reshape(-1, 1, 2)],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    def _split_debug_points_by_segment(self, points, segment_ids):
        if len(points) == 0:
            return []
        if len(segment_ids) != len(points):
            return [points]

        segments = []
        start = 0
        for index in range(1, len(points)):
            if segment_ids[index] != segment_ids[index - 1]:
                segments.append(points[start:index])
                start = index
        segments.append(points[start:])
        return segments

    def _draw_debug_panel(
        self,
        image,
        reference,
        actual,
        ref_history_segments,
        current_ref,
        forward_segment,
        axes,
        title,
    ):
        panel_size = self.trajectory_debug_window_size
        margin = 36.0
        point_sets = [reference]
        point_sets.extend(segment for segment in ref_history_segments if len(segment) > 0)
        point_sets.append(current_ref)
        if forward_segment is not None:
            point_sets.append(forward_segment)
        if len(actual) > 0:
            point_sets.append(actual)
        all_points = np.concatenate(point_sets, axis=0)
        projected = all_points[:, axes]
        bounds_min = projected.min(axis=0)
        bounds_max = projected.max(axis=0)
        extent = np.maximum(bounds_max - bounds_min, 1.0)
        scale = (panel_size - 2.0 * margin) / float(np.max(extent))
        centered_extent = np.array([panel_size, panel_size], dtype=np.float32) / scale
        center = 0.5 * (bounds_min + bounds_max)
        bounds_min = center - 0.5 * centered_extent
        offset = np.array([margin, margin], dtype=np.float32)

        ref_px = self._project_debug_points(reference, bounds_min, scale, offset, axes)
        self._draw_debug_polyline(image, ref_px, (80, 220, 80), thickness=2)
        cv2.circle(image, tuple(ref_px[0]), 6, (255, 255, 40), -1, lineType=cv2.LINE_AA)
        for ref_segment in ref_history_segments:
            ref_history_px = self._project_debug_points(ref_segment, bounds_min, scale, offset, axes)
            self._draw_debug_polyline(image, ref_history_px, (40, 130, 40), thickness=1)

        current_ref_px = self._project_debug_points(current_ref, bounds_min, scale, offset, axes)
        cv2.circle(
            image,
            tuple(current_ref_px[0]),
            6,
            (255, 80, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

        if len(actual) > 0:
            actual_px = self._project_debug_points(actual, bounds_min, scale, offset, axes)
            self._draw_debug_polyline(image, actual_px, (40, 140, 255), thickness=2)
            cv2.circle(image, tuple(actual_px[-1]), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        if forward_segment is not None:
            forward_px = self._project_debug_points(forward_segment, bounds_min, scale, offset, axes)
            cv2.arrowedLine(
                image,
                tuple(forward_px[0]),
                tuple(forward_px[1]),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
                tipLength=0.35,
            )

        cv2.putText(
            image,
            title,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

    def _show_env0_trajectory_debug(self):
        if not self.trajectory_debug_enabled:
            return
        if not self.trajectory_debug_available:
            return
        reference = self._evaluate_reference_numpy(0).T.astype(np.float32)
        actual = np.asarray(self.trajectory_debug_actual_history, dtype=np.float32)
        ref_history = np.asarray(self.trajectory_debug_ref_history, dtype=np.float32)
        ref_history_segments = self._split_debug_points_by_segment(
            ref_history, self.trajectory_debug_ref_segment_history
        )
        current_ref = self.ref_position[0].detach().cpu().numpy().reshape(1, 3).astype(np.float32)
        robot_position = self.obs_dict["robot_position"][0].detach().cpu().numpy()
        robot_forward = quat_axis(self.obs_dict["robot_orientation"], 0)[0].detach().cpu().numpy()
        forward_arrow_length = float(
            getattr(self.task_config, "trajectory_debug_forward_arrow_length_m", 1.0)
        )
        forward_segment = np.stack(
            (
                robot_position,
                robot_position + robot_forward * forward_arrow_length,
            ),
            axis=0,
        ).astype(np.float32)

        panel_size = self.trajectory_debug_window_size
        xy_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        xz_panel = np.zeros_like(xy_panel)
        self._draw_debug_panel(
            xy_panel,
            reference,
            actual,
            ref_history_segments,
            current_ref,
            forward_segment,
            (0, 1),
            "XY top view",
        )
        self._draw_debug_panel(
            xz_panel,
            reference,
            actual,
            ref_history_segments,
            current_ref,
            forward_segment,
            (0, 2),
            "XZ side view",
        )
        frame = np.concatenate((xy_panel, xz_panel), axis=1)

        pos_error = torch.norm(
            self.ref_position[0] - self.obs_dict["robot_position"][0]
        ).item()
        start_distance = torch.norm(
            self.ref_start[0] - self.obs_dict["robot_position"][0]
        ).item()
        tilt_angle = np.rad2deg(
            np.arccos(
                np.clip(float(quat_axis(self.obs_dict["robot_orientation"], 2)[0, 2].item()), -1.0, 1.0)
            )
        )
        cv2.putText(
            frame,
            "green=ref cyan=start magenta=target orange=actual red=current white=front | target_dist={:.2f}m start_dist={:.2f}m z(s/t/c)=({:.2f}/{:.2f}/{:.2f}) tilt={:.1f}deg".format(
                pos_error,
                start_distance,
                float(self.ref_start[0, 2].item()),
                float(self.ref_position[0, 2].item()),
                float(self.obs_dict["robot_position"][0, 2].item()),
                tilt_angle,
            ),
            (16, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.trajectory_debug_window_name, frame)
        cv2.waitKey(self.trajectory_debug_wait_ms)

    def render(self, mode="human"):
        return self.sim_env.render()

    def _make_body_rate_axis_scale(self):
        axis_scale = getattr(self.task_config, "body_rate_command_axis_scale_rad_s", None)
        if axis_scale is None:
            max_rate = float(
                getattr(
                    self.task_config,
                    "body_rate_command_max_rad_s",
                    np.deg2rad(
                        float(getattr(self.task_config, "body_rate_command_max_deg_s", 360.0))
                    ),
                )
            )
            axis_scale = [max_rate, max_rate, max_rate]
        if len(axis_scale) != 3:
            raise ValueError("body_rate_command_axis_scale_rad_s must have 3 values")
        return torch.tensor(
            axis_scale,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

    def _actions_to_rate_commands(self, raw_actions):
        commands = torch.zeros_like(raw_actions)
        controller = self.sim_env.robot_manager.robot.controller
        gravity_mag = torch.norm(controller.gravity, dim=1)
        throttle = torch.clamp(
            raw_actions[:, 0],
            float(self.task_config.thrust_command_min),
            float(self.task_config.thrust_command_max),
        )
        hover_throttle = torch.clamp(self.hover_throttle_current, min=1.0e-6)
        total_thrust_accel = throttle / hover_throttle * gravity_mag
        # LeeRatesController expects commanded vertical acceleration. It adds
        # gravity internally, so hover throttle maps to az=0, not az=g.
        commands[:, 0] = total_thrust_accel - gravity_mag
        commands[:, 1:4] = (
            torch.clamp(raw_actions[:, 1:4], -1.0, 1.0)
            * self.body_rate_command_axis_scale
        )
        return commands

    def step(self, actions):
        clipped_policy_actions = torch.minimum(
            torch.maximum(actions, self.action_low_tensor), self.action_high_tensor
        )
        self._push_action_delay_buffer(clipped_policy_actions)
        self.prev_raw_actions[:] = self.raw_actions
        self.raw_actions[:] = self._get_delayed_raw_actions()
        self.rate_commands[:] = self._actions_to_rate_commands(self.raw_actions)
        self.sim_env.step(actions=self.rate_commands)

        self._compute_rewards_and_dones()
        rewards_to_return = self.rewards.clone()
        terminations_to_return = self.terminations.clone()
        truncations_to_return = self.truncations.clone()

        chained_envs = (
            self.segment_done & ~terminations_to_return & ~truncations_to_return
        ).nonzero(as_tuple=False).squeeze(-1)
        if len(chained_envs) > 0:
            self._sample_reference_trajectories(
                chained_envs, start_from_robot_state=True
            )
            self._evaluate_reference()
            self.prev_position_error_norm[chained_envs] = torch.norm(
                self.ref_position[chained_envs]
                - self.obs_dict["robot_position"][chained_envs],
                dim=1,
            )

        self._push_obs_delay_buffer(self.all_env_ids)
        self._append_env0_trajectory_debug_sample()
        self._show_env0_trajectory_debug()
        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self._reset_task_buffers(reset_envs)
            self._configure_reset_envs(reset_envs)

        if not self.task_config.return_state_before_reset:
            return_tuple = (
                self.get_return_tuple()[0],
                rewards_to_return,
                terminations_to_return,
                truncations_to_return,
                self.infos,
            )
        return return_tuple

    def _compute_heading_error(self):
        path_direction = self.ref_end[:, 0:2] - self.ref_start[:, 0:2]
        target_direction = torch.where(
            torch.norm(self.ref_velocity[:, 0:2], dim=1, keepdim=True) > 0.2,
            self.ref_velocity[:, 0:2],
            path_direction,
        )
        desired_heading = torch.atan2(target_direction[:, 1], target_direction[:, 0])
        yaw = self.obs_dict["robot_euler_angles"][:, 2]
        return torch.atan2(torch.sin(yaw - desired_heading), torch.cos(yaw - desired_heading))

    def _compute_rewards_and_dones(self):
        self._evaluate_reference()
        position_error = self.ref_position - self.obs_dict["robot_position"]
        velocity_error = self.ref_velocity - self.obs_dict["robot_linvel"]
        position_error_norm = torch.norm(position_error, dim=1)
        velocity_error_norm = torch.norm(velocity_error, dim=1)
        heading_error = self._compute_heading_error()
        body_z_world = quat_axis(self.obs_dict["robot_orientation"], 2)
        upright_alignment = body_z_world[:, 2].clamp(-1.0, 1.0)
        body_rate_norm = torch.norm(self.obs_dict["robot_body_angvel"], dim=1)
        max_body_rate = np.deg2rad(float(self.task_config.max_body_rate_deg_s))
        body_rate_violation = torch.clamp(body_rate_norm - max_body_rate, min=0.0)
        upright_threshold = float(self.task_config.upright_cos_threshold)
        tilt_violation = torch.clamp(
            upright_threshold - upright_alignment,
            min=0.0,
        )
        robot_position = self.obs_dict["robot_position"]
        bounds_min = torch.tensor(self.task_config.bounds_min, device=self.device)
        bounds_max = torch.tensor(self.task_config.bounds_max, device=self.device)
        below_min = robot_position < bounds_min
        below_min[:, 2] = False
        out_of_bounds = torch.any(
            below_min | (robot_position > bounds_max),
            dim=1,
        )
        elapsed_steps = torch.clamp(
            self.sim_env.sim_steps - self.ref_segment_start_step,
            min=0,
        )
        segment_done = elapsed_steps.float() * self.policy_dt >= self.ref_duration
        timeouts = self.sim_env.sim_steps >= int(self.task_config.episode_len_steps)
        self.segment_done[:] = segment_done
        self.successes[:] = (
            segment_done
            & (position_error_norm < float(self.task_config.success_position_error_m))
            & (velocity_error_norm < float(self.task_config.success_velocity_error_m_s))
        )
        max_tracking_error_reset_m = float(
            getattr(self.task_config, "max_tracking_error_reset_m", 0.0)
        )
        if max_tracking_error_reset_m > 0.0:
            tracking_lost = position_error_norm > max_tracking_error_reset_m
        else:
            tracking_lost = torch.zeros_like(self.terminations)
        # Keep long chained references alive; leaving the original workspace bounds
        # should not terminate this tracking task.
        instability = body_rate_norm > max_body_rate
        crashes = self.obs_dict["crashes"].clone() | instability | tracking_lost
        self.terminations[:] = crashes
        self.truncations[:] = timeouts
        self.sim_env.collision_tensor[:] = self.sim_env.collision_tensor | tracking_lost

        pos_reward = (
            self.reward_params["pos_reward_gain1"]
            * torch.exp(-self.reward_params["pos_reward_exp1"] * position_error_norm**2)
            + self.reward_params["pos_reward_gain2"]
            * torch.exp(-self.reward_params["pos_reward_exp2"] * position_error_norm**2)
        )
        vel_reward = self.reward_params["vel_reward_gain"] * torch.exp(
            -self.reward_params["vel_reward_exp"] * velocity_error_norm**2
        )
        max_dist = float(self.reward_params["distance_reward_max_m"].item())
        distance_reward = self.reward_params["distance_reward_gain"] * (
            max_dist - torch.clamp(position_error_norm, max=max_dist)
        ) / max_dist

        progress = torch.clamp(
            self.prev_position_error_norm - position_error_norm,
            min=-float(self.reward_params["progress_clip_m"].item()),
            max=float(self.reward_params["progress_clip_m"].item()),
        )
        progress_reward = self.reward_params["progress_gain"] * progress

        robot_velocity = self.obs_dict["robot_linvel"]
        speed_norm = torch.norm(robot_velocity, dim=1)
        velocity_alignment = torch.sum(robot_velocity * position_error, dim=1) / (
            torch.clamp(speed_norm, min=1.0e-6)
            * torch.clamp(position_error_norm, min=1.0e-6)
        )
        velocity_direction_reward = self.reward_params["velocity_direction_gain"] * torch.where(
            (speed_norm > 0.05) & (position_error_norm > 0.10),
            torch.clamp(0.5 * (velocity_alignment + 1.0), 0.0, 1.0),
            torch.zeros_like(position_error_norm),
        )

        heading_reward = self.reward_params["heading_reward_gain"] * torch.exp(
            -self.reward_params["heading_reward_exp"] * heading_error**2
        )
        heading_error_penalty = self.reward_params["heading_error"] * (
            1.0 - torch.cos(heading_error)
        )
        up_error = 1.0 - upright_alignment
        up_reward = self.reward_params["up_reward_gain"] * torch.exp(
            -self.reward_params["up_reward_exp"] * up_error**2
        )
        body_rate_reward = self.reward_params["body_rate_reward_gain"] * torch.exp(
            -self.reward_params["body_rate_reward_exp"] * body_rate_norm**2
        )
        yaw_rate = self.obs_dict["robot_body_angvel"][:, 2]
        yaw_rate_penalty = self.reward_params["yaw_rate"] * yaw_rate**2
        max_pos_reward = (
            self.reward_params["pos_reward_gain1"] + self.reward_params["pos_reward_gain2"]
        )
        tracking_gate = torch.clamp(pos_reward / torch.clamp(max_pos_reward, min=1.0e-6), 0.0, 1.0)
        gated_stability_reward = tracking_gate * (
            heading_reward + up_reward + body_rate_reward
        )

        action_smoothness = torch.sum((self.raw_actions - self.prev_raw_actions) ** 2, dim=1)
        rate_command_norm = torch.sum(self.raw_actions[:, 1:4] ** 2, dim=1)
        yaw_command_norm = self.raw_actions[:, 3] ** 2
        hover_throttle = self.hover_throttle_current
        thrust_command_norm = (self.raw_actions[:, 0] - hover_throttle) ** 2
        constraint_violation = body_rate_violation**2 + tilt_violation**2

        shaped_reward = (
            pos_reward
            + vel_reward
            + distance_reward
            + progress_reward
            + velocity_direction_reward
            + gated_stability_reward
            + heading_error_penalty
            + yaw_rate_penalty
            + self.reward_params["body_rate_command"] * rate_command_norm
            + self.reward_params["yaw_command"] * yaw_command_norm
            + self.reward_params["thrust_command"] * thrust_command_norm
            + self.reward_params["action_smoothness"] * action_smoothness
            + self.reward_params["constraint_violation"] * constraint_violation
            + self.reward_params["success"] * self.successes.float()
        )
        self.rewards[:] = torch.where(
            crashes,
            self.reward_params["crash"] * torch.ones_like(shaped_reward),
            shaped_reward,
        )
        self.prev_position_error_norm[:] = position_error_norm.detach()
        self.infos = {
            "pos_error": position_error_norm.detach(),
            "vel_error": velocity_error_norm.detach(),
            "heading_error": torch.abs(heading_error).detach(),
            "constraint_violation": constraint_violation.detach(),
            "out_of_bounds": out_of_bounds.detach(),
            "reward_pos": pos_reward.detach(),
            "reward_vel": vel_reward.detach(),
            "reward_progress": progress_reward.detach(),
            "reward_direction": velocity_direction_reward.detach(),
            "reward_heading": heading_reward.detach(),
            "reward_heading_error": heading_error_penalty.detach(),
            "reward_yaw_rate": yaw_rate_penalty.detach(),
            "reward_up": up_reward.detach(),
            "reward_body_rate": body_rate_reward.detach(),
            "reward_gated_stability": gated_stability_reward.detach(),
            "success": self.successes.detach(),
            "segment_done": self.segment_done.detach(),
            "tracking_lost": tracking_lost.detach(),
            "max_tracking_error_reset_m": torch.full_like(
                position_error_norm, max_tracking_error_reset_m
            ).detach(),
            "robot_position": robot_position.detach().clone(),
            "robot_velocity": self.obs_dict["robot_linvel"].detach().clone(),
            "ref_position": self.ref_position.detach().clone(),
            "ref_velocity": self.ref_velocity.detach().clone(),
            "sysid_target": self.sysid_target.detach().clone(),
            "obs_delay_steps": self.obs_delay_steps.detach().float(),
            "action_delay_steps": self.action_delay_steps.detach().float(),
            "obs_delay_s": self.obs_delay_steps.detach().float() * self.policy_dt,
            "action_delay_s": self.action_delay_steps.detach().float() * self.policy_dt,
            "terminations": self.terminations.detach().clone(),
            "truncations": self.truncations.detach().clone(),
        }

    def _compute_current_state_observation(self, env_ids=None):
        self._evaluate_reference()
        if env_ids is None:
            env_ids = self.all_env_ids
        robot_position = self.obs_dict["robot_position"]
        robot_velocity = self.obs_dict["robot_linvel"]
        vehicle_orientation = self.obs_dict["robot_vehicle_orientation"]
        position_error_vehicle = quat_rotate_inverse(
            vehicle_orientation[env_ids], self.ref_position[env_ids] - robot_position[env_ids]
        )
        velocity_error_vehicle = quat_rotate_inverse(
            vehicle_orientation[env_ids], self.ref_velocity[env_ids] - robot_velocity[env_ids]
        )
        ref_velocity_vehicle = quat_rotate_inverse(
            vehicle_orientation[env_ids], self.ref_velocity[env_ids]
        )
        ref_accel_vehicle = quat_rotate_inverse(
            vehicle_orientation[env_ids], self.ref_acceleration[env_ids]
        )
        heading_error = self._compute_heading_error()[env_ids]
        body_up_vehicle = quat_rotate_inverse(
            vehicle_orientation[env_ids],
            quat_axis(self.obs_dict["robot_orientation"], 2)[env_ids],
        )
        state = self.current_state_obs
        state[env_ids, 0:3] = position_error_vehicle
        state[env_ids, 3:6] = velocity_error_vehicle
        state[env_ids, 6:9] = ref_velocity_vehicle
        state[env_ids, 9:12] = ref_accel_vehicle
        state[env_ids, 12] = torch.sin(heading_error)
        state[env_ids, 13] = torch.cos(heading_error)
        state[env_ids, 14:17] = body_up_vehicle
        state[env_ids, 17:20] = self.obs_dict["robot_body_angvel"][env_ids]
        state[env_ids, 20:24] = self.prev_raw_actions[env_ids]

    def _fill_obs_delay_buffer(self, env_ids):
        self._compute_current_state_observation(env_ids)
        self.obs_delay_buffer[env_ids] = self.current_state_obs[env_ids, None, :]

    def _push_obs_delay_buffer(self, env_ids=None):
        if env_ids is None:
            env_ids = self.all_env_ids
        self._compute_current_state_observation(env_ids)
        if self.max_obs_delay_steps > 0:
            self.obs_delay_buffer[env_ids, 1:] = self.obs_delay_buffer[
                env_ids, :-1
            ].clone()
        self.obs_delay_buffer[env_ids, 0] = self.current_state_obs[env_ids]

    def process_obs_for_task(self):
        delayed_state = self.obs_delay_buffer[self.all_env_ids, self.obs_delay_steps]
        self.task_obs["state"][:] = delayed_state
        self.task_obs["observations"][:] = delayed_state
        self.task_obs["sysid_target"][:] = self.sysid_target

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

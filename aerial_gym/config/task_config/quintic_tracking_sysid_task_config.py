import numpy as np


class task_config:
    seed = -1
    sim_name = "base_sim_1ms"
    env_name = "quintic_tracking_env"
    robot_name = "base_quadrotor_hover32"
    controller_name = "lee_rates_control_randomized_rates"
    args = {}
    num_envs = 1024
    use_warp = False
    headless = True
    device = "cuda:0"

    observation_space_dim = 24
    privileged_observation_space_dim = 0
    state_observation_dim = 24
    sysid_target_dim = 4
    action_space_dim = 4
    return_state_before_reset = False

    policy_dt = 0.02
    episode_len_steps = 780
    trajectory_duration_min_s = 2.5
    trajectory_duration_max_s = 4.5
    trajectory_position_constraint_m = 10.0
    trajectory_velocity_constraint_m_s = 10.0
    trajectory_acceleration_constraint_m_s2 = 5.0
    trajectory_max_speed_m_s = 15.0
    start_position = [0.0, 0.0, 1.5]
    start_position_jitter = [0.25, 0.25, 0.15]
    start_yaw_jitter_deg = 20.0
    min_reference_height = 0.8
    max_reference_height = 4.0

    thrust_command_min = 0.05
    thrust_command_max = 0.90
    hover_throttle = 0.26
    hover_throttle_min = 0.195
    hover_throttle_max = 0.351
    # Export the learned action with the same public meaning used by MAVLink
    # SET_ATTITUDE_TARGET: throttle in [0, 1] and body-rate setpoints in rad/s.
    # Axis/frame sign differences belong in the deployment bridge, not in the
    # policy action definition.
    body_rate_command_max_rad_s = 1.0
    body_rate_command_axis_scale_rad_s = [1.0, 1.0, 1.0]
    max_body_rate_deg_s = 720.0
    max_tilt_rad = np.deg2rad(65.0)
    upright_cos_threshold = float(np.cos(max_tilt_rad))
    max_tracking_error_reset_m = 12.0
    success_position_error_m = 0.5
    success_velocity_error_m_s = 1.0
    bounds_min = [-10.0, -10.0, 0.0]
    bounds_max = [10.0, 10.0, 8.0]

    show_trajectory_debug = False
    trajectory_debug_interval = 10
    trajectory_debug_history = 500
    trajectory_debug_window_size = 720
    trajectory_debug_wait_ms = 1
    trajectory_debug_window_name = "Env0 Quintic Tracking"
    trajectory_debug_forward_arrow_length_m = 1.0

    randomize_obs_delay = True
    obs_delay_min_s = 0.0
    obs_delay_max_s = 0.06
    randomize_action_delay = True
    action_delay_min_s = 0.0
    action_delay_max_s = 0.06

    # SysID labels are only meaningful if the sampled mass/inertia also affect
    # the simulator dynamics. Keep this enabled unless you are debugging PhysX
    # property writes explicitly.
    randomize_mass_inertia_physics = True
    mass_scale_min = 0.75
    mass_scale_max = 1.35
    inertia_scale_min = [0.65, 0.65, 0.65]
    inertia_scale_max = [1.60, 1.60, 1.60]

    reward_parameters = {
        "pos_reward_gain1": 3.0,
        "pos_reward_exp1": 1.5,
        "pos_reward_gain2": 2.0,
        "pos_reward_exp2": 8.0,
        "vel_reward_gain": 2.0,
        "vel_reward_exp": 0.8,
        "distance_reward_gain": 0.5,
        "distance_reward_max_m": 8.0,
        "progress_gain": 40.0,
        "progress_clip_m": 0.08,
        "velocity_direction_gain": 0.8,
        "heading_reward_gain": 1.2,
        "heading_reward_exp": 3.0,
        "heading_error": -1.5,
        "yaw_rate": -0.05,
        "yaw_command": -0.08,
        "up_reward_gain": 0.35,
        "up_reward_exp": 1.5,
        "body_rate_reward_gain": 0.4,
        "body_rate_reward_exp": 0.08,
        "body_rate_command": -0.015,
        "thrust_command": -0.015,
        "action_smoothness": -0.02,
        "constraint_violation": -3.0,
        "crash": -20.0,
        "success": 10.0,
    }

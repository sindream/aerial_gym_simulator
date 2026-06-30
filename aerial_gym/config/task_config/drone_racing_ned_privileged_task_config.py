import numpy as np

from aerial_gym.config.asset_config.racing_ned_track_asset_config import (
    GATE_COLLISION_DEPTH_METERS,
    GATE_INNER_SIZE_METERS,
    GATE_OUTER_SIZE_METERS,
    NUM_LAPS,
    RACING_NED_TRACK_GATES,
    START_POSITION,
    START_YAW_DEG,
    TRACK_BOUNDS_MAX,
    TRACK_BOUNDS_MIN,
)
from aerial_gym.config.controller_config.lee_controller_config_randomized_rates import (
    control as randomized_lee_controller_config,
)
from aerial_gym.config.sensor_config.camera_config.monorace_camera_config import (
    MonoRaceCameraConfig,
)


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "drone_racing_ned_env"
    robot_name = "base_quadrotor_racing"
    controller_name = "lee_rates_control_randomized_rates"
    camera_config = MonoRaceCameraConfig
    args = {}
    num_envs = 64
    use_warp = True
    headless = True
    device = "cuda:0"

    observation_space_dim = 19
    privileged_observation_space_dim = 19
    state_observation_dim = 19
    critic_state_observation_dim = 19
    rate_gain_target_dim = 3
    image_observation_channels = 4
    image_height = 12
    image_width = 16

    action_space_dim = 4
    thrust_command_min = -1.0
    thrust_command_max = 2.0
    episode_len_steps = 2500
    return_state_before_reset = False

    gate_layout = RACING_NED_TRACK_GATES
    num_laps = NUM_LAPS
    loop_track = False
    start_position = START_POSITION
    start_yaw_deg = START_YAW_DEG
    randomize_start_gate = False
    retry_failed_target_gate = False

    bounds_min = TRACK_BOUNDS_MIN
    bounds_max = TRACK_BOUNDS_MAX

    physical_gate_inner_width = GATE_INNER_SIZE_METERS
    physical_gate_inner_height = GATE_INNER_SIZE_METERS
    physical_gate_outer_width = GATE_OUTER_SIZE_METERS
    physical_gate_outer_height = GATE_OUTER_SIZE_METERS
    gate_pass_half_width = GATE_INNER_SIZE_METERS * 0.5
    gate_pass_half_height = GATE_INNER_SIZE_METERS * 0.5
    gate_collision_depth = GATE_COLLISION_DEPTH_METERS
    progress_gate_offset_m = 0.0
    drone_collision_radius = 0.09

    ground_collision_height = 0.05
    ground_collision_speed = 2.0
    ground_collision_requires_speed = True
    max_body_rate_deg_s = 1700.0

    attitude_max_inclination_rad = np.deg2rad(40.0)
    attitude_max_yaw_rate_rad_s = randomized_lee_controller_config.max_yaw_rate
    body_rate_command_max_deg_s = 360.0

    optical_flow_clip_pixels_per_step = 8.0
    optical_flow_depth_epsilon_m = 0.1
    desired_speed_m_s = 10.0
    depth_observation_noise_std = 0.01
    inverse_depth_epsilon_m = 0.1
    show_env0_optical_flow = False
    show_env0_optical_flow_index = 0
    show_env0_optical_flow_scale = 8
    show_env0_optical_flow_wait_ms = 1
    show_env0_optical_flow_window_name = "Env0 NED Privileged Optical Flow"
    show_env0_reset_reason = True
    num_random_cylinders = 0
    random_cylinder_radius_m = 0.0
    cylinder_gate_exclusion_radius_m = 0.0
    post_gate_goal_distance_m = 5.0
    goal_reach_radius_m = 1.0

    reward_parameters = {
        "lambda_1_progress": 6.0,
        "lambda_2_theta": 1.00,
        "lambda_3_cmd_norm": -0.0005,
        "lambda_4_cmd_delta": -0.0002,
        "lambda_5_speed": 0.00,
        "lambda_6_avoid": 0.0,
        "lambda_7_pass": 40.0,
        "lambda_8_crash": -15.0,
        "avoid_bias_b_omega": 0.5,
        "lambda_9_upright": -2.0,
        "upright_cos_threshold": 0.95,
        "lambda_10_altitude": 0.5,
        "lambda_11_velocity_target_alignment": -0.15,
        "lambda_12_overspeed": -0.08,
        "lambda_13_body_rate": -0.02,
        "body_rate_penalty_reference_rad_s": 4.0,
        "lambda_14_rate_command": -0.03,
        "lambda_15_rate_command_delta": -0.01,
        "lambda_16_yaw_error": -1.20,
        "lambda_17_yaw_rate": -0.20,
        "yaw_rate_penalty_reference_rad_s": 1.5,
        "lambda_18_yaw_command": -0.08,
        "altitude_error_scale_m": 2.0,
        "velocity_alignment_min_speed_m_s": 0.5,
        "overspeed_reference_speed_m_s": 10.0,
        "theta_relax_distance_m": 2.0,
        "theta_near_gate_min_scale": 0.0,
    }

import numpy as np

from aerial_gym.config.asset_config.racing_track_asset_config import (
    GATE_COLLISION_DEPTH_METERS,
    GATE_INNER_SIZE_METERS,
    GATE_OUTER_SIZE_METERS,
    NUM_LAPS,
    RANDOM_CYLINDER_RADIUS_METERS,
    RANDOM_CYLINDER_OBSTACLE_COUNT,
    RACING_TRACK_GATES,
    START_POSITION,
    START_YAW_DEG,
    TRACK_BOUNDS_MAX,
    TRACK_BOUNDS_MIN,
)
from aerial_gym.config.controller_config.lee_controller_config import control as lee_controller_config
from aerial_gym.config.sensor_config.camera_config.monorace_camera_config import (
    MonoRaceCameraConfig,
)


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "drone_racing_env"
    robot_name = "monorace_paper_camera_quadrotor"
    controller_name = "lee_attitude_control"
    args = {}
    num_envs = 64
    use_warp = True
    headless = True
    device = "cuda:0"

    observation_space_dim = 16
    privileged_observation_space_dim = 0
    state_observation_dim = 16
    image_observation_channels = 4
    image_height = 12
    image_width = 16

    action_space_dim = 4
    episode_len_steps = 2500
    return_state_before_reset = False

    gate_layout = RACING_TRACK_GATES
    num_laps = NUM_LAPS
    start_position = START_POSITION
    start_yaw_deg = START_YAW_DEG

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

    attitude_max_inclination_rad = lee_controller_config.max_inclination_angle_rad
    attitude_max_yaw_rate_rad_s = lee_controller_config.max_yaw_rate

    optical_flow_clip_pixels_per_step = 8.0
    optical_flow_depth_epsilon_m = 0.1
    desired_speed_m_s = 10.0
    depth_observation_noise_std = 0.01
    inverse_depth_epsilon_m = 0.1
    show_env0_optical_flow = True
    show_env0_optical_flow_index = 0
    show_env0_optical_flow_scale = 8
    show_env0_optical_flow_wait_ms = 1
    show_env0_optical_flow_window_name = "Env0 Optical Flow"
    show_env0_reset_reason = True
    num_random_cylinders = RANDOM_CYLINDER_OBSTACLE_COUNT
    random_cylinder_radius_m = RANDOM_CYLINDER_RADIUS_METERS
    cylinder_gate_exclusion_radius_m = 0.5 * GATE_OUTER_SIZE_METERS + RANDOM_CYLINDER_RADIUS_METERS
    post_gate_goal_distance_m = 5.0
    goal_reach_radius_m = 1.0

    reward_parameters = {
        "lambda_1_progress": 1.2,
        "lambda_2_theta": 0.05,
        "lambda_3_cmd_norm": -0.001,
        "lambda_4_cmd_delta": -0.0005,
        "lambda_5_speed": 0.0,
        "lambda_6_avoid": -0.01,
        "lambda_7_pass": 12.0,
        "lambda_8_crash": -4.0,
        "avoid_bias_b_omega": 0.5,
        "lambda_9_upright": -0.01,
        "upright_cos_threshold": 1.0,
    }

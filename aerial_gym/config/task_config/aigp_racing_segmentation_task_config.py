import numpy as np

from aerial_gym.config.asset_config.aigp_track_asset_config import (
    AIGP_SELECTED_TRACK_NAME,
    AIGP_TRACK_GATES,
    GATE_COLLISION_DEPTH_METERS,
    GATE_INNER_HEIGHT_METERS,
    GATE_INNER_WIDTH_METERS,
    GATE_OUTER_HEIGHT_METERS,
    GATE_OUTER_WIDTH_METERS,
    NUM_LAPS,
    START_POSITION,
    START_YAW_DEG,
    TRACK_BOUNDS_MAX,
    TRACK_BOUNDS_MIN,
)
from aerial_gym.config.controller_config.lee_controller_config_randomized_rates import (
    control as randomized_lee_controller_config,
)
from aerial_gym.config.sensor_config.camera_config.aigp_segmentation_camera_config import (
    AIGPSegmentationCameraConfig,
)


class task_config:
    seed = -1
    sim_name = "base_sim_1ms"
    env_name = "aigp_racing_env"
    robot_name = "base_quadrotor_aigp_racing"
    controller_name = "lee_rates_control_randomized_rates"
    camera_config = AIGPSegmentationCameraConfig
    args = {}
    num_envs = 128
    use_warp = False
    headless = True
    device = "cuda:0"

    selected_track_name = AIGP_SELECTED_TRACK_NAME

    # Actor gets only segmentation vision and allowed self-state:
    # accelerometer, gyro, previous action. Gate pose/position stays out of
    # actor obs and is used only by reward/reset and critic_state.
    observation_space_dim = 15
    privileged_observation_space_dim = 24
    state_observation_dim = 15
    critic_state_observation_dim = 24
    rate_gain_target_dim = 0
    image_observation_channels = 1
    image_height = AIGPSegmentationCameraConfig.height
    image_width = AIGPSegmentationCameraConfig.width

    action_space_dim = 4
    thrust_command_min = 0.05
    thrust_command_max = 0.90
    hover_throttle = 0.32
    body_rate_command_max_rad_s = 1.0
    episode_len_steps = 3500
    return_state_before_reset = False

    gate_layout = AIGP_TRACK_GATES
    num_laps = NUM_LAPS
    loop_track = False
    start_position = START_POSITION
    start_yaw_deg = START_YAW_DEG
    randomize_start_gate = False
    retry_failed_target_gate = False
    spawn_after_previous_gate_forward_m = 0.5
    spawn_after_previous_gate_forward_jitter_m = 0.25
    spawn_gate_lateral_jitter_m = 0.5
    spawn_gate_vertical_jitter_m = 0.25
    spawn_gate_yaw_jitter_deg = 10.0
    spawn_min_height_m = 0.2

    bounds_min = TRACK_BOUNDS_MIN
    bounds_max = TRACK_BOUNDS_MAX

    physical_gate_inner_width = GATE_INNER_WIDTH_METERS
    physical_gate_inner_height = GATE_INNER_HEIGHT_METERS
    physical_gate_outer_width = GATE_OUTER_WIDTH_METERS
    physical_gate_outer_height = GATE_OUTER_HEIGHT_METERS
    gate_pass_half_width = GATE_INNER_WIDTH_METERS * 0.5
    gate_pass_half_height = GATE_INNER_HEIGHT_METERS * 0.5
    gate_collision_depth = GATE_COLLISION_DEPTH_METERS
    progress_gate_offset_m = 0.0
    drone_collision_radius = 0.14

    ground_collision_height = 0.03
    ground_collision_speed = 1.0
    ground_collision_requires_speed = False
    max_body_rate_deg_s = 900.0

    attitude_max_inclination_rad = np.deg2rad(80.0)
    attitude_max_yaw_rate_rad_s = randomized_lee_controller_config.max_yaw_rate
    body_rate_command_max_deg_s = np.rad2deg(body_rate_command_max_rad_s)

    optical_flow_clip_pixels_per_step = 8.0
    optical_flow_depth_epsilon_m = 0.1
    desired_speed_m_s = 10.0
    depth_observation_noise_std = 0.0
    inverse_depth_epsilon_m = 0.1
    central_flow_crop_height_ratio = 0.5
    central_flow_crop_width_ratio = 0.5
    show_env0_optical_flow = False
    show_env0_optical_flow_index = 0
    show_env0_optical_flow_scale = 8
    show_env0_optical_flow_wait_ms = 1
    show_env0_optical_flow_window_name = "Env0 AIGP Segmentation"
    show_env0_reset_reason = True
    num_random_cylinders = 0
    random_cylinder_radius_m = 0.0
    cylinder_gate_exclusion_radius_m = 0.0
    post_gate_goal_distance_m = 5.0
    goal_reach_radius_m = 1.0

    imu_accel_normalizer_m_s2 = 39.24
    imu_gyro_normalizer_rad_s = 4.0
    gate_relation_position_normalizer_m = 30.0

    reward_parameters = {
        "lambda_1_progress": 6.0,
        "lambda_2_theta": 0.20,
        "lambda_3_cmd_norm": -0.0005,
        "lambda_4_cmd_delta": -0.0002,
        "lambda_5_speed": 0.00,
        "lambda_6_avoid": -0.01,
        "lambda_7_pass": 40.0,
        "lambda_8_crash": -15.0,
        "avoid_bias_b_omega": 0.5,
        "lambda_9_upright": 0.0,
        "upright_cos_threshold": 1.0,
        "lambda_10_altitude": 0.5,
        "lambda_11_velocity_target_alignment": -0.15,
        "lambda_12_overspeed": -0.08,
        "altitude_error_scale_m": 0.0,
        "velocity_alignment_min_speed_m_s": 0.5,
        "overspeed_reference_speed_m_s": 10.0,
        "theta_relax_distance_m": 2.0,
        "theta_near_gate_min_scale": 0.0,
    }

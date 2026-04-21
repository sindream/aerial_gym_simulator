import numpy as np

from aerial_gym.config.asset_config.racing_track_asset_config import (
    GATE_INNER_SIZE_METERS,
    NUM_LAPS,
    RACING_TRACK_GATES,
    START_POSITION,
    START_YAW_DEG,
    TRACK_BOUNDS_MAX,
    TRACK_BOUNDS_MIN,
)
from aerial_gym.config.sensor_config.camera_config.skydreamer_segmentation_camera_config import (
    SkyDreamerSegmentationCameraConfig,
)


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "drone_racing_env"
    robot_name = "monorace_skydreamer_quadrotor"
    controller_name = "no_control"
    collect_robot_name = "monorace_skydreamer_collect_quadrotor"
    collect_controller_name = "lee_position_control_skydreamer_collect"
    args = {}
    num_envs = 64
    use_warp = True
    headless = True
    device = "cuda:0"

    image_height = SkyDreamerSegmentationCameraConfig.height
    image_width = SkyDreamerSegmentationCameraConfig.width
    image_channels = 1
    show_env0_segmentation = True
    show_env0_segmentation_index = 0
    show_env0_segmentation_scale = 1
    show_env0_segmentation_wait_ms = 1
    show_env0_segmentation_window_name = "Env0 Segmentation"

    body_rate_dim = 3
    motor_rpm_dim = 4
    flight_plan_dim = 24
    observation_space_dim = body_rate_dim + motor_rpm_dim + flight_plan_dim
    privileged_observation_space_dim = 34
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

    # Paper-like virtual pre/post gate corridor.
    gate_effective_size = 0.8
    gate_thickness_m = 0.8
    gate_progress_weight = 5.0
    gate_reward_weight = 30.0
    control_frequency_hz = 90.0
    rate_penalty_clip = 17.0
    paper_drone_mass = 0.966
    motor_thrust_constant = paper_drone_mass * 1.55e-6
    motor_omega_min = 341.75
    motor_omega_max = 3100.0
    motor_command_curve_k = 0.5
    initial_motor_speed_ratio_range = [0.25, 0.5]
    action_noise_half_range_train = 0.2
    action_noise_half_range_eval = 0.2
    action_noise_half_range_collect = 0.0

    max_body_rate_deg_s = 1700.0

    train_any_gate_spawn_prob = 0.7
    eval_any_gate_spawn_prob = 0.0
    collect_any_gate_spawn_prob = 0.0
    train_spawn_x_g_range = [-4.0, -2.0]
    eval_spawn_x_g_range = [-4.0, -2.0]
    collect_spawn_x_g_range = [-4.0, -2.0]
    train_spawn_y_g_range = [-1.0, 1.0]
    eval_spawn_y_g_range = [-1.0, 1.0]
    collect_spawn_y_g_range = [-0.3, 0.3]
    train_spawn_z_g_range = [0.0, 1.3]
    eval_spawn_z_g_range = [0.7, 1.3]
    collect_spawn_z_g_range = [0.0, 0.3]
    train_spawn_roll_pitch_yaw_g_range = [-np.pi / 9.0, np.pi / 9.0]
    eval_spawn_roll_pitch_yaw_g_range = [-np.pi / 9.0, np.pi / 9.0]
    collect_spawn_roll_pitch_yaw_g_range = [-np.pi / 36.0, np.pi / 36.0]
    train_spawn_body_rate_range = [-0.1, 0.1]
    eval_spawn_body_rate_range = [-0.1, 0.1]
    collect_spawn_body_rate_range = [-0.02, 0.02]

    camera_erode_probability = 0.5
    camera_erode_hold_steps = 100

    flight_plan_num_gates = 3
    gate_index_random_increment = True

    ground_collision_height = 0.5
    ground_collision_speed = 1.0
    ground_collision_angle_rad = np.pi / 3.0

    reward_parameters = {
        "smoothness_weight": 0.0,
    }

import numpy as np

from aerial_gym.config.asset_config.racing_track_asset_config import (
    GATE_COLLISION_DEPTH_METERS,
    GATE_INNER_SIZE_METERS,
    NUM_LAPS,
    RACING_TRACK_GATES,
    START_POSITION,
    START_YAW_DEG,
    TRACK_BOUNDS_MAX,
    TRACK_BOUNDS_MIN,
)


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "drone_racing_env"
    robot_name = "monorace_paper_quadrotor"
    controller_name = "no_control"
    args = {}
    num_envs = 64
    use_warp = False
    headless = True
    device = "cuda:0"

    observation_space_dim = 20
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 25000
    return_state_before_reset = False

    gate_layout = RACING_TRACK_GATES
    num_laps = NUM_LAPS
    start_position = START_POSITION
    start_yaw_deg = START_YAW_DEG

    bounds_min = TRACK_BOUNDS_MIN
    bounds_max = TRACK_BOUNDS_MAX

    physical_gate_inner_width = GATE_INNER_SIZE_METERS
    physical_gate_inner_height = GATE_INNER_SIZE_METERS
    gate_pass_half_width = GATE_INNER_SIZE_METERS * 0.5
    gate_pass_half_height = GATE_INNER_SIZE_METERS * 0.5
    gate_collision_depth = GATE_COLLISION_DEPTH_METERS
    progress_gate_offset_m = 0.0
    drone_collision_radius = 0.12

    ground_collision_height = 0.05
    ground_collision_speed = 2.0
    ground_collision_requires_speed = True
    max_body_rate_deg_s = 1700.0

    # Position and velocity reset ranges come from the robot init_config.
    # The task aligns the spawn heading to the selected gate after the robot reset.
    reset_gate_index = 0
    reset_yaw_noise_deg = 5.0
    reset_roll_pitch_noise_deg = 0.0

    initial_idle_action = -1.0
    paper_drone_mass = 0.966
    motor_thrust_constant = paper_drone_mass * 1.55e-6
    motor_omega_min = 341.75
    motor_omega_max = 3100.0
    motor_command_curve_k = 0.5

    reward_parameters = {
        # Defaulted to the more robust M23-style reward settings from Table 1.
        "progress_weight": 1.0,
        "max_progress_speed": 20.0,
        "gate_reward": 1.5,
        "angular_rate_weight": 0.0005,
        "gate_offset_weight": 1.5,
        "outside_reposition_reward": 1.5,
        "perception_weight": 0.01,
        "perception_angle_threshold_rad": np.pi / 3.0,
        "motor_smoothness_weight": 0.0,
        "motor_smoothness_deadband": 0.0,
        "low_action_weight": 0.0,
        "crash_penalty": 7.0,
        "success_reward": 0.0,
    }

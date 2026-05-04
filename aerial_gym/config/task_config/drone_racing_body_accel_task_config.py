from aerial_gym.config.task_config.drone_racing_task_config import (
    task_config as drone_racing_task_config,
)
from aerial_gym.config.controller_config.lee_controller_config import control as lee_controller_config


class task_config(drone_racing_task_config):
    controller_name = "lee_attitude_control"
    attitude_max_inclination_rad = drone_racing_task_config.attitude_max_inclination_rad
    attitude_max_yaw_rate_rad_s = lee_controller_config.max_yaw_rate
    action_space_dim = 3

    world_accel_xy_max_m_s2 = 8.0
    world_accel_z_max_m_s2 = 19.62
    randomize_start_gate = True
    spawn_gate_distance_m = 3.0
    spawn_gate_distance_jitter_m = 0.75
    spawn_gate_lateral_jitter_m = 0.6
    spawn_gate_vertical_jitter_m = 0.4
    spawn_gate_yaw_jitter_deg = 15.0
    spawn_min_height_m = 0.25
    heading_velocity_threshold_m_s = 1.0
    heading_yaw_rate_gain = 2.0
    body_accel_thrust_command_min = -1.0
    body_accel_thrust_command_max = 1.0

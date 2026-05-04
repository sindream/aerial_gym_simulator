from aerial_gym.config.controller_config.lee_controller_config import control as lee_controller_config
from aerial_gym.config.task_config.drone_racing_task_config import (
    task_config as drone_racing_task_config,
)


class task_config(drone_racing_task_config):
    controller_name = "lee_attitude_control"
    attitude_max_inclination_rad = drone_racing_task_config.attitude_max_inclination_rad
    attitude_max_yaw_rate_rad_s = lee_controller_config.max_yaw_rate
    action_space_dim = 4

    world_accel_xy_max_m_s2 = 8.0
    world_accel_z_up_max_m_s2 = 12.0
    world_accel_z_down_gravity_scale = 1.0

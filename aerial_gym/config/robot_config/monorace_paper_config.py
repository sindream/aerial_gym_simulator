import numpy as np

from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg


class MonoRacePaperCfg(BaseQuadCfg):
    class init_config(BaseQuadCfg.init_config):
        min_init_state = [
            0.02,
            0.08,
            0.04,
            0.0,
            0.0,
            -np.pi / 12.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        max_init_state = [
            0.06,
            0.12,
            0.08,
            0.0,
            0.0,
            np.pi / 12.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    class sensor_config(BaseQuadCfg.sensor_config):
        enable_camera = False
        camera_config = None

        enable_lidar = False
        lidar_config = None

        enable_imu = False
        imu_config = None

    class robot_asset(BaseQuadCfg.robot_asset):
        name = "monorace_paper_quadrotor"

        min_state_ratio = [
            0.02,
            0.08,
            0.04,
            0.0,
            0.0,
            -np.pi / 12.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        max_state_ratio = [
            0.06,
            0.12,
            0.08,
            0.0,
            0.0,
            np.pi / 12.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    class control_allocator_config(BaseQuadCfg.control_allocator_config):
        class motor_model_config(BaseQuadCfg.control_allocator_config.motor_model_config):
            max_thrust = 14.39
            min_thrust = 0.175

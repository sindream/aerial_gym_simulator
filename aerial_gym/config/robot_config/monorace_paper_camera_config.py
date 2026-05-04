import numpy as np

from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.config.sensor_config.camera_config.monorace_camera_config import (
    MonoRaceCameraConfig,
)
from aerial_gym.config.sensor_config.imu_config.monorace_imu_config import (
    MonoRaceImuConfig,
)


class MonoRacePaperCameraCfg(BaseQuadCfg):
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
        enable_camera = True
        camera_config = MonoRaceCameraConfig

        enable_lidar = False
        lidar_config = None

        enable_imu = True
        imu_config = MonoRaceImuConfig

    class robot_asset(BaseQuadCfg.robot_asset):
        name = "monorace_paper_camera_quadrotor"

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

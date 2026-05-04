from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.config.sensor_config.camera_config.monorace_camera_config import (
    MonoRaceCameraConfig,
)
from aerial_gym.config.sensor_config.imu_config.monorace_imu_config import (
    MonoRaceImuConfig,
)


class BaseQuadRacingCfg(BaseQuadCfg):
    class sensor_config(BaseQuadCfg.sensor_config):
        enable_camera = True
        camera_config = MonoRaceCameraConfig

        enable_lidar = False
        lidar_config = None

        enable_imu = True
        imu_config = MonoRaceImuConfig

    class robot_asset(BaseQuadCfg.robot_asset):
        name = "base_quadrotor_racing"

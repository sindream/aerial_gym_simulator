from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.config.sensor_config.camera_config.aigp_segmentation_camera_config import (
    AIGPSegmentationCameraConfig,
)
from aerial_gym.config.sensor_config.imu_config.monorace_imu_config import (
    MonoRaceImuConfig,
)


NOMINAL_MASS_KG = 0.25
NOMINAL_HOVER_THROTTLE = 0.32
NOMINAL_MAX_THRUST_PER_MOTOR_N = NOMINAL_MASS_KG * 9.81 / (
    4.0 * NOMINAL_HOVER_THROTTLE
)


class BaseQuadAIGPRacingCfg(BaseQuadCfg):
    nominal_mass_kg = NOMINAL_MASS_KG
    nominal_hover_throttle = NOMINAL_HOVER_THROTTLE

    class sensor_config(BaseQuadCfg.sensor_config):
        enable_camera = True
        camera_config = AIGPSegmentationCameraConfig

        enable_lidar = False
        lidar_config = None

        enable_imu = True
        imu_config = MonoRaceImuConfig

    class robot_asset(BaseQuadCfg.robot_asset):
        name = "base_quadrotor_aigp_racing"

    class control_allocator_config(BaseQuadCfg.control_allocator_config):
        class motor_model_config(BaseQuadCfg.control_allocator_config.motor_model_config):
            max_thrust = NOMINAL_MAX_THRUST_PER_MOTOR_N
            min_thrust = 0.0

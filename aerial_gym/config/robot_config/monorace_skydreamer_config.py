import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY
from aerial_gym.config.sensor_config.camera_config.skydreamer_segmentation_camera_config import (
    SkyDreamerSegmentationCameraConfig,
)


class MonoRaceSkyDreamerCfg:
    class init_config:
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

    class sensor_config:
        enable_camera = True
        camera_config = SkyDreamerSegmentationCameraConfig

        enable_lidar = False
        lidar_config = None

        # The SkyDreamer paper does not use accelerometer measurements.
        enable_imu = False
        imu_config = None

    class disturbance:
        enable_disturbance = False
        prob_apply_disturbance = 0.0
        max_force_and_torque_disturbance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class damping:
        linvel_linear_damping_coefficient = [0.10, 0.10, 0.03]
        linvel_quadratic_damping_coefficient = [0.04, 0.04, 0.01]
        angular_linear_damping_coefficient = [0.02, 0.02, 0.02]
        angular_quadratic_damping_coefficient = [0.01, 0.01, 0.01]

    class robot_asset:
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/monorace"
        file = "monorace.urdf"
        name = "monorace_skydreamer_quadrotor"
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        collision_mask = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
        density = 0.000001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 200.0
        max_linear_velocity = 200.0
        armature = 0.001

        semantic_id = 0
        per_link_semantic = False

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

        color = None
        semantic_masked_links = {}
        keep_in_env = True

        min_position_ratio = None
        max_position_ratio = None

        min_euler_angles = [-np.pi, -np.pi, -np.pi]
        max_euler_angles = [np.pi, np.pi, np.pi]

        place_force_sensor = True
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        use_collision_mesh_instead_of_visual = False

    class control_allocator_config:
        num_motors = 4
        force_application_level = "motor_link"

        application_mask = [1, 2, 3, 4]
        motor_directions = [1, -1, 1, -1]

        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.095, -0.095, 0.095, 0.095],
            [-0.095, 0.095, 0.095, -0.095],
            [-0.05, 0.05, -0.05, 0.05],
        ]

        class motor_model_config:
            use_rps = True
            # Match the paper nominal values and allow training-time randomization.
            motor_thrust_constant_min = 1.55e-6 * 0.7
            motor_thrust_constant_max = 1.55e-6 * 1.3
            motor_time_constant_increasing_min = 0.03 * 0.7
            motor_time_constant_increasing_max = 0.03 * 1.3
            motor_time_constant_decreasing_min = 0.03 * 0.7
            motor_time_constant_decreasing_max = 0.03 * 1.3
            max_thrust = 30.0
            min_thrust = 0.05
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.05
            use_discrete_approximation = True

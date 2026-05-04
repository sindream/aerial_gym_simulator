import numpy as np


class control:
    """
    Racing-tuned Lee controller profile for the monorace quadrotor.

    Compared to the default lee_controller_config, this profile:
    - keeps position commands a bit softer in XY
    - increases attitude and yaw damping substantially
    - disables parameter randomization for deterministic diagnostics
    """

    num_actions = 4
    max_inclination_angle_rad = 0.45
    max_yaw_rate = np.pi

    action_limit_min = [-1.0, -0.45, -0.45, -np.pi]
    action_limit_max = [1.0, 0.45, 0.45, np.pi]

    K_pos_tensor_max = [0.0, 0.0, 0.0]
    K_pos_tensor_min = [0.0, 0.0, 0.0]

    K_vel_tensor_max = [0.0, 0.0, 0.0]
    K_vel_tensor_min = [0.0, 0.0, 0.0]

    K_rot_tensor_max = [8.0, 8.0, 4.0]
    K_rot_tensor_min = [8.0, 8.0, 4.0]

    K_angvel_tensor_max = [2.0, 2.0, 1.0]
    K_angvel_tensor_min = [2.0, 2.0, 1.0]

    randomize_params = False

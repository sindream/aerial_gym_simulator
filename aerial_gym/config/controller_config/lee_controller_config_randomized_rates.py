import numpy as np


class control:
    """
    Lee attitude controller with randomized angular-rate gains.

    The nominal controller is intentionally kept untouched. This config is used
    only by the NED-derived privileged racing task so the policy can learn under
    per-environment rate-gain variation.
    """

    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi

    K_pos_tensor_max = [3.0, 3.0, 2.0]
    K_pos_tensor_min = [2.0, 2.0, 1.0]

    K_vel_tensor_max = [3.0, 3.0, 3.0]
    K_vel_tensor_min = [2.0, 2.0, 2.0]

    K_rot_tensor_max = [1.2, 1.2, 0.6]
    K_rot_tensor_min = [0.8, 0.8, 0.4]

    K_angvel_tensor_max = [0.28, 0.28, 0.28]
    K_angvel_tensor_min = [0.07, 0.07, 0.07]

    randomize_params = True

import numpy as np


class control:
    num_actions = 4
    max_inclination_angle_rad = 0.35
    max_yaw_rate = np.pi / 2.0

    action_limit_min = [-4.0, -4.0, -2.0, -np.pi / 2.0]
    action_limit_max = [4.0, 4.0, 2.0, np.pi / 2.0]

    K_pos_tensor_max = [0.0, 0.0, 0.0]
    K_pos_tensor_min = [0.0, 0.0, 0.0]

    K_vel_tensor_max = [1.6, 1.6, 2.4]
    K_vel_tensor_min = [1.6, 1.6, 2.4]

    K_rot_tensor_max = [4.0, 4.0, 1.8]
    K_rot_tensor_min = [4.0, 4.0, 1.8]

    K_angvel_tensor_max = [0.85, 0.85, 0.45]
    K_angvel_tensor_min = [0.85, 0.85, 0.45]

    randomize_params = False

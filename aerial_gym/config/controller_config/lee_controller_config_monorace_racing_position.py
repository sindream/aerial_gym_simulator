import numpy as np


class control:
    num_actions = 4
    max_inclination_angle_rad = 0.20
    max_yaw_rate = np.pi / 2.0
    gravity_compensation_scale = 0.70

    action_limit_min = [-3.0, -3.0, 0.2, -np.pi]
    action_limit_max = [3.0, 3.0, 2.5, np.pi]

    K_pos_tensor_max = [0.25, 0.25, 1.2]
    K_pos_tensor_min = [0.25, 0.25, 1.2]

    K_vel_tensor_max = [1.6, 1.6, 2.8]
    K_vel_tensor_min = [1.6, 1.6, 2.8]

    K_rot_tensor_max = [8.0, 8.0, 4.0]
    K_rot_tensor_min = [8.0, 8.0, 4.0]

    K_angvel_tensor_max = [2.0, 2.0, 1.0]
    K_angvel_tensor_min = [2.0, 2.0, 1.0]

    randomize_params = False

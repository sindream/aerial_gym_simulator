import numpy as np


class control:
    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi / 2.0
    action_limit_min = [-200.0, -200.0, -20.0, -2.0 * np.pi]
    action_limit_max = [200.0, 200.0, 20.0, 2.0 * np.pi]

    # Collection uses an easier, more tightly tracking position loop than training.
    K_pos_tensor_max = [3.0, 3.0, 8.0]
    K_pos_tensor_min = [3.0, 3.0, 8.0]

    K_vel_tensor_max = [2.5, 2.5, 5.0]
    K_vel_tensor_min = [2.5, 2.5, 5.0]

    K_rot_tensor_max = [2.2, 2.2, 1.0]
    K_rot_tensor_min = [2.2, 2.2, 1.0]

    K_angvel_tensor_max = [0.35, 0.35, 0.25]
    K_angvel_tensor_min = [0.35, 0.35, 0.25]

    randomize_params = False

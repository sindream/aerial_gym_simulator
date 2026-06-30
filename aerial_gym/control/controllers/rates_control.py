import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *


from aerial_gym.control.controllers.base_lee_controller import *


class LeeRatesController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Lee body-rate controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired z acceleration
        and body-rate command [az, p, q, r].
        :return: m*g normalized thrust and interial normalized torques
        """
        self.reset_commands()
        gravity_z = self.gravity[:, 2]
        self.wrench_command[:, 2] = (
            command_actions[:, 0] - gravity_z
        ) * self.mass.squeeze(1)
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.robot_orientation, command_actions[:, 1:4]
        )

        return self.wrench_command

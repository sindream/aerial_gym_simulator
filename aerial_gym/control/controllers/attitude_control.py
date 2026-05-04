import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *


from aerial_gym.control.controllers.base_lee_controller import *


class LeeAttitudeController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)
        self.desired_yaw = None

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)
        self.desired_yaw = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32, requires_grad=False
        )
        self.desired_yaw[:] = self.robot_euler_angles[:, 2]

    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super().reset_idx(env_ids)
        self.desired_yaw[env_ids] = self.robot_euler_angles[env_ids, 2]

    def update(self, command_actions):
        """
        Lee attitude controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired thrust, roll, pitch and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        self.reset_commands()
        self.wrench_command[:, 2] = (
            (command_actions[:, 0] + 1.0) * self.mass.squeeze(1) * torch.norm(self.gravity, dim=1)
        )

        self.euler_angle_rates[:, :2] = 0.0
        self.euler_angle_rates[:, 2] = command_actions[:, 3]
        self.desired_body_angvel[:] = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor
        )

        self.desired_yaw[:] = ssa(self.desired_yaw + command_actions[:, 3] * self.dt)

        # quaternion desired
        # desired euler angle is equal to commanded roll, commanded pitch, and integrated yaw
        quat_desired = quat_from_euler_xyz(
            command_actions[:, 1], command_actions[:, 2], self.desired_yaw
        )
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            quat_desired, self.desired_body_angvel
        )

        return self.wrench_command

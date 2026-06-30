from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
    PositionSetpointTask,
)

from aerial_gym.task.position_setpoint_task_sim2real.position_setpoint_task_sim2real import (
    PositionSetpointTaskSim2Real,
)

from aerial_gym.task.position_setpoint_task_sim2real_end_to_end.position_setpoint_task_sim2real_end_to_end import (
    PositionSetpointTaskSim2RealEndToEnd,
)

from aerial_gym.task.position_setpoint_task_sim2real_px4.position_setpoint_task_sim2real_px4 import (
    PositionSetpointTaskSim2RealPX4,
)

from aerial_gym.task.position_setpoint_task_acceleration_sim2real.position_setpoint_task_acceleration_sim2real import (
    PositionSetpointTaskAccelerationSim2Real,
)

from aerial_gym.task.navigation_task.navigation_task import NavigationTask
from aerial_gym.task.drone_racing_task.drone_racing_task import DroneRacingTask
from aerial_gym.task.drone_racing_body_accel_task.drone_racing_body_accel_task import (
    DroneRacingBodyAccelTask,
)
from aerial_gym.task.drone_racing_accel_yawrate_task.drone_racing_accel_yawrate_task import (
    DroneRacingAccelYawrateTask,
)
from aerial_gym.task.drone_racing_paper_task.drone_racing_paper_task import (
    DroneRacingPaperTask,
)
from aerial_gym.task.drone_racing_skydreamer_task.drone_racing_skydreamer_task import (
    DroneRacingSkyDreamerTask,
)
from aerial_gym.task.drone_racing_ned_privileged_task.drone_racing_ned_privileged_task import (
    DroneRacingNedPrivilegedTask,
)
from aerial_gym.task.quintic_tracking_sysid_task.quintic_tracking_sysid_task import (
    QuinticTrackingSysIDTask,
)
from aerial_gym.task.aigp_racing_segmentation_task.aigp_racing_segmentation_task import (
    AIGPRacingSegmentationTask,
)

from aerial_gym.config.task_config.position_setpoint_task_config import (
    task_config as position_setpoint_task_config,
)

from aerial_gym.config.task_config.position_setpoint_task_sim2real_config import (
    task_config as position_setpoint_task_sim2real_config,
)

from aerial_gym.config.task_config.position_setpoint_task_sim2real_end_to_end_config import (
    task_config as position_setpoint_task_sim2real_end_to_end_config,
)

from aerial_gym.config.task_config.position_setpoint_task_sim2real_px4_config import (
    task_config as position_setpoint_task_sim2real_px4_config,
)

from aerial_gym.config.task_config.position_setpoint_task_acceleration_sim2real_config import (
    task_config as position_setpoint_task_acceleration_sim2real_config,
)

from aerial_gym.config.task_config.navigation_task_config import (
    task_config as navigation_task_config,
)
from aerial_gym.config.task_config.drone_racing_task_config import (
    task_config as drone_racing_task_config,
)
from aerial_gym.config.task_config.drone_racing_body_accel_task_config import (
    task_config as drone_racing_body_accel_task_config,
)
from aerial_gym.config.task_config.drone_racing_accel_yawrate_task_config import (
    task_config as drone_racing_accel_yawrate_task_config,
)
from aerial_gym.config.task_config.drone_racing_paper_task_config import (
    task_config as drone_racing_paper_task_config,
)
from aerial_gym.config.task_config.drone_racing_skydreamer_task_config import (
    task_config as drone_racing_skydreamer_task_config,
)
from aerial_gym.config.task_config.drone_racing_ned_privileged_task_config import (
    task_config as drone_racing_ned_privileged_task_config,
)
from aerial_gym.config.task_config.quintic_tracking_sysid_task_config import (
    task_config as quintic_tracking_sysid_task_config,
)
from aerial_gym.config.task_config.aigp_racing_segmentation_task_config import (
    task_config as aigp_racing_segmentation_task_config,
)

from aerial_gym.registry.task_registry import task_registry


task_registry.register_task(
    "position_setpoint_task", PositionSetpointTask, position_setpoint_task_config
)
task_registry.register_task(
    "position_setpoint_task_sim2real",
    PositionSetpointTaskSim2Real,
    position_setpoint_task_sim2real_config,
)

task_registry.register_task(
    "position_setpoint_task_sim2real_end_to_end",
    PositionSetpointTaskSim2RealEndToEnd,
    position_setpoint_task_sim2real_end_to_end_config,
)

task_registry.register_task(
    "position_setpoint_task_sim2real_px4",
    PositionSetpointTaskSim2RealPX4,
    position_setpoint_task_sim2real_px4_config,
)

task_registry.register_task(
    "position_setpoint_task_acceleration_sim2real",
    PositionSetpointTaskAccelerationSim2Real,
    position_setpoint_task_acceleration_sim2real_config,
)

task_registry.register_task("navigation_task", NavigationTask, navigation_task_config)
task_registry.register_task("drone_racing_task", DroneRacingTask, drone_racing_task_config)
task_registry.register_task(
    "drone_racing_body_accel_task",
    DroneRacingBodyAccelTask,
    drone_racing_body_accel_task_config,
)
task_registry.register_task(
    "drone_racing_accel_yawrate_task",
    DroneRacingAccelYawrateTask,
    drone_racing_accel_yawrate_task_config,
)
task_registry.register_task(
    "drone_racing_paper_task", DroneRacingPaperTask, drone_racing_paper_task_config
)
task_registry.register_task(
    "drone_racing_skydreamer_task",
    DroneRacingSkyDreamerTask,
    drone_racing_skydreamer_task_config,
)
task_registry.register_task(
    "drone_racing_ned_privileged_task",
    DroneRacingNedPrivilegedTask,
    drone_racing_ned_privileged_task_config,
)
task_registry.register_task(
    "quintic_tracking_sysid_task",
    QuinticTrackingSysIDTask,
    quintic_tracking_sysid_task_config,
)
task_registry.register_task(
    "aigp_racing_segmentation_task",
    AIGPRacingSegmentationTask,
    aigp_racing_segmentation_task_config,
)


from aerial_gym.task.position_setpoint_task_reconfigurable.position_setpoint_task_reconfigurable import (
    PositionSetpointTaskReconfigurable,
)

from aerial_gym.config.task_config.position_setpoint_task_config_reconfigurable import (
    task_config as position_setpoint_task_config_reconfigurable,
)

from aerial_gym.task.position_setpoint_task_morphy.position_setpoint_task_morphy import (
    PositionSetpointTaskMorphy,
)

from aerial_gym.config.task_config.position_setpoint_task_morphy_config import (
    task_config as position_setpoint_task_config_morphy,
)


task_registry.register_task(
    "position_setpoint_task_reconfigurable",
    PositionSetpointTaskReconfigurable,
    position_setpoint_task_config_reconfigurable,
)

task_registry.register_task(
    "position_setpoint_task_morphy",
    PositionSetpointTaskMorphy,
    position_setpoint_task_config_morphy,
)


## Uncomment this to use custom tasks

# from aerial_gym.task.custom_task.custom_task import CustomTask
# task_registry.register_task("custom_task", CustomTask, custom_task.task_config)

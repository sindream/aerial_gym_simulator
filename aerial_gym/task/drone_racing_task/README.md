# Drone Racing Optical Flow Task

This document summarizes the current optical-flow-based drone racing task setup used by `drone_racing_task`.

## Entry Points

- Task implementation:
  - `aerial_gym/task/drone_racing_task/drone_racing_task.py`
- Task config:
  - `aerial_gym/config/task_config/drone_racing_task_config.py`
- Environment config:
  - `aerial_gym/config/env_config/drone_racing_env.py`
- RL-Games training config:
  - `aerial_gym/rl_training/rl_games/ppo_drone_racing_multimodal.yaml`
- RL-Games runner:
  - `aerial_gym/rl_training/rl_games/runner_multimodal.py`

## Current Task Summary

- Task name registered in the registry: `drone_racing_task`
- Controller: `lee_attitude_control`
- Robot: `monorace_paper_camera_quadrotor`
- Action space: 4D attitude command
  - channel 0: thrust command in `[-1, 1]`
  - channel 1: roll target scaled by `attitude_max_inclination_rad`
  - channel 2: pitch target scaled by `attitude_max_inclination_rad`
  - channel 3: yaw-rate target scaled by `attitude_max_yaw_rate_rad_s`
- Episode length: `2500` steps
- Goal logic:
  - pass all active gates in order
  - after the last gate, activate a virtual goal point `5.0 m` ahead of the last gate
  - success when the robot reaches the goal radius `1.0 m`

## Observation Structure

The policy receives a multimodal observation:

- `state`: 16D vector
- `img_observation`: 4 x 12 x 16 optical flow tensor

### State Vector

Current `state` layout:

- `0:3`: first target relative position in body frame
- `3:6`: second target relative position in body frame
- `6:9`: body linear velocity
- `9:13`: robot orientation quaternion
- `13:16`: body angular velocity

The first target is the current gate or final goal.
The second target is the next gate or final goal.

## Optical Flow Input

### Camera Setup

Defined in `aerial_gym/config/sensor_config/camera_config/monorace_camera_config.py`:

- render resolution: `48 x 64`
- horizontal FOV: `150 deg`
- camera position relative to robot: `[0.12, 0.0, 0.02]`
- depth range:
  - min: `0.05 m`
  - max: `40.0 m`

### Optical Flow Generation

Optical flow is computed analytically in `drone_racing_task.py` using:

- depth image
- body linear velocity
- body angular velocity
- pinhole camera intrinsics

The flow is then normalized before entering the network:

- divide by `optical_flow_clip_pixels_per_step`
- clamp to `[-1, 1]`

Current setting:

- `optical_flow_clip_pixels_per_step = 8.0`

Interpretation:

- `8 px/step -> 1.0`
- `-8 px/step -> -1.0`
- larger magnitude values saturate at `-1.0` or `1.0`

### Dual-Resolution Optical Flow

The network input uses two optical flow views:

- full flow resized from `48 x 64` to `12 x 16`
- center crop resized to `12 x 16`

These are concatenated into a 4-channel tensor:

- channel 0: full flow `u`
- channel 1: full flow `v`
- channel 2: center flow `u`
- channel 3: center flow `v`

Current center crop ratio:

- height ratio: `0.5`
- width ratio: `0.5`

That means the center crop is currently:

- `24 x 32` from the original `48 x 64`
- then resized to `12 x 16`

### Debug Visualization

If enabled in the task config, env 0 displays optical flow with OpenCV.

Current flags:

- `show_env0_optical_flow = True`
- `show_env0_optical_flow_index = 0`
- `show_env0_optical_flow_scale = 8`
- `show_env0_optical_flow_wait_ms = 1`

## Network Structure

Defined in `aerial_gym/rl_training/rl_games/ppo_drone_racing_multimodal.yaml`.

### Image Branch

Current image CNN:

1. `Conv2d(4 -> 16, k=5, s=2, p=2)`
2. `Conv2d(16 -> 32, k=5, s=2, p=2)`
3. `Conv2d(32 -> 64, k=3, s=2, p=1)`
4. `Conv2d(64 -> 64, k=3, s=2, p=1)`
5. `Conv2d(64 -> 64, k=3, s=2, p=1)`

For input `4 x 12 x 16`, the spatial sizes become:

- `12 x 16 -> 6 x 8 -> 3 x 4 -> 2 x 2 -> 1 x 1 -> 1 x 1`

After the CNN:

- flatten
- `image_mlp = [256]`

### State Branch

- `state_mlp = [64, 64]`

### Fusion and Recurrent Core

- concatenate image feature and state feature
- `GRU(units=128, layers=1)`
- GRU is applied before the actor/critic MLPs

### Actor/Critic Heads

- actor MLP: `[192, 96]`
- critic MLP: `[192, 96]`

### Input Normalization

- image input: no running-mean/std normalization in RL-Games
- vector state: RL-Games running-mean/std normalization enabled because `normalize_input: True`

So the image branch uses the task-side fixed flow normalization only.

## Reward and Reset Logic

### Current Reward Terms

Defined in `drone_racing_task_config.py`:

- progress reward
- target yaw alignment reward
- command magnitude penalty
- command delta penalty
- obstacle avoidance penalty from nearest depth distance
- gate pass reward
- soft upright penalty
- crash penalty

Current key values:

- `lambda_1_progress = 1.2`
- `lambda_2_theta = 0.05`
- `lambda_3_cmd_norm = -0.001`
- `lambda_4_cmd_delta = -0.0005`
- `lambda_5_speed = 0.0`
- `lambda_6_avoid = -0.01`
- `lambda_7_pass = 12.0`
- `lambda_8_crash = -4.0`
- `lambda_9_upright = -0.01`

### Current Reset Conditions

Current episode termination logic is intentionally simplified:

- failure reset if either:
  - environment contact collision is reported by `obs_dict["crashes"]`
  - the robot crosses the gate plane outside the valid gate opening
- success reset if:
  - final goal after the last gate is reached
- timeout reset if:
  - `sim_steps >= episode_len_steps`

The following are still computed for debugging but are not currently used as reset conditions:

- `gate_collision`
- `ground_collision`
- `out_of_bounds`
- `excessive_body_rate`

### Reset Reason Debug Print

When enabled, env 0 prints reset reasons in the console:

- `show_env0_reset_reason = True`

Useful fields in the log:

- `contact`
- `wrong_gate_cross`
- `gate_collision`
- `ground`
- `oob`
- `body_rate`

## Obstacles and Track Editing

### Main Track and Obstacle Config

Edit this file:

- `aerial_gym/config/asset_config/racing_track_asset_config.py`

Important variables:

- `RACING_TRACK_GATES`
  - gate positions, yaw angles, semantic ids
- `RANDOM_CYLINDER_OBSTACLE_COUNT`
  - current value: `10`
- `RANDOM_CYLINDER_RADIUS_METERS`
  - current value: `0.40`
- `RANDOM_CYLINDER_HEIGHT_METERS`
  - current value: `4.5`
- `TRACK_BOUNDS_MIN`
- `TRACK_BOUNDS_MAX`
- `START_POSITION`
- `START_YAW_DEG`

### Current Active Gates

Right now only 4 gates are active in `RACING_TRACK_GATES`.
The later gates are commented out.

### Gate Asset

Gate geometry and collision asset:

- `resources/models/environment_assets/racing/gate_1p5m.urdf`

Current relevant sizes:

- gate inner size: `1.5 m`
- gate outer size: `3.0 m`
- gate collision depth: `0.08 m`

### Cylinder Obstacle Asset

Obstacle geometry:

- `resources/models/environment_assets/racing/cylinder_obstacle.urdf`

Placement ranges are controlled by:

- `RaceCylinderObstacleAssetParams` in `racing_track_asset_config.py`

### Gate Exclusion Rule for Cylinders

Random cylinders are additionally moved away from gates inside:

- `DroneRacingTask._resample_cylinders_away_from_gates()`

The exclusion radius is controlled by:

- `cylinder_gate_exclusion_radius_m`

## Robot Collision Setup

Robot URDF:

- `resources/robots/monorace/monorace.urdf`

Current collision behavior:

- base body collision is enabled
- propeller collision geometry has been removed
- propellers remain visual-only

This means environment contact collisions should now be driven by the body collision mesh instead of propeller collision cylinders.

## Training Command

Main training command:

```bash
python aerial_gym/rl_training/rl_games/runner_multimodal.py \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_drone_racing_multimodal.yaml \
  --headless false \
  --use_warp false \
  --num_envs 32
```

Notes:

- `--headless false` shows the viewer if graphics are available
- for this task, using `--use_warp false` is recommended when visually testing behavior because otherwise ground contact can be missed
- `--num_envs` overrides the YAML env count
- `runner_multimodal.py` will create `runs/` and `nn/` directories automatically

For headless training:

```bash
python aerial_gym/rl_training/rl_games/runner_multimodal.py \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_drone_racing_multimodal.yaml \
  --headless true \
  --use_warp false \
  --num_envs 32
```

## Play / Evaluation Command

Example play command:

```bash
python aerial_gym/rl_training/rl_games/runner_multimodal.py \
  --play \
  --file aerial_gym/rl_training/rl_games/ppo_drone_racing_multimodal.yaml \
  --checkpoint runs/<run_name>/nn/<checkpoint_name>.pth \
  --headless false \
  --use_warp false \
  --num_envs 1
```

If `--use_warp true` is used during testing, ground contact may not be detected as expected in this task setup.

## What To Edit For Common Changes

### Change gates

- edit `RACING_TRACK_GATES` in `aerial_gym/config/asset_config/racing_track_asset_config.py`

### Change random obstacle count or size

- edit `RANDOM_CYLINDER_OBSTACLE_COUNT`
- edit `RANDOM_CYLINDER_RADIUS_METERS`
- edit `RANDOM_CYLINDER_HEIGHT_METERS`

### Change optical flow scaling

- edit `optical_flow_clip_pixels_per_step` in `aerial_gym/config/task_config/drone_racing_task_config.py`

### Change camera resolution or FOV

- edit `aerial_gym/config/sensor_config/camera_config/monorace_camera_config.py`

### Change crop ratio

- edit the task config values used by:
  - `central_flow_crop_height_ratio`
  - `central_flow_crop_width_ratio`

If not explicitly added to the config, the task currently defaults both to `0.5`.

### Change reward weights

- edit `reward_parameters` in `aerial_gym/config/task_config/drone_racing_task_config.py`

### Change reset behavior

- edit `_compute_reward_terms()` in `aerial_gym/task/drone_racing_task/drone_racing_task.py`

## Commit Reminder

Training artifacts should not be committed.
Typical generated outputs to exclude:

- `runs/`
- `wandb/`
- `rollout_viz/`
- checkpoints like `*.pth`

# Drone Racing Task Notes

This document summarizes the current optical-flow racing setup around `drone_racing_task` and its accel-command variants.

## Entry Points

- Task:
  - `aerial_gym/task/drone_racing_task/drone_racing_task.py`
- Accel + yaw-rate variant:
  - `aerial_gym/task/drone_racing_accel_yawrate_task/drone_racing_accel_yawrate_task.py`
- Task configs:
  - `aerial_gym/config/task_config/drone_racing_task_config.py`
  - `aerial_gym/config/task_config/drone_racing_accel_yawrate_task_config.py`
  - `aerial_gym/config/task_config/drone_racing_body_accel_task_config.py`
- Environment config:
  - `aerial_gym/config/env_config/drone_racing_env.py`
- Track / gate layout:
  - `aerial_gym/config/asset_config/racing_track_asset_config.py`
  - `resources/models/environment_assets/racing/gate_1p5m.urdf`
- Training entry:
  - `aerial_gym/rl_training/rl_games/runner_multimodal.py`

## RL-Games Setup

This task uses repo-local RL-Games multimodal extensions instead of patching the installed package:

- `aerial_gym/rl_training/rl_games/aerial_multimodal_models.py`
- `aerial_gym/rl_training/rl_games/aerial_multimodal_network_builder.py`

Use `runner_multimodal.py` for the racing tasks.

## Current Robot / Controller

Current default racing task uses:

- robot: `base_quadrotor_racing`
- controller: `lee_attitude_control`
- camera config: `MonoRaceCameraConfig`

`base_quadrotor_racing` keeps the base quad dynamics path and swaps in the racing camera / IMU stack.

## Track

Current track is a 12-gate loop.

- gate centers and yaw come from:
  - `aerial_gym/config/asset_config/racing_track_asset_config.py`
- gate geometry is matched to the task logic:
  - inner opening: `4.0 m`
  - outer size: `4.64 m`
  - collision depth: `0.45 m`

Loop behavior:

- after the last gate, target wraps to gate 1 again
- observation also wraps correctly:
  - current target -> gate 1
  - second target -> gate 2

Gate layout visualization:

- script:
  - `aerial_gym/task/drone_racing_task/visualize_gate_layout.py`
- generated view:
  - `aerial_gym/task/drone_racing_task/gate_layout_current.svg`

## Reset / Spawn

Current reset behavior:

- random start gate is enabled
- robot spawns in front of the selected gate with jitter

Important values:

- `spawn_gate_distance_m = 5.0`
- `spawn_gate_distance_jitter_m = 1.0`
- `spawn_gate_lateral_jitter_m = 0.6`
- `spawn_gate_vertical_jitter_m = 0.4`
- `spawn_gate_yaw_jitter_deg = 15.0`

## Observation

Policy observation is multimodal:

- `state`: 16D vector
- `img_observation`: `4 x 12 x 16` optical flow tensor

### State Layout

For the default `drone_racing_task`:

- `0:3`: current target relative position in `vehicle frame`
- `3:6`: second target relative position in `vehicle frame`
- `6:9`: body linear velocity
- `9:13`: robot orientation quaternion
- `13:16`: body angular velocity

For `drone_racing_accel_yawrate_task`:

- `0:3`: current target relative position in `vehicle frame`
- `3:6`: second target relative position in `vehicle frame`
- `6:9`: vehicle linear velocity
- `9:13`: robot orientation quaternion
- `13:16`: body angular velocity

Notes:

- `vehicle frame` here means yaw-only rotated frame
- target relative positions are computed as:
  - world target delta
  - rotated by `robot_vehicle_orientation`

### Input Normalization

- vector state:
  - RL-Games running mean/std normalization is enabled via `normalize_input: True`
- optical flow image:
  - no running mean/std normalization
  - task-side fixed normalization only
  - divide by `optical_flow_clip_pixels_per_step`
  - clamp to `[-1, 1]`

Current optical-flow normalization:

- `optical_flow_clip_pixels_per_step = 8.0`

## Camera / Optical Flow

Current camera setup comes from `MonoRaceCameraConfig`:

- render resolution: `48 x 64`
- policy resolution after resize: `12 x 16`
- horizontal FOV: `150 deg`
- depth range: `0.05 m ~ 40.0 m`

Network input is dual optical flow:

- channels `0:2`: full-frame flow `(u, v)`
- channels `2:4`: center crop flow `(u, v)`

Center crop ratios:

- height ratio: `0.5`
- width ratio: `0.5`

## Action Spaces

### 1. Default Attitude Task

Task:

- `drone_racing_task`

Policy action:

- 4D
- `[thrust, roll, pitch, yaw_rate]`

Mapping:

- thrust channel range in task space: `[-1, 2]`
- controller mapping:
  - `thrust = (u + 1) * m g`
- so effective commanded thrust range is:
  - `0 ~ 3 m g`

Roll / pitch limits:

- `attitude_max_inclination_rad = 40 deg`

### 2. Accel + Yaw-Rate Task

Task:

- `drone_racing_accel_yawrate_task`

Policy action:

- 4D
- `[a_x, a_y, a_z, yaw_rate]`

Interpretation:

- accel command is built in `vehicle frame`
- then rotated to world with `robot_vehicle_orientation`
- gravity is applied in world
- final command is converted to:
  - thrust
  - roll
  - pitch
  - yaw_rate
  and passed into the same `lee_attitude_control`

Vertical accel handling:

- downward accel lower bound is limited by `-g * world_accel_z_down_gravity_scale`
- upward accel is limited by `world_accel_z_up_max_m_s2`
- final thrust command is clamped to `[-1, 2]`

## Reward

Current reward terms live in:

- `aerial_gym/config/task_config/drone_racing_task_config.py`

Main terms:

- progress reward
- yaw-to-target reward
- command magnitude penalty
- command delta penalty
- gate pass reward
- crash penalty
- upright penalty
- altitude-to-target reward

Current notable values:

- `lambda_1_progress = 5.0`
- `lambda_2_theta = 0.20`
- `lambda_7_pass = 30.0`
- `lambda_8_crash = -4.0`
- `lambda_9_upright = -0.02`
- `lambda_10_altitude = 0.25`

Current yaw-alignment shaping is distance-relaxed:

- for distances beyond `3 m`, full yaw reward weight is used
- near the gate, yaw reward fades down to `25%`

This is controlled by:

- `theta_relax_distance_m = 3.0`
- `theta_near_gate_min_scale = 0.25`

## Reset Logic

Current failure reset logic is intentionally simple:

- simulator contact collision
- wrong gate crossing

Notes:

- `gate_collision`, `ground_collision`, `out_of_bounds`, and `excessive_body_rate` are still computed and logged
- but they are not currently added into the crash boolean

## Physics Notes

Current env config uses:

- `num_physics_steps_per_env_step_mean = 2`

Important:

- this does **not** change `sim.dt` itself
- it does change how many physics steps happen per env / policy step
- so it effectively changes the control update period seen by the policy

If you want to change true simulator timestep resolution instead, adjust:

- `aerial_gym/config/sim_config/base_sim_config.py`

## Training Commands

Default attitude task:

```bash
python aerial_gym/rl_training/rl_games/runner_multimodal.py \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_drone_racing_multimodal.yaml \
  --num_envs 32 \
  --headless true \
  --use_warp false
```

Accel + yaw-rate task:

```bash
python aerial_gym/rl_training/rl_games/runner_multimodal.py \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_drone_racing_multimodal_accel_yawrate.yaml \
  --num_envs 32 \
  --headless true \
  --use_warp false
```

## Commit Notes

Generated local debug artifacts that should not be committed:

- `yaw_controller_test.png`
- `yaw_position_test.png`
- `aerial_gym/task/drone_racing_task/gate_layout_current.svg`

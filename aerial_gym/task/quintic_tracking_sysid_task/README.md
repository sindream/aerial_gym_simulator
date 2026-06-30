# Quintic Tracking SysID Task

This task is separate from the drone-racing tasks. It trains a body-rate policy to
track a reset-sampled single-segment quintic reference trajectory while estimating
latent system parameters.

## Environment

- Task name: `quintic_tracking_sysid_task`
- Sim config: `base_sim_1ms`
- Env config: `quintic_tracking_env`
- Robot: `base_quadrotor_hover32`
- Physics step: `0.001 s`
- Policy step: `0.020 s`
- Action hold: `20` physics steps per policy action
- Ground plane: disabled
- Lower z bound: disabled for resets
- Tilt constraint penalty threshold: `80 deg`
- Controller: `lee_rates_control_randomized_rates`

## Action

The policy outputs a 4D action:

```text
[throttle, p_des, q_des, r_des]
```

- `throttle` is clipped to `[0.05, 0.90]`. The nominal robot motor max thrust is
  chosen so the nominal 0.25 kg quad hovers at `0.26` throttle.
- Per-reset mass randomization is preserved, so each env's hover throttle is
  `0.26 * mass_scale`, clipped to `[0.195, 0.351]`.
- The throttle is converted to the Lee rates controller acceleration input with
  the env-specific hover throttle as `az = throttle / hover_throttle_env * g - g`,
  so `throttle=hover_throttle_env` maps to hover. This also makes the maximum
  achievable commanded acceleration env-dependent, matching fixed-range throttle
  behavior on real vehicles.
- `p_des`, `q_des`, `r_des` are normalized body-rate commands. By default this
  task uses the AIGP/MAVLink sysid effective axis scale
  `[-2.4, 2.4, 2.2] rad/s`, so the learned action sees the same signed
  body-rate response as the deployment bridge instead of an ideal symmetric
  Lee-rate setpoint.

## Observation

The reference path is sampled in the drone vehicle frame and rotated into world
coordinates with the current vehicle yaw. Each segment is a true quintic
polynomial with configurable boundary limits:

- endpoint displacement: `trajectory_position_constraint_m`
- endpoint velocity: `trajectory_velocity_constraint_m_s`
- endpoint acceleration: `trajectory_acceleration_constraint_m_s2`
- sampled profile speed cap: `trajectory_max_speed_m_s`

When a segment finishes, the task does not reset the episode. It samples the
next quintic segment from the current drone position and velocity, then keeps the
same episode running until timeout/crash.

The actor observation intentionally does not concatenate true system parameters.
The 24D state contains:

- vehicle-frame position error
- vehicle-frame velocity error
- vehicle-frame reference velocity
- vehicle-frame reference acceleration
- heading error as `sin/cos`
- vehicle-frame body-up projection vector
- body angular velocity
- previous raw action

The returned actor observation can be delayed independently per environment.
By default `obs_delay_s` is uniformly randomized from `0.00` to `0.04` seconds
in 50 Hz policy-step increments.

The `sysid_target` key is a separate 4D supervised label. These are effective
closed-loop parameters that are more directly identifiable from policy history
than separated physical parameters under a low-level body-rate controller:

```text
[Kp / Ixx, Kq / Iyy, Kr / Izz, hover_throttle]
```

All four labels are normalized to `[0, 1]` using the configured per-reset
randomization ranges before the auxiliary MSE loss is applied.

The custom rl_games network uses this key only for auxiliary MSE loss. The
predicted SysID latent is concatenated into the downstream actor/critic MLPs.
Inference still works if the label key is absent because the target dimension is
declared in the yaml.

## Domain Randomization

On reset, each environment samples:

- mass scale
- diagonal inertia scale
- hover throttle
- angular-rate gain through the randomized rates controller
- observation delay
- action delay

The task always synchronizes randomized labels and the global mass/inertia tensors
used by the controller. Runtime PhysX rigid-body property writes are enabled by
default with `randomize_mass_inertia_physics=True`, so mass/inertia labels also
affect the simulator dynamics. If the PhysX property write fails, training stops
instead of silently continuing with physically meaningless SysID labels.

Action delay is also randomized from `0.00` to `0.04` seconds by default. The
policy action is stored in a delay buffer, and the delayed raw action is what gets
converted to `[az, p, q, r]` for the Lee rates controller.

## Reward

The reward contains:

- setpoint-style dense position tracking rewards: `exp(-||p_ref - p||^2)`
- velocity tracking reward: `exp(-||v_ref - v||^2)`
- progress reward: previous position error minus current position error
- velocity-direction reward toward the current reference point
- heading reward that points yaw along the reference trajectory's direction of travel
- upright and low-body-rate rewards gated by position tracking quality
- body-rate command, thrust command, and action-smoothness penalties
- constraint violation penalty
- crash/instability replacement penalty
- success bonus

Jerk is not used in either observation or reward.

## Commands

Train:

```bash
python aerial_gym/rl_training/rl_games/train_quintic_tracking_sysid.py \
  --num_envs 1024 \
  --seed 10
```

Train with viewer:

```bash
python aerial_gym/rl_training/rl_games/train_quintic_tracking_sysid.py \
  --num_envs 16 \
  --gui
```

Train with live env0 reference/actual trajectory visualization:

```bash
python aerial_gym/rl_training/rl_games/train_quintic_tracking_sysid.py \
  --num_envs 1024 \
  --show_trajectory \
  --trajectory_debug_interval 10
```

Evaluate a checkpoint through rl_games play:

```bash
python aerial_gym/rl_training/rl_games/evaluate_quintic_tracking_sysid.py \
  --checkpoint runs/quintic_tracking_sysid_bodyrate_<run>/nn/quintic_tracking_sysid_bodyrate.pth \
  --num_envs 16
```

Plot reference vs actual rollout:

```bash
python aerial_gym/rl_training/rl_games/visualize_quintic_tracking_rollout.py \
  --checkpoint runs/quintic_tracking_sysid_bodyrate_<run>/nn/quintic_tracking_sysid_bodyrate.pth \
  --output-dir rollout_viz_quintic
```

Smoke-test the plotter without a checkpoint:

```bash
python aerial_gym/rl_training/rl_games/visualize_quintic_tracking_rollout.py \
  --random-policy \
  --max-steps 40 \
  --output-dir rollout_viz_quintic_random
```

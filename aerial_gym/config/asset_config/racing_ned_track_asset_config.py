from aerial_gym import AERIAL_GYM_DIRECTORY

import numpy as np


TRACK_BOUNDS_MIN = [-10.0, -15.0, -26.0]
TRACK_BOUNDS_MAX = [170.0, 15.0, 6.0]

NUM_LAPS = 1

GATE_INNER_SIZE_METERS = 2.7
GATE_OUTER_SIZE_METERS = 3.2
GATE_COLLISION_DEPTH_METERS = 0.45

RANDOM_CYLINDER_OBSTACLE_COUNT = 0
RANDOM_HORIZONTAL_CYLINDER_OBSTACLE_COUNT = 0
RANDOM_TILTED_CYLINDER_OBSTACLE_COUNT = 0
TOTAL_RANDOM_CYLINDER_OBSTACLE_COUNT = 0
RANDOM_CYLINDER_RADIUS_METERS = 0.0
RANDOM_CYLINDER_HEIGHT_METERS = 0.0

START_POSITION = [0.0, 0.0, 2.98]
START_YAW_DEG = 0.0

# Converted from the external simulator packet with sim_x=-N, sim_y=E,
# sim_z=3.0-D. NED D is positive downward, while Isaac Gym z is positive upward.
# The packet normal is perpendicular to the race line, so the
# pass direction yaw is set from the previous point toward the current gate.
RACING_NED_TRACK_GATES = [
    {"name": "ned_gate_01", "position": [23.30, -0.40, 3.05], "yaw_deg": -0.98, "semantic_id": 201},
    {"name": "ned_gate_02", "position": [46.89, -2.50, -2.05], "yaw_deg": -5.09, "semantic_id": 202},
    {"name": "ned_gate_03", "position": [74.59, 1.20, -10.65], "yaw_deg": 7.61, "semantic_id": 203},
    {"name": "ned_gate_04", "position": [111.49, -5.10, -21.55], "yaw_deg": -9.69, "semantic_id": 204},
    {"name": "ned_gate_05", "position": [135.49, -0.80, -22.34], "yaw_deg": 10.16, "semantic_id": 205},
    {"name": "ned_gate_06", "position": [159.19, -4.40, -22.95], "yaw_deg": -8.64, "semantic_id": 206},
]


def world_to_ratio(position):
    ratio = []
    for value, low, high in zip(position, TRACK_BOUNDS_MIN, TRACK_BOUNDS_MAX):
        ratio.append((value - low) / (high - low))
    return ratio


def make_gate_asset_params(gate_definition):
    position_ratio = world_to_ratio(gate_definition["position"])
    yaw_rad = np.deg2rad(gate_definition["yaw_deg"])
    attrs = {
        "num_assets": 1,
        "asset_folder": f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/racing",
        "file": "gate_2p7m.urdf",
        "min_position_ratio": position_ratio,
        "max_position_ratio": position_ratio,
        "collision_mask": 1,
        "disable_gravity": False,
        "replace_cylinder_with_capsule": False,
        "flip_visual_attachments": True,
        "density": 0.001,
        "angular_damping": 0.1,
        "linear_damping": 0.1,
        "max_angular_velocity": 100.0,
        "max_linear_velocity": 100.0,
        "armature": 0.001,
        "collapse_fixed_joints": False,
        "fix_base_link": True,
        "specific_filepath": None,
        "color": [255, 115, 25],
        "keep_in_env": True,
        "body_semantic_label": 0,
        "link_semantic_label": 0,
        "per_link_semantic": False,
        "semantic_masked_links": {},
        "place_force_sensor": False,
        "force_sensor_parent_link": "base_link",
        "force_sensor_transform": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "use_collision_mesh_instead_of_visual": False,
        "semantic_id": gate_definition["semantic_id"],
        "min_state_ratio": [
            position_ratio[0],
            position_ratio[1],
            position_ratio[2],
            0.0,
            0.0,
            yaw_rad,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "max_state_ratio": [
            position_ratio[0],
            position_ratio[1],
            position_ratio[2],
            0.0,
            0.0,
            yaw_rad,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    }
    return type(
        gate_definition["name"].title().replace("_", "") + "AssetParams",
        (),
        attrs,
    )


TRACK_GATE_ASSET_CLASSES = [
    make_gate_asset_params(gate_definition)
    for gate_definition in RACING_NED_TRACK_GATES
]

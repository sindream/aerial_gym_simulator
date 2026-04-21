from aerial_gym import AERIAL_GYM_DIRECTORY

import numpy as np


TRACK_BOUNDS_MIN = [0.0, -15.0, 0.0]
TRACK_BOUNDS_MAX = [100.0, 15.0, 6.0]

TRACK_LENGTH_METERS = 100.0
TRACK_WIDTH_METERS = 30.0
NUM_LAPS = 2

GATE_INNER_SIZE_METERS = 1.5
GATE_OUTER_SIZE_METERS = 3.0
GATE_COLLISION_DEPTH_METERS = 0.08

RANDOM_CYLINDER_OBSTACLE_COUNT = 10
RANDOM_CYLINDER_RADIUS_METERS = 0.40
RANDOM_CYLINDER_HEIGHT_METERS = 4.5
RANDOM_CYLINDER_SEMANTIC_ID = 150

START_POSITION = [2.0, -12.0, 0.25]
START_YAW_DEG = 0.0

RACING_TRACK_GATES = [
    {"name": "gate_01", "position": [8.0, -12.0, 1.6], "yaw_deg": 0.0, "semantic_id": 101},
    {"name": "gate_02", "position": [24.0, -8.0, 1.8], "yaw_deg": 20.0, "semantic_id": 102},
    {"name": "gate_03", "position": [41.0, -2.0, 1.8], "yaw_deg": 35.0, "semantic_id": 103},
    {"name": "gate_04", "position": [60.0, 6.0, 2.0], "yaw_deg": 55.0, "semantic_id": 104},
    # {"name": "gate_05", "position": [80.0, 9.0, 3.3], "yaw_deg": -10.0, "semantic_id": 105},
    # {"name": "gate_06", "position": [72.0, -3.0, 1.2], "yaw_deg": 180.0, "semantic_id": 106},
    # {"name": "gate_07", "position": [52.0, -10.0, 1.6], "yaw_deg": -150.0, "semantic_id": 107},
    # {"name": "gate_08", "position": [33.0, -12.0, 1.7], "yaw_deg": -125.0, "semantic_id": 108},
    # {"name": "gate_09", "position": [17.0, -6.0, 1.7], "yaw_deg": 120.0, "semantic_id": 109},
    # {"name": "gate_10", "position": [22.0, 7.0, 2.0], "yaw_deg": 25.0, "semantic_id": 110},
    # {"name": "gate_11", "position": [10.0, 0.0, 1.7], "yaw_deg": -100.0, "semantic_id": 111},
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
        "file": "gate_1p5m.urdf",
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


TRACK_GATE_ASSET_CLASSES = []
for gate_definition in RACING_TRACK_GATES:
    TRACK_GATE_ASSET_CLASSES.append(make_gate_asset_params(gate_definition))


class RaceCylinderObstacleAssetParams:
    num_assets = RANDOM_CYLINDER_OBSTACLE_COUNT
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/racing"
    file = "cylinder_obstacle.urdf"

    collision_mask = 1
    disable_gravity = False
    replace_cylinder_with_capsule = False
    flip_visual_attachments = True
    density = 0.001
    angular_damping = 0.1
    linear_damping = 0.1
    max_angular_velocity = 100.0
    max_linear_velocity = 100.0
    armature = 0.001
    collapse_fixed_joints = True
    fix_base_link = True
    specific_filepath = None
    color = [110, 190, 255]
    keep_in_env = True
    body_semantic_label = 0
    link_semantic_label = 0
    per_link_semantic = False
    semantic_masked_links = {}
    place_force_sensor = False
    force_sensor_parent_link = "base_link"
    force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    use_collision_mesh_instead_of_visual = False
    semantic_id = RANDOM_CYLINDER_SEMANTIC_ID
    min_state_ratio = [
        0.18,
        0.10,
        (0.5 * RANDOM_CYLINDER_HEIGHT_METERS - TRACK_BOUNDS_MIN[2])
        / (TRACK_BOUNDS_MAX[2] - TRACK_BOUNDS_MIN[2]),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.62,
        0.90,
        (0.5 * RANDOM_CYLINDER_HEIGHT_METERS - TRACK_BOUNDS_MIN[2])
        / (TRACK_BOUNDS_MAX[2] - TRACK_BOUNDS_MIN[2]),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

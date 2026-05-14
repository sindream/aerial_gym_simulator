from aerial_gym import AERIAL_GYM_DIRECTORY

import numpy as np


TRACK_BOUNDS_MIN = [-25.0, -50.0, 0.0]
TRACK_BOUNDS_MAX = [60.0, 30.0, 14.0]

TRACK_LENGTH_METERS = 85.0
TRACK_WIDTH_METERS = 80.0
NUM_LAPS = 1

GATE_INNER_SIZE_METERS = 4.0
GATE_OUTER_SIZE_METERS = 4.64
GATE_COLLISION_DEPTH_METERS = 0.45

RANDOM_CYLINDER_OBSTACLE_COUNT = 12
RANDOM_HORIZONTAL_CYLINDER_OBSTACLE_COUNT = 12
RANDOM_TILTED_CYLINDER_OBSTACLE_COUNT = 12
TOTAL_RANDOM_CYLINDER_OBSTACLE_COUNT = (
    RANDOM_CYLINDER_OBSTACLE_COUNT
    + RANDOM_HORIZONTAL_CYLINDER_OBSTACLE_COUNT
    + RANDOM_TILTED_CYLINDER_OBSTACLE_COUNT
)
RANDOM_CYLINDER_RADIUS_METERS = 0.55
RANDOM_CYLINDER_HEIGHT_METERS = 9.0
RANDOM_CYLINDER_SEMANTIC_ID = 150

START_POSITION = [0.0, 0.0, 3.0]
START_YAW_DEG = 0.0

RACING_TRACK_GATES = [
    {"name": "gate_01", "position": [13.5, -0.5, 4.4], "yaw_deg": 0.0, "semantic_id": 101},
    {"name": "gate_02", "position": [36.0, -8.0, 6.9], "yaw_deg": -33.43, "semantic_id": 102},
    {"name": "gate_03", "position": [53.0, -23.0, 9.9], "yaw_deg": -86.42, "semantic_id": 103},
    {"name": "gate_04", "position": [50.0, -42.0, 11.4], "yaw_deg": -98.97, "semantic_id": 104},
    {"name": "gate_05", "position": [50.0, -41.5, 5.4], "yaw_deg": 90.0, "semantic_id": 105},
    {"name": "gate_06", "position": [40.0, -32.0, 6.4], "yaw_deg": 136.47, "semantic_id": 106},
    {"name": "gate_07", "position": [30.0, -20.5, 8.4], "yaw_deg": 131.01, "semantic_id": 107},
    {"name": "gate_08", "position": [23.0, -6.0, 8.4], "yaw_deg": 115.77, "semantic_id": 108},
    {"name": "gate_09", "position": [9.5, 16.5, 5.9], "yaw_deg": 150.96, "semantic_id": 109},
    {"name": "gate_10", "position": [-4.0, 23.0, 5.9], "yaw_deg": 199.29, "semantic_id": 110},
    {"name": "gate_11", "position": [-16.0, 15.5, 3.9], "yaw_deg": -87.99, "semantic_id": 111},
    {"name": "gate_12", "position": [-10.5, 1.0, 4.4], "yaw_deg": -9.23, "semantic_id": 112},
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


class RaceHorizontalCylinderObstacleAssetParams:
    num_assets = RANDOM_HORIZONTAL_CYLINDER_OBSTACLE_COUNT
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
    color = [90, 220, 160]
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
        np.deg2rad(75.0),
        np.deg2rad(-15.0),
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
        np.deg2rad(105.0),
        np.deg2rad(15.0),
        2.0 * np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


class RaceTiltedCylinderObstacleAssetParams:
    num_assets = RANDOM_TILTED_CYLINDER_OBSTACLE_COUNT
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
    color = [150, 210, 110]
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
        np.deg2rad(25.0),
        np.deg2rad(-50.0),
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
        np.deg2rad(65.0),
        np.deg2rad(50.0),
        2.0 * np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

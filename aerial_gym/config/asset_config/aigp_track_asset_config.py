from aerial_gym import AERIAL_GYM_DIRECTORY

import math
import numpy as np


AIGP_SELECTED_TRACK_NAME = "arsenal_track01"

GATE_INNER_WIDTH_METERS = 1.5
GATE_INNER_HEIGHT_METERS = 1.5
GATE_OUTER_WIDTH_METERS = 2.7
GATE_OUTER_HEIGHT_METERS = 2.7
GATE_COLLISION_DEPTH_METERS = 0.26
START_DISTANCE_BEFORE_FIRST_GATE_METERS = 4.0
TRACK_MARGIN_METERS = 12.0
GROUND_MARGIN_METERS = 2.0
NUM_LAPS = 1


_RAW_AIGP_TRACKS = {
    "anduril_track01": [
        ("anduril_gate_00", [528.90, -98.40, 77.80], -90.0),
        ("anduril_gate_01", [505.30, -96.30, 72.70], -90.0),
        ("anduril_gate_02", [477.60, -100.00, 64.10], -90.0),
        ("anduril_gate_03", [440.70, -93.70, 53.20], -90.0),
        ("anduril_gate_04", [416.70, -98.00, 52.41], -90.0),
        ("anduril_gate_05", [393.00, -94.40, 51.80], -90.0),
    ],
    "arsenal_track01": [
        ("arsenal_gate_00", [10.06, -97.99, 1.10], 0.0),
        ("arsenal_gate_01", [18.92, -82.31, 3.56], 30.0),
        ("arsenal_gate_02", [21.62, -74.11, 2.86], 10.0),
        ("arsenal_gate_03", [13.46, -65.08, 1.70], -50.0),
        ("arsenal_gate_04", [-3.30, -50.40, 1.00], -50.0),
        ("arsenal_gate_05", [-11.80, -30.60, 1.00], 0.0),
        ("arsenal_gate_06", [-11.80, -15.70, 6.10], 0.0),
        ("arsenal_gate_07", [-1.80, -5.30, 6.50], 90.0),
        ("arsenal_gate_08", [9.40, -1.80, 5.70], 70.0),
        ("arsenal_gate_09", [20.00, 5.40, 5.20], 40.0),
        ("arsenal_gate_10", [19.10, 15.28, 4.60], -20.0),
        ("arsenal_gate_11", [13.80, 30.40, 1.10], -20.0),
        ("arsenal_gate_12", [0.70, 43.60, 3.20], -60.0),
        ("arsenal_gate_13", [-9.10, 52.40, 3.70], 0.0),
        ("arsenal_gate_14", [-3.30, 62.50, 3.70], 40.0),
        ("arsenal_gate_15", [6.39, 72.63, 1.64], 20.0),
        ("arsenal_gate_16", [7.80, 93.20, 1.10], 0.0),
    ],
}


def _ue_yaw_to_rh_yaw_deg(ue_yaw_deg):
    # Positions are converted from UE left-handed z-up to RH z-up by flipping y.
    # A yaw rotation therefore changes sign in the RH training frame. The AIGP
    # gate mesh's zero-yaw opening normal is UE +Y, while our URDF gate normal is
    # local +X, so add a 90 degree mesh-frame offset.
    return 90.0 - float(ue_yaw_deg)


def _build_gate_layout(track_name):
    raw_gates = _RAW_AIGP_TRACKS[track_name]
    gates = []
    semantic_base = 300 if track_name == "arsenal_track01" else 400
    for index, (name, position, ue_yaw_deg) in enumerate(raw_gates):
        gates.append(
            {
                "name": name,
                "position": position,
                "yaw_deg": _ue_yaw_to_rh_yaw_deg(ue_yaw_deg),
                "ue_yaw_deg": float(ue_yaw_deg),
                "semantic_id": semantic_base + index + 1,
            }
        )
    return gates


def _compute_start_pose(gates):
    first_gate = np.array(gates[0]["position"], dtype=np.float32)
    yaw_rad = math.radians(gates[0]["yaw_deg"])
    backward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)
    start_position = first_gate - START_DISTANCE_BEFORE_FIRST_GATE_METERS * backward
    return start_position.tolist(), gates[0]["yaw_deg"]


def _compute_bounds(gates, start_position):
    positions = np.array([gate["position"] for gate in gates] + [start_position], dtype=np.float32)
    lower = positions.min(axis=0) - TRACK_MARGIN_METERS
    upper = positions.max(axis=0) + TRACK_MARGIN_METERS
    lower[2] = min(lower[2], 0.0) - GROUND_MARGIN_METERS
    upper[2] += GATE_OUTER_HEIGHT_METERS * 0.5
    return lower.tolist(), upper.tolist()


AIGP_TRACK_GATES = _build_gate_layout(AIGP_SELECTED_TRACK_NAME)
START_POSITION, START_YAW_DEG = _compute_start_pose(AIGP_TRACK_GATES)
TRACK_BOUNDS_MIN, TRACK_BOUNDS_MAX = _compute_bounds(AIGP_TRACK_GATES, START_POSITION)
TRACK_EXTENTS = [
    TRACK_BOUNDS_MAX[index] - TRACK_BOUNDS_MIN[index]
    for index in range(3)
]
ENV_SPACING = max(25.0, max(TRACK_EXTENTS[0], TRACK_EXTENTS[1]) + 20.0)


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
        "file": "gate_aigp_2p7_outer_1p5_inner.urdf",
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
    for gate_definition in AIGP_TRACK_GATES
]

import ast
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TRACK_CONFIG_PATH = REPO_ROOT / "aerial_gym/config/asset_config/racing_track_asset_config.py"
OUTPUT_PATH = REPO_ROOT / "aerial_gym/task/drone_racing_task/gate_layout_current.svg"


def _literal_from_assignments(module, name):
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"Could not find literal assignment for {name}")


def _load_track_data():
    module = ast.parse(TRACK_CONFIG_PATH.read_text())
    gates = _literal_from_assignments(module, "RACING_TRACK_GATES")
    start_position = _literal_from_assignments(module, "START_POSITION")
    start_yaw_deg = _literal_from_assignments(module, "START_YAW_DEG")
    return gates, start_position, start_yaw_deg


def _bounds(values, margin_ratio=0.10):
    vmin = min(values)
    vmax = max(values)
    span = max(vmax - vmin, 1.0)
    margin = span * margin_ratio
    return vmin - margin, vmax + margin


def _map_xy(x, y, x_bounds, y_bounds, width, height, pad):
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    sx = (width - 2 * pad) / (x1 - x0)
    sy = (height - 2 * pad) / (y1 - y0)
    px = pad + (x - x0) * sx
    py = height - pad - (y - y0) * sy
    return px, py


def _map_xz(x, z, x_bounds, z_bounds, width, height, pad):
    x0, x1 = x_bounds
    z0, z1 = z_bounds
    sx = (width - 2 * pad) / (x1 - x0)
    sz = (height - 2 * pad) / (z1 - z0)
    px = pad + (x - x0) * sx
    py = height - pad - (z - z0) * sz
    return px, py


def _project_oblique(x, y, z, x_bounds, y_bounds, z_bounds, width, height, pad):
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    z0, z1 = z_bounds
    sx = (width - 2 * pad) / max(x1 - x0, 1.0)
    sy = (width - 2 * pad) / max(y1 - y0, 1.0)
    sz = (height - 2 * pad) / max(z1 - z0, 1.0)

    xn = (x - x0) * sx
    yn = (y - y0) * sy
    zn = (z - z0) * sz

    px = pad + xn + 0.55 * yn
    py = height - pad - zn + 0.30 * yn
    return px, py


def _svg_header(width, height):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<defs>\n'
        '  <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">\n'
        '    <path d="M0,0 L0,6 L9,3 z" fill="#d9480f"/>\n'
        "  </marker>\n"
        "</defs>\n"
    )


def _render():
    gates, start_position, start_yaw_deg = _load_track_data()

    xs = [gate["position"][0] for gate in gates] + [start_position[0]]
    ys = [gate["position"][1] for gate in gates] + [start_position[1]]
    zs = [gate["position"][2] for gate in gates] + [start_position[2]]

    x_bounds = _bounds(xs)
    y_bounds = _bounds(ys)
    z_bounds = _bounds(zs)

    panel_w = 520
    panel_h = 560
    pad = 60
    gap = 40
    width = panel_w * 3 + gap * 2
    height = panel_h + 70

    lines = [_svg_header(width, height)]
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#fffdf8"/>\n')
    lines.append(
        '<text x="24" y="32" font-family="monospace" font-size="24" fill="#2d2d2d">'
        "Current Racing Gate Layout"
        "</text>\n"
    )

    left_x = 0
    mid_x = panel_w + gap
    right_x = panel_w * 2 + gap * 2
    panel_y = 50
    lines.append(
        f'<rect x="{left_x + 8}" y="{panel_y}" width="{panel_w - 16}" height="{panel_h}" '
        'fill="#ffffff" stroke="#dadada"/>\n'
    )
    lines.append(
        f'<rect x="{mid_x + 8}" y="{panel_y}" width="{panel_w - 16}" height="{panel_h}" '
        'fill="#ffffff" stroke="#dadada"/>\n'
    )
    lines.append(
        f'<rect x="{right_x + 8}" y="{panel_y}" width="{panel_w - 16}" height="{panel_h}" '
        'fill="#ffffff" stroke="#dadada"/>\n'
    )
    lines.append(
        f'<text x="{left_x + 24}" y="{panel_y + 28}" font-family="monospace" font-size="18" fill="#444">'
        "Top View (XY)"
        "</text>\n"
    )
    lines.append(
        f'<text x="{mid_x + 24}" y="{panel_y + 28}" font-family="monospace" font-size="18" fill="#444">'
        "Side View (XZ)"
        "</text>\n"
    )
    lines.append(
        f'<text x="{right_x + 24}" y="{panel_y + 28}" font-family="monospace" font-size="18" fill="#444">'
        "Oblique View (XYZ)"
        "</text>\n"
    )

    poly_xy = []
    poly_xz = []
    poly_oblique = []
    for gate in gates:
        x, y, z = gate["position"]
        px, py = _map_xy(x, y, x_bounds, y_bounds, panel_w, panel_h, pad)
        qx, qz = _map_xz(x, z, x_bounds, z_bounds, panel_w, panel_h, pad)
        ox, oy = _project_oblique(x, y, z, x_bounds, y_bounds, z_bounds, panel_w, panel_h, pad)
        poly_xy.append(f"{left_x + px:.1f},{panel_y + py:.1f}")
        poly_xz.append(f"{mid_x + qx:.1f},{panel_y + qz:.1f}")
        poly_oblique.append(f"{right_x + ox:.1f},{panel_y + oy:.1f}")

    lines.append(
        f'<polyline points="{" ".join(poly_xy)}" fill="none" stroke="#7c8aa5" stroke-width="2.5"/>\n'
    )
    lines.append(
        f'<polyline points="{" ".join(poly_xz)}" fill="none" stroke="#7c8aa5" stroke-width="2.5"/>\n'
    )
    lines.append(
        f'<polyline points="{" ".join(poly_oblique)}" fill="none" stroke="#7c8aa5" stroke-width="2.5"/>\n'
    )

    first_xy = poly_xy[0]
    last_xy = poly_xy[-1]
    first_x, first_y = first_xy.split(",")
    last_x, last_y = last_xy.split(",")
    lines.append(
        f'<line x1="{last_x}" y1="{last_y}" x2="{first_x}" y2="{first_y}" '
        'stroke="#a0a0a0" stroke-width="2" stroke-dasharray="8 6"/>\n'
    )

    sx, sy = _map_xy(start_position[0], start_position[1], x_bounds, y_bounds, panel_w, panel_h, pad)
    lines.append(
        f'<circle cx="{left_x + sx:.1f}" cy="{panel_y + sy:.1f}" r="8" fill="#2f9e44" stroke="#1b5e20"/>\n'
    )
    lines.append(
        f'<text x="{left_x + sx + 12:.1f}" y="{panel_y + sy - 10:.1f}" font-family="monospace" font-size="15" fill="#1b5e20">'
        f"START yaw={start_yaw_deg:.1f}"
        "</text>\n"
    )

    for i, gate in enumerate(gates):
        x, y, z = gate["position"]
        yaw_deg = gate["yaw_deg"]
        yaw = math.radians(yaw_deg)
        dir_x = math.cos(yaw)
        dir_y = math.sin(yaw)

        px, py = _map_xy(x, y, x_bounds, y_bounds, panel_w, panel_h, pad)
        qx, qz = _map_xz(x, z, x_bounds, z_bounds, panel_w, panel_h, pad)
        ox, oy = _project_oblique(x, y, z, x_bounds, y_bounds, z_bounds, panel_w, panel_h, pad)

        start_px = left_x + px
        start_py = panel_y + py
        arrow_scale = 34.0
        end_px = start_px + dir_x * arrow_scale
        end_py = start_py - dir_y * arrow_scale

        lines.append(
            f'<circle cx="{start_px:.1f}" cy="{start_py:.1f}" r="7" fill="#f08c00" stroke="#9a4d00"/>\n'
        )
        lines.append(
            f'<line x1="{start_px:.1f}" y1="{start_py:.1f}" x2="{end_px:.1f}" y2="{end_py:.1f}" '
            'stroke="#d9480f" stroke-width="3" marker-end="url(#arrow)"/>\n'
        )
        lines.append(
            f'<text x="{start_px + 10:.1f}" y="{start_py - 10:.1f}" font-family="monospace" font-size="14" fill="#222">'
            f"{i+1:02d}"
            "</text>\n"
        )
        lines.append(
            f'<circle cx="{right_x + qx:.1f}" cy="{panel_y + qz:.1f}" r="6" fill="#f08c00" stroke="#9a4d00"/>\n'
        )
        side_start_x = mid_x + qx
        side_start_y = panel_y + qz
        side_end_x = side_start_x + dir_x * arrow_scale
        side_end_y = side_start_y
        lines.append(
            f'<line x1="{side_start_x:.1f}" y1="{side_start_y:.1f}" x2="{side_end_x:.1f}" y2="{side_end_y:.1f}" '
            'stroke="#d9480f" stroke-width="3" marker-end="url(#arrow)"/>\n'
        )
        lines.append(
            f'<text x="{mid_x + qx + 10:.1f}" y="{panel_y + qz - 10:.1f}" font-family="monospace" font-size="14" fill="#222">'
            f"{i+1:02d} z={z:.1f} yaw={yaw_deg:.1f}"
            "</text>\n"
        )
        oblique_start_x = right_x + ox
        oblique_start_y = panel_y + oy
        oblique_end_x = oblique_start_x + dir_x * arrow_scale
        oblique_end_y = oblique_start_y - dir_y * arrow_scale * 0.55
        lines.append(
            f'<circle cx="{oblique_start_x:.1f}" cy="{oblique_start_y:.1f}" r="6" fill="#f08c00" stroke="#9a4d00"/>\n'
        )
        lines.append(
            f'<line x1="{oblique_start_x:.1f}" y1="{oblique_start_y:.1f}" x2="{oblique_end_x:.1f}" y2="{oblique_end_y:.1f}" '
            'stroke="#d9480f" stroke-width="3" marker-end="url(#arrow)"/>\n'
        )
        lines.append(
            f'<text x="{oblique_start_x + 10:.1f}" y="{oblique_start_y - 10:.1f}" font-family="monospace" font-size="14" fill="#222">'
            f"{i+1:02d}"
            "</text>\n"
        )

    legend_y = panel_y + panel_h + 24
    lines.append(
        f'<text x="24" y="{legend_y}" font-family="monospace" font-size="14" fill="#555">'
        "Orange circle: gate center, orange arrow: gate pass direction (local +x), gray dashed line: loop from last gate back to first"
        "</text>\n"
    )
    lines.append("</svg>\n")

    OUTPUT_PATH.write_text("".join(lines))
    return OUTPUT_PATH


if __name__ == "__main__":
    path = _render()
    print(path)

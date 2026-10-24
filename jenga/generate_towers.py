import random
import csv
import json
import math
import xml.etree.ElementTree as ET
import mujoco

# Constants
NUMBER_BLOCKS = 27
NUMBER_TOWERS = 50000
OUTPUT_FILE = "towers_dataset.csv"
MAX_LAYERS = 27

# Generate one tower with exactly 27 blocks and 1–3 blocks per layer.
def generate_tower(number_blocks=NUMBER_BLOCKS):
    remaining = number_blocks
    blocks_per_layer = []

    while remaining > 0:
        count = random.randint(1, min(3, remaining))
        blocks_per_layer.append(count)
        remaining -= count

    blocks = []

    # Create the block coordinates for each layer.
    for layer_index, count in enumerate(blocks_per_layer):
        layer = layer_index + 1
        z = 0.0155 + layer_index * 0.0305
        yaw = 0 if layer_index % 2 == 0 else 90

        if yaw == 0:
            positions = [(-0.0505, 0), (0, 0), (0.0505, 0)]
        else:
            positions = [(0, 0.0505), (0, 0), (0, -0.0505)]

        chosen_positions = random.sample(positions, count)

        for slot, (x, y) in enumerate(chosen_positions, start=1):
            blocks.append({
                "layer": layer,
                "slot": slot,
                "x": x + random.uniform(-0.0025, 0.0025),
                "y": y + random.uniform(-0.0025, 0.0025),
                "z": z,
                "yaw": yaw + random.uniform(-1, 1),
            })

    return blocks, blocks_per_layer


def center_of_mass(blocks):
    """Calculate the center of mass, assuming all blocks have equal mass."""
    n = len(blocks)
    return (
        sum(block["x"] for block in blocks) / n,
        sum(block["y"] for block in blocks) / n,
        sum(block["z"] for block in blocks) / n,
    )

# Full block dimensions in metres
BLOCK_SIZE = (0.05, 0.15, 0.03)


def cross_product(origin, point_a, point_b):
    """Return the 2D cross product of vectors origin->point_a and origin->point_b."""
    return (
        (point_a[0] - origin[0]) * (point_b[1] - origin[1])
        - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
    )


def convex_hull(points):
    """Calculate the convex hull of a set of 2D points."""
    points = sorted(set(points))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross_product(
            lower[-2], lower[-1], point
        ) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross_product(
            upper[-2], upper[-1], point
        ) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def block_corners(block):
    """Return the four XY corners of a block, accounting for its yaw."""
    import math

    x, y = block["x"], block["y"]
    yaw = math.radians(block["yaw"])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    half_x = BLOCK_SIZE[0] / 2
    half_y = BLOCK_SIZE[1] / 2

    corners = []
    for local_x, local_y in (
        (-half_x, -half_y),
        (-half_x, half_y),
        (half_x, -half_y),
        (half_x, half_y),
    ):
        corners.append((
            x + local_x * cos_yaw - local_y * sin_yaw,
            y + local_x * sin_yaw + local_y * cos_yaw,
        ))

    return corners


def point_in_convex_polygon(point, polygon, tolerance=1e-9):
    """Return whether a 2D point is inside or on the boundary of a convex polygon."""
    crosses = [
        cross_product(polygon[i], polygon[(i + 1) % len(polygon)], point)
        for i in range(len(polygon))
    ]

    return (
        all(value >= -tolerance for value in crosses)
        or all(value <= tolerance for value in crosses)
    )


def is_stable(blocks):
    """Check the tower's geometric stability at every layer interface."""
    layers = sorted({block["layer"] for block in blocks})

    # Check each layer as support for all blocks above it.
    for support_layer in layers[:-1]:
        support_blocks = [
            block for block in blocks
            if block["layer"] == support_layer
        ]
        upper_blocks = [
            block for block in blocks
            if block["layer"] > support_layer
        ]

        support_corners = [
            corner
            for block in support_blocks
            for corner in block_corners(block)
        ]
        support_region = convex_hull(support_corners)

        upper_com = (
            sum(block["x"] for block in upper_blocks) / len(upper_blocks),
            sum(block["y"] for block in upper_blocks) / len(upper_blocks),
        )

        if not point_in_convex_polygon(upper_com, support_region):
            return False

    return True

# def simulate_stability(
#     blocks,
#     seconds=20.0,
#     check_interval=3.0,
#     max_horizontal_movement=0.01,
#     max_vertical_movement=0.015,
#     max_rotation_degrees=15.0,
# ):
#     """Return True if the tower stays within the movement thresholds."""
#     root = ET.Element("mujoco", model="jenga_test")
#     ET.SubElement(root, "compiler", angle="degree", coordinate="local")
#     ET.SubElement(
#         root,
#         "option",
#         gravity="0 0 -9.81",
#         timestep="0.002",
#         iterations="200",
#         solver="Newton",
#         cone="pyramidal",
#         impratio="30",
#     )

#     default = ET.SubElement(root, "default")
#     ET.SubElement(
#         default,
#         "geom",
#         density="650",
#         margin="0.0005",
#         gap="0.0005",
#     )

#     world = ET.SubElement(root, "worldbody")
#     ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1")

#     ET.SubElement(
#         world,
#         "geom",
#         name="floor",
#         type="plane",
#         size="0.3 0.3 0.01",
#         friction="0.8 0.001 0.0001",
#     )

#     for block in blocks:
#         name = f"b{block['layer']}_{block['slot']}"
#         body = ET.SubElement(
#             world,
#             "body",
#             name=name,
#             pos=f"{block['x']} {block['y']} {block['z']}",
#             euler=f"0 0 {block['yaw']}",
#         )
#         ET.SubElement(body, "joint", type="free")
#         ET.SubElement(
#             body,
#             "geom",
#             type="box",
#             size="0.025 0.075 0.015",
#             friction="0.32 0.04 0.001",
#         )

#     xml_string = ET.tostring(root, encoding="unicode")
#     model = mujoco.MjModel.from_xml_string(xml_string)
#     data = mujoco.MjData(model)

#     mujoco.mj_forward(model, data)

#     body_ids = {}
#     initial_poses = {}

#     for block in blocks:
#         name = f"b{block['layer']}_{block['slot']}"
#         body_id = mujoco.mj_name2id(
#             model, mujoco.mjtObj.mjOBJ_BODY, name
#         )
#         body_ids[name] = body_id
#         initial_poses[name] = (
#             data.xpos[body_id].copy(),
#             data.xquat[body_id].copy(),
#         )

#     total_steps = round(seconds / model.opt.timestep)
#     steps_per_check = max(
#         1, round(check_interval / model.opt.timestep)
#     )
#     max_rotation = math.radians(max_rotation_degrees)

#     for start_step in range(0, total_steps, steps_per_check):
#         steps = min(steps_per_check, total_steps - start_step)
#         mujoco.mj_step(model, data, nstep=steps)
#         mujoco.mj_forward(model, data)

#         for name, body_id in body_ids.items():
#             initial_pos, initial_quat = initial_poses[name]
#             current_pos = data.xpos[body_id]
#             current_quat = data.xquat[body_id]

#             horizontal_movement = math.hypot(
#                 current_pos[0] - initial_pos[0],
#                 current_pos[1] - initial_pos[1],
#             )
#             vertical_movement = abs(current_pos[2] - initial_pos[2])

#             quat_dot = abs(sum(
#                 initial_quat[i] * current_quat[i]
#                 for i in range(4)
#             ))
#             quat_dot = min(1.0, max(-1.0, quat_dot))
#             rotation = 2 * math.acos(quat_dot)

#             if (
#                 horizontal_movement > max_horizontal_movement
#                 or vertical_movement > max_vertical_movement
#                 or rotation > max_rotation
#             ):
#                 return False

#     return True

def tower_to_row(tower_id, blocks, blocks_per_layer):
    """Create one CSV row for a tower."""
    row = {"Tower_ID": tower_id}

    for layer in range(1, MAX_LAYERS + 1):
        if layer <= len(blocks_per_layer):
            layer_blocks = [
                [block["x"], block["y"], block["z"], block["yaw"]]
                for block in blocks
                if block["layer"] == layer
            ]
            row[f"Layer_{layer}"] = json.dumps(layer_blocks)
        else:
            row[f"Layer_{layer}"] = ""

    row["Stable"] = int(is_stable(blocks))
    # row["Stable_10s"] = int(simulate_stability(blocks))
    row["CoM"] = json.dumps(center_of_mass(blocks))
    return row


def generate_dataset(number_towers=NUMBER_TOWERS, output_file=OUTPUT_FILE):
    """Generate the dataset and write it to a CSV file."""
    fieldnames = (
        ["Tower_ID"]
        + [f"Layer_{layer}" for layer in range(1, MAX_LAYERS + 1)]
        # + ["Stable", "Stable_10s", "CoM"]
        + ["Stable", "CoM"]
    )

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for tower_id in range(1, number_towers + 1):
            blocks, blocks_per_layer = generate_tower()
            row = tower_to_row(tower_id, blocks, blocks_per_layer)
            writer.writerow(row)

    print(f"Generated {number_towers} towers in {output_file}")


if __name__ == "__main__":
    random.seed(1)
    generate_dataset()
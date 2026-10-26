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

def generate_tower(number_blocks=NUMBER_BLOCKS, mode="random"):
    remaining = number_blocks
    blocks_per_layer = []

    while remaining > 0:
        max_count = min(3, remaining)
        choices = list(range(1, max_count + 1))

        if mode == "compact":
            weights = [1, 3, 6][:max_count]
            count = random.choices(choices, weights=weights)[0]

        elif mode == "sparse":
            weights = [6, 3, 1][:max_count]
            count = random.choices(choices, weights=weights)[0]

        else:
            count = random.choice(choices)

        blocks_per_layer.append(count)
        remaining -= count

    blocks = []

    for layer_index, count in enumerate(blocks_per_layer):
        layer = layer_index + 1
        z = 0.0155 + layer_index * 0.0305
        yaw = 0 if layer_index % 2 == 0 else 90

        if yaw == 0:
            positions = [
                (-0.0505, 0),
                (0, 0),
                (0.0505, 0)
            ]
        else:
            positions = [
                (0, 0.0505),
                (0, 0),
                (0, -0.0505)
            ]

        # All modes can produce asymmetric configurations.
        if mode == "compact":
            if count == 1:
                chosen_positions = random.choices(
                    positions, weights=[1, 4, 1], k=1
                )
            elif count == 2:
                chosen_positions = (
                    [positions[0], positions[2]]
                    if random.random() < 0.6
                    else random.sample(positions, 2)
                )
            else:
                chosen_positions = positions.copy()
        else:
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

def generate_dataset(
    number_towers=NUMBER_TOWERS,
    output_file=OUTPUT_FILE,
    stable_ratio=0.5
):
    target_stable = round(number_towers * stable_ratio)
    target_unstable = number_towers - target_stable

    stable_count = 0
    unstable_count = 0
    attempts = 0

    fieldnames = (
        ["Tower_ID"]
        + [f"Layer_{layer}" for layer in range(1, MAX_LAYERS + 1)]
        + ["Stable", "CoM"]
    )

    modes = ["random", "compact", "sparse"]
    mode_weights = [0.5, 0.35, 0.15]

    # Keep rows in memory so we can shuffle before writing.
    rows = []

    while (
        stable_count < target_stable
        or unstable_count < target_unstable
    ):
        attempts += 1

        mode = random.choices(
            modes,
            weights=mode_weights,
            k=1
        )[0]

        blocks, blocks_per_layer = generate_tower(mode=mode)
        stable = int(is_stable(blocks))

        if stable == 1:
            if stable_count >= target_stable:
                continue
            stable_count += 1
        else:
            if unstable_count >= target_unstable:
                continue
            unstable_count += 1

        row = tower_to_row(
            len(rows) + 1,
            blocks,
            blocks_per_layer
        )

        rows.append(row)

        if len(rows) % 1000 == 0:
            print(
                f"Generated: {len(rows)}/{number_towers} | "
                f"Stable: {stable_count} | "
                f"Unstable: {unstable_count} | "
                f"Attempts: {attempts}"
            )

    # Shuffle rows, preserving labels and coordinates together.
    random.shuffle(rows)

    with open(
        output_file, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for tower_id, row in enumerate(rows, start=1):
            row["Tower_ID"] = tower_id
            writer.writerow(row)

    print("\nDataset completed!")
    print("Stable:", stable_count)
    print("Unstable:", unstable_count)
    print("Total attempts:", attempts)


if __name__ == "__main__":
    random.seed(1)
    generate_dataset()
import csv
import json
import xml.etree.ElementTree as ET

INPUT_FILE = "towers_dataset.csv"
OUTPUT_FILE = "jenga_test.xml"
TOWER_ID = 40000 # Change this to select another tower

BLOCK_SIZE = (0.05, 0.15, 0.03)  # Full dimensions in metres
PLANE_SIZE = (0.3, 0.3, 0.01)
FRICTION = (0.32, 0.04, 0.001)


def vec(values):
    return " ".join(f"{value:g}" for value in values)


def load_tower(dataset_file, tower_id):
    """Load one tower from the dataset and return its blocks."""
    with open(dataset_file, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if int(row["Tower_ID"]) == tower_id:
                blocks = []

                for column, cell in row.items():
                    if not column.startswith("Layer_") or not cell:
                        continue

                    layer = int(column.split("_")[1])
                    layer_blocks = json.loads(cell)

                    for slot, block_data in enumerate(layer_blocks, start=1):
                        x, y, z, yaw = block_data
                        blocks.append({
                            "layer": layer,
                            "slot": slot,
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "yaw": float(yaw),
                        })

                return blocks

    raise ValueError(f"Tower_ID {tower_id} was not found in {dataset_file}")


blocks = load_tower(INPUT_FILE, TOWER_ID)

root = ET.Element("mujoco", model=f"jenga_tower_{TOWER_ID}")
ET.SubElement(root, "compiler", angle="degree", coordinate="local")
ET.SubElement(
    root,
    "option",
    gravity="0 0 -9.81",
    timestep="0.002",
    iterations="200",
    solver="Newton",
    cone="pyramidal",
    impratio="30",
)

default = ET.SubElement(root, "default")
ET.SubElement(default, "geom", density="650", margin="0.0005", gap="0.0005")

world = ET.SubElement(root, "worldbody")
ET.SubElement(world, "light", diffuse=".5 .5 .5", pos="0 0 4", dir="0 0 -1")

ET.SubElement(
    world,
    "geom",
    type="plane",
    size=vec(PLANE_SIZE),
    rgba="0.96 0.93 0.73 1",
    friction="0.8 0.001 0.0001",
)

half_size = tuple(size / 2 for size in BLOCK_SIZE)

for block in blocks:
    body = ET.SubElement(
        world,
        "body",
        name=f"b{block['layer']}_{block['slot']}",
        pos=vec((block["x"], block["y"], block["z"])),
        euler=vec((0, 0, block["yaw"])),
    )
    ET.SubElement(body, "joint", type="free")
    ET.SubElement(
        body,
        "geom",
        type="box",
        size=vec(half_size),
        rgba="0.68 0.85 0.90 1",
        friction=vec(FRICTION),
    )

ET.indent(root, space="  ")
ET.ElementTree(root).write(
    OUTPUT_FILE,
    encoding="utf-8",
    xml_declaration=True,
)

print(f"Wrote {OUTPUT_FILE} for Tower_ID {TOWER_ID} with {len(blocks)} blocks")
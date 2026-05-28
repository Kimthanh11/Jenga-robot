import random
OUTPUT_FILE = "jenga.xml"

LAYERS = 18
BLOCKS_PER_LAYER = 3

BLOCK_SIZE = (0.25, 0.75, 0.15)
SIDE_SPACING = 0.51

START_Z = BLOCK_SIZE[2]
LAYER_HEIGHT = 2 * BLOCK_SIZE[2]

GRAVITY = (0, 0, -9.81)
PLANE_SIZE = (3, 3, 0.1)
PLANE_COLOR = (0.96, 0.93, 0.73, 1)

FRICTION = (0.3, 0.001, 0.0001)

COLOR_A = (0.68, 0.85, 0.90, 1.0)
COLOR_B = (0.96, 0.96, 0.95, 1.0)

random.seed(0)

def vec(values):
    return " ".join(f"{v:g}" for v in values)


parts = [f"""
<mujoco model="jenga">

    <compiler angle="degree" coordinate="local"/>

    <option gravity="{vec(GRAVITY)}" timestep="0.005" iterations="100" solver="Newton" />

    <default>
        <geom density="650"
         solref="0.002 1"
         solimp="0.95 0.99 0.001" />
    </default>

    <worldbody>

        <light diffuse=".5 .5 .5" pos="0 0 8" dir="0 0 -1"/>

        <geom
            type="plane"
            size="{vec(PLANE_SIZE)}"
            rgba="{vec(PLANE_COLOR)}"
            {'friction="' + vec(FRICTION) + '"'}
        />

        <body name="hook" pos="1.5 0.51 4.05">
            <joint name="hook_slide" type="slide" axis="1 0 0"/>

            <geom type="box"
                size="0.18 0.015 0.04"
                pos="0 0 0"
                rgba="0.1 0.1 0.9 1"
                density="2000"/>

            <geom type="box"
                size="0.06 0.14 0.05"
                pos="-0.20 0 0"
                rgba="1 0 0 1"
                density="2000"/>
        </body>


"""]

for layer in range(1, LAYERS + 1):

    parts.append(f"""
        <!-- LAYER {layer} -->
                """)

    for block in range(1, BLOCKS_PER_LAYER + 1):

        z = START_Z + (layer - 1) * LAYER_HEIGHT

        if layer % 2 == 1:
            x_positions = [-SIDE_SPACING, 0, SIDE_SPACING]
            x = x_positions[block - 1]
            y = 0
            euler = ""
        else:
            y_positions = [SIDE_SPACING, 0, -SIDE_SPACING]
            x = 0
            y = y_positions[block - 1]
            euler = 'euler="0 0 90"'

        if layer % 2 == 1:
            color = COLOR_A if block in (1, 3) else COLOR_B
        else:
            color = COLOR_B if block in (1, 3) else COLOR_A

        random_friction = random.uniform(0.1, 0.6)

        # paramter meaning: sliding (fritcion against sliding), torsional (rotatinoal force, block rotates harder) rolling(friction aganist rolling, mostly not relevant for blocks?)
        friction = f'friction="{random_friction:.3f} 0.01 0.001"'

        parts.append(f"""
        <body name="b{layer}_{block}" pos="{x:g} {y:g} {z:g}" {euler}>
            <joint type="free"/>

            <geom
                type="box"
                size="{vec(BLOCK_SIZE)}"
                rgba="{vec(color)}"
                {friction}
            />
        </body>
""")

parts.append("""
    </worldbody>

    <actuator>
        <motor joint="hook_slide" ctrlrange="-1 1" gear="5000"/>
    </actuator>

</mujoco>
""")

xml = "".join(parts)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.write(xml)

print("Wrote", OUTPUT_FILE)
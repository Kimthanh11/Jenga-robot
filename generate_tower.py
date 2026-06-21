import random
OUTPUT_FILE = "jenga.xml"

# Game configuration settings
LAYERS = 9
BLOCKS_PER_LAYER = 3

# EXACT JENGA BLOCK SIZE
BLOCK_SIZE = (0.05, 0.152, 0.03)

SIDE_SPACING = BLOCK_SIZE[0] + 0.0005  
START_Z = (BLOCK_SIZE[2] / 2) + 0.0005  
LAYER_HEIGHT = BLOCK_SIZE[2] + 0.0005   

GRAVITY = (0, 0, -9.81)

# THE AUTOMATIC USER-VIEW ZOOM FIX
PLANE_SIZE = (0.3, 0.3, 0.01)  
PLANE_COLOR = (0.96, 0.93, 0.73, 1)

# Friction properties: [Sliding friction, Torsional friction, Rolling friction]
FRICTION = (0.8, 0.001, 0.0001)

COLOR_A = (0.68, 0.85, 0.90, 1.0)
COLOR_B = (0.96, 0.96, 0.95, 1.0)

random.seed(0)

def vec(values):
    return " ".join(f"{v:g}" for v in values)

# Initialize XML builder list with the environment header elements
parts = [f"""
<mujoco model="jenga">

    <compiler angle="degree" coordinate="local"/>

    <option gravity="{vec(GRAVITY)}" timestep="0.002" iterations="200" solver="Newton" />

    <default>
        <geom density="650"
              margin="0.0005"
              gap="0.0005"
/>
    </default>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 4" dir="0 0 -1"/>

        <geom
            type="plane"
            size="{vec(PLANE_SIZE)}"
            rgba="{vec(PLANE_COLOR)}"
            {'friction="' + vec(FRICTION) + '"'}
        />

        <body name="hook" pos="0.15 0.05 0.16">
            <joint name="hook_slide" type="slide" axis="1 0 0" damping="2"/>
             <joint name="hook_slide_y" type="slide" axis="0 1 0" damping="2"/>
              <joint name="hook_slide_z" type="slide" axis="0 0 1" damping="2"/>

            <geom type="box"
                size="0.04 0.005 0.01"
                pos="0 0 0"
                rgba="0.1 0.1 0.9 1"
                density="2000"
                contype="0"
                conaffinity="0"/>

            <geom type="box"
                size="0.006 0.004 0.004"
                pos="-0.05 0 0"
                rgba="1 0 0 1"
                density="2000"/>
        </body>

        <body name="hook2" pos="0.15 -0.05 0.107">
            <joint name="hook_slide2" type="slide" axis="1 0 0" damping="2"/>

            <geom type="box"
                size="0.04 0.005 0.01"
                pos="0 0 0"
                rgba="0.1 0.1 0.9 1"
                density="2000"
                contype="0"
                conaffinity="0"/>

            <geom type="box"
                size="0.006 0.004 0.004"
                pos="-0.05 0 0"
                rgba="1 0 0 1"
                density="2000"/>
        </body>

        <body name="hook3" pos="0.15 0 0.107">
            <joint name="hook_slide3" type="slide" axis="1 0 0" damping="2"/>

            <geom type="box"
                size="0.04 0.005 0.01"
                pos="0 0 0"
                rgba="0.1 0.1 0.9 1"
                density="2000"
                contype="0"
                conaffinity="0"/>

            <geom type="box"
                size="0.006 0.004 0.004"
                pos="-0.05 0 0"
                rgba="1 0 0 1"
                density="2000"/>
        </body>
"""]

# Generate the Jenga blocks layer-by-layer
for layer in range(1, LAYERS + 1):

    parts.append(f"""
        """)

    for block in range(1, BLOCKS_PER_LAYER + 1):
        # Calculate exact center height coordinate for the active layer
        z = START_Z + (layer - 1) * LAYER_HEIGHT

        # Odd layers run parallel to the X-axis
        if layer % 2 == 1:
            x_positions = [-SIDE_SPACING, 0, SIDE_SPACING]
            x = x_positions[block - 1]
            y = 0
            euler = ""
        # Even layers rotate 90-degrees and run parallel to the Y-axis
        else:
            y_positions = [SIDE_SPACING, 0, -SIDE_SPACING]
            x = 0
            y = y_positions[block - 1]
            euler = 'euler="0 0 90"'

        # Alternate the coloring pattern sequentially by layer and block position
        if layer % 2 == 1:
            color = COLOR_A if block in (1, 3) else COLOR_B
        else:
            color = COLOR_B if block in (1, 3) else COLOR_A

        # Randomize sliding friction
        random_sliding_friction = random.uniform(0.25, 0.5)

        # Randomize torsional friction 
        random_torsional_friction = random.uniform(0.01, 0.06)

        rolling_friction = 0.0015

        friction = f'friction="{random_sliding_friction:.3f} {random_torsional_friction:.3f} {rolling_friction}"'

        # --- THE MUJOCO HALF-SIZE CONVERSION ---
        # Crucial step: MuJoCo box geoms measure from their center out to their edge.
        # We cut your target size in half here so that when MuJoCo doubles it internally,
        # the blocks come out to your exact requested dimensions.
        mujoco_half_size = [v / 2 for v in BLOCK_SIZE]

        # Append the block node string to XML
        parts.append(f"""
        <body name="b{layer}_{block}" pos="{x:g} {y:g} {z:g}" {euler}>
            <joint type="free"/>

            <geom
                type="box"
                size="{vec(mujoco_half_size)}"
                rgba="{vec(color)}"
                {friction}
            />
        </body>
""")

# Add pushers ---------------------------------------------------------------
def block_center(layer, position):
    z = START_Z + (layer - 1) * LAYER_HEIGHT

    if layer % 2 == 1:
        x = [-SIDE_SPACING, 0, SIDE_SPACING][position - 1]
        y = 0
    else:
        x = 0
        y = [SIDE_SPACING, 0, -SIDE_SPACING][position - 1]

    return x, y, z

def add_hook(parts, layer, position, name):
    x, y, z = block_center(layer, position)

    offset = 0.15

    if layer % 2 == 0:
        # odd layer: push along X
        hook_x = offset
        hook_y = y
        angle = 0
    else:
        # even layer: push along Y
        hook_x = x
        hook_y = -offset
        angle = -90

    parts.append(f"""
        <body name="{name}"
              pos="{hook_x:g} {hook_y:g} {z:g}"
              euler="0 0 {angle}">
            <joint name="{name}" type="slide" axis="1 0 0" damping="2"/>

            <geom type="box"
                size="0.04 0.005 0.01"
                pos="0 0 0"
                rgba="0.1 0.1 0.9 1"
                density="2000"/>

            <geom type="box"
                size="0.01 0.01 0.01"
                pos="-0.05 0 0"
                rgba="1 0 0 1"
                density="2000"/>
        </body>
""")
    
hooks = [
    (8, 3),
    (7, 1),
    (1, 2),
]

for i, (layer, position) in enumerate(hooks, start=1):
    joint_name = "hook_slide" if i == 1 else f"hook_slide{i}"
    add_hook(parts, layer, position, joint_name)

# Append the closing root elements to finalize the valid XML pattern
# gear="5" means the motor can push with a maximum force of 5N
parts.append("""
    </worldbody>

    <actuator>
        <motor name="hook_x_motor" joint="hook_slide" ctrlrange="-1 1" gear="5"/>
        <position name="hook_y_pos" joint="hook_slide_y" ctrlrange="-0.05 0.05" />
        <position name="hook_z_pos" joint="hook_slide_z" ctrlrange="-0.05 0.05" />
        <motor joint="hook_slide2" ctrlrange="-1 1" gear="5"/>
        <motor joint="hook_slide3" ctrlrange="-1 1" gear="5"/>
    </actuator>

</mujoco>
""")

# ---------------------------------------------------------------


xml = "".join(parts)

# Compile and dump the array string directly into a physical MJCF XML file
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.write(xml)

print("Wrote", OUTPUT_FILE)

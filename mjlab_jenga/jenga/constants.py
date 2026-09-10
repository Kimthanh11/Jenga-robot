import math
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pathlib import Path

# Tower Configurations
LAYERS = 9
BLOCKS_PER_LAYER = 3

BLOCK_SIZE = (0.05, 0.15, 0.03)
BLOCK_HALF_SIZE = tuple(v / 2 for v in BLOCK_SIZE)
# Per-block build-time domain randomization. Keep this small: larger shape
# variation can create unrealistic overlaps in the stacked tower.
BLOCK_DENSITY = 650.0
BLOCK_DENSITY_RANDOMIZATION = 0.0
BLOCK_SIZE_RANDOMIZATION = (0.0, 0.0, 0.0)
RESET_DENSITY_RANDOMIZATION = 0.15
RESET_FRICTION_SLIDING_RANGE = (0.28, 0.48)
RESET_FRICTION_TORSIONAL_RANGE = (0.012, 0.055)
RESET_FRICTION_ROLLING_RANGE = (0.001, 0.001)
TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT = 0.012
TOWER_SUCCESS_MAX_BLOCK_VERTICAL_SHIFT = 0.008
# Out-of-plane tilt of any non-target block. These limits always described tipping;
# the measurement, not the numbers, was wrong.
TOWER_SUCCESS_MAX_BLOCK_ROTATION = math.radians(8.0)
TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT = 0.025
TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT = 0.015
TOWER_DAMAGE_MAX_BLOCK_ROTATION = math.radians(15.0)
CONTACT_X_LIMIT = 0.01
CONTACT_Y_LIMIT = BLOCK_HALF_SIZE[1]
CONTACT_Z_LIMIT = 0.006
CONTACT_FACE_Y = -CONTACT_Y_LIMIT
PUSH_X_VELOCITY_SCALE = 0.03
PUSH_X_VELOCITY_CLIP = (-0.05, 0.05)
PUSH_ACTION_DEADZONE = 0.08
PUSH_VELOCITY_CHANGE_PER_STEP = 0.006
HOOK_CONTACT_SENSOR_NAME = "hook_contact"
CONTACT_FORCE_OBS_NORMALIZER = 5.0
CONTACT_FORCE_OBS_CLIP = 2.0

SIDE_SPACING = BLOCK_SIZE[0] + 0.0005
START_Z = (BLOCK_SIZE[2] / 2) + 0.0005
LAYER_HEIGHT = BLOCK_SIZE[2] + 0.0005

# Horizontal block displacement is measured against the nominal spawn pose, so a tower
# that slides across the floor as one rigid piece counts as damaged even though nothing
# about it came apart. With this enabled the bottom layer's drift is subtracted first,
# making the measure "how far has this block moved relative to the tower's base" --
# shear and blocks sliding out of their layer still count, rigid translation does not.
# Off by default until the A/B says it helps. Layer 1 is never a target and never a
# missing-block candidate, so the base reference is always intact.
TOWER_SHIFT_RELATIVE_TO_BASE = False

# Contact softness of the block geoms, as MuJoCo solref = (timeconst, dampratio).
# None keeps MuJoCo's default (0.02, 1). Lower timeconst means a stiffer contact;
# MuJoCo clamps it to at least 2 * timestep, so 0.004 is the floor here.
#
# This is the remaining lever on the drag ratio. impratio took it from 0.81 to 0.19 by
# stiffening the FRICTION constraints; solref governs how far the contacts deform in
# the first place. Success at full extraction needs drag below 0.107 (12 mm of
# neighbour movement over 112.5 mm of target movement), and the per-target drags
# predict the sweep outcomes exactly: b3_1 at 0.098 succeeds, b6_3 at 0.149, b6_1 at
# 0.181 and b7_1 at 0.252 all fail.
BLOCK_SOLREF: tuple[float, float] | None = None

MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP = 0
MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS = 600_000
MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY = 0.05
MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY = 0.35
MISSING_BLOCK_DOUBLE_BEGIN_STEP = 250_000
MISSING_BLOCK_TRIPLE_BEGIN_STEP = 500_000
FORCED_MISSING_BLOCK_COUNT: int | None = None
# Evaluation can provide an exact, target-valid set of patterns. They are assigned in
# a deterministic cycle over the vectorized environments; training leaves this unset.
FORCED_MISSING_PATTERN_IDS: tuple[int, ...] | None = None
FORCED_MISSING_PATTERN_OFFSET = 0
MISSING_BLOCK_PARK_OFFSET = (1.5, 1.5, 0.5)
MISSING_BLOCK_PARK_SPACING = 0.2
RANDOM_TARGET_BLOCK_BEGIN_STEP = 0
RANDOM_TARGET_BLOCK_RAMP_STEPS = 1
RANDOM_TARGET_BLOCK_START_PROBABILITY = 1.0
RANDOM_TARGET_BLOCK_END_PROBABILITY = 1.0
RANDOM_TARGET_WITH_MISSING_BEGIN_STEP = 0
RANDOM_TARGET_WITH_MISSING_RAMP_STEPS = 1
RANDOM_TARGET_WITH_MISSING_START_PROBABILITY = 1.0
RANDOM_TARGET_WITH_MISSING_END_PROBABILITY = 1.0
FIXED_TARGET_BLOCK_NAME = "b6_1"
# b1_1 and b1_3 are deliberately absent: the scripted full-push controller cannot
# extract them. They end in step_cap rather than damage -- the tower holds, the block
# simply stops after ~25 mm with the actuator at 5-6 N, at or above its stall force.
# Layer 1 carries the whole tower, so this is a real force limit, not a policy failure,
# and keeping them would hand PPO episodes it cannot win. Revisit if the push actuator
# is ever given a larger force budget.
RANDOM_TARGET_BLOCK_NAMES = (
    "b2_1",
    "b2_2",
    "b2_3",
    "b3_1",
    "b9_1",
    "b9_2",
    "b9_3",
)
# Candidate blocks are admitted only if a scripted push can extract them safely. A
# safe success requires every non-target block to remain within
# TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT while the target travels the full success
# distance, giving a drag-ratio threshold of 12 / 112.5 = 0.107. Measured drag ratios
# reproduce the outcome of the scripted sweep in feasibility_sweep.py:
#
#     b3_1  0.098  extractable        b6_3  0.149  not extractable
#     b6_1  0.181  not extractable    b7_1  0.252  not extractable
#
# b6_1, b6_2, b6_3, b7_1 and b7_3 exceed the threshold and are excluded. Contact
# parameters were calibrated first (see calibrate_drag.py): impratio reduced the drag
# ratio from 0.81 to 0.26, which is what makes any target solvable. Beyond that, a
# solref sweep over the block geoms yields a shallow optimum at 0.01 worth 13 percent
# (0.232 against 0.267 at the default), with the simulation diverging at 0.04 and
# above. That does not close the remaining factor of 2.2: at a drag ratio of 0.232 a
# neighbouring block still travels 26 mm over a full 112.5 mm extraction against a
# 12 mm allowance.
#
# The calibrated scripted controller cannot extract these blocks under the 12 mm
# stability allowance. This is an empirical exclusion for the present actuator and
# controller family, not a proof that no controller could extract them.
HOOK_BASE_POS = (0.15, 0.05, 0.16)
HOOK_TIP_LOCAL_X = -0.056
HOOK_APPROACH_GAP = 0.02
HOOK_BOTTOM_LAYER_Z_LIFT = 0.006

COLOR_A = (0.68, 0.85, 0.90, 1.0)
COLOR_B = (0.96, 0.96, 0.95, 1.0)

# get the Scene configurations
_JENGA_XML = Path(__file__).parent.parent / "jenga.xml"
_HOOK1_CFG = SceneEntityCfg("hook", joint_names=("hook_slide",))
_HOOK2_CFG = SceneEntityCfg("jenga", joint_names=("hook_slide2",))
_HOOK3_CFG = SceneEntityCfg("jenga", joint_names=("hook_slide3",))
_HOOK_Y_CFG = SceneEntityCfg("hook", joint_names=("hook_slide_y",))
_HOOK_Z_CFG = SceneEntityCfg("hook", joint_names=("hook_slide_z",))
_TARGET_BLOCK_CFG = SceneEntityCfg("b6_1", body_names=("b6_1",))
_REF_BLOCK_1_CFG = SceneEntityCfg("b6_2", body_names=("b6_2",))
_REF_BLOCK_2_CFG = SceneEntityCfg("b6_3", body_names=("b6_3",))
_HOOK_YAW_CFG = SceneEntityCfg("hook", joint_names=("hook_yaw",))
_HOOK_ALL_CFG = SceneEntityCfg(
    "hook",
    joint_names=("hook_slide", "hook_slide_y", "hook_slide_z", "hook_yaw"),
    preserve_order=True,
)
_HOOK_TIP_CFG = SceneEntityCfg("hook", site_names=("hook_tip",))
_HOOK_JOINT_ORDER = ("hook_slide", "hook_slide_y", "hook_slide_z", "hook_yaw")


MISSING_BLOCK_SINGLE_PATTERNS = (
    ("b4_1",),
    ("b4_3",),
    ("b5_2",),
)
MISSING_BLOCK_DOUBLE_PATTERNS = (
    ("b4_1", "b5_2"),
    ("b4_3", "b5_2"),
    ("b3_2", "b5_1"),
    ("b3_3", "b5_3"),
)
MISSING_BLOCK_TRIPLE_PATTERNS = (
    ("b3_2", "b4_1", "b5_2"),
    ("b3_3", "b4_3", "b5_2"),
    ("b4_1", "b5_2", "b8_2"),
)
MISSING_BLOCK_PATTERNS = (
    (),
    *MISSING_BLOCK_SINGLE_PATTERNS,
    *MISSING_BLOCK_DOUBLE_PATTERNS,
    *MISSING_BLOCK_TRIPLE_PATTERNS,
)
MISSING_BLOCK_CANDIDATES = tuple(
    sorted({block_name for pattern in MISSING_BLOCK_PATTERNS for block_name in pattern})
)

# Fraction of the block length that counts as extracted. START == END made this a
# no-op: 0.75 * 0.15 m = 112.5 mm was required from step 0, and the scripted controller
# needs 400-900 contact steps to get there, so early policies never saw a success at
# all. Ramping from 2 cm gives every target a reachable signal early -- even the ones
# that only reach 79-106 mm before tripping tower_damage.
SUCCESS_CURRICULUM_START = 0.1333   # 2.0 cm
SUCCESS_CURRICULUM_END = 0.75       # 11.25 cm
# 200k was slower than learning needed at the easy end: the first run reached the
# maximum attainable return within ~70 iterations at 2 cm and held it all the way
# through 4.1 cm at iteration 1400. Since a 12-hour job only covers ~1000
# iterations, that slack costs whole days of wall clock. Raise it again if the
# return starts lagging the ramp.
SUCCESS_CURRICULUM_STEPS = 120_000
TOUCH_CURRICULUM_START = 1.0
TOUCH_CURRICULUM_END = 1.0
TOUCH_CURRICULUM_BEGIN_STEP = 0
TOUCH_CURRICULUM_STEPS = 1
YAW_CURRICULUM_START = 0.10
YAW_CURRICULUM_END = 0.6
YAW_CURRICULUM_BEGIN_STEP = 0
YAW_CURRICULUM_STEPS = 600_000
YAW_ACTION_SCALE = 0.06
YAW_TARGET_LIMIT = 0.6
ACTION_CLIP = 1.0
HOOK_SLIDE_Y_TARGET_RANGE = (-0.13, 0.23)
HOOK_SLIDE_Z_TARGET_RANGE = (-0.17, 0.13)
PROGRESS_REWARD_WEIGHT = 8.0
SUCCESS_REWARD_WEIGHT = 10.0
TOWER_INSTABILITY_REWARD_WEIGHT = -2.0
TOWER_DAMAGE_REWARD_WEIGHT = -20.0
TIMEOUT_REWARD_WEIGHT = -5.0
STUCK_REWARD_WEIGHT = -0.005
ACTION_RATE_REWARD_WEIGHT = -0.0005
ACTION_MAGNITUDE_REWARD_WEIGHT = -0.00005
STUCK_CONTACT_FORCE_THRESHOLD = 2.0
STUCK_BLOCK_SPEED_THRESHOLD = 0.002
STUCK_GRACE_STEPS = 20
TOWER_INSTABILITY_GRACE_STEPS = 10


__all__ = [
    # Tower Configurations & Dimensions
    "LAYERS",
    "BLOCKS_PER_LAYER",
    "BLOCK_SIZE",
    "BLOCK_HALF_SIZE",
    "BLOCK_DENSITY",
    "BLOCK_DENSITY_RANDOMIZATION",
    "BLOCK_SIZE_RANDOMIZATION",
    "RESET_DENSITY_RANDOMIZATION",
    "RESET_FRICTION_SLIDING_RANGE",
    "RESET_FRICTION_TORSIONAL_RANGE",
    "RESET_FRICTION_ROLLING_RANGE",
    "SIDE_SPACING",
    "START_Z",
    "LAYER_HEIGHT",
    "COLOR_A",
    "COLOR_B",
    # Scene Configurations & Paths
    "_JENGA_XML",
    "_HOOK1_CFG",
    "_HOOK2_CFG",
    "_HOOK3_CFG",
    "_HOOK_Y_CFG",
    "_HOOK_Z_CFG",
    "_TARGET_BLOCK_CFG",
    "_REF_BLOCK_1_CFG",
    "_REF_BLOCK_2_CFG",
    "_HOOK_YAW_CFG",
    "_HOOK_ALL_CFG",
    "_HOOK_TIP_CFG",
    "_HOOK_JOINT_ORDER",
    # Reference Positions
    # "_START_REF_POS",
    # "_START_TARGET_REL_POS",
    # Tool Geometry & Limits
    "HOOK_BASE_POS",
    "HOOK_TIP_LOCAL_X",
    "HOOK_APPROACH_GAP",
    "HOOK_BOTTOM_LAYER_Z_LIFT",
    "HOOK_CONTACT_SENSOR_NAME",
    "HOOK_SLIDE_Y_TARGET_RANGE",
    "HOOK_SLIDE_Z_TARGET_RANGE",
    # Action & Contact Bounds
    "ACTION_CLIP",
    "CONTACT_X_LIMIT",
    "CONTACT_Y_LIMIT",
    "CONTACT_Z_LIMIT",
    "CONTACT_FACE_Y",
    "PUSH_X_VELOCITY_SCALE",
    "PUSH_X_VELOCITY_CLIP",
    "PUSH_ACTION_DEADZONE",
    "PUSH_VELOCITY_CHANGE_PER_STEP",
    "YAW_ACTION_SCALE",
    "YAW_TARGET_LIMIT",
    # Stability & Damage Thresholds
    "TOWER_SUCCESS_MAX_BLOCK_HORIZONTAL_SHIFT",
    "TOWER_SUCCESS_MAX_BLOCK_VERTICAL_SHIFT",
    "TOWER_SUCCESS_MAX_BLOCK_ROTATION",
    "TOWER_DAMAGE_MAX_BLOCK_HORIZONTAL_SHIFT",
    "TOWER_DAMAGE_MAX_BLOCK_VERTICAL_SHIFT",
    "TOWER_DAMAGE_MAX_BLOCK_ROTATION",
    "TOWER_SHIFT_RELATIVE_TO_BASE",
    "BLOCK_SOLREF",
    # Normalization, Contact & Stuck Thresholds
    "CONTACT_FORCE_OBS_NORMALIZER",
    "CONTACT_FORCE_OBS_CLIP",
    "STUCK_CONTACT_FORCE_THRESHOLD",
    "STUCK_BLOCK_SPEED_THRESHOLD",
    "STUCK_GRACE_STEPS",
    "TOWER_INSTABILITY_GRACE_STEPS",
    # Reward Weights
    "PROGRESS_REWARD_WEIGHT",
    "SUCCESS_REWARD_WEIGHT",
    "TOWER_INSTABILITY_REWARD_WEIGHT",
    "TOWER_DAMAGE_REWARD_WEIGHT",
    "TIMEOUT_REWARD_WEIGHT",
    "STUCK_REWARD_WEIGHT",
    "ACTION_RATE_REWARD_WEIGHT",
    "ACTION_MAGNITUDE_REWARD_WEIGHT",
    # Target Names & Feasibility Sets
    "FIXED_TARGET_BLOCK_NAME",
    "RANDOM_TARGET_BLOCK_NAMES",
    # Missing Block Randomization & Patterns
    "MISSING_BLOCK_RANDOMIZATION_BEGIN_STEP",
    "MISSING_BLOCK_RANDOMIZATION_RAMP_STEPS",
    "MISSING_BLOCK_RANDOMIZATION_START_PROBABILITY",
    "MISSING_BLOCK_RANDOMIZATION_END_PROBABILITY",
    "MISSING_BLOCK_DOUBLE_BEGIN_STEP",
    "MISSING_BLOCK_TRIPLE_BEGIN_STEP",
    "FORCED_MISSING_BLOCK_COUNT",
    "FORCED_MISSING_PATTERN_IDS",
    "FORCED_MISSING_PATTERN_OFFSET",
    "MISSING_BLOCK_PARK_OFFSET",
    "MISSING_BLOCK_PARK_SPACING",
    "MISSING_BLOCK_SINGLE_PATTERNS",
    "MISSING_BLOCK_DOUBLE_PATTERNS",
    "MISSING_BLOCK_TRIPLE_PATTERNS",
    "MISSING_BLOCK_PATTERNS",
    "MISSING_BLOCK_CANDIDATES",
    # Target Block Randomization Schedule
    "RANDOM_TARGET_BLOCK_BEGIN_STEP",
    "RANDOM_TARGET_BLOCK_RAMP_STEPS",
    "RANDOM_TARGET_BLOCK_START_PROBABILITY",
    "RANDOM_TARGET_BLOCK_END_PROBABILITY",
    "RANDOM_TARGET_WITH_MISSING_BEGIN_STEP",
    "RANDOM_TARGET_WITH_MISSING_RAMP_STEPS",
    "RANDOM_TARGET_WITH_MISSING_START_PROBABILITY",
    "RANDOM_TARGET_WITH_MISSING_END_PROBABILITY",
    # Curricula Schedules (Extraction, Contact, Yaw)
    "SUCCESS_CURRICULUM_START",
    "SUCCESS_CURRICULUM_END",
    "SUCCESS_CURRICULUM_STEPS",
    "TOUCH_CURRICULUM_START",
    "TOUCH_CURRICULUM_END",
    "TOUCH_CURRICULUM_BEGIN_STEP",
    "TOUCH_CURRICULUM_STEPS",
    "YAW_CURRICULUM_START",
    "YAW_CURRICULUM_END",
    "YAW_CURRICULUM_BEGIN_STEP",
    "YAW_CURRICULUM_STEPS",
]
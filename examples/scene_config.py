"""Scene settings shared by the examples: window, camera, colours, slope geometry, spawner.

Solver constants (gravity, contact, friction, sleep) live in ceaulver/config.py.
"""

WINDOW_W = 1280
WINDOW_H = 720

# Slope geometry
SLOPE_ANGLE   = 40    # degrees from horizontal
SLOPE_TOP_Y   = 120.0   # Y at the top edge (x=0)
SLOPE_Z_MIN   = -90.0
SLOPE_Z_MAX   = 150.0
# Spawner
SPAWN_INTERVAL = 0.5   # seconds between spawns when active
MAX_CHUNKS     = 1000
SPAWN_BATCH    = 10      # blocks per spawn tick
VIEW_MARGIN    = 10.0   # how far outside frame spawn/despawn should be

# Block size distribution — multivariate log-normal
SPAWN_LOG_MU    = .7    # log-space mean per axis (median he ≈ 1.0)
SPAWN_LOG_SIGMA = 0.4   # log-space std dev (controls size spread)
SPAWN_SHAPE_RHO = 0.15   # inter-axis correlation (0=independent, 1=cubic)
SPAWN_HE_MIN    = 0.5   # minimum half-extent (no single-voxel blocks)

# Diagnostic: invisible vertical wall at the bottom of the slope, to see
# how piles actually form. Normal is -X (uphill); position is
# X_wall = _x_base + WALL_X_OFFSET. Wall is not rendered.
ENABLE_BOTTOM_WALL = False
WALL_X_OFFSET      = 90.0   # units past the slope-base X where the wall sits

# Camera (head-on: eye along slope normal)
ISO_EYE_DIST = 400.0
ISO_SCALE    = 70.0
ISO_NEAR     = 1.0
ISO_FAR      = 1200.0

# Rendering
EDGE_COLOR = (1.0, 1.0, 1.0)
BG_COLOR   = (0.02, 0.02, 0.02)
EDGE_BIAS  = 5e-4

# Slope-mu transition band (scree.py)
# Where the slope mu changes, as fractions of the VIEW height measured from the bottom
# of the frame (0 = bottom edge, 1 = top edge; values outside 0..1 are off-screen).
# Below LO the slope has SLOPE_MU_HI, above HI it has SLOPE_MU_LO, linear in between.
# Live in the tuning panel; G toggles the guide lines that mark them on the slope.
SLOPE_MU_BAND_LO    = 0
SLOPE_MU_BAND_HI    = 0.8
GUIDE_COLOR         = (0.45, 0.45, 0.45)   # monochrome: grey, under the white block edges

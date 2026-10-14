"""wall.py — an upright wall of randomly sized blocks, eroding face-first.

Geometry, in the WALL's frame (x into the wall, y up, z along the face). The wall's
height is cut once into horizontal BEDDING strips (LMIN..LMAX tall, fixed for the whole
wall); each strip is a ring around the period, filled into the wall in batches
BATCH_LAYERS (3) nominal layers deep, the batch boundaries staggered strip to strip (the
first batch a random fraction of one). Each batch is cut in 3D -- opened at a random z,
pieces over LMAX in any direction cut, then volume-weighted random cuts at 35-65% of the
longest side, never under LMIN -- so blocks are 0.6-4 in every direction, most span two
nominal layers, and no depth plane runs the wall's height: the layers interlock (no
curtains), the space is filled exactly. A block's NOMINAL LAYER (the "layer" everything
below uses) is the one its depth centre is in (pitch DPITCH). The wall is vertical on
flat ground (WALL_LEAN tilts both). All of the scene's bookkeeping -- tiling, spawning,
face tracking, deletion -- runs on the GPU; the host reads back scalars and a small face
histogram per frame.

Wake (Perlin): a block may come loose once the wake front F reaches its key. A peel moves a target one layer; F slides toward it at SWEEP_RATE
(1 / PEEL_WAIT layers/s) x (1 - free-fall count, FREEFALL_AVG_T (15 s) rolling average, / FREEFALL_BURST) x (1 - fully woken
layers still standing / STANDING_MAX), stopped by a burst or H, so blocks wake continuously as F sweeps the
noise map. The target starts at 1 (one peel, F from 0) and moves one layer per peel: automatically when fewer than FREEFALL_MAX (3) blocks
are in free fall (awake, > FREEFALL_V, touching nothing; upper third of the wall only, not off
the shelf) and PEEL_WAIT (1.5 s) has passed
since the last peel and F has reached the target, and none for BURST_FREEZE (3 s) after the count was over
FREEFALL_BURST (10) -- the collapse is paced by its own debris; H stops it, P peels by hand.
The key is c + PEEL*(1 - y/H) + PEEL*NOISE_A*N(c, z), N a 2D fractal Perlin noise
(NOISE_OCT octaves, NOISE_ROUGH) over (layer, z), periodic in z -- PEEL (5) layers deep at a time, always top to bottom within a
column, the loose/frozen boundary a noisy contour moving down each layer.

Face tracking keeps the old cube grid as a LINE GRID: ROWS row bands x WIDTH columns,
each block assigned to the cell its centre is in (a cell may hold 0, 1 or more blocks
of a layer).

Wake gate (per column, GATE_ON -- currently OFF): blocks of (depth layer c, column k) may wake only once
the highest in-wall block of the column in front (c-1, k) is WALL_GATE (1/4 of the wall
height) below its own highest in-wall block -- the slabs come down in sequence, not together.

Sleep: a cube may fall asleep only when the forces on it balance (net force and torque
under WALL_BALANCE = 25% of its weight, torque x its largest half-extent) and it is slow, or when it is forced asleep;
everything else stays awake. Every cube spawns ASLEEP. The face is re-found continuously: along each line of
cubes running into the wall (one per row and column), the outermost two cubes still "in
the wall" never sleep (`fields.nosleep_f`), the third sleeps and wakes normally, and
every deeper cube is unconditionally asleep (`fields.nowake_f`) -- so the newest layer
is the last to wake. "In the wall" = asleep, or in continuous contact at low speed
for at least CONT_T -- AND still within MAX_DISP of where it spawned (otherwise a cube
lying slow in the debris at the foot would count as the face).

Sabotage: when a depth layer wakes -- the wall's face (median face layer over the lines
still standing) reaches it -- its blocks whose bottom edge is in the lowest UNDERMINE
row bands are deleted, once, blocks still in place only.
The slab above is undermined rather than crushed.

Sides (WALL_PERIODIC=1, default): z is PERIODIC -- a cube leaving one side re-enters
the other, and cubes touch across the seam (`fields.per_on`: wrapped hash grid, minimum
image in every contact). The period is WIDTH cube pitches, so spacing is uniform across
the seam. WALL_PERIODIC=0 falls back to: two FRICTIONLESS barriers at z = +-(half width), so the wall cannot collapse
sideways -- only from the front. They contain the cubes without holding them up.

Shelf (edge: SHELF in front of the frontmost standing layer, receding toward it at a
constant SHELF_SPEED, faster when it lags (gap / SHELF_CATCH); a block that has fallen off it -- centre past the edge and below the surface --
is ignored by it for good, `fields.noslope_f`): the wall stands on a shelf whose front edge is SHELF cube edges in front of the
closest sleeping layer (the first unconditionally-asleep one), measured from that
layer's front face; the edge only ever recedes. A cube whose centre is past the edge
has no support (`fields.slope_umin_f`) and falls; any cube more than DROP_MAX (10 cube
edges) below the shelf plane is deleted.

Pacing: BEHIND (2) sleeping layers are kept behind the face's two awake layers and its
one free layer, the face being the median face layer over the lines still standing;
up to SPAWN_PER_FRAME new layers a frame, spawned asleep at the back.

Run:
    uv run examples/wall.py
    uv run examples/wall.py --check     # headless, prints stats
    uv run examples/wall.py --capture   # record WALL_CAP_T=30 s of
        video: runs/<date>_wall_capture/wall.mp4 (native framebuffer, 60 fps, video time =
        wall time) + wall.csv (a row per frame and per printed line, wallclock-stamped).
        The window is not resizable.
Env:
    WALL_ROWS=30  WALL_WIDTH=60  WALL_DEPTH=6 (initial)  WALL_HE=1.0 (half the depth)
    WALL_LMIN=0.3  WALL_LMAX=2.0  (face side limits, in block depths 2 HE)
    WALL_MU_S=0.595 WALL_MU_K=0.425 (block-block static / kinetic mu)  WALL_LEAN=-1 (tilt past vertical, deg; + = face leans out)
    WALL_PEEL=3  WALL_STALL=3  WALL_FORCE_WAKE=1
    WALL_BEHIND=2   WALL_MU=4.10 (ground)  WALL_SHELF=7.84 (shelf depth, block depths)   WALL_VIEW (ortho half-height)   WALL_T=60 (--check)

Rendering: each cube face is drawn flat, its grey looked up from a LUT by the face's
world orientation. WALL_LUT=lights (default): two lights at 45 deg elevation, 45 deg
either side of head-on, grey, up-facing faces one flat top tone; motion blur from a velocity buffer
(B toggles, WALL_SHUTTER=1/120 s); soft: camera-facing faces white; bands: soft, posterised. W toggles back to the wireframe.

Controls: SPC pause   R reset   F static friction on/off   W shaded/wireframe
          B motion blur on/off   H hold the wake (no new layer comes loose, no undermining)
          LEFT / RIGHT (held) orbit the camera horizontally (+-YAW_MAX)
          C cycle colours: all blocks -> awake blocks only (sleeping ones grey) -> grey (shaded mode)
          left click: delete the block under the cursor
          O (MANUAL_LAYERS) wake the next 2x2 strip: one row band of the next layer, top down
          P peel one layer (holds the wake, loosens one more layer, undermines one block)   ESC quit
"""
from console import info, onoff, STATUS   # first: before taichi is imported
import os
import sys
import time as _time
_T_LAUNCH = _time.time()       # --capture: the CSV's t_wall is measured from here

import numpy as np

from ceaulver import config
import scene_config
from ceaulver import params, fields, solver
from ceaulver.fields import n_f, sleep_f, pos_f, vel_f, R_f, cntct_f, nosleep_f, nowake_f, MAX_N
from ceaulver.solver import Solver

ROWS    = int(os.environ.get("WALL_ROWS", "30"))
WIDTH   = int(os.environ.get("WALL_WIDTH", "60"))
PERIODIC = os.environ.get("WALL_PERIODIC", "1") != "0"   # z wraps: exit left, re-enter right
DEPTH0  = int(os.environ.get("WALL_DEPTH", "0"))    # 0 = awake + free + BEHIND
HE      = float(os.environ.get("WALL_HE", "1.0"))
# Each depth layer's face is tiled with NB = ROWS*WIDTH random rectangles (sides between
# LMIN and LMAX); every block is 2 HE deep.
LMIN    = float(os.environ.get("WALL_LMIN", "0.3")) * 2.0 * HE   # shortest face side
LMAX    = float(os.environ.get("WALL_LMAX", "2.0")) * 2.0 * HE   # longest face side
HE_MAX  = max(HE, 0.5 * LMAX)      # largest half-extent: sizes the broadphase cell
LEAN    = float(os.environ.get("WALL_LEAN", "-1"))   # tilt past vertical, deg (face leans out)
MU_S    = float(os.environ.get("WALL_MU_S", "0.595"))  # block-block static mu (production 1.15)
MU_K    = float(os.environ.get("WALL_MU_K", "0.425"))  # block-block kinetic mu (production 0.15)
MU_TEST = 1.0                      # TEST: block-block friction x this (1 = off)
MU_S *= MU_TEST; MU_K *= MU_TEST
BEHIND  = int(os.environ.get("WALL_BEHIND", "2"))   # SLEEPING layers kept behind the face
GROUND_MU = float(os.environ.get("WALL_MU", "4.10"))
AWAKE_LAYERS = 2      # outer layers per line that never sleep
FREE_LAYERS  = 1      # the next layer in: normal sleep/wake. Everything deeper: never wakes
DEPTH0 = DEPTH0 or (AWAKE_LAYERS + FREE_LAYERS + BEHIND)
GAP     = 0.02 * 2.0 * HE          # between neighbours in a row and between depth layers
ROW_H   = 2.0 * HE                 # row band of the line grid (face tracking, undermining)
PITCH   = 2.0 * HE + GAP           # the line grid's column width (and the period, WIDTH pitches)
DHE     = HE                       # depth half-extent of every block
DPITCH  = 2.0 * DHE + GAP          # depth pitch between layers
TILT    = LEAN
V_LOW   = 0.5 * 2.0 * HE           # "low speed": half an edge per second
CONT_T  = 0.5                      # continuous contact at low speed for this long (s)
V_FALL  = 3.0 * 2.0 * HE           # leaving the wall: faster than this (s.t. jitter does not)
MAX_DISP = HE                      # still within half an edge of the spawn position
DT      = 1.0 / 60.0
EYE_DIR = (-1.0, 0.0875, 0.0)      # camera: head on, raised 5 deg (tan 5 = 0.0875)
WALL_FOV = 30.0                    # deg: perspective camera (depth reads when orbiting); 0 = orthographic
YAW_RATE = 30.0                    # deg/s the camera orbits while LEFT / RIGHT is held
YAW_MAX  = 70.0                    # deg either side of head-on
SHUTTER   = float(os.environ.get("WALL_SHUTTER", str(1.0 / 120.0)))   # motion blur, seconds
SPAWN_WAIT = 2.0                   # s a spawn waits for debris to clear its slab, then deletes it (hidden
                                   # behind the BEHIND frozen layers)
SPAWN_PER_FRAME = 3                # pacing may add up to this many back layers per frame
SHELF     = float(os.environ.get("WALL_SHELF", "7.84"))   # shelf depth, cube edges
SHELF_SPEED = 0.2                  # units/s the shelf edge recedes toward its target (constant) ...
SHELF_CATCH = 5.0                  # ... or faster when lagging: closes the gap within ~this many s
SHELF_Q     = 0.1                  # the frontmost standing layer = this quantile of the face over lines
SHELF_ROW0  = ROWS // 3            # ... counting only lines from this row band up (foot stubs ignored)
PERIOD    = WIDTH * (2.0 * HE + 0.02 * 2.0 * HE)      # periodic z: WIDTH pitches
NZ        = int(np.floor(PERIOD / (2.0 * HE_MAX * np.sqrt(3.0))))
assert not (os.environ.get("WALL_PERIODIC", "1") != "0") or NZ >= 3, \
    f"periodic z needs >= 3 grid cells across the period; WIDTH={WIDTH} gives {NZ}"
SIDE_W    = 0.5 * PERIOD            # barrier |z|: the tiled width
DROP_MAX  = float(os.environ.get("WALL_DROP", "10")) * 2.0 * HE   # deleted once this far below the shelf
GRACE_T   = 0.5                    # s a woken cube may not sleep (else it re-sleeps at once)
BALANCE   = float(os.environ.get("WALL_BALANCE", "0.25"))  # may sleep only if |F_net| < this x weight
GATE      = float(os.environ.get("WALL_GATE", "0.25")) * ROWS * ROW_H   # a column wakes only once the
                                                      # column in front has dropped this far (1/4 wall height)
GATE_ON   = False               # per-column wake gate; off: the noise map + per-line rule decide
DESPAWN_EVERY = 0.1                # s between batched debris deletions (a cull is serial)
UNDERMINE = int(os.environ.get("WALL_UNDERMINE", "0"))  # row bands a layer's undermined block is taken from
WAKE_MODE   = "rate"            # "rate": the front releases WAKE_BLOCKS blocks/s in noise-key order,
                                # constant; "paced": peels paced by free fall (below)
WAKE_BLOCKS = 50.0              # blocks per second the wake releases, constant (rate mode)
KEY_BINS    = 256               # rate mode: key histogram ahead of the front
KEY_WIN     = 0.25              # ... over this key window (~2 blocks per bin at ~1800 per key unit)
MANUAL_LAYERS = False           # TEST: all asleep at start; each P wakes the next WHOLE layer (no noise,
                                # no auto peeling); combined with WAKE_ALL_LOOSE it stays awake
WAKE_ALL_LOOSE = True           # TEST: every block the wake front has ever reached stays awake
FORCE_WAKE = os.environ.get("WALL_FORCE_WAKE", "1") != "0"  # in-wall blocks IN FRONT of the median face stay awake
STALL     = float(os.environ.get("WALL_STALL", "3"))  # s without the face advancing -> undermine one more block
# Noise-driven wake: a block may come loose once the wake front F (median face + PEEL)
# reaches its key = c + PEEL*(1 - y/H) + PEEL*NOISE_A*N(c, z), N = 2D Perlin in +-1/2 over
# (layer, z), periodic in z. Higher blocks of a column always wake first.
NOISE_A   = 0.5                 # noise amplitude, in units of the PEEL span
NOISE_LC  = 9.0                 # noise cell depth, layers (1/3 the frequency of z)
NOISE_LZ  = 16.0 / 3.0          # noise cell width along the face, world units (rounded to fit the period)
NOISE_SEED = 7
NOISE_OCT  = 5                  # fractal (fBm) octaves: each 2x the frequency ...
NOISE_ROUGH = 0.7               # ... and this x the amplitude of the one before (higher = more jagged)
# The noise term of the wake key, in layers (negative = woken earlier = the wake digs in):
#   NOISE_BASE * nn - NOISE_GOUGE * max(0, -nn - 1)^2,   nn = the fBm normalised to unit std
# -- general unevenness of ~+-NOISE_BASE, plus one-sided GOUGES where the noise is over one
# std low, ~4 layers deep at the rare ~3 std spots.
NOISE_BASE  = 0.5
NOISE_GOUGE = 0.625
FBM_STD     = 0.2               # measured std of _fbm2 (normalises nn)
PEEL      = int(os.environ.get("WALL_PEEL", "5"))     # layers the noise-driven wake spans
FREEFALL_MAX = 3                # peel only while fewer blocks than this are in free fall
FREEFALL_V   = 1.0              # free fall: awake, faster than this, touching nothing (spawn-gap settle reaches ~1.3)
PEEL_WAIT    = 1.5              # s: at least this long between two peels
FREEFALL_BURST  = 10            # more than this many in free fall ...
SWEEP_RATE      = 1.0 / PEEL_WAIT   # layers/s the wake front sweeps with nothing in free fall;
                                    # slowed linearly by the free-fall count, 0 at FREEFALL_BURST
STANDING_MAX    = 3.0           # ... and by the fully woken layers still standing, 0 at this many
FREEFALL_AVG_T  = 15.0          # s: the sweep speed and the peel rule use the free-fall count
                                # averaged over this rolling window (the burst freeze does not)
BURST_FREEZE    = 3.0           # ... freezes peeling for this long (s), from the last frame over it
PIVOT   = np.array([0.0, float(config.GROUND_Y), 0.0])   # front-bottom edge of the wall
_height = ROWS * ROW_H
VIEW_SCALE = float(os.environ.get("WALL_VIEW", str(0.62 * _height + 6.0)))
LOOK_AT = (0.0, 0.45 * _height, 0.0)

params.CHUNK_MU_S = MU_S              # before _FRICTION_ON / _FRICTION_OFF capture them
params.CAP_CONSERVATIVE = 0.0         # EXPERIMENTAL (T-11), OFF: saturating w'(V) removed the pump but
                                      # saturated ~91% of awake contacts (W'max too low for corner/edge
                                      # contacts) and did not stop the ringing
params.SLEEP_IMBALANCE = 0.0          # T-16 relative force gate OFF: at 0.1 a jumbled debris heap rarely
                                      # balances that well -> 7K debris never slept. Debris uses the plain gate
params.ZETA_N = 1.0                   # normal Rayleigh damping (production 0.3): kills the ~9 Hz
                                      # rocking of loaded, kept-awake face blocks (also before capture)
SUBSTEPS = 2 * config.PHYSICS_SUBSTEPS   # 4: ringing blocks had contact omega*dt ~2.6 (p90 5.2) at 2
GRAVITY  = config.GRAVITY
params.GRAVITY = GRAVITY                 # read live by sync_params
# stiffness and yield ceiling are derived from g and the substep dt (config.derive_contact):
# re-derive for this scene's values
params.FENG_KHERTZ, params.VMAX_CONTACT = config.derive_contact(
    config.CONTACT_SAFETY, config.PILE_DEPTH, GRAVITY, config.SPAWN_HE_MAX, SUBSTEPS,
    config.YIELD_STRAIN)
STIFF_X = 0.5                            # softer contact against ringing (omega / sqrt 2); the
params.FENG_KHERTZ *= STIFF_X            # yield ceiling is kept, so it yields ~1.6x deeper
# Block mass ~ sqrt(face area) (a unit-cube face keeps m = 8), floored at MASS_FLOOR: small
# blocks ring first -- contact stiffness / mass grows as 1 / size^2 at fixed depth
MASS_FLOOR = 4.0
params.CHUNK_MU = MU_K
# the ground is the runtime "slope" plane, with the Feng contact
params.SLOPE_MU_LO = params.SLOPE_MU_HI = GROUND_MU

_a = np.radians(TILT)
RZ = np.array([[np.cos(_a), -np.sin(_a), 0.0], [np.sin(_a), np.cos(_a), 0.0], [0.0, 0.0, 1.0]])
GROUND_N = RZ @ np.array([0.0, 1.0, 0.0])
T_DIR    = RZ @ np.array([1.0, 0.0, 0.0])     # in-plane, into the wall (wall-frame x)

_FRICTION_ON = {k: getattr(params, k) for k in
                ("SHEAR_K_FRAC", "PLANE_SHEAR_K", "CHUNK_MU_S", "ZETA_T", "ZETA_N")}
_FRICTION_OFF = dict(SHEAR_K_FRAC=0.0, PLANE_SHEAR_K=0.0, CHUNK_MU_S=params.CHUNK_MU,
                     ZETA_T=0.0, ZETA_N=0.0)


def set_friction(on):
    for k, v in (_FRICTION_ON if on else _FRICTION_OFF).items():
        setattr(params, k, v)


_PARAM_KEYS = ("CHUNK_MU", "CHUNK_MU_S", "SLOPE_MU_HI", "SHEAR_K_FRAC", "PLANE_SHEAR_K",
               "ZETA_T", "ZETA_N", "CHUNK_ROLL", "SLEEP_SPEED", "SLEEP_OMEGA")


def _params():
    return "  ".join(f"{k}={getattr(params, k)}" for k in _PARAM_KEYS if hasattr(params, k))


# ---- the scene's per-block state and bookkeeping, all on the GPU. Per frame the host
# reads back only scalars and a HWIN-int face histogram (for the medians pacing needs).
import taichi as ti

NB    = ROWS * WIDTH        # blocks per layer (kept from the cube grid)
DENS  = NB / (ROWS * ROW_H * PERIOD * DPITCH)   # blocks per unit volume (keeps the count per layer)
NL    = ROWS * WIDTH        # lines into the wall: one per (row, column) cell of that grid
RING  = 512                 # layers of the per-(layer, column) top-height table (mod RING)
HWIN  = 256                 # face histogram window (layers below back+1-HWIN clamp to bin 0)
BIG   = 1 << 30
QH    = 1024.0              # height quantisation for the integer atomic max
GATE_Q = int(GATE * QH)
JOB_SLOTS = 64              # strip batches tiled per launch (>= the number of strips)
MAXB      = 512             # blocks per strip batch, max
BATCH_LAYERS = 3            # a strip is filled in batches this many (nominal) layers deep
DEL_MAX = 16384             # deleted-block positions kept for the wake-near pass
_RZL = RZ.tolist(); _GNL = GROUND_N.tolist(); _PVL = PIVOT.tolist(); _TDL = T_DIR.tolist()

s_c     = ti.field(ti.i32, MAX_N)            # depth layer
s_line  = ti.field(ti.i32, MAX_N)            # row * WIDTH + column of the block's centre
s_kcol  = ti.field(ti.i32, MAX_N)            # column of the block's centre
s_ylo   = ti.field(ti.f32, MAX_N)            # bottom edge, wall frame (undermining)
s_spawn = ti.Vector.field(3, ti.f32, MAX_N)  # spawn position (base period)
s_cont  = ti.field(ti.f32, MAX_N)            # time in continuous slow contact
s_prev  = ti.field(ti.i32, MAX_N)            # in the wall last pass (hysteresis)
s_in    = ti.field(ti.i32, MAX_N)            # in the wall this pass
s_grace = ti.field(ti.f32, MAX_N)            # s left during which a woken block may not sleep
s_key   = ti.field(ti.i32, MAX_N)            # scratch: random pick of the undermined block
s_wkey  = ti.field(ti.f32, MAX_N)            # wake key: loose once the wake front F reaches it
s_loose = ti.field(ti.i32, MAX_N)            # sticky: the wake front has reached this block
s_hok   = ti.field(ti.i32, MAX_N)            # H: loose when the hold began (may stay loose while held)
l_first  = ti.field(ti.i32, NL)
l_second = ti.field(ti.i32, NL)
l_third  = ti.field(ti.i32, NL)
top_q = ti.field(ti.i32, RING * WIDTH)       # highest in-wall block per (layer % RING, column)
cnt   = ti.field(ti.i32, 8)                  # 0 kept awake, 1 lines standing, 2 deleted, 3 fallen, 4 undermine pick, 5 free fall,
                                             # 6 face blocks, 7 face blocks awake
hist  = ti.field(ti.i32, HWIN)
hist_up = ti.field(ti.i32, HWIN)             # the same, lines in the upper part only (shelf)
del_p = ti.Vector.field(4, ti.f32, DEL_MAX)  # deleted blocks: centre, largest half-extent
t_box = ti.Vector.field(6, ti.f32, (JOB_SLOTS, MAXB))   # 3D tiling: (y0, y1, z0, z1, x0, x1)
t_m   = ti.field(ti.i32, JOB_SLOTS)                     # pieces cut per job
t_job = ti.Vector.field(4, ti.f32, JOB_SLOTS)           # job region: (y0, y1, x0, x1), ring in z
t_tgt = ti.field(ti.i32, JOB_SLOTS)                     # blocks wanted per job


@ti.func
def _disp(i):
    d = pos_f[i] - s_spawn[i]
    if ti.static(PERIODIC):
        d[2] -= PERIOD * ti.round(d[2] / PERIOD)
    return d.norm()


@ti.func
def _hmax(i):
    h = fields.he_f[i]
    return ti.max(h[0], ti.max(h[1], h[2]))


@ti.kernel
def _k_wrap(L: ti.f32, n: ti.i32):
    for i in range(n):
        z = pos_f[i][2]
        if z < -0.5 * L or z >= 0.5 * L:
            pos_f[i][2] = z - L * ti.floor((z + 0.5 * L) / L)


@ti.kernel
def _k_settle(n0: ti.i32, n1: ti.i32):
    for i in range(n0, n1):
        sleep_f[i] = 1
        vel_f[i] = ti.Vector([0.0, 0.0, 0.0])
        fields.omega_f[i] = ti.Vector([0.0, 0.0, 0.0])


# ---- the face: first / second / third in-wall layer per line, the per-column gate ----
@ti.kernel
def _k_face_reset():
    for l in range(NL):
        l_first[l] = BIG; l_second[l] = BIG; l_third[l] = BIG
    for q in range(RING * WIDTH):
        top_q[q] = -BIG
    for b in range(HWIN):
        hist[b] = 0; hist_up[b] = 0
    cnt[0] = 0; cnt[1] = 0; cnt[5] = 0; cnt[6] = 0; cnt[7] = 0


@ti.kernel
def _k_inwall(dt: ti.f32, n: ti.i32):
    # hysteresis: a block JOINS the wall by continuous slow contact (or being asleep) and
    # LEAVES it only by being displaced or actually falling -- jitter in place does not
    # move the face
    for i in range(n):
        sp = vel_f[i].norm()
        if sp < V_LOW and cntct_f[i] == 1:
            s_cont[i] += dt
        else:
            s_cont[i] = 0.0
        inw = 0
        if _disp(i) < MAX_DISP:
            if sleep_f[i] == 1 or s_cont[i] >= CONT_T or (s_prev[i] == 1 and sp <= V_FALL):
                inw = 1
        s_in[i] = inw; s_prev[i] = inw
        if inw == 1:
            ti.atomic_min(l_first[s_line[i]], s_c[i])


@ti.kernel
def _k_second(n: ti.i32):
    for i in range(n):
        if s_in[i] == 1 and s_c[i] > l_first[s_line[i]]:
            ti.atomic_min(l_second[s_line[i]], s_c[i])


@ti.kernel
def _k_third(n: ti.i32):
    gn = ti.Vector(_GNL); pv = ti.Vector(_PVL)
    for i in range(n):
        if s_in[i] == 1:
            if s_c[i] > l_second[s_line[i]]:
                ti.atomic_min(l_third[s_line[i]], s_c[i])
            h = (pos_f[i] - pv).dot(gn)                     # height above the shelf plane
            ti.atomic_max(top_q[(s_c[i] % RING) * WIDTH + s_kcol[i]], int(ti.floor(h * QH)))


@ti.kernel
def _k_classify(dt: ti.f32, n: ti.i32, g: ti.f32, wfront: ti.f32, fmed: ti.i32,
                hold: ti.i32, hfront: ti.f32, manual: ti.i32, mlayer: ti.i32,
                olayer: ti.i32, ocut: ti.f32):
    # the outermost two in-wall blocks per line never sleep, the third sleeps and wakes
    # normally, every deeper one is unconditionally asleep. Per-column wake gate: (layer
    # c, column k) may only be kept / free once the highest in-wall block of (c-1, k) is
    # GATE below the highest of (c, k) (none left in front: open). Only force-balanced
    # blocks may sleep: net force or torque over BALANCE x weight (x largest half-extent)
    # keeps a block awake, the frozen layers aside.
    for i in range(n):
        c = s_c[i]; l = s_line[i]
        inw = s_in[i] == 1
        keep = inw and (c == l_first[l] or c == l_second[l])
        free = inw and c == l_third[l]
        own = top_q[(c % RING) * WIDTH + s_kcol[i]]
        front = -BIG
        if c > 0:
            front = top_q[((c - 1) % RING) * WIDTH + s_kcol[i]]
        unlocked = True
        if ti.static(GATE_ON):
            unlocked = front < own - GATE_Q
        woke = s_wkey[i] <= wfront                  # the noise-driven wake front has reached it
        if manual == 1:
            # TEST: whole layers woken by P; layer `olayer` woken by O from the top down --
            # every block whose top edge is above the cut `ocut`
            woke = c <= mlayer or (c == olayer and s_ylo[i] + 2.0 * fields.he_f[i][1] > ocut)
        if hold == 1:
            # held: only what was loose when H was pressed, or what a P peel has since
            # brought in (key beyond the front at the time) -- the line rule may not keep
            # freeing the next block as the one in front falls
            woke = woke and (s_hok[i] == 1 or s_wkey[i] > hfront)
        keep = keep and unlocked and woke
        free = free and unlocked and woke
        w = fields.mass_f[i] * g
        imb = ti.max(fields.F_f[i].norm() / w, fields.tau_f[i].norm() / (w * _hmax(i)))
        if ti.static(FORCE_WAKE):
            # the layers out in front of the face (the staircase steps left standing) are
            # awake, gate or not -- and NEVER sleep: a sleeping block is rigid, so the load
            # that crumbles a woken layer would never act on it
            keep = keep or (inw and c < fmed and woke)
        if woke:
            s_loose[i] = 1
        if ti.static(WAKE_ALL_LOOSE):
            # TEST: loosened once = awake for good -- in the wall or not, any line position
            keep = keep or s_loose[i] == 1
        frozen = inw and not keep and not free
        gr = ti.max(s_grace[i] - dt, 0.0)
        s_grace[i] = gr
        # the balance rule is for blocks still in the wall (T-16: no re-freeze mid-slip);
        # debris sleeps by the solver's own gate (+ SLEEP_IMBALANCE)
        unbal = imb > BALANCE and not frozen and inw
        nosleep_f[i] = 1 if (keep or gr > 0.0 or unbal) else 0
        nowake_f[i] = 1 if frozen else 0
        if keep:
            sleep_f[i] = 0
            ti.atomic_add(cnt[0], 1)
        # in free fall -- counting only the upper third of the wall, and not blocks that
        # fell off the shelf
        if (sleep_f[i] == 0 and fields.cntct_f[i] == 0 and fields.gnd_f[i] == 0
                and fields.slp_f[i] == 0 and vel_f[i].norm() > FREEFALL_V
                and fields.noslope_f[i] == 0
                and (pos_f[i] - ti.Vector(_PVL)).dot(ti.Vector(_GNL)) > ROWS * ROW_H * 2.0 / 3.0):
            ti.atomic_add(cnt[5], 1)
        if frozen:                                       # asleep means at rest
            sleep_f[i] = 1
            vel_f[i] = ti.Vector([0.0, 0.0, 0.0])
            fields.omega_f[i] = ti.Vector([0.0, 0.0, 0.0])
        if inw and c == l_first[l]:                      # a face block: frontmost standing of its line
            ti.atomic_add(cnt[6], 1)
            if sleep_f[i] == 0:
                ti.atomic_add(cnt[7], 1)



@ti.kernel
def _k_lines(back: ti.i32, out: ti.types.ndarray()):
    """Face histogram over the lines still standing -> out[0:HWIN], out[HWIN] = lines
    standing, out[HWIN+1] = kept awake. Bin b = face layer back+2-HWIN+b."""
    base = back + 2 - HWIN
    for l in range(NL):
        f = l_first[l]
        if f < BIG:
            ti.atomic_add(cnt[1], 1)
            ti.atomic_add(hist[ti.min(ti.max(f - base, 0), HWIN - 1)], 1)
            if l // WIDTH >= SHELF_ROW0:                # standing up the wall, not a stub at its foot
                ti.atomic_add(hist_up[ti.min(ti.max(f - base, 0), HWIN - 1)], 1)
    for b in range(HWIN):
        out[b] = hist[b]; out[HWIN + 3 + b] = hist_up[b]
    out[HWIN] = cnt[1]; out[HWIN + 1] = cnt[0]; out[HWIN + 2] = cnt[5]
    out[2 * HWIN + 3] = cnt[6]; out[2 * HWIN + 4] = cnt[7]


# ---- deleting blocks: mark (drop far below + remember where), wake near, cull ----
@ti.func
def _drop(i):
    d = ti.atomic_add(cnt[2], 1)
    if d < DEL_MAX:
        p = pos_f[i]
        del_p[d] = ti.Vector([p[0], p[1], p[2], _hmax(i)])
    pos_f[i][1] = -1e6


@ti.kernel
def _k_mark_undermine(n: ti.i32, lo: ti.i32, hi: ti.i32):
    """Delete ONE block of layers lo..hi: a random one of those in their bottom UNDERMINE
    row bands still in place (smallest random key wins)."""
    cnt[4] = BIG
    for i in range(n):
        s_key[i] = BIG
        if (s_c[i] >= lo and s_c[i] <= hi and s_ylo[i] < UNDERMINE * ROW_H - 1e-4
                and pos_f[i][1] > -1e5 and _disp(i) < MAX_DISP):
            s_key[i] = int(ti.random() * (BIG - 1))
            ti.atomic_min(cnt[4], s_key[i])
    for i in range(n):
        if s_key[i] < BIG and s_key[i] == cnt[4]:
            _drop(i)


@ti.kernel
def _k_mark_below(n: ti.i32):
    gn = ti.Vector(_GNL); pv = ti.Vector(_PVL)
    for i in range(n):
        if pos_f[i][1] > -1e5 and (pos_f[i] - pv).dot(gn) < -DROP_MAX:
            _drop(i)


# Block colours: a CYCLIC continuous colormap (matplotlib's "twilight": its ends meet,
# perceptually uniform) indexed by each block's SPAWN depth -- continuous through the wall,
# wrapping every COLOR_CYCLE layers; a falling block keeps its colour. Each entry is
# normalised to brightest channel 1: the colormap gives the hue, the LUT the brightness.
LAYER_COLORS = 1                # colour mode at start: 0 grey, 1 all blocks, 2 awake blocks only; C cycles
COLORMAP     = "twilight_r"          # reversed order
COLOR_CYCLE  = 12.0             # nominal layers per full turn of the colormap
_NPAL = 256
def _palette():
    import matplotlib
    cols = matplotlib.colormaps[COLORMAP](np.linspace(0.0, 1.0, _NPAL, endpoint=False))[:, :3]
    return (cols / cols.max(axis=1, keepdims=True)).astype(np.float32)
pal_f = ti.Vector.field(3, ti.f32, _NPAL)
pal_f.from_numpy(_palette())


@ti.kernel
def _k_tint(out: ti.types.ndarray(), n: ti.i32, awake_only: ti.i32):
    td = ti.Vector(_TDL); pv = ti.Vector(_PVL)
    for i in range(n):
        u = (s_spawn[i] - pv).dot(td) / (COLOR_CYCLE * DPITCH)     # spawn depth, in turns
        col = pal_f[int((u - ti.floor(u)) * _NPAL) % _NPAL]
        if awake_only == 1 and sleep_f[i] == 1:
            col = ti.Vector([1.0, 1.0, 1.0])                         # asleep: grey (the LUT as is)
        out[i, 0] = col[0]; out[i, 1] = col[1]; out[i, 2] = col[2]


def layer_tint(n, awake_only=False):
    out = np.empty((n, 3), dtype=np.float32)
    if n:
        _k_tint(out, n, 1 if awake_only else 0)
    return out


pick_t = ti.field(ti.i32, ())                # click picking: nearest hit (quantised ray t) ...
pick_i = ti.field(ti.i32, ())                # ... and its block


@ti.kernel
def _k_pick(n: ti.i32, o: ti.types.vector(3, ti.f32), d: ti.types.vector(3, ti.f32),
            k: ti.i32, L: ti.f32):
    """Nearest block hit by the ray o + t d, over the 2k+1 periodic copies on screen
    (slab test in each block's own frame)."""
    pick_t[None] = BIG; pick_i[None] = -1
    for i in range(n):
        if pos_f[i][1] > -1e5:
            R = R_f[i]; h = fields.he_f[i]
            ld = R.transpose() @ d
            best = 1e30
            for kk in range(-k, k + 1):
                lo = R.transpose() @ (o - pos_f[i] - ti.Vector([0.0, 0.0, kk * L]))
                t0 = -1e30; t1 = 1e30
                for a in ti.static(range(3)):
                    if ti.abs(ld[a]) < 1e-9:
                        if ti.abs(lo[a]) > h[a]:
                            t0 = 1e30
                    else:
                        ta = (-h[a] - lo[a]) / ld[a]; tb = (h[a] - lo[a]) / ld[a]
                        t0 = ti.max(t0, ti.min(ta, tb)); t1 = ti.min(t1, ti.max(ta, tb))
                if t0 <= t1 and t1 > 0.0:
                    best = ti.min(best, ti.max(t0, 0.0))
            if best < 1e29:
                ti.atomic_min(pick_t[None], int(best * 64.0))
    for i in range(n):
        if pos_f[i][1] > -1e5:
            R = R_f[i]; h = fields.he_f[i]
            ld = R.transpose() @ d
            for kk in range(-k, k + 1):
                lo = R.transpose() @ (o - pos_f[i] - ti.Vector([0.0, 0.0, kk * L]))
                t0 = -1e30; t1 = 1e30
                for a in ti.static(range(3)):
                    if ti.abs(ld[a]) < 1e-9:
                        if ti.abs(lo[a]) > h[a]:
                            t0 = 1e30
                    else:
                        ta = (-h[a] - lo[a]) / ld[a]; tb = (h[a] - lo[a]) / ld[a]
                        t0 = ti.max(t0, ti.min(ta, tb)); t1 = ti.min(t1, ti.max(ta, tb))
                if t0 <= t1 and t1 > 0.0 and int(ti.max(t0, 0.0) * 64.0) == pick_t[None]:
                    pick_i[None] = i


@ti.kernel
def _k_spawn_clear(n: ti.i32, u0: ti.f32, u1: ti.f32, h0: ti.f32, h1: ti.f32, drop: ti.i32) -> ti.i32:
    """Blocks in the way of a new strip batch: extent into the wall (along T_DIR)
    overlapping [u0, u1] and height overlapping [h0, h1]. drop = 1: delete them."""
    pv = ti.Vector(_PVL); td = ti.Vector(_TDL); gn = ti.Vector(_GNL)
    hits = 0
    for i in range(n):
        if pos_f[i][1] > -1e5:
            R = R_f[i]; h = fields.he_f[i]
            eu = (ti.abs(R.transpose() @ td)).dot(h)            # half-extent along T_DIR
            eh = (ti.abs(R.transpose() @ gn)).dot(h)            # ... along the wall's up
            d = pos_f[i] - pv
            u = d.dot(td); hh = d.dot(gn)
            if u + eu > u0 and u - eu < u1 and hh + eh > h0 and hh - eh < h1:
                hits += 1
                if drop == 1:
                    _drop(i)
    return hits


kh = ti.field(ti.i32, KEY_BINS)


@ti.kernel
def _k_key_hist(n: ti.i32, lo: ti.f32, w: ti.f32, out: ti.types.ndarray()):
    """Histogram of the wake keys in (lo, lo + w] -- the blocks the front reaches next."""
    for b in range(KEY_BINS):
        kh[b] = 0
    for i in range(n):
        k = s_wkey[i]
        if k > lo and k <= lo + w and pos_f[i][1] > -1e5:
            ti.atomic_add(kh[ti.min(int((k - lo) / w * KEY_BINS), KEY_BINS - 1)], 1)
    for b in range(KEY_BINS):
        out[b] = kh[b]


@ti.kernel
def _k_drop_one(i: ti.i32):
    if pos_f[i][1] > -1e5:
        _drop(i)


@ti.kernel
def _k_wake_near(n: ti.i32):
    """Deleting a block is not a wake event: a sleeper resting on it is never re-checked
    and hangs in mid-air. Wake every non-frozen block within touching range (+0.5) of a
    deleted one, and keep it from sleeping for GRACE_T."""
    m = ti.min(cnt[2], DEL_MAX)
    for i in range(n):
        if pos_f[i][1] > -1e5 and nowake_f[i] == 0:
            p = pos_f[i]; hi = _hmax(i)
            for d in range(m):
                q = del_p[d]
                dv = ti.Vector([q[0] - p[0], q[1] - p[1], q[2] - p[2]])
                if ti.static(PERIODIC):
                    dv[2] -= PERIOD * ti.round(dv[2] / PERIOD)
                if dv.norm() < hi + q[3] + 0.5:
                    sleep_f[i] = 0; nosleep_f[i] = 1; s_grace[i] = GRACE_T
                    break


@ti.kernel
def _k_swap_scene(n0: ti.i32):
    """Replay solver._cull_swap (same removal list, same order) on the scene fields."""
    nr = fields.cull_n_remove[None]
    nn = n0
    ti.loop_config(serialize=True)
    for k in range(nr):
        ri = fields.cull_remove[nr - 1 - k]
        last = nn - 1
        if ri < last:
            s_c[ri] = s_c[last]; s_line[ri] = s_line[last]; s_kcol[ri] = s_kcol[last]
            s_ylo[ri] = s_ylo[last]; s_spawn[ri] = s_spawn[last]; s_cont[ri] = s_cont[last]
            s_prev[ri] = s_prev[last]; s_in[ri] = s_in[last]; s_grace[ri] = s_grace[last]
            s_wkey[ri] = s_wkey[last]; s_hok[ri] = s_hok[last]; s_loose[ri] = s_loose[last]
        nn -= 1


@ti.kernel
def _k_off_shelf(n: ti.i32, u_edge: ti.f32):
    """A block whose centre is past the shelf edge AND below the shelf surface has fallen
    off: the shelf ignores it from now on (fields.noslope_f). Otherwise debris sliding
    back under the shelf would find itself deep inside the plane and be launched."""
    gn = ti.Vector(_GNL); pv = ti.Vector(_PVL); td = ti.Vector(_TDL)
    for i in range(n):
        d = pos_f[i] - pv
        if d.dot(td) < u_edge and d.dot(gn) < 0.0:
            fields.noslope_f[i] = 1


@ti.kernel
def _k_snap_hold(n: ti.i32):
    for i in range(n):
        s_hok[i] = 1 if nowake_f[i] == 0 else 0      # not frozen = loose right now


@ti.kernel
def _k_fallen(n: ti.i32):
    cnt[3] = 0
    for i in range(n):
        if _disp(i) > 2.0 * HE:
            ti.atomic_add(cnt[3], 1)


# ---- spawning: random guillotine tiling of each layer's face, on the GPU ----
@ti.func
def _split3(j, i, m):
    """Cut box i of job j across its longest side at a random 35-65% (never leaving a
    side under LMIN); the second part goes to slot m."""
    b = t_box[j, i]
    ly = b[1] - b[0]; lz = b[3] - b[2]; lx = b[5] - b[4]
    L = ti.max(ly, ti.max(lz, lx))
    lo = ti.max(0.35, LMIN / L); hi = ti.min(0.65, 1.0 - LMIN / L)
    f = lo + (hi - lo) * ti.random()
    a = b; c2 = b
    if ly >= lz and ly >= lx:
        a[1] = b[0] + f * ly; c2[0] = a[1]
    elif lz >= lx:
        a[3] = b[2] + f * lz; c2[2] = a[3]
    else:
        a[5] = b[4] + f * lx; c2[4] = a[5]
    t_box[j, i] = a
    t_box[j, m] = c2


@ti.func
def _maxlen3(b):
    return ti.max(b[1] - b[0], ti.max(b[3] - b[2], b[5] - b[4]))


@ti.func
def _vol3(b):
    return (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4])


@ti.func
def _vmax3(j, m):
    v = 0.0
    for i in range(m):
        b = t_box[j, i]
        if _maxlen3(b) >= 2.0 * LMIN:
            v = ti.max(v, _vol3(b))
    return v


@ti.kernel
def _k_tile3(nj: ti.i32):
    """One thread per strip batch: the region (a bedding strip y0..y1, a ring in z, depth
    x0..x1) is cut into t_tgt boxes. The ring is opened at a random z (no seam); then every
    piece longer than LMAX in any direction is cut (queue order); then pieces picked at
    random weighted by volume are cut across their longest side, until the target count.
    Depth is cut like any other direction, so blocks span 1 or 2 nominal layers and their
    depth faces are staggered -- no layer is a curtain."""
    for j in range(nj):
        jb = t_job[j]
        z0 = 0.0                                   # side barriers: no block straddles them
        if ti.static(PERIODIC):
            z0 = ti.random() * PERIOD              # periodic: open the ring anywhere (no seam)
        t_box[j, 0] = ti.Vector([jb[0], jb[1], z0, z0 + PERIOD, jb[2], jb[3]])
        m = 1
        tgt = ti.min(t_tgt[j], MAXB)
        k = 0
        while k < m and m < tgt:
            if _maxlen3(t_box[j, k]) > LMAX:
                _split3(j, k, m); m += 1
            else:
                k += 1
        vmax = _vmax3(j, m)
        since = 0
        while m < tgt and vmax > 0.0:
            i = ti.min(int(ti.random() * m), m - 1)
            b = t_box[j, i]
            if _maxlen3(b) >= 2.0 * LMIN and ti.random() * vmax < _vol3(b):
                _split3(j, i, m); m += 1; since += 1
                if since >= 32:
                    vmax = _vmax3(j, m); since = 0
        t_m[j] = m


_NZC = max(1, int(round(PERIOD / NOISE_LZ)))   # noise cells across the period (z wraps), octave 0
_FBM_NORM = 1.0 / float(np.sqrt(sum(NOISE_ROUGH ** (2 * o) for o in range(NOISE_OCT))))
_SEEDH = (NOISE_SEED * 1442695041) & 0x7fffffff


@ti.func
def _hash2(ix, iz):
    h = (ix * 374761393 + iz * 668265263 + _SEEDH) & 0x7fffffff
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7fffffff
    return h ^ (h >> 16)


@ti.func
def _grad(ix, iz, fx, fz):
    a = (_hash2(ix, iz) & 0xffff) * (6.283185307 / 65536.0)
    return ti.cos(a) * fx + ti.sin(a) * fz


@ti.func
def _perlin2(x, z, nzc):
    """2D gradient (Perlin) noise, periodic in z with nzc cells; roughly in +-0.7."""
    x0 = int(ti.floor(x)); z0 = int(ti.floor(z))
    fx = x - x0; fz = z - z0
    ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0)
    za = ((z0 % nzc) + nzc) % nzc; zb = (za + 1) % nzc
    n00 = _grad(x0, za, fx, fz);       n10 = _grad(x0 + 1, za, fx - 1.0, fz)
    n01 = _grad(x0, zb, fx, fz - 1.0); n11 = _grad(x0 + 1, zb, fx - 1.0, fz - 1.0)
    nx0 = n00 + ux * (n10 - n00); nx1 = n01 + ux * (n11 - n01)
    return nx0 + uz * (nx1 - nx0)


@ti.func
def _fbm2(x, z):
    """NOISE_OCT octaves of _perlin2, octave o at 2^o the frequency (still periodic in z:
    _NZC * 2^o cells) and NOISE_ROUGH^o the amplitude; normalised so the spread matches
    one octave. x in octave-0 cells, z in [0, _NZC)."""
    acc = 0.0; amp = 1.0; f = 1.0; nzc = _NZC
    for o in ti.static(range(NOISE_OCT)):
        acc += amp * _perlin2(x * f + 17.0 * o, z * f, nzc)
        amp *= NOISE_ROUGH; f *= 2.0; nzc *= 2
    return acc * _FBM_NORM


@ti.kernel
def _k_stage3(jj: ti.i32, off: ti.i32, k: ti.i32, base: ti.i32, mref: ti.f32):
    """Boxes off..off+k of job jj -> the solver's spawn staging (sp_*) + this scene's fields for
    blocks base+off.. . GAP between neighbours in every direction. A block's nominal layer
    is the one its depth centre is in."""
    Rm = ti.Matrix(_RZL); pv = ti.Vector(_PVL)
    for q in range(k):
        r = t_box[jj, off + q]
        y = 0.5 * (r[0] + r[1])
        zb = 0.5 * (r[2] + r[3])
        zb = zb - PERIOD * ti.floor(zb / PERIOD)      # wrap the centre into [0, P): seam-straddling blocks
        xc = 0.5 * (r[4] + r[5])
        c = int(ti.floor(xc / DPITCH))                # nominal layer
        h = ti.Vector([0.5 * (r[5] - r[4]) - 0.5 * GAP, 0.5 * (r[1] - r[0]) - 0.5 * GAP,
                       0.5 * (r[3] - r[2]) - 0.5 * GAP])
        pw = Rm @ ti.Vector([xc, y, zb - 0.5 * PERIOD]) + pv
        fields.sp_pos[q] = pw; fields.sp_he[q] = h; fields.sp_R[q] = Rm
        m = ti.max(MASS_FLOOR, 8.0 * h[0] * ti.sqrt(h[1] * h[2]))   # ~ depth x sqrt(face area), floored
        Il = ti.Vector([3.0 / (m * (h[1] * h[1] + h[2] * h[2])),
                        3.0 / (m * (h[0] * h[0] + h[2] * h[2])),
                        3.0 / (m * (h[0] * h[0] + h[1] * h[1]))])
        fields.sp_mass[q] = m; fields.sp_Iil[q] = Il
        Imn = (1.0 / Il[0] + 1.0 / Il[1] + 1.0 / Il[2]) / 3.0
        ms = ti.max(1.0, (mref / m) ** (2.0 / 3.0))
        fields.sp_rs[q] = ti.Vector([ti.min(10.0, ti.max(1.0, Imn * Il[0])) * ms,
                                     ti.min(10.0, ti.max(1.0, Imn * Il[1])) * ms,
                                     ti.min(10.0, ti.max(1.0, Imn * Il[2])) * ms])
        j = base + off + q
        row = ti.min(ti.max(int(ti.floor(y / ROW_H)), 0), ROWS - 1)
        col = ti.min(ti.max(int(ti.floor(zb / PITCH)), 0), WIDTH - 1)
        s_c[j] = c; s_line[j] = row * WIDTH + col; s_kcol[j] = col; s_ylo[j] = r[0]
        s_spawn[j] = pw; s_cont[j] = 0.0; s_prev[j] = 1; s_in[j] = 1; s_grace[j] = 0.0
        yh = ti.min(ti.max(y / (ROWS * ROW_H), 0.0), 1.0)
        nn = _fbm2(c / NOISE_LC, zb / PERIOD * _NZC) / FBM_STD
        gouge = ti.max(0.0, -nn - 1.0)
        s_wkey[j] = c + PEEL * (1.0 - yh) + NOISE_BASE * nn - NOISE_GOUGE * gouge * gouge
        s_hok[j] = 0; s_loose[j] = 0
    fields.sp_n[None] = k


class WallScene:
    def __init__(self):
        self.cm = Solver(max_he=HE_MAX,
                               slope_normal=tuple(GROUND_N.tolist()),
                               slope_d=float(GROUND_N @ PIVOT),
                               slope_mu_y_range=(-1e6, 1e6), slope_feng=True)
        self.cm.spawn_box(pos=(0, 1000, 0), half_extents=(1, 1, 1))
        self.cm.snapshot_ke(); self.cm.step(1e-4)
        self.cm.get_instance_data(); self.cm.cull(despawn_y=500.0)
        self._out = np.zeros(2 * HWIN + 5, dtype=np.int32)
        self.reset()

    def reset(self):
        self.cm.clear()
        self.sim_time = 0.0
        self.back = -1                 # deepest layer spawned
        self.woken_upto = -1           # layers <= this have been undermined
        self.t_advance = 0.0           # sim time the face last advanced (or a stall undermine)
        self.hold = False              # H: no deeper layer comes loose, no undermining
        self.hold_front = 0.0          # the wake front when H was pressed
        self.front = 0.0               # wake front F: slides toward `target` at 1 / PEEL_WAIT layers/s
        self.target = 1.0              # +1 layer per peel (auto-paced, or P); starts one peel in
        self.mlayer = -1               # MANUAL_LAYERS: deepest layer woken by P (-1: all asleep)
        self.budget = 0.0              # rate mode: blocks owed to the wake
        self.extra = 0.0               # rate mode: blocks added by P (not capped)
        self._kh = np.zeros(KEY_BINS, dtype=np.int32)
        self.olayer = -1               # layer being woken row by row by O (-1: none)
        self.ocut = 0.0                # O: blocks of olayer with their top above this are awake
        if MANUAL_LAYERS:
            self.front = -1e9; self.target = 0.0
        self.manual = False            # a P peel is sliding in (it slides even while held)
        self.t_peel = 0.0              # sim time of the last peel
        self.t_unfreeze = 0.0          # no peeling before this (a free-fall burst froze it)
        self.n_freefall = 0
        from collections import deque
        self.ff_hist = deque()         # (dt, free-fall count) per tick, last FREEFALL_AVG_T s
        self.ff_sum = 0.0; self.ff_t = 0.0
        self.ff_avg = 0.0
        self.removed = 0
        self.despawned = 0
        self.shelf_u = -1e30           # shelf front edge, wall-frame x (only recedes)
        self.n_alive = 0; self.n_keep = 0
        self.hist = np.zeros(HWIN, dtype=np.int64)
        fields.slope_t_f[None] = T_DIR.tolist(); fields.slope_o_f[None] = PIVOT.tolist()
        fields.slope_umin_f[None] = -1e30
        if PERIODIC:
            # the period must be a whole number (>= 3) of grid cells: stretch the cell
            fields.side_on[None] = 0
            fields.grid_cell_f[None] = float(PERIOD / NZ)
            fields.per_L[None] = float(PERIOD); fields.per_nz[None] = NZ; fields.per_on[None] = 1
            self.cm.invalidate_grid()
        else:
            fields.per_on[None] = 0
            fields.side_w[None] = float(SIDE_W); fields.side_on[None] = 1
        self.capped = False
        self.t_blocked = None          # sim time a spawn first found its slab occupied
        self.n_tiles = 0               # periodic rendering: copies either side (set by the window)
        self.next_despawn = 0.0
        # bedding: the wall's height cut once into strips (fixed for the whole wall), each
        # filled into the wall in BATCH_LAYERS-deep batches whose depth boundaries are
        # staggered strip to strip (first batch: a random fraction of one)
        rng = np.random.default_rng()
        beds = [(0.0, ROWS * ROW_H)]
        while any(b1 - b0 > LMAX for b0, b1 in beds):
            i = next(i for i, (b0, b1) in enumerate(beds) if b1 - b0 > LMAX)
            b0, b1 = beds[i]; hgt = b1 - b0
            f = rng.uniform(max(0.35, LMIN / hgt), min(0.65, 1.0 - LMIN / hgt))
            beds[i:i + 1] = [(b0, b0 + f * hgt), (b0 + f * hgt, b1)]
        self.beds = np.array(beds)
        self.fill = np.zeros(len(beds))                  # depth filled so far, per strip
        self.first = rng.uniform(0.3, 1.0, len(beds)) * BATCH_LAYERS * DPITCH   # first batch depth
        self.blocked = np.full(len(beds), -1.0)          # sim time a strip's batch found debris (-1: no)
        while self.spawn_to(DEPTH0 * DPITCH):
            pass

    # ---- spawning -----------------------------------------------------------------------
    def spawn_to(self, depth):
        """One batch for every strip whose filled depth is short of `depth` (wall-frame x),
        each into clear space only (debris in its region makes it wait; after SPAWN_WAIT
        the debris is deleted -- it sits behind the BEHIND frozen layers, unseen). Updates
        `back`, the deepest nominal layer filled across every strip. Returns True if
        anything was spawned."""
        todo = np.where(self.fill < depth - 1e-6)[0][:JOB_SLOTS]
        if len(todo) == 0:
            return False
        n0 = n_f[None]
        jobs = []
        for s_ in todo:
            x0 = self.fill[s_]
            x1 = x0 + (self.first[s_] if x0 == 0.0 else BATCH_LAYERS * DPITCH)
            y0, y1 = self.beds[s_]
            if n0 and _k_spawn_clear(n0, x0, x1, y0, y1, 0) > 0:
                if self.blocked[s_] < 0:
                    self.blocked[s_] = self.sim_time
                if self.sim_time - self.blocked[s_] < SPAWN_WAIT:
                    continue
                _k_spawn_clear(n0, x0, x1, y0, y1, 1)
                self._clicked = True                   # cull them at the next tick
            self.blocked[s_] = -1.0
            tgt = max(1, int(round((y1 - y0) * PERIOD * (x1 - x0) * DENS)))
            jobs.append((s_, y0, y1, x0, x1, tgt))
        if not jobs:
            return False
        self.capped = n0 + sum(j[5] for j in jobs) > MAX_N - 64   # re-checked: deletions free room
        if self.capped:
            return False
        t_job.from_numpy(np.array([[j[1], j[2], j[3], j[4]] for j in jobs]
                                  + [[0, 0, 0, 0]] * (JOB_SLOTS - len(jobs)), dtype=np.float32))
        t_tgt.from_numpy(np.array([j[5] for j in jobs] + [0] * (JOB_SLOTS - len(jobs)), dtype=np.int32))
        _k_tile3(len(jobs))
        tm = t_m.to_numpy()
        base = n0
        mref = float(params.ROLLING_M_REF)
        for jj, j in enumerate(jobs):
            m = int(tm[jj])
            for off in range(0, m, fields.SPAWN_MAX):
                k = min(fields.SPAWN_MAX, m - off)
                _k_stage3(jj, off, k, base, mref)
                solver._spawn_commit(base + off)
            base += m
            self.fill[j[0]] = j[4]
        n_f[None] = base
        _k_settle(n0, base)
        self.cm.invalidate_grid()
        self.back = int(np.floor(self.fill.min() / DPITCH)) - 1
        return True

    # ---- the face -------------------------------------------------------------------------
    def face_pass(self, dt):
        n = n_f[None]
        _k_face_reset()
        if n:
            _k_inwall(dt, n); _k_second(n); _k_third(n)
            fm = self.face_quantile(0.5)
            _k_classify(dt, n, abs(float(params.GRAVITY)),
                        self.front,
                        fm if fm is not None else self.back + 1,
                        1 if self.hold else 0, self.hold_front,
                        1 if MANUAL_LAYERS else 0, self.mlayer,
                        self.olayer, float(self.ocut))
        _k_lines(self.back, self._out)                 # the one per-frame readback
        self.hist = self._out[:HWIN].astype(np.int64)
        self.n_alive = int(self._out[HWIN]); self.n_keep = int(self._out[HWIN + 1])
        self.n_freefall = int(self._out[HWIN + 2])
        self.hist_up = self._out[HWIN + 3:2 * HWIN + 3].astype(np.int64)
        self.n_face = int(self._out[2 * HWIN + 3]); self.n_face_awake = int(self._out[2 * HWIN + 4])

    def face_quantile(self, q):
        """Face layer at quantile q over the lines still standing (None if none stand)."""
        n = self.n_alive
        if n == 0:
            return None
        k = min(n - 1, int(np.floor(q * (n - 1))))
        return self.back + 2 - HWIN + int(np.searchsorted(np.cumsum(self.hist), k + 1))

    def set_hold(self, on):
        """Hold the wake: the loose window stops where it is and undermining (incl. the
        stall timer) stops; what is already moving keeps falling and settling."""
        self.hold = on
        if on:                                    # snapshot what is loose now
            _k_snap_hold(n_f[None])
            self.hold_front = self.front
        self.t_advance = self.sim_time            # releasing does not fire the stall timer at once

    def wake_layer(self):
        """MANUAL_LAYERS: wake the next whole layer (never within BEHIND of the deepest spawned one);
        a layer O has started is finished first."""
        if self.olayer >= 0:
            self.mlayer = self.olayer; self.olayer = -1
        elif self.mlayer + 1 <= self.back - BEHIND:    # keep BEHIND frozen layers before the spawn slots
            self.mlayer += 1
        self.target = float(self.mlayer + 1)           # spawning paces behind it
        return self.mlayer

    def wake_row(self):
        """MANUAL_LAYERS: wake the next 2 x 2 strip -- one row band (ROW_H) of the next
        layer, from the top down, full width. Returns (layer, cut height)."""
        if self.olayer < 0:
            if self.mlayer + 1 > self.back - BEHIND:
                return self.mlayer, 0.0
            self.olayer = self.mlayer + 1
            self.ocut = ROWS * ROW_H
            self.target = float(self.olayer + 1)       # spawning paces behind it
        self.ocut -= ROW_H
        L, cut = self.olayer, self.ocut
        if self.ocut <= 1e-6:                          # down to the ground: the layer is done
            self.mlayer = self.olayer; self.olayer = -1
        return L, cut

    def peel_face(self):
        """Rate mode, P: advance the wake by one full wall face -- NB blocks, released in
        noise-key order at the next frame (also while held)."""
        self.extra += NB
        return NB

    def peel_one(self):
        """One manual step: hold the wake, let one more layer come loose, and undermine
        one block at or in front of the face (deleted at the next tick's flush)."""
        if not self.hold:
            self.set_hold(True)
        if self.target + 1.0 <= self.back - BEHIND:    # keep BEHIND frozen layers before the spawn slots
            self.target += 1.0
            self.t_peel = self.sim_time
            self.manual = True
        med = self.face_median()
        face = int(np.round(med)) if med is not None else self.back + 1
        d0 = self._deleted()
        _k_mark_undermine(n_f[None], 0, face)
        self.removed += self._deleted() - d0
        self._peeled = True
        return self.target

    def delete_picked(self, o, d, k):
        """Delete the block the ray o + t d hits first (k periodic copies either side).
        Returns its index, or -1. Culled at the next tick, its neighbours woken."""
        n = n_f[None]
        if n == 0:
            return -1
        _k_pick(n, ti.Vector(list(map(float, o))), ti.Vector(list(map(float, d))), int(k), float(PERIOD))
        i = int(pick_i[None])
        if i >= 0:
            _k_drop_one(i)
            self._clicked = True
        return i

    def face_median(self):
        """Median face layer over the lines still standing (None if none stand)."""
        n = self.n_alive
        if n == 0:
            return None
        cum = np.cumsum(self.hist)
        base = self.back + 2 - HWIN
        lo = base + int(np.searchsorted(cum, (n - 1) // 2 + 1))
        hi = base + int(np.searchsorted(cum, n // 2 + 1))
        return 0.5 * (lo + hi)

    # ---- deleting -----------------------------------------------------------------------
    def _deleted(self):
        return int(cnt[2])

    def flush_deletes(self):
        """Cull the dropped blocks; replay the solver's swaps on the scene fields."""
        nd = self._deleted()
        if nd == 0:
            return False
        n0 = n_f[None]
        _k_wake_near(n0)
        solver._cull_mark(-1e5)
        _k_swap_scene(n0)
        solver._cull_swap()
        self.cm.invalidate_grid()
        cnt[2] = 0
        return True

    def sabotage(self):
        """A depth layer is undermined when the wall's face (median over the lines still
        standing) reaches it: ONE random block of its bottom UNDERMINE row bands, still
        in place, is deleted, once; and every STALL s the face does not advance, one more."""
        med = self.face_median()
        if med is None:
            return False
        face = int(np.round(med))
        if face <= self.woken_upto:
            # stalled: every STALL s without the face advancing, one more block -- from
            # the face layer or the steps standing in front of it (awake, so they fall)
            if self.sim_time - self.t_advance < STALL:
                return False
            self.t_advance = self.sim_time
            d0 = self._deleted()
            _k_mark_undermine(n_f[None], 0, face)
            self.removed += self._deleted() - d0
            return True
        d0 = self._deleted()
        for c in range(self.woken_upto + 1, face + 1):
            _k_mark_undermine(n_f[None], c, c)
        self.woken_upto = face
        self.t_advance = self.sim_time
        self.removed += self._deleted() - d0
        return True

    def shelf(self, dt):
        """The shelf edge's target is SHELF edges in front of the frontmost layer still
        standing (the SHELF_Q quantile of the face over the lines still standing from row
        band SHELF_ROW0 up -- not the minimum, and not the foot, where leftover stubs of a
        collapsed layer stay "in the wall" and would pin the shelf forever); the edge recedes
        toward it at SHELF_SPEED, never jumps, never advances. Every DESPAWN_EVERY,
        delete what is DROP_MAX below the shelf."""
        hu = getattr(self, "hist_up", None)
        n_up = int(hu.sum()) if hu is not None else 0
        if n_up:
            k = min(n_up - 1, int(np.floor(SHELF_Q * (n_up - 1))))
            front = self.back + 2 - HWIN + int(np.searchsorted(np.cumsum(hu), k + 1))
        else:
            front = self.back + 1
        u_tgt = front * DPITCH - SHELF * 2.0 * HE
        if self.shelf_u < -1e29:
            self.shelf_u = u_tgt                       # start where it belongs
        elif self.shelf_u < u_tgt:
            v = max(SHELF_SPEED, (u_tgt - self.shelf_u) / SHELF_CATCH)   # catch up when the wall outpaces it
            self.shelf_u = min(u_tgt, self.shelf_u + v * dt)
        fields.slope_umin_f[None] = float(self.shelf_u)
        _k_off_shelf(n_f[None], float(self.shelf_u))
        if self.sim_time < self.next_despawn:     # deletions are batched
            return False
        self.next_despawn = self.sim_time + DESPAWN_EVERY
        d0 = self._deleted()
        _k_mark_below(n_f[None])
        nd = self._deleted() - d0
        self.despawned += nd
        return nd > 0

    def pace(self):
        """Keep BEHIND *sleeping* layers behind the face's awake and free layers -- and
        behind the wake front, which runs on its own clock and can be ahead of the
        material that has actually fallen. Up to SPAWN_PER_FRAME layers a frame."""
        med = self.face_median()
        face = int(np.ceil(med)) if med is not None else self.back + 1   # every line down: regrow
        # ... and far enough behind the wake target that the next peel (which needs
        # target + 1 <= back - BEHIND) is never refused: back >= ceil(target) + 1 + BEHIND
        face = max(face, int(np.ceil(self.target)) + 1 - (AWAKE_LAYERS + FREE_LAYERS - 1))
        need = face + AWAKE_LAYERS + FREE_LAYERS - 1 + BEHIND     # back must reach this layer
        if self.back < need:
            self.spawn_to((need + 1) * DPITCH)

    # ---- stepping -----------------------------------------------------------------------
    def advance(self, dt):
        self.tick(dt)
        self.physics(dt)

    def wake_rate(self, dt):
        """Rate mode: move the front over the next WAKE_BLOCKS * dt blocks in key order
        (fractional budget carried), never within BEHIND of the deepest spawned layer. P adds
        a whole face."""
        self.wake_bps = WAKE_BLOCKS
        if not self.hold:                             # constant rate
            self.budget += WAKE_BLOCKS * dt
        self.budget += self.extra; self.extra = 0.0          # P: a whole face's worth, now
        cap = float(self.back - BEHIND)
        n = n_f[None]
        while self.budget >= 1.0 and self.front < cap and n:
            _k_key_hist(n, self.front, KEY_WIN, self._kh)
            cum = np.cumsum(self._kh)
            if cum[-1] == 0:                          # nothing in the window: skip ahead
                self.front = min(cap, self.front + KEY_WIN)
                continue
            k = int(np.searchsorted(cum, self.budget))   # first bin reaching the budget
            if k >= KEY_BINS:                         # the whole window is within budget
                self.front = min(cap, self.front + KEY_WIN); self.budget -= cum[-1]
                continue
            self.front = min(cap, self.front + (k + 1) * KEY_WIN / KEY_BINS)
            self.budget -= cum[k]
            break
        # an overshoot (a bin holds whole blocks) is a debt carried to the next frames;
        # no backlog burst after the front was capped (a P that the cap blocked waits)
        if self.front >= cap and self.budget > WAKE_BLOCKS:
            self.extra = self.budget - WAKE_BLOCKS
        self.budget = min(self.budget, WAKE_BLOCKS)
        self.target = self.front                      # spawning paces behind the front

    def tick(self, dt):
        self.sim_time += dt
        if WAKE_MODE == "rate" and not MANUAL_LAYERS:
            self.wake_rate(dt)
        # pace the collapse: peel one layer when fewer than FREEFALL_MAX blocks are in
        # free fall (last pass) and PEEL_WAIT has passed since the previous peel
        # free fall over FREEFALL_BURST freezes peeling for BURST_FREEZE s
        # rolling, time-weighted average of the free-fall count
        self.ff_hist.append((dt, self.n_freefall))
        self.ff_sum += dt * self.n_freefall; self.ff_t += dt
        while self.ff_hist and self.ff_t - self.ff_hist[0][0] >= FREEFALL_AVG_T:
            d0, n0 = self.ff_hist.popleft()
            self.ff_sum -= d0 * n0; self.ff_t -= d0
        self.ff_avg = self.ff_sum / self.ff_t if self.ff_t > 0 else 0.0
        if self.n_freefall > FREEFALL_BURST:
            self.t_unfreeze = self.sim_time + BURST_FREEZE
        frozen = (self.hold or self.sim_time < self.t_unfreeze or MANUAL_LAYERS
                  or WAKE_MODE == "rate")
        if (not frozen and self.ff_avg < FREEFALL_MAX
                and self.sim_time - self.t_peel >= PEEL_WAIT
                and self.front >= self.target - 1e-6
                and self.target + 1.0 <= self.back - BEHIND):   # keep BEHIND frozen layers before the spawn slots
            self.target += 1.0
            self.t_peel = self.sim_time
        # a peel is not a jump: the front sweeps the noise map toward the target, one layer
        # per PEEL_WAIT, so blocks wake a few per frame as it passes their keys. It stops
        # while held or frozen by a free-fall burst.
        burst = self.sim_time < self.t_unfreeze
        if not MANUAL_LAYERS and WAKE_MODE != "rate" and not burst and (not self.hold or self.manual):       # P slides even while held
            # slower the more is falling, and the more fully woken layers still stand: a
            # layer is woken to the ground once F passes it by PEEL; the median face is the
            # frontmost layer still standing
            med = self.face_median()
            standing = max(0.0, self.front - PEEL - med + 1.0) if med is not None else 0.0
            rate = (SWEEP_RATE * max(0.0, 1.0 - self.ff_avg / FREEFALL_BURST)
                    * max(0.0, 1.0 - standing / STANDING_MAX))
            self.front = min(self.target, self.front + rate * dt)
            if self.front >= self.target - 1e-6:
                self.manual = False
        n = n_f[None]
        if PERIODIC and n:
            _k_wrap(float(PERIOD), n)
            self.cm.invalidate_grid(pairs_valid=True)   # a wrap by one period moves no pair
        self.face_pass(dt)
        marked = False if self.hold else self.sabotage()
        marked = getattr(self, "_peeled", False) or getattr(self, "_clicked", False) or marked
        self._peeled = False; self._clicked = False
        marked = self.shelf(dt) or marked
        if marked and self.flush_deletes():       # indices moved: rewrite the flags
            self.face_pass(0.0)
        self.pace()

    def physics(self, dt):
        self.cm.snapshot_ke()
        for _ in range(SUBSTEPS):
            self.cm.step(dt / SUBSTEPS)

    def stats(self):
        n = n_f[None]
        _k_fallen(n)
        first = l_first.to_numpy()
        alive = first < BIG
        f = first[alive] if alive.any() else np.array([self.back + 1])
        return dict(n=n, awake=int((sleep_f.to_numpy()[:n] == 0).sum()),
                    kept=self.n_keep, fallen=int(cnt[3]),
                    removed=self.removed, sabotaged=self.woken_upto + 1,
                    despawned=self.despawned, shelf=self.shelf_u,
                    sleeping_layers=int(self.back - (int(np.ceil(f.mean())) + AWAKE_LAYERS
                                                      + FREE_LAYERS - 1)),
                    lines=int(alive.sum()),
                    face_min=int(f.min()), face_med=float(np.median(f)),
                    face_max=int(f.max()), back=self.back, capped=self.capped)

    def tiles(self, aspect):
        """How many periods either side of the base copy cover the window width."""
        if not PERIODIC:
            return 0
        return int(np.ceil(VIEW_SCALE * aspect * 1.6 / PERIOD)) + 1

    def view_center(self):
        """Middle of the shelf, between its edge and the closest sleeping layer, at mid
        wall height -- the face recedes, the camera follows."""
        u_face = self.shelf_u + SHELF * 2.0 * HE if self.shelf_u > -1e29 else 0.0
        u = 0.5 * (max(self.shelf_u, -SHELF * 2.0 * HE) + u_face)
        return PIVOT + T_DIR * u + np.array([0.0, 0.45 * _height, 0.0])

    def ground_outline(self):
        """The shelf: from its front edge back past the wall, as wide as the wall + margin."""
        u0 = max(self.shelf_u, -SHELF * 2.0 * HE)
        u1 = (self.back + 2) * DPITCH
        w = (self.n_tiles + 0.5) * PERIOD if PERIODIC else 0.5 * WIDTH * PITCH + 4.0 * HE
        sq = np.array([[u0, 0, -w], [u1, 0, -w], [u1, 0, w], [u0, 0, w]])
        q = sq @ RZ.T + PIVOT
        segs = [[q[i], q[(i + 1) % 4]] for i in range(4)]
        # the two side barriers, as rectangles in the planes z = +-SIDE_W
        for zs in (() if PERIODIC else (-SIDE_W, SIDE_W)):   # periodic: no seams drawn
            r = np.array([[u0, 0, zs], [u1, 0, zs], [u1, _height, zs], [u0, _height, zs]]) @ RZ.T + PIVOT
            segs += [[r[i], r[(i + 1) % 4]] for i in range(4)]
        return np.array(segs)


def _fmt(st, on):
    return (f"blocks={st['n']:5d} awake={st['awake']:4d} kept-awake={st['kept']:4d} "
            f"undermined layers={st['sabotaged']} (cubes removed {st['removed']})  "
            f"shelf edge x={st['shelf']:.1f} despawned={st['despawned']}  "
            f"sleeping layers behind face={st['sleeping_layers']}  "
            f"fallen={st['fallen']:5d}  lines standing={st['lines']:3d}/{ROWS * WIDTH}  "
            f"face layer min/med/max={st['face_min']}/"
            f"{st['face_med']:.1f}/{st['face_max']}  back layer={st['back']}"
            f"{'  SPAWN CAPPED (MAX_N)' if st['capped'] else ''}  "
            f"static friction {'ON' if on else 'OFF'}")


def _header():
    return (f"rows={ROWS} width={WIDTH} ({NB} blocks/layer, face sides {LMIN:.2f}-{LMAX:.2f}, depth {2 * HE:g}) depth0={DEPTH0} "
            f"{'periodic z, period ' + format(PERIOD, '.2f') + ' (' + str(NZ) + ' cells)' if PERIODIC else 'side barriers'}  "
            f"lean={LEAN:g} deg mu_s={MU_S} mu_k={MU_K} undermine={UNDERMINE} rows behind={BEHIND} ground mu={GROUND_MU} "
            f"sub={SUBSTEPS} g={params.GRAVITY:.0f}")


def check():
    T = float(os.environ.get("WALL_T", "60"))
    on = "--off" not in sys.argv
    set_friction(on)
    sc = WallScene()
    print(f"[wall --check] {_header()} T={T}s")
    t0 = _time.perf_counter(); k = 0
    while sc.sim_time < T:
        sc.advance(DT); k += 1
        if k % int(5 / DT) == 0:
            print(f"t={sc.sim_time:6.1f}  " + _fmt(sc.stats(), on))
    print("  params: " + _params())
    print(f"wall {(_time.perf_counter() - t0):.1f}s for {T:.0f}s sim")


def run_window(scene_cls=None, title=None):
    """The wall window. scene_cls: a WallScene subclass -- same
    keys, rendering and capture over another solver."""
    title_ = title
    import moderngl_window as mglw
    import taichi as ti
    from render import Renderer
    from shading import make_lut, motion_data

    CAPTURE = "--capture" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--capture"]  # moderngl-window parses argv too
    cap = None
    if CAPTURE:
        from pathlib import Path
        import capture
        cap = capture.WindowCapture("wall", Path(__file__).resolve().parent, _T_LAUNCH)
        CAP_T = float(os.environ.get("WALL_CAP_T", "30"))

    class WallWindow(mglw.WindowConfig):
        title = title_ or "wall — face erosion"
        gl_version = (3, 3)
        window_size = (scene_config.WINDOW_W, scene_config.WINDOW_H)
        aspect_ratio = None
        resizable = not CAPTURE        # a capture's frame size is fixed at the first frame

        def __init__(self, **kw):
            super().__init__(**kw)
            self.friction = True
            set_friction(True)
            self.scene = (scene_cls or WallScene)()
            scene_config.ISO_SCALE = VIEW_SCALE
            self.renderer = Renderer(self.ctx, LOOK_AT, slope_normal=(1.0, 0.0, 0.0),
                                     eye_dir=EYE_DIR)
            self.renderer.enable_lut_shading(make_lut(os.environ.get("WALL_LUT", "lights"),
                                                       eye=EYE_DIR, top_n=RZ @ np.array([0.0, 1.0, 0.0])))
            self.renderer.enable_motion_blur(shutter=SHUTTER)
            self.paused = False
            self.layer_colors = LAYER_COLORS
            self.yaw = 0.0; self._turn = 0                    # camera orbit (LEFT / RIGHT)
            self.renderer.persp_fov = WALL_FOV or None
            self._prof = {k: 0.0 for k in ("physics", "cull", "spawn", "geom", "render", "total")}
            self._prof_n = 0
            info(f"Wall: {_header()}  view half-height={VIEW_SCALE:.1f}")
            STATUS.show_controls(lambda: (
                f"F   = static friction {onoff(self.friction)}\n"
                f"B   = motion blur {onoff(self.renderer.blur)}\n"
                "W   = shaded/wireframe\n"
                "H   = hold\n"
                "P   = peel one layer\n"
                "SPC = pause\n"
                "R   = reset\n"
                "ESC = quit"))

        def on_render(self, time, frametime):
            STATUS.poll()
            if self._turn:                                       # orbit the camera horizontally
                self.yaw = float(np.clip(self.yaw + self._turn * YAW_RATE * frametime, -YAW_MAX, YAW_MAX))
                a = np.radians(self.yaw); e = np.asarray(EYE_DIR, float)
                self.renderer._eye_dir = (e[0] * np.cos(a) - e[2] * np.sin(a), e[1],
                                          e[0] * np.sin(a) + e[2] * np.cos(a))
            t_frame = _time.perf_counter()
            if not self.paused:
                dt = min(frametime, DT)
                t0 = _time.perf_counter()
                self.scene.physics(dt)
                ti.sync()
                t1 = _time.perf_counter()
                t2 = t1
                self.scene.tick(dt)                 # face tracking + paced spawning
                t3 = _time.perf_counter()
            inst, count = self.scene.cm.get_instance_data()
            mot = motion_data(count) if count else None     # velocity, omega per block
            # periodic space: the base period (positions already wrapped on the GPU) drawn
            # 2k+1 times, offset by whole periods -- on the GPU, no copies uploaded
            if PERIODIC:
                # seen at an angle the periodic wall spans more of the window: more copies
                k = self.scene.tiles(self.wnd.width / max(self.wnd.height, 1)
                                     / max(np.cos(np.radians(self.yaw)), 0.3))
                self.scene.n_tiles = k
                self.renderer.set_copies([(0.0, 0.0, kk * PERIOD) for kk in range(-k, k + 1)])
            tint = (layer_tint(count, self.layer_colors == 2) if (self.layer_colors and count) else None)
            self.renderer.update_instances(inst[:count], count, motion=mot, tint=tint)
            self.renderer.set_guides(self.scene.ground_outline())
            self.renderer.show_guides = not self.renderer.shaded   # shaded: solids only
            # follow the receding face, smoothed (~1 s)
            tgt = self.scene.view_center()
            cur = np.asarray(self.renderer._look_center, dtype=float)
            a = min(1.0, frametime / 1.0)
            self.renderer._look_center = tuple((cur + a * (tgt - cur)).tolist())
            t4 = _time.perf_counter()
            self.renderer.draw(self.wnd.width, self.wnd.height)
            t5 = _time.perf_counter()
            if cap is not None and not cap.done:
                cap.frame(self.wnd, self.scene.sim_time, n_f[None], t_frame)
                if cap.video_time() >= CAP_T:
                    cap.finish(f"{CAP_T:.0f} s of video")
                    self.wnd.close()
                    return
            if self.paused:
                return
            p = self._prof
            STATUS.tick(t1 - t0)
            p["physics"] += t1 - t0; p["cull"] += t2 - t1; p["spawn"] += t3 - t2
            p["geom"] += t4 - t3; p["render"] += t5 - t4; p["total"] += t5 - t_frame
            self._prof_n += 1
            if self._prof_n >= 120:
                n = self._prof_n
                st = self.scene.stats()
                info(f"--- Profile ({st['n']} blocks, {st['awake']} awake, {n} frames) ---")
                for k in ["physics", "cull", "spawn", "geom", "render", "total"]:
                    info(f"  {k:8s}: {p[k]/n*1000:6.2f} ms/frame")
                info(f"  wall: t={self.scene.sim_time:6.1f}  " + _fmt(st, self.friction))
                info("  params: " + _params())
                for k in p: p[k] = 0.0
                self._prof_n = 0

        def on_mouse_press_event(self, x, y, button):
            """Left click: delete the block under the cursor."""
            if button != 1:
                return
            import glm
            w, h = self.wnd.width, self.wnd.height
            inv = glm.inverse(self.renderer._mvp(w, h))
            nx = 2.0 * x / max(w, 1) - 1.0; ny = 1.0 - 2.0 * y / max(h, 1)
            a = inv * glm.vec4(nx, ny, -1.0, 1.0); b = inv * glm.vec4(nx, ny, 1.0, 1.0)
            a = np.array([a.x, a.y, a.z]) / a.w; b = np.array([b.x, b.y, b.z]) / b.w
            d = (b - a) / np.linalg.norm(b - a)
            i = self.scene.delete_picked(a, d, self.scene.n_tiles if PERIODIC else 0)
            info(f"[click] deleted block {i}" if i >= 0 else "[click] no block under the cursor")

        def on_key_event(self, key, action, modifiers):
            k = self.wnd.keys
            if key in (k.LEFT, k.RIGHT):                          # held: orbit the camera
                d = -1 if key == k.LEFT else 1
                if action == k.ACTION_PRESS:
                    self._turn = d
                elif action == k.ACTION_RELEASE and self._turn == d:
                    self._turn = 0
                return
            if action != k.ACTION_PRESS:
                return
            if key == k.ESCAPE:
                self.wnd.close()
            elif key == k.SPACE:
                self.paused = not self.paused
                info("[PAUSED]" if self.paused else "[RUNNING]")
            elif key == k.R:
                self.scene.reset(); info("[RESET]")
            elif key == k.B:
                self.renderer.blur = not self.renderer.blur
                info(f"[motion blur {'ON' if self.renderer.blur else 'OFF'}]")
            elif key == k.W:
                self.renderer.shaded = not self.renderer.shaded
                info(f"[{'LUT-shaded' if self.renderer.shaded else 'wireframe'}]")
            elif key == k.H:
                self.scene.set_hold(not self.scene.hold)
                info(f"[wake {'HELD' if self.scene.hold else 'released'}]")
            elif key == k.C:
                self.layer_colors = (self.layer_colors + 1) % 3      # all -> awake only -> grey
                info(f"[colours: {('grey', 'all blocks', 'awake blocks only')[self.layer_colors]}]")
            elif key == k.O and MANUAL_LAYERS:
                L, cut = self.scene.wake_row()
                info(f"[layer {L}: awake above {cut:.0f}]")
            elif key == k.P and MANUAL_LAYERS:
                info(f"[layer {self.scene.wake_layer()} woken]")
            elif key == k.P and WAKE_MODE == "rate" and not MANUAL_LAYERS:
                info(f"[peel: wake advanced by {self.scene.peel_face()} blocks (one wall face)]")
            elif key == k.P:
                d = self.scene.peel_one()
                info(f"[peel: wake front F={d:.0f}; wake HELD]")
            elif key == k.F:
                self.friction = not self.friction
                set_friction(self.friction)
                info(f"[static friction {'ON' if self.friction else 'OFF'}]")

    import traceback
    code = 0
    try:
        mglw.run_window_config(WallWindow)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(); code = 1
    finally:
        STATUS.end()
        if cap is not None:
            try:
                cap.finish("window closed")      # ESC / crash: still a valid mp4 + csv
            except Exception:
                traceback.print_exc(); code = 1
        try:
            ti.reset()
        except Exception:
            pass
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        run_window()

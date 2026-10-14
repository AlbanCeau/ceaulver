"""Taichi fields: block state, boundaries, broadphase grid, pair list, shear cache, probes."""

import math as _math

import numpy as np
import taichi as ti

from .params import (
    MAX_NEIGH,
)

MAX_N = 32768
# Per-block kernels loop `for i in range(MAX_N): if i >= n_f[None]: continue` rather than
# `range(n_f[None])`: on Metal a loop whose bound is read from a field costs ~8 us per dispatch
# whatever n is, a constant 32K-thread loop with the guard ~3 us. Pair and
# cell loops keep the dynamic bound: their constant maximum (262K-885K threads) costs more.

pos_f     = ti.Vector.field(3, ti.f32, MAX_N)
vel_f     = ti.Vector.field(3, ti.f32, MAX_N)
omega_f   = ti.Vector.field(3, ti.f32, MAX_N)
R_f       = ti.Matrix.field(3, 3, ti.f32, MAX_N)
he_f      = ti.Vector.field(3, ti.f32, MAX_N)
mass_f    = ti.field(ti.f32, MAX_N)
Iil_f     = ti.Vector.field(3, ti.f32, MAX_N)   # I_inv_local (diagonal)
Iiw_f     = ti.Matrix.field(3, 3, ti.f32, MAX_N) # I_inv_world
rs_f      = ti.Vector.field(3, ti.f32, MAX_N)   # rolling_scale per axis
sleep_f   = ti.field(ti.i32, MAX_N)
nosleep_f = ti.field(ti.i32, MAX_N)   # 1 = this block may not fall asleep (scene-driven; default 0)
nowake_f  = ti.field(ti.i32, MAX_N)   # 1 = this sleeper may not be woken by impact/force (default 0)
birth_f   = ti.field(ti.f32, MAX_N)   # scene time the block was spawned (scenes set birth_now; colour only)
birth_now = ti.field(ti.f32, ())      # stamped on every block spawned from now on
noslope_f = ti.field(ti.i32, MAX_N)   # 1 = the slope plane ignores this block (scene-driven, e.g. it fell
                                      # off a shelf edge: re-entering under the plane would launch it). Default 0
n_f       = ti.field(ti.i32, ())

ke_pre_f  = ti.field(ti.f32, MAX_N)           # KE snapshot before substep
support_f = ti.field(ti.f32, MAX_N)           # support radius along slope normal
active_f  = ti.field(ti.i32, MAX_N)           # 1 = active (needs substeps), 0 = ballistic

F_f       = ti.Vector.field(3, ti.f32, MAX_N)
tau_f     = ti.Vector.field(3, ti.f32, MAX_N)
fsq_f     = ti.field(ti.f32, MAX_N)              # sum |F_k|^2 over this eval's forces (SLEEP_IMBALANCE)
gnd_f     = ti.field(ti.i32, MAX_N)
slp_f     = ti.field(ti.i32, MAX_N)
cntct_f   = ti.field(ti.i32, MAX_N)
cnt_n_f   = ti.field(ti.i32, MAX_N)   # per-block contact count (freeze gate)
n_above   = ti.field(ti.i32, MAX_N)   # per-block count of contacts with a block resting ABOVE it (freeze gate)
active    = ti.field(ti.i32, MAX_N)   # island sleeping: 1 = process fully (awake OR adjacent-to-awake);
                                      # 0 = stable-frozen interior -> skip its boundary force, contacts & wake.
# island-sleeping recheck mode (how a SLEEPING block re-earns a wake-check):
#   0 = ONE-RING (default): awake blocks promote their sleeping neighbours to active each step,
#       + a periodic full re-check. Optimal when the awake set is small/clustered or zero
#       (active -> 0 when fully settled -> the B scan-skip then skips everything). But a SPREAD
#       awake set (jitter throughout a pile) blankets the whole pile to active (measured 85 awake
#       -> 5623/5623), defeating the scan-skip.
#   1 = STAGGERED: each step rechecks 1/ISLAND_RECHECK of ALL blocks round-robin (i%R == frame%R).
#       Immune to the blanket (active ~ awake + N/R regardless of spread), at a constant ~N/R
#       overhead even when fully settled. Better for a large, partially-awake (cascade) scene.
# Default is 1 (STAGGERED): measurements showed the one-ring blanket is
# the dominant settled/large-scene cost (a scattered awake front promotes the whole accumulated pile
# to active -> cost grows with N, not the moving front; reproduced the live 5623->5623-active 71 ms
# case). Staggered makes active track the front (~awake + N/R), flat in pile size. Set
# fields.island_stag[None]=0 for one-ring.
island_stag = ti.field(ti.i32, ())
island_stag[None] = 1

slope_n_f = ti.Vector.field(3, ti.f32, ())
slope_d_f = ti.field(ti.f32, ())
slope_on  = ti.field(ti.i32, ())
slope_feng_on = ti.field(ti.i32, ()) # 1: the slope plane uses the Feng contact (as the ground)
# Optional edge on the slope plane (a shelf): it acts only on blocks whose centre
# satisfies (p - slope_o).slope_t >= slope_umin. Unbounded by default (umin = -1e30).
slope_t_f    = ti.Vector.field(3, ti.f32, ())
slope_o_f    = ti.Vector.field(3, ti.f32, ())
slope_umin_f = ti.field(ti.f32, ())
slope_mu_y_hi = ti.field(ti.f32, ())
slope_mu_y_lo = ti.field(ti.f32, ())

# Optional invisible vertical barrier at the bottom of the slope.
# Same plane equation as slope (n . x >= d), force computed by plane_force().
wall_n_f  = ti.Vector.field(3, ti.f32, ())
wall_d_f  = ti.field(ti.f32, ())
wall_on   = ti.field(ti.i32, ())

# Optional invisible vertical cylindrical confinement (vat). Infinite height,
# axis is the Y direction at (cyl_cx, cyl_cz).
cyl_on    = ti.field(ti.i32, ())
cyl_R_f   = ti.field(ti.f32, ())
cyl_cx_f  = ti.field(ti.f32, ())
cyl_cz_f  = ti.field(ti.f32, ())

# Optional square-prism confinement: 4 flat vertical walls at x,z = +/- box_w,
# centred at origin. Flat planes -> conservative per-corner contact (no cylinder
# tangent-plane artifact), while still keeping a pile dense.
box_on = ti.field(ti.i32, ())
box_w  = ti.field(ti.f32, ())
# Optional FRICTIONLESS side barriers at z = +-side_w (scene-driven; off by default):
# contain a wall laterally without holding it up.
side_on = ti.field(ti.i32, ())
side_w  = ti.field(ti.f32, ())
# Optional PERIODIC z (scene-driven; off by default). Period per_L must equal
# per_nz * grid cell exactly, per_nz >= 3: the hash wraps the z cell index modulo
# per_nz, and every pair geometry uses the minimum image of p_j - p_i in z.
per_on = ti.field(ti.i32, ())
per_L  = ti.field(ti.f32, ())
per_nz = ti.field(ti.i32, ())

# Ledge bounds (finite ground): x in [ledge_x0, ledge_x1], z in [ledge_z0, ledge_z1]
ledge_on  = ti.field(ti.i32, ())
ledge_x0  = ti.field(ti.f32, ())
ledge_x1  = ti.field(ti.f32, ())
ledge_z0  = ti.field(ti.f32, ())
ledge_z1  = ti.field(ti.f32, ())

# Spawn staging buffer (small, CPU→GPU for new blocks only)
SPAWN_MAX = 64
sp_pos  = ti.Vector.field(3, ti.f32, SPAWN_MAX)
sp_he   = ti.Vector.field(3, ti.f32, SPAWN_MAX)
sp_R    = ti.Matrix.field(3, 3, ti.f32, SPAWN_MAX)
sp_mass = ti.field(ti.f32, SPAWN_MAX)
sp_Iil  = ti.Vector.field(3, ti.f32, SPAWN_MAX)
sp_rs   = ti.Vector.field(3, ti.f32, SPAWN_MAX)
sp_n    = ti.field(ti.i32, ())

# Cull (GPU-side swap-with-last)
cull_remove = ti.field(ti.i32, MAX_N)  # indices to remove (written by _cull_mark)
cull_n_remove = ti.field(ti.i32, ())   # how many to remove

# Spatial hash grid for broadphase
MAX_BLOCK_HE = 5.0
GRID_CELL    = 2.0 * MAX_BLOCK_HE * _math.sqrt(3.0)  # default cell (covers worst-case rotated he=5 AABB)
NUM_BUCKETS  = 4096
MAX_PER_CELL = 128   # dense piles overflow a 17.3-cell (sized for he=5) with
                     # he~1 blocks; an unconstrained pile spreads, but keep headroom vs dropped contacts

# Cell size is PER-WORLD (runtime), set from the world's actual max half-extent in
# Solver(max_he=...). The 3x3x3 neighbour scan requires cell >= 2*max_he*sqrt(3)
# (worst-case rotated AABB), so a world of small blocks (e.g. the vat, he<=1.4) must use a
# small cell or the whole footprint collapses into a few overflowing buckets.
grid_cell_f = ti.field(ti.f32, ())

grid_cnt  = ti.field(ti.i32, NUM_BUCKETS)
grid_data = ti.field(ti.i32, (NUM_BUCKETS, MAX_PER_CELL))
# (x, y, z, bounding radius |he|) of the block in each grid slot, written by _build_grid.
# _feng_cells rejects most candidates on this alone: measured ~103 candidates per block
# against ~6 contacts, and each rejection used to read pos/R/he of j from scattered memory.
grid_pr   = ti.Vector.field(4, ti.f32, (NUM_BUCKETS, MAX_PER_CELL))
grid_cell_f[None] = GRID_CELL

MAX_PAIRS = 8 * MAX_N     # candidate pairs per force eval (pass 1 -> pass 2 of _collisions_feng)
pair_i = ti.field(ti.i32, MAX_PAIRS)
pair_j = ti.field(ti.i32, MAX_PAIRS)
pair_s = ti.field(ti.i32, MAX_PAIRS)     # shear-spring slot in row pair_i (-1 = none)
pair_n = ti.field(ti.i32, ())
pair_overflow = ti.field(ti.i32, ())     # debug: pairs dropped because MAX_PAIRS was full
MAX_CAND = 32                            # per-block candidate list (_feng_cells -> _feng_claim)
cand_j = ti.field(ti.i32, (MAX_N, MAX_CAND))
cand_n = ti.field(ti.i32, MAX_N)
cand_overflow = ti.field(ti.i32, ())     # debug: candidates dropped because a block's list was full
# FENG_FACELIST: the (pair, face) threads that survive the slab cull, as p*12 + face.
face_list = ti.field(ti.i32, MAX_PAIRS * 12)
face_n = ti.field(ti.i32, ())
reuse_dmax = ti.field(ti.f32, ())       # FENG_REUSE: max over blocks of acc_f
acc_f = ti.field(ti.f32, MAX_N)         # FENG_REUSE: per block, motion bound accumulated since the rebuild
act_build = ti.field(ti.i32, MAX_N)     # FENG_REUSE: 1 = the list holds this block's pairs (active at the rebuild, or appended since)
reuse_mode = ti.field(ti.i32, ())       # FENG_REUSE: this eval -- 0 rebuild, 1 keep + append newly active blocks
reuse_force = ti.field(ti.i32, ())      # FENG_REUSE: host asked for a rebuild (invalidate_grid)
topo_gen = ti.field(ti.i32, ())         # bumped by every kernel that adds, removes or re-indexes blocks
topo_build = ti.field(ti.i32, ())       # FENG_REUSE: topo_gen at the last rebuild
reuse_rebuilds = ti.field(ti.i32, ())   # FENG_REUSE debug: evals that rebuilt (cumulative)
reuse_evals = ti.field(ti.i32, ())      # FENG_REUSE debug: evals (cumulative)
reuse_force[None] = 1
# Per-pair sums of the 12 face contributions (_feng_faces -> _feng_forces).
facc_V  = ti.field(ti.f32, MAX_PAIRS)
facc_c  = ti.Vector.field(3, ti.f32, MAX_PAIRS)
facc_a  = ti.Vector.field(3, ti.f32, MAX_PAIRS)
facc_gA = ti.Vector.field(3, ti.f32, MAX_PAIRS)
facc_gB = ti.Vector.field(3, ti.f32, MAX_PAIRS)

# Cundall shear-spring per-block neighbour cache (T-05). Each
# pair is processed only by thread i, so this cache is written by exactly
# one thread per pair and needs no atomics (unlike the grid above).
shear_j    = ti.field(ti.i32, (MAX_N, MAX_NEIGH))            # neighbour index, -1 = empty slot
shear_F    = ti.Vector.field(3, ti.f32, (MAX_N, MAX_NEIGH))  # accumulated tangential force (world)
shear_n    = ti.Vector.field(3, ti.f32, (MAX_N, MAX_NEIGH))  # contact normal at last update
shear_seen = ti.field(ti.i32, (MAX_N, MAX_NEIGH))            # frame id of last activity (eviction)
cur_frame  = ti.field(ti.i32, ())                            # incremented once per step()
shear_overflow = ti.field(ti.i32, ())                        # debug: pairs with no free slot
shear_maxocc   = ti.field(ti.i32, ())                        # debug: highest slot index used + 1
# 1 = this _collisions_feng / _prep call advances the shear springs. Within step() it is
# 0 for the eval at x(t) and 1 for the eval at x(t+dt), so the spring moves once per step
# by the drift's slip (v_half*dt) instead of twice. Set INSIDE kernels step() already
# launches (_kick_drift -> 1, _evict_shear -> 0): a Python-side `shear_upd[None] = v` is
# its own launch + sync, measured 0.106 ms each, 4 per frame at sub 2. Starts at 1, so
# scripts that call the force kernels without step() keep advancing it; after a step()
# it reads 0 until the next _kick_drift.
shear_upd      = ti.field(ti.i32, ())
# Boundary spring store: accumulated tangential force (world) per block, per
# plane (0 = ground, 1 = slope), per corner. No neighbour key, no ownership and no
# reseat test: the plane and the corner's identity are both fixed. Zeroed when
# the corner stops penetrating.
N_PLANE_SPRING = 2
shear_pl_F = ti.Vector.field(3, ti.f32, (MAX_N, N_PLANE_SPRING, 8))
dbg_omdt_blk   = ti.field(ti.f32, MAX_N)                     # debug (DEBUG_OMDT): per block, max omega*dt of its contacts
max_omdt       = ti.field(ti.f32, ())                        # debug: max contact omega*dt (ringing margin; >2 = ring)
# perf-probe contact counters (runtime-gated by dbg_on; ZERO cost when dbg_on=0). Reset per step
# by the probe. Tell us whether _collisions_feng is broadphase-iteration-bound vs clip-bound.
dbg_on   = ti.field(ti.i32, ())    # 1 = count this step (probe only)
dbg_cand = ti.field(ti.i32, ())    # candidate pairs examined (after j>i)
dbg_skip = ti.field(ti.i32, ())    # pairs skipped by island sleeping (both inactive)
dbg_clip = ti.field(ti.i32, ())    # pairs that ran the full Feng clip (passed AABB + SAT)
# Model-fidelity probes (same dbg_on gate, same zero cost when off). The deep-overlap
# cap is NOT a gradient -- scaling -grad w by a configuration-dependent scalar leaves a
# field that does net work around a closed loop (measured: +3.7e3 where the uncapped law
# gives 6e-2; figures/006_friction/loop_work.py). These count how often production is
# actually in that regime, and how often the boundary spring's adaptive-K clamp binds.
dbg_ncon = ti.field(ti.i32, ())    # block-block contacts with Vc > 0
dbg_fr   = ti.field(ti.f32, 12)    # T-05 friction probe: last block-block pair evaluated
dbg_fv   = ti.field(ti.f32, (4, 15))  # T-05: per-eval vt, Fs0, Fs, nhat, cc (world vectors)
dbg_cap  = ti.field(ti.i32, ())    # ... of those, ones where VMAX_CONTACT bound
dbg_capblk = ti.field(ti.i32, MAX_N)  # ... per block: its contacts that bound (dbg_on only; never reset here)
dbg_pcor = ti.field(ti.i32, ())    # boundary (plane) corners in penetration
dbg_pclm = ti.field(ti.i32, ())    # ... of those, ones where Ke = alpha*me/dt^2 bound
shear_upd[None] = 1

_CS = ti.Vector.field(3, ti.f32, 8)
_CS.from_numpy(np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=np.float32))


GEOM_MAX = MAX_N        # the render instance buffer must hold every block
INST_FLOATS = 15  # pos(3) + he(3) + R columns(9)

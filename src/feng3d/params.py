"""Taichi init, solver tunables, and the packed runtime parameter field."""

import os as _os
import math as _math

import numpy as np
import taichi as ti

from . import config

# Offline cache speeds production startup. But diagnostics flip *captured* Python
# constants (CHUNK_MU, rolling resistance, sleep thresholds, contact damping) at
# import time, and a cached kernel can be reloaded from disk without re-capturing
# them -- so the override silently doesn't apply. Diagnostics set TI_NO_CACHE=1
# before importing this module to force a clean recompile.
# Startup: Taichi re-traces every kernel's Python source at each launch (the
# offline-cache key is a hash of the traced IR), and for every syntax node it
# eagerly formats a source-position message -- the line wrapped to 80 columns
# plus a ^^^ underline -- that is only read if the kernel fails to compile.
# Profiled, that formatting is about a third of the trace. Replace it with
# file / line / raw source line. TI_FULL_ERRORS=1 restores Taichi's own.
# Touches a Taichi 1.7.4 internal: skipped silently if it is not there.
def _cheap_pos_info(self, node):
    return (f'File "{self.file}", line {node.lineno + self.lineno_offset}, '
            f'in {self.func.func.__name__}:\n{self.src[node.lineno - 1]}\n')

if _os.environ.get("TI_FULL_ERRORS") != "1":
    try:
        from taichi.lang.ast.ast_transformer_utils import ASTTransformerContext as _Ctx
        if hasattr(_Ctx, "get_pos_info"):
            _Ctx.get_pos_info = _cheap_pos_info
    except Exception:
        pass

# First (uncached) launch: ~13 s, nearly all of it Taichi's optimiser on the
# boundary kernels. TI_ADVANCED_OPTIMIZATION=0 (Taichi's own switch) cuts that
# to ~3 s; no runtime cost was measured on a 1.4K-block pile, larger scenes untested.
#
# Taichi is initialised here unless the caller already did: to choose the backend or
# other settings, call ti.init(..., default_fp=ti.f32) BEFORE importing any feng3d
# module other than config. The solver assumes f32.
from taichi.lang import impl as _ti_impl
if _ti_impl.get_runtime().prog is None:
    import contextlib as _ctx, io as _io
    # ti.init prints "[Taichi] Starting on arch=..." unconditionally; FENG3D_VERBOSE=1 shows it
    _quiet = _os.environ.get("FENG3D_VERBOSE", "0") != "1"
    with _ctx.redirect_stdout(_io.StringIO()) if _quiet else _ctx.nullcontext():
        ti.init(arch=ti.gpu, default_fp=ti.f32, log_level=ti.WARN,
                offline_cache=(_os.environ.get("TI_NO_CACHE") != "1"))
    if _os.environ.get("TI_NO_CACHE") == "1":
        print("[feng3d] TI_NO_CACHE=1 -> offline_cache disabled (kernels compiled fresh)")

# ==================================================================
# Constants
# ==================================================================

# Scene physics lives in config.py -- these are rebinds, not values. Editing a
# number here changes nothing; edit config.py. The module-level names are kept so
# the existing diagnostic override idiom (params.CHUNK_MU = 0.0) still works:
# sync_params() runs at the top of every step() and pushes whatever these names
# currently hold to the GPU, so an override takes effect on the next step with no
# kernel recompile (which is why it no longer needs TI_NO_CACHE=1 to be seen).

GRAVITY         = config.GRAVITY
GROUND_Y        = config.GROUND_Y
SLEEP_SPEED     = config.SLEEP_SPEED
SLEEP_OMEGA     = config.SLEEP_OMEGA
FREEZE_ABOVE    = config.FREEZE_ABOVE
FREEZE_CONTACTS = config.FREEZE_CONTACTS
ISLAND_RECHECK  = config.ISLAND_RECHECK
WAKE_VEL        = config.WAKE_VEL
WAKE_FORCE      = config.WAKE_FORCE
# EXPERIMENTAL (T-16) sleep gate: relative force balance. Default OFF = the speed/omega gate alone.
# > 0: a block may sleep only if it is ALSO balanced relative to the forces acting on it:
#   |sum F_k| < SLEEP_IMBALANCE * sqrt(sum |F_k|^2)                (net force)
#   |sum tau_k| < SLEEP_IMBALANCE * h_max * sqrt(sum |F_k|^2)      (net torque, at the
#                                                                    block's force scale)
# over every force this eval (gravity, boundary planes, each contact; fsq_f). A tipping
# or slipping block is out of balance and cannot be re-frozen in the step that woke it;
# a loaded pile block, whose large contact forces cancel, is judged against THEM, not
# its weight. (Torque uses the force scale x size, not sum |tau_k|^2: a block resting
# flat has every torque ~0 and would never pass.)
SLEEP_IMBALANCE = getattr(config, "SLEEP_IMBALANCE", 0.0)
# EXPERIMENTAL (T-11): conservative yield ceiling. 0 (default) = production: the force is
# rescaled by fcap/|F| when it exceeds fcap -- |F| = w'|a_c| depends on the contact
# geometry, so the rescaled field is not a gradient and does net work around closed loops
# (T-11 loop_work; sustained ringing of load-bearing contacts in wall.py). 1 = saturate the
# potential's slope instead: w'(V) = min(kH V^1.5, W'max), W'max = fcap / (largest face
# area of the lighter block) -- an exact -grad w(V), and a full-face contact on that face
# still saturates at fcap.
CAP_CONSERVATIVE = getattr(config, "CAP_CONSERVATIVE", 0.0)
MAX_SPEED       = config.MAX_SPEED
MAX_OMEGA       = config.MAX_OMEGA
ROLLING_RESIST  = config.ROLLING_RESIST
ROLLING_M_REF   = config.ROLLING_M_REF
CONTACT_TOL     = config.CONTACT_TOL
STIFFNESS_ALPHA = config.STIFFNESS_ALPHA
FRICTION_VISC   = config.FRICTION_VISC

def _damping_coeff(e):
    ln_e = _math.log(max(e, 1e-4))
    return -2.0 * ln_e / _math.sqrt(_math.pi**2 + ln_e**2)

# Derived ONCE from the config restitution, then treated as a plain tunable: a diagnostic
# may set it directly (params._SLOPE_D = 0.0), so sync_params() pushes whatever it holds
# and never recomputes it from config -- otherwise the override would be clobbered.
_SLOPE_D  = _damping_coeff(config.SLOPE_RESTITUTION)

SOIL_MU     = config.SOIL_MU
SLOPE_K     = config.SLOPE_K;     SLOPE_MU = config.SLOPE_MU
SLOPE_MU_LO = config.SLOPE_MU_LO; SLOPE_MU_HI = config.SLOPE_MU_HI
CHUNK_MU    = config.CHUNK_MU
CHUNK_ROLL  = config.CHUNK_ROLL
SLOPE_ROLL  = config.SLOPE_ROLL
PEN_MAX     = config.PEN_MAX

# Safeguard: cap the per-contact spring impulse so a DEEP overlap (e.g. a broadphase-dropped
# contact, V huge) pushes apart gently instead of flinging a block at 40 m/s. Bounds the
# lighter block's velocity change from one contact per step. Only bites when pathologically
# deep; the normal shallow regime is untouched (so conservation there is preserved).
VMAX_CONTACT = config.VMAX_CONTACT

# Hertzian-VOLUME contact (Poschel & Schwager 2005, eq. 2.10/2.12 adapted to Feng overlap
# volume): a single fixed nonlinear potential
# w(V) = (2/5) k V^{5/2} -> wprime = k V^{3/2}. For a face contact V=A*delta this gives the
# Hertz depth-stiffening |F| ~ delta^{3/2} (sublinear penetration under load -> deep piles
# bottom out less) while staying a clean -grad w(V_c) potential (conservative, no SAT flip).
# FENG_KHERTZ sets stiffness; D_HERTZ is the P&S sqrt(V)*v dashpot ratio.
FENG_KHERTZ = config.FENG_KHERTZ   # Hertz-volume stiffness: wprime = (FENG_KHERTZ/(AcA*AcB)) * V^{3/2}
D_HERTZ     = config.D_HERTZ    # viscoelastic damping COEFFICIENT (De = D_HERTZ*sqrt(V*me), P&S 2.12 form),
                        # repulsive-clamped. NOT a 0-1 ratio.

# Cundall (1979) static-friction shear spring (T-05). The tangential contact
# force accumulates across substeps so a settled pile holds a static value
# instead of collapsing to zero when v_t -> 0 (pure-kinetic Coulomb has no rest
# configuration). k_s = k_n in Cundall Eq. 17, realised per-contact as
# SHEAR_K_FRAC * K_eff; the Coulomb cap is Eq. 19. State lives in a per-block
# neighbour cache (see fields below); the spring is reset when the contact
# normal reseats (n_last . n_now < SHEAR_FLIP_DOT).
MAX_NEIGH      = 16      # shear-cache slots per block (dense pile core ~6-12 contacts)
SHEAR_K_FRAC   = config.SHEAR_K_FRAC     # k_s = SHEAR_K_FRAC * k_tangent. ZERO DISABLES the
                         # spring and falls back to the stateless kinetic branch. A runtime
                         # parameter (sync_params), so a diagnostic can A/B it in
                         # one process with no recompile.
SHEAR_FLIP_DOT = 0.5     # reset F_s if the contact normal rotates past ~60 deg
# ZETA_N > 0 replaces the P&S point dashpot with a Rayleigh dashpot on dV_c/dt
#   (T-12): F = -c*Vdot*(a_c, gA, gB), c = 2*ZETA_N*sqrt(k_V*m_V) -- Tsuji, Tanaka &
#   Ishida 1992's constant-zeta Hertz damping, written for the volume potential.
#   The production dashpot is ~1 % of critical on a resting face (T-05).
# ZETA_T > 0 adds Cundall & Strack's shear dashpot to the spring, active only
#   while the contact is not slipping (PFC convention).
ZETA_N = config.ZETA_N
ZETA_T = config.ZETA_T
# Static friction coefficient for the shear spring (T-05). The stored force may
# reach CHUNK_MU_S*Fn before the contact slips; on slip it drops to CHUNK_MU*Fn
# (kinetic) and may re-stick next step if it stays under the static limit --
# the two-coefficient spring scheme (Luding 2008, Granular Matter 10, [verify
# section]). Value in config.py.
CHUNK_MU_S = config.CHUNK_MU_S
# Boundary (ground / slope plane) static friction, T-05 step 2. A Cundall spring per
# contact point (slope: each corner; ground: one): k_s = PLANE_SHEAR_K * its normal stiffness. One
# coefficient (the plane's mu) is both the static cap and the kinetic value --
# boundary mu is already 1.1-12. 0 = the stateless law production shipped.
# Shares ZETA_T (shear dashpot) and the once-per-step update with the block spring.
PLANE_SHEAR_K = config.PLANE_SHEAR_K

# Fracture energy sink: max KE gain per substep = FRAC_BUDGET * m * g * he_max
# Physically: contact stress above this threshold does fracture work, not rebound.
FRAC_BUDGET = config.FRAC_BUDGET

# --- Compile-time switches (read from the environment at import) ---
FENG_FACECULL = int(_os.environ.get("FACECULL", "1"))   # skip clip faces provably outside the other box (exact speed opt)
FENG_FACELIST = int(_os.environ.get("FENG_FACELIST", "1"))  # 1: _feng_facelist culls faces per pair, _feng_faces clips only the survivors
# FENG_REUSE: the pair list is kept across force evals and steps (Verlet-list skin). A rebuild
# keeps pairs within FENG_SKIN of touching; every drift adds each block's motion bound to acc_f
# and reuse_dmax tracks the largest; the list is rebuilt when 2 * reuse_dmax > FENG_SKIN (two
# blocks that each moved less than half the skin cannot have closed a gap the skin covered, so
# no contact onset is missed), when blocks were added/removed (topo_gen), or when the host says
# so (invalidate_grid). Island sleeping changes the active set between rebuilds: a block that
# became active since the rebuild (act_build == 0) scans and appends its pairs with the blocks
# that were inactive then -- every other pair of it is already listed. Pairs whose two blocks
# are both inactive now stay listed and are skipped before the clip.
FENG_REUSE = int(_os.environ.get("FENG_REUSE", "1"))
FENG_SKIN = float(_os.environ.get("FENG_SKIN", "0.1"))
USE_GRAPH = int(_os.environ.get("USE_GRAPH", "1"))   # 1: step() replays one recorded ti.graph instead of ~17 Python kernel calls
FENG_DEDUPE = int(_os.environ.get("FENG_DEDUPE", "1"))   # T-10 bucket dedupe; env 0 = off (perf A/B only, compile-time)
DEBUG_OMDT     = False   # gate the omega*dt probe: it's a GLOBAL atomic_max PER CONTACT -> serializes
                         # the GPU kernel. OFF in production; diagnostics set it True before first step.

# --- Runtime scene parameters (packed) ----------------------------------
# One f32 field holding every live tunable, so the kernels read them as GPU
# uniforms instead of baking them in as compile-time literals. That is what makes
# config.py authoritative: a changed value is an upload, not a recompile, so it
# cannot be masked by a kernel reloaded from the Taichi offline cache -- the
# failure mode the TI_NO_CACHE=1 dance in the diagnostics exists to dodge.
# Index order must match _PARAM_NAMES.
_P_GRAVITY          =  0
_P_GROUND_Y         =  1
_P_SOIL_MU          =  2
_P_SLOPE_K          =  3
_P_SLOPE_MU         =  4
_P_SLOPE_MU_LO      =  5
_P_SLOPE_MU_HI      =  6
_P_SLOPE_D          =  7
_P_STIFFNESS_ALPHA  =  8
_P_PEN_MAX          =  9
_P_FENG_KHERTZ      = 10
_P_D_HERTZ          = 11
_P_CHUNK_MU         = 12
_P_VMAX_CONTACT     = 13
_P_FRICTION_VISC    = 14
_P_ROLLING_RESIST   = 15
_P_CHUNK_ROLL       = 16
_P_SLOPE_ROLL       = 17
_P_ROLLING_M_REF    = 18
_P_SLEEP_SPEED      = 19
_P_SLEEP_OMEGA      = 20
_P_FREEZE_ABOVE     = 21
_P_FREEZE_CONTACTS  = 22
_P_WAKE_VEL         = 23
_P_WAKE_FORCE       = 24
_P_MAX_SPEED        = 25
_P_MAX_OMEGA        = 26
_P_CONTACT_TOL      = 27
_P_FRAC_BUDGET      = 28
_P_SHEAR_K_FRAC     = 29
_P_SHEAR_FLIP_DOT   = 30
_P_ZETA_N           = 31
_P_ZETA_T           = 32
_P_CHUNK_MU_S       = 33
_P_PLANE_SHEAR_K    = 34
_P_SLEEP_IMBALANCE  = 35
_P_CAP_CONSERVATIVE = 36

_PARAM_NAMES = ("GRAVITY",
                "GROUND_Y",
                "SOIL_MU",
                "SLOPE_K",
                "SLOPE_MU",
                "SLOPE_MU_LO",
                "SLOPE_MU_HI",
                "_SLOPE_D",
                "STIFFNESS_ALPHA",
                "PEN_MAX",
                "FENG_KHERTZ",
                "D_HERTZ",
                "CHUNK_MU",
                "VMAX_CONTACT",
                "FRICTION_VISC",
                "ROLLING_RESIST",
                "CHUNK_ROLL",
                "SLOPE_ROLL",
                "ROLLING_M_REF",
                "SLEEP_SPEED",
                "SLEEP_OMEGA",
                "FREEZE_ABOVE",
                "FREEZE_CONTACTS",
                "WAKE_VEL",
                "WAKE_FORCE",
                "MAX_SPEED",
                "MAX_OMEGA",
                "CONTACT_TOL",
                "FRAC_BUDGET",
                "SHEAR_K_FRAC",
                "SHEAR_FLIP_DOT",
                "ZETA_N",
                "ZETA_T",
                "CHUNK_MU_S",
                "PLANE_SHEAR_K",
                "SLEEP_IMBALANCE",
                "CAP_CONSERVATIVE",)
N_PARAMS = len(_PARAM_NAMES)
PRM = ti.field(ti.f32, N_PARAMS)
island_recheck_f = ti.field(ti.i32, ())   # kept separate: used in an integer modulo

_prm_cache = None

def sync_params():
    """Push the module-level tunables to the GPU. No-op when nothing changed."""
    global _prm_cache
    g = globals()
    vals = tuple(float(g[n]) for n in _PARAM_NAMES) + (int(ISLAND_RECHECK),)
    if vals == _prm_cache:
        return
    _prm_cache = vals
    PRM.from_numpy(np.array(vals[:-1], dtype=np.float32))
    island_recheck_f[None] = vals[-1]

sync_params()

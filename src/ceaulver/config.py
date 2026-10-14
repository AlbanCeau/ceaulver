"""Solver defaults. params.py rebinds these at import and pushes them to the GPU as runtime
uniforms (params.sync_params(), at the top of every step()).
"""

# --- Sizing: these two also feed derive_contact() below ---
SPAWN_HE_MAX    = 2.5   # max half-extent: caps the log-normal tail so the broadphase cell can be
                        # sized to 2.5 (cell ~8.7) instead of MAX_BLOCK_HE=5 (cell 17.3). A coarse
                        # cell forces piled blocks to scan ~128 scattered neighbours each -> the perf
                        # killer. Lower = finer grid = faster (but fewer big blocks).
PHYSICS_SUBSTEPS = 2

# ==========================================================================
# Solver physics
#
# Single source of truth for the scene. params.py rebinds these at import and
# pushes them to the GPU as runtime uniforms (params.sync_params(), called at
# the top of every step()), so editing this file changes the sim on the next
# launch — no Taichi kernel recompile, and immune to the offline-cache
# staleness that TI_NO_CACHE=1 exists to work around.
# ==========================================================================

# --- World ---
GRAVITY  = -40.0
GROUND_Y =   0.0

# --- Boundary contact ---
# The ground is the Feng overlap-volume contact, as block-block (stiffness and damping are
# FENG_KHERTZ / ZETA_N below); only its friction is set here. Slope / wall / cylinder / box:
# linear per-corner penalty spring, restitution targeted via _damping_coeff() in params.py.
SOIL_MU             = 7.5    # homogeneous with the slope surface
SLOPE_K             = 30000.0
SLOPE_MU            = 3
SLOPE_RESTITUTION   = 0.03   # -> params._SLOPE_D
SLOPE_MU_LO         = 0.5   # mu ABOVE mu_y_hi -- the top of the slope, where blocks spawn
                             # (the _LO/_HI names are the wrong way round: _LO applies high up).
                             # With static friction, mu > tan(40) = 0.84 holds a
                             # spawned block where it lands, out of view, forever. 0.5 slides from
                             # rest at g*(sin40 - 0.5*cos40) = 10.4 units/s^2.
SLOPE_MU_HI         = 2
STIFFNESS_ALPHA     = 0.25   # K_eff = min(K, STIFFNESS_ALPHA * m_eff / dt^2)
PEN_MAX             = 1.5    # per-corner penetration clamp

# --- Block-block contact: Hertzian-volume (Feng 2021 overlap volume, with the
#     Poschel & Schwager 2005 eq. 2.10/2.12 depth-stiffening exponent) ---
# Sized from the block population + scene, not hand-tuned. FENG_KHERTZ (elastic
# branch) and VMAX_CONTACT (yield ceiling) are one contact model, not two knobs:
# Hertz up to the ceiling, perfectly plastic above it -- Thornton 1997 / Walton &
# Braun 1986, and what rock does. Three inputs, both constants follow.
#
# The yield ceiling must clear the force a load-bearing contact has to supply,
#   fcap = VMAX_CONTACT*min(m_i,m_j)/dt  >=  CONTACT_SAFETY*(PILE_DEPTH-1)*m*g
# which fixes VMAX_CONTACT; setting the elastic branch to reach that ceiling at a
# penetration of YIELD_STRAIN*(2*SPAWN_HE_MAX) fixes FENG_KHERTZ. dt cancels out
# of the second one, so PHYSICS_SUBSTEPS moves the cap but not the stiffness.
#
# CONTACT_SAFETY  measured ceiling ~3. Above it the cap stops being a rare safety
#                 net: at 10, blocks retained 778 -> 616, p90 speed 5.5 -> 14.1,
#                 awake fraction ~2x (awake blocks integrate, so that is frame time).
#                 NB omega*dt runs 10-18 in this scene: the contact is nowhere near
#                 resolved and does not need to be -- the yield ceiling is what makes
#                 it stable. With the ceiling removed the same stiffness sheds 70% of
#                 the blocks in 45 s.
# PILE_DEPTH      blocks of load the bottom contact of a pile carries, +1.
# YIELD_STRAIN    penetration at yield as a fraction of the largest block's edge.
#                 The look knob -- smaller = stiffer = crisper. Measured ceiling:
#                 0.26% settles 3/3 runs, 0.165% 2/3, 0.101% 1/3. Below ~0.26% the
#                 pile never goes quiet, so it never sleeps either.
CONTACT_SAFETY = 3.0
PILE_DEPTH     = 4
YIELD_STRAIN   = 0.0026


def derive_contact(safety, pile_depth, gravity, he_max, substeps, yield_strain):
    """-> (FENG_KHERTZ, VMAX_CONTACT). See the commentary above for the derivation.

    A function, not module-level arithmetic, so tuning.py can re-derive the pair when a
    slider moves any of the six inputs. Writing FENG_KHERTZ without re-deriving
    VMAX_CONTACT (or vice versa) moves the elastic branch and the yield ceiling
    independently, which is exactly the inconsistency the derivation exists to prevent.
    """
    load   = safety * (pile_depth - 1) * abs(gravity)   # force per unit mass
    sub_dt = 1.0 / (60.0 * substeps)                    # scree.py caps frame dt at 1/60
    return (2.0**0.5 * load * he_max**0.5 / yield_strain**1.5,   # FENG_KHERTZ
            load * sub_dt)                                       # VMAX_CONTACT


FENG_KHERTZ, VMAX_CONTACT = derive_contact(
    CONTACT_SAFETY, PILE_DEPTH, GRAVITY, SPAWN_HE_MAX, PHYSICS_SUBSTEPS, YIELD_STRAIN)
                           # FENG_KHERTZ  6.07e6 at the values above: Hertz-volume
                           #   stiffness, wprime = (FENG_KHERTZ/(AcA*AcB)) * V^{3/2}
                           # VMAX_CONTACT 3.0 at the values above: deep-overlap
                           #   safeguard, caps one contact's spring impulse
D_HERTZ      = 150.0     # viscoelastic damping COEFFICIENT (De = D_HERTZ*sqrt(V*me), P&S 2.12 form),
                         # repulsive-clamped. NOT a 0-1 ratio.
CHUNK_MU     = 0.15      # block-block Coulomb friction. Deliberately far below the boundary
                         # mu (SOIL_MU / SLOPE_MU_LO/HI = 7.5): blocks are meant to lock into each
                         # other geometrically, not by friction. NB the ratio is not 7.5/0.15 in
                         # effect -- friction is min(mu*Fn, FRICTION_VISC*vtm, me*vtm/dt), and for
                         # a he~0.8 pair the me*vtm/dt stability clamp is ~98*vtm, so the Coulomb
                         # branch only binds when mu < 0.6*vtm. Above mu~0.5 the term is dead at
                         # settling speeds (measured: 1.0 and 0.75 give the same pile). 0.15 is the
                         # knee -- 0.1 and 0.15 measure alike, so mu is fully engaged without going
                         # gratuitously low. pile_test N=400: on-ground 42% -> 51%, max radius
                         # 20.1 -> 23.3 vs mu=1.0 (flatter, wider scree); still settles to 0% awake.
# Static friction (T-05). Switched ON 2026-10-07 with the set measured on the
# 100-block pile (creep 0.014-0.021 -> 0.0004-0.0005 units/s) and the slope rig
# (tile holds at 45 deg, slides at 55 deg). Set SHEAR_K_FRAC = PLANE_SHEAR_K = 0
# to get the stateless law back.
SHEAR_K_FRAC  = 1.0      # block-block Cundall spring, k_s = this * k_tangent
PLANE_SHEAR_K = 1.0      # ground/slope per-corner spring, k_s = this * Ke
CHUNK_MU_S    = 1.15     # block-block static mu (stick limit); CHUNK_MU is the kinetic one.
                         # Lowest value that holds a free 1x0.5x1 tile at 45 deg in slope_hold.py
                         # at sub 2 and 4, placed or dropped (1.10 fails 3 of 4; ideal tan45 = 1.0 --
                         # the 15% is the touchdown bounce and the spring loading transient).
ZETA_T        = 1.0      # shear dashpot, x critical; needed so loading does not overshoot mu_s
ZETA_N        = 0.3      # Rayleigh normal dashpot on dV_c/dt, x critical (0 = P&S point dashpot)

# --- Friction regularisation (shared by boundary and block-block) ---
FRICTION_VISC = 1500.0   # viscous cap on the Coulomb force at low sliding speed

# --- Rolling resistance (per-axis exponential decay on omega; a settling aid,
#     not a contact force) ---
ROLLING_RESIST = 6.0    # on the ground. Needed so
                        # omega DECAYS -> settled blocks clear the SLEEP_OMEGA gate -> sleep ->
                        # island sleeping engages. Contact friction alone arrests spin too slowly.
CHUNK_ROLL     = 3.0    # in block-block contact
SLOPE_ROLL     = 2.0    # on the slope
ROLLING_M_REF  = 50.0   # mass reference for the per-block rolling/sleep scale

# --- Sleep / freeze / wake ---
SLEEP_SPEED     =   1.0
SLEEP_OMEGA     =   0.5
FREEZE_ABOVE    =   1   # Aggressive freeze of the buried quasi-static core: a slow block with
FREEZE_CONTACTS =   4   # >=FREEZE_CONTACTS contacts AND >=FREEZE_ABOVE blocks resting on top is
                        # load-bearing; its residual spin is numerical, so freeze it regardless of
                        # omega (the omega sleep-gate is what roll=0 broke). Velocity-gated (never
                        # freeze a fast block = active avalanche); WAKE_FORCE re-wakes it if support
                        # is lost (so no floaters). This is the perf lever: sleeping core skips
                        # integration.
WAKE_VEL        =   2.0  # wake a sleeping block only on a real impact (relative speed > this).
WAKE_FORCE      =   0.5  # also wake a sleeper when it leaves force equilibrium: net force >
                         # WAKE_FORCE*weight (lost support -> would fall) or net torque >
                         # WAKE_FORCE*weight*he (about to tip). Keeps stable piles frozen
                         # (balanced -> stay asleep) but no floaters when support is removed.
                         # Resting/jiggling neighbours don't wake it -> settled piles freeze solid.
ISLAND_RECHECK  =  20   # island sleeping: every Nth step, evaluate ALL blocks (not just the live
                        # region) so a frozen block that silently lost support (culled/teleported
                        # supporter) can't float longer than N steps. ~1/N full-cost steps.
CONTACT_TOL     =   1.0
MAX_SPEED       = 300.0
MAX_OMEGA       =  30.0

# --- Fracture energy sink ---
FRAC_BUDGET = 50.0   # max KE gain per substep = FRAC_BUDGET * m * g * he_max.
                     # Physically: contact stress above this threshold does fracture work, not
                     # rebound. 50 is a loose safety net (only catches v>~30 m/s
                     # pathologies), not a brake on normal bounces.

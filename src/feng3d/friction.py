"""Friction: Coulomb kinetic law, Cundall shear springs (block-block and boundary), spring cache."""

import taichi as ti

from .params import (
    MAX_NEIGH, PRM, _P_CHUNK_MU, _P_CHUNK_MU_S, _P_FRICTION_VISC, _P_PLANE_SHEAR_K,
    _P_SHEAR_FLIP_DOT, _P_SHEAR_K_FRAC, _P_STIFFNESS_ALPHA, _P_ZETA_T,
)
from .fields import (
    Iiw_f, MAX_N, N_PLANE_SPRING, cur_frame, dbg_fr, dbg_fv, dbg_on, mass_f, n_f, pos_f,
    shear_F, shear_j, shear_maxocc, shear_n, shear_overflow, shear_pl_F, shear_seen,
    shear_upd, topo_gen,
)

# ==================================================================
# Cundall shear-spring cache lifecycle (T-05)
# ==================================================================

@ti.kernel
def _init_shear():
    topo_gen[None] += 1                  # clear(): the world restarts
    for i, s in shear_j:
        shear_j[i, s] = -1
        shear_F[i, s] = ti.Vector([0.0, 0.0, 0.0])
        shear_n[i, s] = ti.Vector([0.0, 0.0, 0.0])
        shear_seen[i, s] = -1
    for i, p, k in shear_pl_F:
        shear_pl_F[i, p, k] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def _clear_plane_spring(idx: ti.i32):
    topo_gen[None] += 1                  # spawn_box: a new block
    for p, k in ti.ndrange(N_PLANE_SPRING, 8):
        shear_pl_F[idx, p, k] = ti.Vector([0.0, 0.0, 0.0])
    # and the block-block cache row, as _spawn_commit does: _evict_shear only sweeps rows
    # < n_f, so a reused row may still hold the springs of the block culled from it.
    for s in range(MAX_NEIGH):
        shear_j[idx, s] = -1
        shear_F[idx, s] = ti.Vector([0.0, 0.0, 0.0])
        shear_n[idx, s] = ti.Vector([0.0, 0.0, 0.0])
        shear_seen[idx, s] = -1

@ti.kernel
def _evict_shear():
    # Run once per frame (after both substeps). Clear any cached pair that was
    # not touched this frame — i.e. the two blocks stopped overlapping.
    shear_upd[None] = 0          # step done: the next step's first eval only reads the springs
    fid = cur_frame[None]
    # live rows only: a struct-for over shear_j swept all MAX_N x MAX_NEIGH slots every step.
    # Rows >= n_f are reset by _spawn_commit before reuse (and carried by _cull_swap).
    for i in range(MAX_N):
        if i >= n_f[None]: continue
        for s in range(MAX_NEIGH):
            if shear_j[i, s] != -1 and shear_seen[i, s] < fid:
                shear_j[i, s] = -1
                shear_F[i, s] = ti.Vector([0.0, 0.0, 0.0])


# ==================================================================
# Coulomb kinetic law (stateless)
# ==================================================================

@ti.func
def kinetic_friction(vt: ti.math.vec3, vtm: ti.f32, cap: ti.f32, me: ti.f32, dt: ti.f32) -> ti.math.vec3:
    # clamp at me*vtm/dt so the (explicit) viscous friction can't reverse the
    # tangential velocity -> c*dt/me <= 1, stable for light blocks (mirrors the
    # normal-damping clamp; without it FRICTION_VISC*dt/m > 2 rings small rocks)
    ftm = 0.0
    if vtm > 1e-6:
        ftm = ti.min(ti.min(cap, PRM[_P_FRICTION_VISC] * vtm), me * vtm / dt)
    return (-ftm / ti.max(vtm, 1e-12)) * vt

# ==================================================================
# Boundary (plane) friction: per-corner Cundall spring
# ==================================================================

@ti.func
def plane_friction(ci: int, pid: ti.template(), k: ti.template(), dt: ti.f32,
                   pn: ti.math.vec3, rc: ti.math.vec3, vc: ti.math.vec3, vn: ti.f32,
                   me: ti.f32, Ke: ti.f32, Fn: ti.f32, mu: ti.f32) -> ti.math.vec3:
    vt = vc - vn * pn; vtm = vt.norm()
    Fkin = kinetic_friction(vt, vtm, mu * Fn, me, dt)
    Ffr = Fkin
    if ti.static(pid >= 0):
        # --- per-corner Cundall spring (T-05 step 2) ---
        # Same law as the block-block spring: once-per-step increment
        # (shear_upd), stick below mu*Fn, slip to mu*Fn, shear dashpot
        # (ZETA_T) clipped so spring + dashpot never exceeds the cap.
        # Single-assignment vectors, scalar masks (Taichi 1.7.4, see the
        # block-block spring).
        pks = PRM[_P_PLANE_SHEAR_K]
        spr = 0.0
        if pks > 0.0:
            spr = 1.0
        pupd = 1.0 * shear_upd[None]
        P0 = shear_pl_F[ci, pid, k]
        ksp = pks * Ke
        Pt1 = (P0 - P0.dot(pn) * pn) - (ksp * dt * pupd) * vt
        ptd = vt / ti.max(vtm, 1e-12)
        prx = rc.cross(ptd)
        pme = 1.0 / (1.0/mass_f[ci] + prx.dot(Iiw_f[ci] @ prx))
        pct = ti.min(2.0 * PRM[_P_ZETA_T] * ti.sqrt(ksp * pme), pme / dt)
        Pdt0 = (-pct) * vt
        pfm = Pt1.norm()
        pcap = mu * Fn
        pslip = 0.0
        if pfm > pcap and pfm > 1e-12:
            pslip = 1.0
        Pst = Pt1 * (1.0 - pslip + (pcap / ti.max(pfm, 1e-12)) * pslip)
        Ptot = Pt1 + Pdt0
        ptm = Ptot.norm()
        ptc = 1.0
        if ptm > pcap and ptm > 1e-12:
            ptc = pcap / ptm
        Pdt = (Ptot * ptc - Pt1) * (1.0 - pslip)
        if spr > 0.5 and pupd > 0.5:
            shear_pl_F[ci, pid, k] = Pst
        Ffr = (Pst + Pdt) * spr + Fkin * (1.0 - spr)
    return Ffr

@ti.func
def plane_spring_release(ci: int, pid: ti.template(), k: ti.template()):
    if ti.static(pid >= 0):
        if shear_upd[None] == 1:
            shear_pl_F[ci, pid, k] = ti.Vector([0.0, 0.0, 0.0])   # contact broken

# ==================================================================
# Block-block friction: Cundall & Strack shear spring
# ==================================================================

@ti.func
def claim_shear_slot(i: int, j: int, touch: int) -> ti.i32:
    # Find or claim the shear-cache slot of pair (i, j) in row i. -1 = spring off or cache full.
    ksf = PRM[_P_SHEAR_K_FRAC]
    cslot = -1
    free = -1
    if ksf > 0.0:
        for sc in range(MAX_NEIGH):
            if shear_j[i, sc] == j:
                cslot = sc
            elif shear_j[i, sc] == -1 and free == -1:
                free = sc
        # a skin-only pair keeps a spring it already has but never allocates one: the
        # cache (MAX_NEIGH per block) is for contacts, not near neighbours
        if cslot == -1 and free != -1 and touch == 1:
            cslot = free
            shear_j[i, cslot] = j
            shear_F[i, cslot] = ti.Vector([0.0, 0.0, 0.0])
            shear_n[i, cslot] = ti.Vector([0.0, 0.0, 0.0])   # normal unknown here; reseat test zeroes F (already 0)
            # OWNERSHIP HANDOVER. Island sleeping's staggered
            # recheck makes a sleeping neighbour active for one
            # step in ISLAND_RECHECK, which hands the pair to the
            # lower index and back. Without this the spring
            # restarted from zero at every handover (measured:
            # a cube on a sleeping slab lost its hold every 20
            # steps and slid at 0.18 units/s). Adopt the other
            # row's entry, sign-flipped: it holds the force on
            # j and the normal from j's side. Read-only on row
            # j -- this thread is the pair's only owner this
            # eval, and the stale entry ages out via
            # _evict_shear, so no new write race.
            for sc in range(MAX_NEIGH):
                if shear_j[j, sc] == i:
                    shear_F[i, cslot] = -shear_F[j, sc]
                    shear_n[i, cslot] = -shear_n[j, sc]
        if cslot == -1:
            ti.atomic_add(shear_overflow[None], 1)   # cache full
    return cslot

@ti.func
def pair_friction(i: int, j: int, cslot: int, dt: ti.f32, nhat: ti.math.vec3,
                  vrel: ti.math.vec3, vn: ti.f32, ra: ti.math.vec3, rb: ti.math.vec3,
                  cc: ti.math.vec3, me: ti.f32, Fn_mag: ti.f32, kH: ti.f32, an: ti.f32,
                  Vc: ti.f32) -> ti.math.vec3:
    # --- block-block Coulomb friction ---
    # Tangential, OPPOSES the relative sliding -> dissipative (NOT along
    # it: the old box-box anti-friction-sign bug injected energy). Applied
    # at the contact point cc, so its lever arm arrests ROTATION through
    # contact (the goal -> no rolling-resistance hack). Coulomb-capped at
    # CHUNK_MU*|Fn|, viscous-regularized, and me*vtm/dt-clamped like the
    # boundary friction so it can't ring light blocks.
    mi = mass_f[i]; mj = mass_f[j]
    vt = vrel - vn*nhat
    vtm = vt.norm()
    ksf = PRM[_P_SHEAR_K_FRAC]
    # --- Cundall & Strack (1979) shear spring (T-05) ---
    # ksf == 0 keeps the stateless kinetic law. The spring accumulates the tangential
    # force while the contact persists, so a settled pile holds a
    # static value instead of collapsing as v_t -> 0. The kinetic
    # law cannot: measured, it binds on me*vtm/dt 497 times in
    # 514, and that term is PROPORTIONAL to the slip, so it
    # vanishes exactly where it is needed. slope_test confirms no
    # rest configuration at any mu -- raising CHUNK_MU 20x, from
    # below tan(40 deg) to well above it, left the creep at
    # 2.3-2.6 units/s against a gripped threshold of 0.05.
    #
    # THE CACHE ROW IS THIS THREAD'S OWN `i`, not min(i,j).
    # `_collisions_feng` parallelises over i and one thread owns
    # each pair, so a row keyed by i has exactly one writer and
    # the slot search needs no atomics; min(i,j) would be shared
    # between two threads and could hand one pair another pair's
    # spring. Ownership flips when the lower-index block freezes
    # (island sleeping), but the normal reverses with it, so the
    # reseat test zeroes the stale value rather than carrying a
    # sign error.
    #
    # EVERY VECTOR BELOW IS SINGLE-ASSIGNMENT and only scalars
    # branch: Taichi 1.7.4 rejects a vector local re-assigned
    # inside a nested conditional in this kernel (LocalStoreStmt
    # assertion), so the spring/kinetic choice is a mask, not an
    # if/else.
    spring = 0.0
    if cslot != -1:
        spring = 1.0
    sidx = ti.max(cslot, 0)    # safe index, masked out when spring == 0
    # the stateless magnitude, used when the spring is off or the
    # cache had no free cslot
    F_kin = kinetic_friction(vt, vtm, PRM[_P_CHUNK_MU]*Fn_mag, me, dt)
    # the spring, Cundall eq 17 with the eq 19 slip cap
    keep = spring
    if shear_n[i, sidx].dot(nhat) < PRM[_P_SHEAR_FLIP_DOT]:
        keep = 0.0                                   # reseated
    k_tan = 1.5 * kH * an * an * ti.sqrt(Vc)
    ks = ti.min(ksf * k_tan, PRM[_P_STIFFNESS_ALPHA] * me / (dt*dt))
    Fs0 = shear_F[i, sidx] * keep
    upd = 1.0 * shear_upd[None]
    Fs1 = (Fs0 - Fs0.dot(nhat) * nhat) - (ks * dt * spring * upd) * vt
    # NOTE the me*vtm/dt clamp is deliberately NOT applied to the
    # spring: it is proportional to the slip, so on an accumulated
    # force it would drive the spring to zero exactly at rest,
    # which is the one thing this branch exists to prevent.
    # Cundall & Strack shear dashpot. Tangential effective mass
    # along t-hat at cc (not the normal me the ks clamp uses --
    # T-09(c)).
    zt = PRM[_P_ZETA_T]
    tdir = vt / ti.max(vtm, 1e-12)
    rxta = ra.cross(tdir); rxtb = rb.cross(tdir)
    me_t = 1.0/(1.0/mi + rxta.dot(Iiw_f[i]@rxta)
                + 1.0/mj + rxtb.dot(Iiw_f[j]@rxtb))
    ct = ti.min(2.0 * zt * ti.sqrt(ks * me_t), me_t / dt)
    Fdt0 = (-ct) * vt
    # STICK/SLIP. The slip decision is the SPRING's (its elastic
    # state, Cundall eq 19, PFC convention); the dashpot is CLIPPED so
    # spring + dashpot never exceeds mu_s*Fn. Two failures fixed:
    #  - dashpot outside the cap: clamped at me_t*vt/dt it arrests any
    #    slide in one step, so it held a tile at 55 deg against
    #    mu_s = 1.2 (tan 55 = 1.43) -- friction set by the timestep;
    #  - slip tested on the total: a block placed at rest picks up
    #    v ~ g*dt before the spring loads, and me_t*v/dt alone exceeds
    #    mu_s*Fn, so nothing ever stuck.
    # Slip: force -> mu_k*Fn along the spring trial, dashpot off.
    fsm = Fs1.norm()
    cap = PRM[_P_CHUNK_MU_S] * Fn_mag
    slip = 0.0
    if fsm > cap and fsm > 1e-12:
        slip = 1.0
    kin = PRM[_P_CHUNK_MU] * Fn_mag / ti.max(fsm, 1e-12)
    Fs = Fs1 * (1.0 - slip + kin * slip)
    tot = Fs1 + Fdt0
    tm = tot.norm()
    tcl = 1.0
    if tm > cap and tm > 1e-12:
        tcl = cap / tm
    Fdt = (tot * tcl - Fs1) * (1.0 - slip)
    if dbg_on[None] == 1:
        dbg_fr[0] = Fn_mag; dbg_fr[1] = cap; dbg_fr[2] = ks
        dbg_fr[3] = k_tan; dbg_fr[4] = vtm; dbg_fr[5] = keep
        dbg_fr[6] = Fs0.norm(); dbg_fr[7] = fsm; dbg_fr[8] = Fs.norm()
        dbg_fr[9] = Vc; dbg_fr[10] += 1.0; dbg_fr[11] = me
        ke = ti.min(int(dbg_fr[10]) - 1, 3)
        for d3 in ti.static(range(3)):
            dbg_fv[ke, d3] = vt[d3]; dbg_fv[ke, 3 + d3] = Fs0[d3]
            dbg_fv[ke, 6 + d3] = Fs[d3]; dbg_fv[ke, 9 + d3] = nhat[d3]
            dbg_fv[ke, 12 + d3] = cc[d3] + pos_f[i][d3]   # back to world
    if cslot != -1 and upd > 0.5:
        shear_F[i, sidx] = Fs
    if cslot != -1:
        shear_n[i, sidx] = nhat
        shear_seen[i, sidx] = cur_frame[None]
        ti.atomic_max(shear_maxocc[None], cslot + 1)
    return (Fs + Fdt) * spring + F_kin * (1.0 - spring)

"""Velocity Verlet (kick-drift-kick): two force evals per step, KE clamp, sleep, ballistic path."""

import taichi as ti

from .params import (
    FENG_FACELIST, FENG_REUSE, FENG_SKIN, PRM, USE_GRAPH, _P_CHUNK_ROLL, _P_CONTACT_TOL,
    _P_FRAC_BUDGET, _P_FREEZE_ABOVE, _P_FREEZE_CONTACTS, _P_GRAVITY, _P_GROUND_Y,
    _P_MAX_OMEGA, _P_MAX_SPEED, _P_ROLLING_M_REF, _P_ROLLING_RESIST, _P_SLEEP_IMBALANCE,
    _P_SLEEP_OMEGA, _P_SLEEP_SPEED, _P_SLOPE_ROLL, _P_WAKE_FORCE,
)
from .fields import (
    F_f, Iiw_f, MAX_N, MAX_PER_CELL, R_f, acc_f, active, active_f, cnt_n_f, cntct_f,
    cur_frame, fsq_f, gnd_f, grid_cell_f, grid_cnt, grid_data, he_f, ke_pre_f, ledge_on,
    mass_f, max_omdt, n_above, n_f, nosleep_f, noslope_f, nowake_f, omega_f, pos_f,
    reuse_dmax, rs_f, shear_maxocc, shear_overflow, shear_upd, sleep_f, slope_d_f,
    slope_n_f, slope_o_f, slope_on, slope_t_f, slope_umin_f, slp_f, support_f, tau_f, vel_f,
)
from .friction import (
    _evict_shear,
)
from .boundary import (
    BOUNDARY_KERNELS,
)
from .broadphase import (
    _build_grid, _feng_cells, _feng_claim, _grid_hash, _mark_active,
)
from .contact import (
    _feng_facelist, _feng_faces, _feng_forces,
)


@ti.kernel
def _begin_step(fid: ti.i32):
    # Frame id + per-frame debug counters, in ONE asynchronous launch. They were four
    # Python-side field writes per step, each its own launch + sync (measured ~0.27 ms
    # per step together, ~1.1 ms/frame at sub 2).
    cur_frame[None] = fid
    shear_overflow[None] = 0
    shear_maxocc[None] = 0
    max_omdt[None] = 0.0

@ti.kernel
def _kick_drift(dt: ti.f32):
    shear_upd[None] = 1          # the next force eval is at x(t+dt): advance the springs
    for i in range(MAX_N):
        if i >= n_f[None]: continue
        if sleep_f[i] == 1:
            # island sleeping: frozen-interior block (no awake neighbour) has its forces
            # intentionally NOT computed this step -> F_f is incomplete, do NOT wake-check it.
            # It stays frozen until a neighbour wakes (then _mark_active promotes it next step).
            if active[i] == 0:
                continue
            # stay asleep unless OUT OF EQUILIBRIUM (lost support -> would fall, or tipping);
            # real impacts are handled by the WAKE_VEL wake in _collisions_feng. A settled
            # pile block has gravity balanced by contacts -> net force ~0 -> stays frozen.
            hmx = ti.max(he_f[i][0], ti.max(he_f[i][1], he_f[i][2]))
            wf = PRM[_P_WAKE_FORCE] * mass_f[i] * ti.abs(PRM[_P_GRAVITY])
            if (F_f[i].norm() > wf or tau_f[i].norm() > wf * hmx) and nowake_f[i] == 0:
                sleep_f[i] = 0
            else:
                continue
        vel_f[i] += F_f[i] / mass_f[i] * (dt*0.5)
        omega_f[i] += Iiw_f[i] @ tau_f[i] * (dt*0.5)
        pos_f[i] += vel_f[i] * dt
        dtheta = omega_f[i] * dt
        if ti.static(FENG_REUSE == 1):
            # no point of the box moves further than this (a rotation by theta moves a point at
            # radius r by a chord <= theta * r; |he| bounds r)
            acc_f[i] += vel_f[i].norm() * dt + dtheta.norm() * he_f[i].norm()
            ti.atomic_max(reuse_dmax[None], acc_f[i])
        theta = dtheta.norm()
        if theta >= 1e-7:
            k = dtheta / theta
            c = ti.cos(theta); s = ti.sin(theta); t = 1.0 - c
            d00 = c+t*k[0]*k[0];     d01 = t*k[0]*k[1]-s*k[2]; d02 = t*k[0]*k[2]+s*k[1]
            d10 = t*k[1]*k[0]+s*k[2]; d11 = c+t*k[1]*k[1];     d12 = t*k[1]*k[2]-s*k[0]
            d20 = t*k[2]*k[0]-s*k[1]; d21 = t*k[2]*k[1]+s*k[0]; d22 = c+t*k[2]*k[2]
            for col in ti.static(range(3)):
                r0 = R_f[i][0, col]; r1 = R_f[i][1, col]; r2 = R_f[i][2, col]
                R_f[i][0, col] = d00*r0 + d01*r1 + d02*r2
                R_f[i][1, col] = d10*r0 + d11*r1 + d12*r2
                R_f[i][2, col] = d20*r0 + d21*r1 + d22*r2

@ti.kernel
def _snapshot_ke():
    for i in range(MAX_N):
        if i >= n_f[None]: continue
        if sleep_f[i] == 1:
            ke_pre_f[i] = 0.0
        else:
            v2 = vel_f[i].dot(vel_f[i])
            if v2 < 1e-8:
                ke_pre_f[i] = -1.0
            else:
                ke_pre_f[i] = 0.5 * mass_f[i] * v2

@ti.kernel
def _kick_clamp_sleep(dt: ti.f32):
    for i in range(MAX_N):
        if i >= n_f[None]: continue
        if sleep_f[i] == 1:
            # same wake rules as _kick_drift
            if active[i] == 0:
                continue
            hmx = ti.max(he_f[i][0], ti.max(he_f[i][1], he_f[i][2]))
            wf = PRM[_P_WAKE_FORCE] * mass_f[i] * ti.abs(PRM[_P_GRAVITY])
            if (F_f[i].norm() > wf or tau_f[i].norm() > wf * hmx) and nowake_f[i] == 0:
                sleep_f[i] = 0
            else:
                continue
        vel_f[i] += F_f[i] / mass_f[i] * (dt*0.5)
        omega_f[i] += Iiw_f[i] @ tau_f[i] * (dt*0.5)
        ke_now = 0.5 * mass_f[i] * vel_f[i].dot(vel_f[i])
        he_max = ti.max(he_f[i][0], ti.max(he_f[i][1], he_f[i][2]))
        grav_work = mass_f[i] * ti.abs(PRM[_P_GRAVITY]) * he_max * dt
        budget = grav_work
        if ke_pre_f[i] >= 0.0:
            budget = ke_pre_f[i] + PRM[_P_FRAC_BUDGET] * mass_f[i] * ti.abs(PRM[_P_GRAVITY]) * he_max + grav_work
        if ke_now > budget and ke_now > 1e-6:
            scale = ti.sqrt(budget / ke_now)
            vel_f[i] *= scale
            omega_f[i] *= scale
        Ri = R_f[i]; rsi = rs_f[i]
        if slp_f[i]==1:
            ob = Ri.transpose() @ omega_f[i]
            for d in ti.static(range(3)):
                ob[d] *= ti.max(0.0, 1.0 - PRM[_P_SLOPE_ROLL]*rsi[d]*dt)
            omega_f[i] = Ri @ ob
        elif gnd_f[i]==1:
            ob = Ri.transpose() @ omega_f[i]
            for d in ti.static(range(3)):
                ob[d] *= ti.max(0.0, 1.0 - PRM[_P_ROLLING_RESIST]*rsi[d]*dt)
            omega_f[i] = Ri @ ob
        if cntct_f[i]==1:
            ob = Ri.transpose() @ omega_f[i]
            for d in ti.static(range(3)):
                ob[d] *= ti.max(0.0, 1.0 - PRM[_P_CHUNK_ROLL]*rsi[d]*dt)
            omega_f[i] = Ri @ ob
        sp = vel_f[i].norm(); om = omega_f[i].norm()
        if sp > PRM[_P_MAX_SPEED]: vel_f[i] *= PRM[_P_MAX_SPEED]/sp
        if om > PRM[_P_MAX_OMEGA]: omega_f[i] *= PRM[_P_MAX_OMEGA]/om
        ss = ti.max(1.0, (PRM[_P_ROLLING_M_REF]/mass_f[i])**(1.0/3.0))
        # slp_f added (T-05): with the boundary spring a block resting only on the slope
        # plane is genuinely at rest (stored friction balances m*g*sin(theta)), so it may
        # sleep. Without the spring its at-rest friction is zero and the WAKE_FORCE check
        # (0.5 < sin 40) can wake it again.
        if (gnd_f[i]==1 or cntct_f[i]==1 or slp_f[i]==1) and nosleep_f[i] == 0:
            # buried = load-bearing quasi-static core: many contacts + something resting on top.
            # Such a block's spin is numerical, so freeze it on the speed gate ALONE (bypass the
            # omega gate that roll=0 broke). Normal sleep still needs BOTH speed and omega low.
            buried = (n_above[i] >= PRM[_P_FREEZE_ABOVE] and cnt_n_f[i] >= PRM[_P_FREEZE_CONTACTS])
            calm = sp < PRM[_P_SLEEP_SPEED]*ss and (om < PRM[_P_SLEEP_OMEGA]*ss or buried)
            rimb = PRM[_P_SLEEP_IMBALANCE]
            if calm and rimb > 0.0:                   # T-16: balanced relative to its own forces
                hmx = ti.max(he_f[i][0], ti.max(he_f[i][1], he_f[i][2]))
                fs = rimb * ti.sqrt(fsq_f[i])
                calm = F_f[i].norm() < fs and tau_f[i].norm() < fs * hmx
            if calm:
                vel_f[i] = ti.Vector([0.0,0.0,0.0])
                omega_f[i] = ti.Vector([0.0,0.0,0.0])
                sleep_f[i] = 1

@ti.kernel
def _compute_support():
    sn = slope_n_f[None]
    for i in range(n_f[None]):
        R = R_f[i]; h = he_f[i]
        s = 0.0
        for j in ti.static(range(3)):
            s += ti.abs(R[0,j]*sn[0] + R[1,j]*sn[1] + R[2,j]*sn[2]) * h[j]
        support_f[i] = s

@ti.kernel
def _classify():
    nn = n_f[None]
    gn = ti.Vector([0.0, 1.0, 0.0])
    for i in range(nn):
        if sleep_f[i] == 1:
            active_f[i] = 0
            continue
        pi = pos_f[i]
        hi = he_f[i]
        Ri = R_f[i]
        radius = 0.0
        for ax in ti.static(range(3)):
            radius += ti.abs(Ri[0,ax]*gn[0] + Ri[1,ax]*gn[1] + Ri[2,ax]*gn[2]) * hi[ax]
        near_surface = 0
        if ledge_on[None] == 1:
            gnd_dist = pi.dot(gn) - PRM[_P_GROUND_Y]
            if gnd_dist - radius < PRM[_P_CONTACT_TOL]:
                near_surface = 1
        if slope_on[None] == 1 and noslope_f[i] == 0 and (pi - slope_o_f[None]).dot(slope_t_f[None]) >= slope_umin_f[None]:
            sn = slope_n_f[None]
            s_radius = 0.0
            for ax in ti.static(range(3)):
                s_radius += ti.abs(Ri[0,ax]*sn[0] + Ri[1,ax]*sn[1] + Ri[2,ax]*sn[2]) * hi[ax]
            slp_dist = pi.dot(sn) - slope_d_f[None]
            if slp_dist - s_radius < PRM[_P_CONTACT_TOL]:
                near_surface = 1
        if near_surface == 1:
            active_f[i] = 1
            continue
        cx = int(ti.floor(pi[0] / grid_cell_f[None]))
        cy = int(ti.floor(pi[1] / grid_cell_f[None]))
        cz = int(ti.floor(pi[2] / grid_cell_f[None]))
        has_neighbor = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    bucket = _grid_hash(cx+dx, cy+dy, cz+dz)
                    cnt = grid_cnt[bucket]
                    if cnt > MAX_PER_CELL: cnt = MAX_PER_CELL
                    for slot in range(cnt):
                        j = grid_data[bucket, slot]
                        if j != i and sleep_f[j] == 0:
                            has_neighbor = 1
        if has_neighbor == 1:
            active_f[i] = 1
        else:
            active_f[i] = 0

@ti.kernel
def _ballistic_step(dt: ti.f32):
    for i in range(n_f[None]):
        if active_f[i] == 1 or sleep_f[i] == 1: continue
        pos_f[i] += vel_f[i] * dt + 0.5 * PRM[_P_GRAVITY] * dt * dt * ti.Vector([0.0, 1.0, 0.0])
        vel_f[i][1] += PRM[_P_GRAVITY] * dt
        dtheta = omega_f[i] * dt
        theta = dtheta.norm()
        if theta >= 1e-7:
            k = dtheta / theta
            c = ti.cos(theta); s = ti.sin(theta); t = 1.0 - c
            d00 = c+t*k[0]*k[0];     d01 = t*k[0]*k[1]-s*k[2]; d02 = t*k[0]*k[2]+s*k[1]
            d10 = t*k[1]*k[0]+s*k[2]; d11 = c+t*k[1]*k[1];     d12 = t*k[1]*k[2]-s*k[0]
            d20 = t*k[2]*k[0]-s*k[1]; d21 = t*k[2]*k[1]+s*k[0]; d22 = c+t*k[2]*k[2]
            for col in ti.static(range(3)):
                r0 = R_f[i][0, col]; r1 = R_f[i][1, col]; r2 = R_f[i][2, col]
                R_f[i][0, col] = d00*r0 + d01*r1 + d02*r2
                R_f[i][1, col] = d10*r0 + d11*r1 + d12*r2
                R_f[i][2, col] = d20*r0 + d21*r1 + d22*r2

@ti.kernel
def _reortho():
    for i in range(MAX_N):
        if i >= n_f[None]: continue
        c0 = ti.Vector([R_f[i][0,0], R_f[i][1,0], R_f[i][2,0]]).normalized()
        c1_raw = ti.Vector([R_f[i][0,1], R_f[i][1,1], R_f[i][2,1]])
        c1 = (c1_raw - c1_raw.dot(c0)*c0).normalized()
        c2 = c0.cross(c1)
        R_f[i][0,0] = c0[0]; R_f[i][1,0] = c0[1]; R_f[i][2,0] = c0[2]
        R_f[i][0,1] = c1[0]; R_f[i][1,1] = c1[1]; R_f[i][2,1] = c1[2]
        R_f[i][0,2] = c2[0]; R_f[i][1,2] = c2[1]; R_f[i][2,2] = c2[2]


# ==================================================================
# One step
# ==================================================================

def _collisions_feng(dt):
    # Feng energy-conserving overlap-VOLUME contact (Force = -grad w(V_c)).
    _feng_cells()              # per (block, cell): broadphase -> per-block candidates
    _feng_claim()              # per block: shear-slot claim -> pair list
    if FENG_FACELIST:
        _feng_facelist()       # per pair: face cull -> list of surviving (pair, face)
    _feng_faces()              # per (pair, face): the clip
    _feng_forces(dt)           # per pair: forces from the face sums


_step_graphs = {}

def _step_graph(rebuild_first):
    """One substep as a recorded ti.graph (USE_GRAPH). Measured on Metal: a Python kernel
    call costs ~16 us of dispatch inside step(), whatever the kernel does (an EMPTY world's
    step() took 274 us for ~17 launches); a graph replays a dispatch for ~2.3 us. The kernel
    sequence is exactly step()'s; the Python-side choices are fixed per graph (cached by key)."""
    key = (rebuild_first, FENG_FACELIST, FENG_REUSE, FENG_SKIN)
    g = _step_graphs.get(key)
    if g is not None:
        return g
    gb = ti.graph.GraphBuilder()
    a_dt = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "dt", ti.f32)
    a_fid = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "fid", ti.i32)
    a_full = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "full", ti.i32)
    gb.dispatch(_begin_step, a_fid)
    if rebuild_first:
        gb.dispatch(_build_grid)
    for half in range(2):
        gb.dispatch(_mark_active, a_full)
        for kern in BOUNDARY_KERNELS:
            gb.dispatch(kern, a_dt)
        gb.dispatch(_feng_cells); gb.dispatch(_feng_claim)
        if FENG_FACELIST:
            gb.dispatch(_feng_facelist)
        gb.dispatch(_feng_faces); gb.dispatch(_feng_forces, a_dt)
        if half == 0:
            gb.dispatch(_kick_drift, a_dt)
            gb.dispatch(_build_grid)
    gb.dispatch(_kick_clamp_sleep, a_dt)
    gb.dispatch(_evict_shear)
    g = gb.compile()
    _step_graphs[key] = g
    return g


def _forces(dt):
    for kern in BOUNDARY_KERNELS:    # reset + gravity, then one kernel per boundary type
        kern(dt)
    _collisions_feng(dt)


def _step_python(dt, fid, rebuild_first):
    _begin_step(fid)
    # island sleeping: tag the live region; frozen interior is skipped (B scan-skip). The
    # floater safety net depends on the recheck mode: one-ring needs a periodic full=1 step;
    # staggered (island_stag=1) carries its own per-step net so full stays 0 (no scan spike).
    full = -1                  # _mark_active decides from island_stag / frame id on the GPU
    # Two force evals per step: eval at x(t), half-kick + drift, eval at x(t+dt), half-kick.
    # NB the single-eval velocity-Verlet (plan D: carry F across steps for the opening kick)
    # was TRIED and REVERTED. Carrying the force makes the velocity-dependent DAMPING half a
    # step stale, which under-damps the pile so it never reaches the sleep gate (pile_test under
    # g=-40: vmax pinned ~7 m/s, the pile never goes quiet, and the residual jitter wake-
    # cascades the WHOLE pile to active -> 42 ms vs this scheme's full settle ~10 ms). Re-
    # evaluating the (cheap broadphase-skipped) force each half keeps the damping current, so
    # settle-and-hold scenes actually sleep -- the prerequisite for the scan-skip to pay off.
    if rebuild_first:
        _build_grid()
    _mark_active(full)
    _forces(dt)
    _kick_drift(dt)
    _build_grid()
    _mark_active(full)
    _forces(dt)
    _kick_clamp_sleep(dt)
    _evict_shear()   # drop pairs that stopped touching this frame


def step(dt, fid, rebuild_first):
    if USE_GRAPH:
        _step_graph(rebuild_first).run({"dt": float(dt), "fid": int(fid), "full": -1})
    else:
        _step_python(dt, fid, rebuild_first)

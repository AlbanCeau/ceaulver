"""Broadphase: spatial hash grid, island-sleeping active set, candidate pairs -> pair list."""

import taichi as ti

from .params import (
    FENG_DEDUPE, FENG_REUSE, FENG_SKIN, island_recheck_f,
)
from .fields import (
    MAX_CAND, MAX_N, MAX_PAIRS, MAX_PER_CELL, NUM_BUCKETS, R_f, acc_f, act_build, active,
    cand_j, cand_n, cand_overflow, cur_frame, dbg_cand, dbg_clip, dbg_on, dbg_skip, facc_V,
    facc_a, facc_c, facc_gA, facc_gB, grid_cell_f, grid_cnt, grid_data, grid_pr, he_f,
    island_stag, n_f, pair_i, pair_j, pair_n, pair_overflow, pair_s, per_L, per_nz, per_on,
    pos_f, reuse_dmax, reuse_evals, reuse_force, reuse_mode, reuse_rebuilds, sleep_f,
    topo_build, topo_gen,
)
from .friction import (
    claim_shear_slot,
)

# ==================================================================
# Spatial hash grid
# ==================================================================

@ti.func
def _mi_z(dz: ti.f32) -> ti.f32:
    """Minimum image of a z separation (identity unless periodic)."""
    r = dz
    if per_on[None] == 1:
        L = per_L[None]
        r = dz - L * ti.round(dz / L)
    return r

@ti.func
def _mi(d):
    """Minimum image of a separation vector p_j - p_i, in z."""
    return ti.Vector([d[0], d[1], _mi_z(d[2])])

@ti.func
def _grid_hash(cx: ti.i32, cy: ti.i32, cz_in: ti.i32) -> ti.i32:
    cz = cz_in
    if per_on[None] == 1:                     # periodic z: wrap the cell index
        nz = per_nz[None]
        cz = ((cz_in % nz) + nz) % nz
    h = cx * 73856093 ^ cy * 19349663 ^ cz * 83492791
    return h & (NUM_BUCKETS - 1)

@ti.kernel
def _build_grid():
    for b in range(NUM_BUCKETS):
        grid_cnt[b] = 0
    for i in range(MAX_N):
        if i >= n_f[None]: continue
        p = pos_f[i]
        cx = int(ti.floor(p[0] / grid_cell_f[None]))
        cy = int(ti.floor(p[1] / grid_cell_f[None]))
        cz = int(ti.floor(p[2] / grid_cell_f[None]))
        slot = ti.atomic_add(grid_cnt[_grid_hash(cx, cy, cz)], 1)
        if slot < MAX_PER_CELL:
            grid_data[_grid_hash(cx, cy, cz), slot] = i
            grid_pr[_grid_hash(cx, cy, cz), slot] = ti.Vector([p[0], p[1], p[2], he_f[i].norm()])

@ti.kernel
def _mark_active(full: ti.i32):
    # Island sleeping. active=1 == this block SCANS the broadphase, gets boundary force,
    # integrates, and is wake-checked. active=0 == frozen interior: skipped everywhere (the B
    # scan-skip). A SLEEPING block re-earns a wake-check two ways (see island_stag):
    #   one-ring (island_stag==0): promoted if it sits in an awake block's neighbour cells, plus
    #     the periodic full=1 re-check. Optimal when awake is small/clustered/zero.
    #   staggered (island_stag==1): rechecked when i%R == frame%R (1/R of all blocks per step).
    #     Immune to the one-ring blanket on a spread awake set.
    # Either way a frozen block still gets full support + an impact (WAKE_VEL) wake from its AWAKE
    # neighbours scanning IT (pair ownership in _collisions_feng); only quasi-static support loss
    # waits for the recheck (<= ISLAND_RECHECK steps).
    nn = n_f[None]
    stag = island_stag[None]
    irc = island_recheck_f[None]
    rc = cur_frame[None] % irc
    # full == -1: decide here (step() passes this). Reading island_stag from Python every
    # step was a host read, i.e. a full GPU drain per step.
    fl = full
    if full == -1:
        fl = 0
        if stag == 0 and rc == 0:
            fl = 1
    for i in range(MAX_N):
        if i >= nn: continue
        if fl == 1:
            active[i] = 1
        elif sleep_f[i] == 0:
            active[i] = 1                                  # awake -> active
        elif stag == 1 and (i % irc) == rc:
            active[i] = 1                                  # staggered floater recheck
        else:
            active[i] = 0                                  # provisionally frozen
    for i in range(MAX_N):                                    # one-ring promotion (island_stag==0 only)
        if i >= nn: continue
        if stag == 0 and fl == 0 and sleep_f[i] == 0:
            pi = pos_f[i]
            cx = int(ti.floor(pi[0] / grid_cell_f[None]))
            cy = int(ti.floor(pi[1] / grid_cell_f[None]))
            cz = int(ti.floor(pi[2] / grid_cell_f[None]))
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        bucket = _grid_hash(cx + dx, cy + dy, cz + dz)
                        cnt = grid_cnt[bucket]
                        if cnt > MAX_PER_CELL: cnt = MAX_PER_CELL
                        for slot in range(cnt):
                            j = grid_data[bucket, slot]
                            if sleep_f[j] == 1:
                                active[j] = 1               # idempotent -> no atomic needed

@ti.kernel
def _feng_cells():
    # PASS 1a of _collisions_feng: one thread per (block, neighbour cell), 27 per block
    # (as one thread per block the 27-cell walk, ~103 candidates, ran serially on only
    # ~1000 threads). Survivors of sphere / ownership / AABB / SAT go into
    # the block's own candidate list; the shear-slot claim stays per block (_feng_claim) so
    # each cache row keeps exactly one writer (T-05).
    # FENG_REUSE (see its definition): every test is widened by the skin; mode 1 keeps the
    # list and only blocks that became active since the rebuild scan, for partners that were
    # inactive then (the pairs nobody listed).
    nn = n_f[None]
    skin = 0.0
    mode = 0
    if ti.static(FENG_REUSE == 1):
        skin = FENG_SKIN
        reuse_evals[None] += 1
        if (reuse_force[None] == 0 and topo_gen[None] == topo_build[None]
                and 2.0 * reuse_dmax[None] <= FENG_SKIN):
            mode = 1
        else:
            reuse_rebuilds[None] += 1
            reuse_dmax[None] = 0.0; reuse_force[None] = 0; topo_build[None] = topo_gen[None]
    reuse_mode[None] = mode
    if mode == 0:
        pair_n[None] = 0
    for t in range(nn * 27):
        i = t // 27
        o = t % 27
        if active[i] == 0:
            if o == 0 and dbg_on[None] == 1: ti.atomic_add(dbg_skip[None], 1)
            continue
        if ti.static(FENG_REUSE == 1):
            if mode == 1 and act_build[i] == 1: continue     # its pairs are listed
        dx = o // 9 - 1; dy = (o // 3) % 3 - 1; dz = o % 3 - 1
        pi = pos_f[i]
        r_i = he_f[i].norm()                 # bounding-sphere radius of i
        cx = int(ti.floor(pi[0] / grid_cell_f[None]))
        cy = int(ti.floor(pi[1] / grid_cell_f[None]))
        cz = int(ti.floor(pi[2] / grid_cell_f[None]))
        bucket = _grid_hash(cx + dx, cy + dy, cz + dz)
        cnt = grid_cnt[bucket]
        if cnt > MAX_PER_CELL: cnt = MAX_PER_CELL
        if ti.static(FENG_DEDUPE == 1):      # T-10: skip a bucket an earlier offset already owns
            for q in range(o):
                if _grid_hash(cx + q // 9 - 1, cy + (q // 3) % 3 - 1, cz + q % 3 - 1) == bucket:
                    cnt = 0
        for slot in range(cnt):
            # bounding-sphere reject from the grid's own compact record:
            # exact (a box lies inside its |he| sphere, so no overlap is
            # ever rejected), and it skips the scattered reads of j.
            pr = grid_pr[bucket, slot]
            dsx = pr[0] - pi[0]; dsy = pr[1] - pi[1]; dsz = _mi_z(pr[2] - pi[2])
            rs = pr[3] + r_i + 1e-4 + skin
            if dsx*dsx + dsy*dsy + dsz*dsz > rs*rs: continue
            j = grid_data[bucket, slot]
            # pair ownership (i is active here): both-active -> the LOWER
            # index owns (dedup). active(i)-inactive(j) -> i owns regardless of order
            # (the inactive j never scans), so the active block always gets full
            # support + a valid wake check. both-inactive never reach here.
            if j == i: continue
            if ti.static(FENG_REUSE == 1):
                if mode == 1 and act_build[j] == 1: continue   # listed from j's side
            if active[j] == 1 and j < i: continue
            if dbg_on[None] == 1: ti.atomic_add(dbg_cand[None], 1)   # probe: candidate pair
            # --- AABB broadphase ---
            dd = _mi(pos_f[j] - pos_f[i])
            aabb_ok = 1
            for k in ti.static(range(3)):
                hi_ = 0.0; hj_ = 0.0
                for a in ti.static(range(3)):
                    hi_ += ti.abs(R_f[i][k,a]) * he_f[i][a]
                    hj_ += ti.abs(R_f[j][k,a]) * he_f[j][a]
                if ti.abs(dd[k]) > hi_ + hj_ + skin: aabb_ok = 0
            if aabb_ok == 0: continue
            # --- cheap 6-face-axis SAT early-out ---
            # The clip has no early-out; without this it runs the full
            # 12-face x 6-plane clip even for AABB-overlapping pairs that
            # don't actually intersect (V=0). Face-axis separation => the
            # OBBs are truly disjoint, so this only skips genuine misses.
            Ri = R_f[i]; Rj = R_f[j]; hi2 = he_f[i]; hj2 = he_f[j]
            sep = 0
            touch = 1                        # exact SAT (no skin): only touching pairs claim a slot
            for k in ti.static(range(3)):
                ax = ti.Vector([Ri[0,k], Ri[1,k], Ri[2,k]])
                rj = (ti.abs(ax[0]*Rj[0,0]+ax[1]*Rj[1,0]+ax[2]*Rj[2,0])*hj2[0]
                    + ti.abs(ax[0]*Rj[0,1]+ax[1]*Rj[1,1]+ax[2]*Rj[2,1])*hj2[1]
                    + ti.abs(ax[0]*Rj[0,2]+ax[1]*Rj[1,2]+ax[2]*Rj[2,2])*hj2[2])
                if ti.abs(ax.dot(dd)) > hi2[k] + rj + skin: sep = 1
                if ti.abs(ax.dot(dd)) > hi2[k] + rj: touch = 0
                ax2 = ti.Vector([Rj[0,k], Rj[1,k], Rj[2,k]])
                ri = (ti.abs(ax2[0]*Ri[0,0]+ax2[1]*Ri[1,0]+ax2[2]*Ri[2,0])*hi2[0]
                    + ti.abs(ax2[0]*Ri[0,1]+ax2[1]*Ri[1,1]+ax2[2]*Ri[2,1])*hi2[1]
                    + ti.abs(ax2[0]*Ri[0,2]+ax2[1]*Ri[1,2]+ax2[2]*Ri[2,2])*hi2[2])
                if ti.abs(ax2.dot(dd)) > ri + hj2[k] + skin: sep = 1
                if ti.abs(ax2.dot(dd)) > ri + hj2[k]: touch = 0
            if sep == 1: continue
            if dbg_on[None] == 1: ti.atomic_add(dbg_clip[None], 1)   # probe: ran the full clip
            kk = ti.atomic_add(cand_n[i], 1)
            if kk < MAX_CAND:
                cand_j[i, kk] = ti.select(touch == 1, j, -j - 1)   # skin-only pair: encoded negative
            else:
                ti.atomic_add(cand_overflow[None], 1)


@ti.kernel
def _feng_claim():
    # PASS 1b: one thread per block -- claim the shear slot for each of its
    # candidates (single writer per cache row), append to the pair list, and
    # reset the block's candidate counter for the next eval.
    nn = n_f[None]
    mode = reuse_mode[None]
    for i in range(MAX_N):
        if i >= nn: continue
        if ti.static(FENG_REUSE == 1):
            if mode == 0:
                acc_f[i] = 0.0; act_build[i] = active[i]
            elif active[i] == 1:
                act_build[i] = 1         # appended now (or already listed)
        m = ti.min(cand_n[i], MAX_CAND)
        for kk in range(m):
            j = cand_j[i, kk]
            touch = 1
            if j < 0:                    # skin-only candidate (FENG_REUSE): not touching yet
                j = -j - 1; touch = 0
            cslot = claim_shear_slot(i, j, touch)
            pidx = ti.atomic_add(pair_n[None], 1)
            if pidx < MAX_PAIRS:
                pair_i[pidx] = i; pair_j[pidx] = j; pair_s[pidx] = cslot
                facc_V[pidx] = 0.0
                facc_c[pidx] = ti.Vector([0.0, 0.0, 0.0])
                facc_a[pidx] = ti.Vector([0.0, 0.0, 0.0])
                facc_gA[pidx] = ti.Vector([0.0, 0.0, 0.0])
                facc_gB[pidx] = ti.Vector([0.0, 0.0, 0.0])
            else:
                ti.atomic_add(pair_overflow[None], 1)
        cand_n[i] = 0

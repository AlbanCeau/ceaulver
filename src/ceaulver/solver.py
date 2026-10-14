"""Solver: the host-side interface (spawn, cull, step, instance export)."""

import time as _time
import math as _math

import numpy as np
import taichi as ti

from . import params
from .params import (
    FENG_REUSE, MAX_NEIGH, sync_params,
)
from .fields import (
    GEOM_MAX, INST_FLOATS, Iil_f, Iiw_f, MAX_BLOCK_HE, MAX_N, N_PLANE_SPRING, R_f,
    SPAWN_MAX, active_f, birth_f, birth_now, box_on, box_w, cull_n_remove, cull_remove,
    cyl_R_f, cyl_cx_f, cyl_cz_f, cyl_on, grid_cell_f, he_f, ledge_on, ledge_x0, ledge_x1,
    ledge_z0, ledge_z1, mass_f, n_f, nosleep_f, noslope_f, nowake_f, omega_f, per_L, per_nz,
    per_on, pos_f, reuse_force, rs_f, shear_F, shear_j, shear_maxocc, shear_n,
    shear_overflow, shear_pl_F, shear_seen, side_on, sleep_f, slope_d_f, slope_mu_y_hi,
    slope_feng_on, slope_mu_y_lo, slope_n_f, slope_o_f, slope_on, slope_t_f, slope_umin_f, sp_Iil, sp_R,
    sp_he, sp_mass, sp_n, sp_pos, sp_rs, topo_gen, vel_f, wall_d_f, wall_n_f, wall_on,
)
from .friction import (
    _clear_plane_spring, _init_shear,
)
from .broadphase import (
    _build_grid,
)
from .integrator import (
    _ballistic_step, _classify, _compute_support, _reortho, _snapshot_ke,
)
from .integrator import step as _integrator_step


@ti.kernel
def _export_instances(out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    nn = ti.min(n_f[None], GEOM_MAX)
    for i in range(nn):
        p = pos_f[i]; h = he_f[i]; R = R_f[i]
        out[i, 0] = p[0]; out[i, 1] = p[1]; out[i, 2] = p[2]
        out[i, 3] = h[0]; out[i, 4] = h[1]; out[i, 5] = h[2]
        for c in ti.static(range(3)):
            for r in ti.static(range(3)):
                out[i, 6 + c*3 + r] = R[r, c]

@ti.kernel
def _cull_mark(despawn_y: ti.f32):
    cull_n_remove[None] = 0
    ti.loop_config(serialize=True)
    for i in range(n_f[None]):
        if pos_f[i][1] <= despawn_y:
            idx = cull_n_remove[None]
            cull_remove[idx] = i
            cull_n_remove[None] = idx + 1

@ti.kernel
def _cull_set(idx: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    # host-side cull (Solver.cull): the removal list, ascending, as _cull_mark builds it
    cull_n_remove[None] = idx.shape[0]
    for k in range(idx.shape[0]):
        cull_remove[k] = idx[k]

@ti.kernel
def _cull_swap():
    topo_gen[None] += 1
    nr = cull_n_remove[None]
    nn = n_f[None]
    ti.loop_config(serialize=True)
    for k in range(nr):
        ri = cull_remove[nr - 1 - k]
        last = nn - 1
        if ri < last:
            pos_f[ri] = pos_f[last]
            vel_f[ri] = vel_f[last]
            omega_f[ri] = omega_f[last]
            R_f[ri] = R_f[last]
            he_f[ri] = he_f[last]
            mass_f[ri] = mass_f[last]
            Iil_f[ri] = Iil_f[last]
            Iiw_f[ri] = Iiw_f[last]
            rs_f[ri] = rs_f[last]
            sleep_f[ri] = sleep_f[last]
            noslope_f[ri] = noslope_f[last]
            birth_f[ri] = birth_f[last]
            # Carry the shear cache so the swapped-in block keeps its own state.
            # (Other blocks' cached indices that referenced `last` go stale, but
            #  the per-frame eviction + normal-flip reset heal that within a frame.)
            for s in range(MAX_NEIGH):
                shear_j[ri, s] = shear_j[last, s]
                shear_F[ri, s] = shear_F[last, s]
                shear_n[ri, s] = shear_n[last, s]
                shear_seen[ri, s] = shear_seen[last, s]
            for pp_ in range(N_PLANE_SPRING):
                for kc_ in range(8):
                    shear_pl_F[ri, pp_, kc_] = shear_pl_F[last, pp_, kc_]
        nn -= 1
    n_f[None] = nn

@ti.kernel
def _spawn_commit(base: ti.i32):
    topo_gen[None] += 1
    for i in range(sp_n[None]):
        j = base + i
        pos_f[j] = sp_pos[i]
        vel_f[j] = ti.Vector([0.0, 0.0, 0.0])
        omega_f[j] = ti.Vector([0.0, 0.0, 0.0])
        he_f[j] = sp_he[i]
        R_f[j] = sp_R[i]
        mass_f[j] = sp_mass[i]
        Iil_f[j] = sp_Iil[i]
        d = sp_Iil[i]
        for row in ti.static(range(3)):
            for col in ti.static(range(3)):
                Iiw_f[j][row, col] = (d[0]*sp_R[i][row,0]*sp_R[i][col,0]
                                     + d[1]*sp_R[i][row,1]*sp_R[i][col,1]
                                     + d[2]*sp_R[i][row,2]*sp_R[i][col,2])
        rs_f[j] = sp_rs[i]
        sleep_f[j] = 0
        noslope_f[j] = 0
        birth_f[j] = birth_now[None]
        active_f[j] = 1
        # Fresh blocks start with an empty shear cache (slot may be reused).
        for s in range(MAX_NEIGH):
            shear_j[j, s] = -1
            shear_F[j, s] = ti.Vector([0.0, 0.0, 0.0])
            shear_n[j, s] = ti.Vector([0.0, 0.0, 0.0])
            shear_seen[j, s] = -1
        for pp_ in range(N_PLANE_SPRING):
            for kc_ in range(8):
                shear_pl_F[j, pp_, kc_] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def _apply_impulse(ix: ti.f32, iy: ti.f32, iz: ti.f32):
    imp = ti.Vector([ix, iy, iz])
    for i in range(n_f[None]):
        vel_f[i] += imp; sleep_f[i] = 0

class Solver:
    def __init__(self, slope_normal=None, slope_d=0.0, ledge_bounds=None,
                 slope_mu_y_range=None, wall=None, cylinder=None, box=None,
                 max_he=None, slope_feng=False):
        n_f[None] = 0
        # Host mirror of n_f. Only this class changes the block count (spawn here, cull here),
        # so it always knows it; a host read of n_f[None] is a launch + full GPU drain
        # (~0.15 ms on Metal).
        self._n = 0
        self._exp = None          # last get_instance_data rows (host cull reads their y)
        reuse_force[None] = 1
        self.last_solver_ms = 0.0
        # Per-world broadphase cell: size to this world's largest block so the 3x3x3
        # neighbour scan stays correct (cell >= 2*max_he*sqrt(3)) AND the grid is fine
        # enough that dense piles don't overflow a bucket. Default = the he=5 sizing.
        self._grid_max_he = float(max_he) if max_he is not None else MAX_BLOCK_HE
        grid_cell_f[None] = 2.0 * self._grid_max_he * _math.sqrt(3.0)
        self._spawn_he_warned = False
        self._grid_valid = False
        self._step_count = 0
        self._frame_id = 0
        if slope_normal is not None:
            slope_n_f[None] = list(slope_normal)
            slope_d_f[None] = float(slope_d)
            slope_on[None] = 1
        else:
            slope_on[None] = 0
        slope_feng_on[None] = 1 if slope_feng else 0
        slope_t_f[None] = [1.0, 0.0, 0.0]; slope_o_f[None] = [0.0, 0.0, 0.0]
        slope_umin_f[None] = -1e30          # no shelf edge unless a scene sets one
        if wall is not None:
            wall_n_f[None] = list(wall[0])
            wall_d_f[None] = float(wall[1])
            wall_on[None] = 1
        else:
            wall_on[None] = 0
        if cylinder is not None:
            cyl_cx_f[None] = float(cylinder[0])
            cyl_cz_f[None] = float(cylinder[1])
            cyl_R_f[None]  = float(cylinder[2])
            cyl_on[None]   = 1
        else:
            cyl_on[None]   = 0
        if box is not None:
            box_w[None]  = float(box)
            box_on[None] = 1
        else:
            box_on[None] = 0
        side_on[None] = 0                  # frictionless side barriers: a scene turns them on
        per_on[None] = 0; per_L[None] = 0.0; per_nz[None] = 1   # periodic z: a scene turns it on
        if slope_mu_y_range is not None:
            slope_mu_y_lo[None] = float(slope_mu_y_range[0])
            slope_mu_y_hi[None] = float(slope_mu_y_range[1])
        else:
            slope_mu_y_lo[None] = 0.0
            slope_mu_y_hi[None] = 120.0
        if ledge_bounds is not None:
            x0, x1, z0, z1 = ledge_bounds
            ledge_x0[None] = float(x0)
            ledge_x1[None] = float(x1)
            ledge_z0[None] = float(z0)
            ledge_z1[None] = float(z1)
            ledge_on[None] = 1
        else:
            ledge_on[None] = 0
        self._inst_buf = np.zeros((GEOM_MAX, INST_FLOATS), dtype=np.float32)
        sync_params()
        _init_shear()

    def set_slope_mu_band(self, y_lo, y_hi):
        """Move the slope-mu transition band (world y) at runtime."""
        slope_mu_y_lo[None] = float(y_lo)
        slope_mu_y_hi[None] = float(y_hi)

    def clear(self):
        n_f[None] = 0
        self._n = 0
        self._exp = None
        nosleep_f.fill(0); nowake_f.fill(0); noslope_f.fill(0)
        self._grid_valid = False
        self._frame_id = 0
        _init_shear()

    @property
    def shear_overflow(self):
        """Debug: number of contact pairs that found no free cache slot last frame."""
        return shear_overflow[None]

    @property
    def shear_maxocc(self):
        """Debug: highest per-block cache slot index used last frame (+1)."""
        return shear_maxocc[None]

    def spawn_box(self, pos, half_extents, velocity=(0,0,0), omega=(0,0,0), R=None):
        idx = self._n
        if idx >= MAX_N: return
        if not self._spawn_he_warned and max(half_extents) > self._grid_max_he + 1e-6:
            print(f"[ceaulver] WARNING: block he={max(half_extents):.2f} exceeds the world's "
                  f"max_he={self._grid_max_he:.2f}; the broadphase cell is sized for max_he and "
                  f"will MISS contacts for this block. Construct Solver(max_he=...) larger.")
            self._spawn_he_warned = True
        pos_f[idx] = list(pos)
        vel_f[idx] = list(velocity)
        omega_f[idx] = list(omega)
        he_f[idx] = list(half_extents)
        sleep_f[idx] = 0
        noslope_f[idx] = 0
        birth_f[idx] = birth_now[None]
        if R is not None:
            R_f[idx] = R.tolist() if hasattr(R, 'tolist') else R
        else:
            R_f[idx] = [[1,0,0],[0,1,0],[0,0,1]]
        a,b,c = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
        m = 8.0*a*b*c
        mass_f[idx] = m
        Il = [3/(m*(b*b+c*c)), 3/(m*(a*a+c*c)), 3/(m*(a*a+b*b))]
        Iil_f[idx] = Il
        Iiw_f[idx] = [[Il[0],0,0],[0,Il[1],0],[0,0,Il[2]]]
        Iloc = [1/Il[0], 1/Il[1], 1/Il[2]]
        Imn = sum(Iloc)/3
        ms = max(1.0, (params.ROLLING_M_REF/m)**(2/3))
        rs_f[idx] = [min(10, max(1, Imn/Iloc[d]))*ms for d in range(3)]
        _clear_plane_spring(idx)
        n_f[None] = idx + 1
        self._n = idx + 1
        self._grid_valid = False

    def spawn_boxes(self, positions, half_extents_list, R_list):
        idx = self._n
        k = min(len(positions), MAX_N - idx, SPAWN_MAX)
        if k <= 0: return
        p_stg = np.zeros((SPAWN_MAX, 3), dtype=np.float32)
        h_stg = np.zeros((SPAWN_MAX, 3), dtype=np.float32)
        r_stg = np.zeros((SPAWN_MAX, 3, 3), dtype=np.float32)
        m_stg = np.zeros(SPAWN_MAX, dtype=np.float32)
        il_stg = np.zeros((SPAWN_MAX, 3), dtype=np.float32)
        rs_stg = np.zeros((SPAWN_MAX, 3), dtype=np.float32)
        for i in range(k):
            he = half_extents_list[i]
            R = R_list[i]
            p_stg[i] = positions[i]
            h_stg[i] = he
            r_stg[i] = R
            a, b, c = float(he[0]), float(he[1]), float(he[2])
            m = 8.0 * a * b * c
            m_stg[i] = m
            Il = [3/(m*(b*b+c*c)), 3/(m*(a*a+c*c)), 3/(m*(a*a+b*b))]
            il_stg[i] = Il
            Iloc = [1/Il[0], 1/Il[1], 1/Il[2]]
            Imn = sum(Iloc) / 3
            ms = max(1.0, (params.ROLLING_M_REF / m) ** (2/3))
            rs_stg[i] = [min(10, max(1, Imn/Iloc[d])) * ms for d in range(3)]
        sp_pos.from_numpy(p_stg)
        sp_he.from_numpy(h_stg)
        sp_R.from_numpy(r_stg)
        sp_mass.from_numpy(m_stg)
        sp_Iil.from_numpy(il_stg)
        sp_rs.from_numpy(rs_stg)
        sp_n[None] = k
        _spawn_commit(idx)
        n_f[None] = idx + k
        self._n = idx + k
        self._grid_valid = False

    def apply_impulse(self, impulse):
        _apply_impulse(float(impulse[0]), float(impulse[1]), float(impulse[2]))

    def cull(self, despawn_y=-50.0):
        if self._n == 0: return
        if self._exp is not None:
            # Host cull from the rows the renderer already read back (column 1 = y): no GPU
            # round trip unless something is removed. The rows are the previous frame's
            # positions, so a block leaves one frame late. Their indices are still valid:
            # since that export, blocks were only appended (spawn), never moved (cull is here).
            rm = np.flatnonzero(self._exp[:, 1] <= despawn_y).astype(np.int32)
            self._exp = None
            if rm.size == 0: return
            _cull_set(rm)
            nr = int(rm.size)
        else:
            _cull_mark(float(despawn_y))
            nr = cull_n_remove[None]
            if nr == 0: return
        _cull_swap()
        self._n -= nr
        self._grid_valid = False

    def cull_cached(self, pn, n, despawn_y=-50.0):
        self.cull(despawn_y)

    def snapshot_ke(self):
        if self._n > 0:
            _snapshot_ke()

    def compute_support(self):
        if self._n > 0:
            _compute_support()

    def get_R(self, idx):
        return R_f.to_numpy()[idx]

    def classify(self):
        if self._n == 0: return
        if not self._grid_valid:
            _build_grid()
            self._grid_valid = True
        _classify()

    def ballistic_step(self, dt):
        if self._n == 0: return
        _ballistic_step(float(dt))

    def invalidate_grid(self, pairs_valid=False):
        # The contract for code that changes blocks behind this class (wall.py drives
        # _spawn_commit / _cull_swap itself): call this afterwards. It also re-reads the block
        # count into the host mirror and drops the export rows the host cull would read, and
        # (FENG_REUSE) makes the next eval rebuild the pair list. pairs_valid=True: the change
        # moved blocks without changing any pair's geometry -- a periodic wrap by exactly one
        # period (every pair test uses the minimum image) -- so the list may be kept. Adding,
        # removing or re-indexing blocks is caught on the GPU (topo_gen) either way.
        self._grid_valid = False
        if FENG_REUSE and not pairs_valid:
            reuse_force[None] = 1        # an asynchronous write: no GPU drain
        self._n = n_f[None]
        self._exp = None

    def step(self, dt):
        # config.py (or a diagnostic's params.X override) is authoritative; this is a
        # tuple compare and an early return unless something actually changed.
        sync_params()
        # NO host reads or writes of fields in here: each is a launch + sync that stops the
        # CPU queueing ahead of the GPU. (The old `if n_f[None] == 0: return` was one -- every
        # kernel already loops over n_f, so an empty world costs a few empty launches.)
        t0 = _time.perf_counter()
        # T-05: one step() == one shear-cache "frame".
        self._frame_id += 1
        _integrator_step(dt, self._frame_id, not self._grid_valid)
        self._grid_valid = True
        self._step_count += 1
        if self._step_count % 8 == 0: _reortho()
        self.last_solver_ms = (_time.perf_counter() - t0) * 1e3

    def get_instance_data(self):
        n = min(self._n, GEOM_MAX)
        if n == 0:
            self._exp = None
            return self._inst_buf[:0], 0
        # Pass only the live rows: Taichi copies an external numpy array to the GPU and back
        # whole on every call, so the full GEOM_MAX buffer cost ~0.63 ms at ANY block count.
        out = self._inst_buf[:n]
        _export_instances(out)
        self._exp = out
        return out, n

    @property
    def chunk_count(self):
        return self._n


# ==================================================================
# Geometry helpers (Python/numpy — for rendering only)
# ==================================================================

def _rodrigues(dtheta):
    theta = float(np.linalg.norm(dtheta))
    if theta < 1e-12: return np.eye(3)
    k = dtheta / theta
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

_LC = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=np.float64)
_TI = np.array([0,1,2,0,2,3, 4,6,5,4,7,6, 0,4,5,0,5,1, 3,2,6,3,6,7, 0,3,7,0,7,4, 1,5,6,1,6,2])
_EI = np.array([0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 0,4,1,5,2,6,3,7])

def _box_geometry(center, half, R):
    c = ((_LC * half) @ R.T + center).astype(np.float32)
    return c[_TI], c[_EI]

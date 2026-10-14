"""picking.py — click a block, delete it (any ceaulver scene).

pick(o, d): the nearest block hit by the ray o + t d (slab test in each block's own
frame; optional periodic copies along z). drop(i): send block i far below the world --
the scene's next cull removes it, and the blocks it supported wake through the
solver's own checks (force imbalance / island recheck; see integrator._kick_drift).
ray_from_click: the ray through a window point, from the renderer's view-projection.
"""
import numpy as np
import taichi as ti

from ceaulver import fields
from ceaulver.fields import pos_f, R_f, n_f

_BIG = 1 << 30
pick_t = ti.field(ti.i32, ())
pick_i = ti.field(ti.i32, ())


@ti.func
def _hit(i, o, d, off):
    R = R_f[i]; h = fields.he_f[i]
    ld = R.transpose() @ d
    lo = R.transpose() @ (o - pos_f[i] - off)
    t0 = -1e30; t1 = 1e30
    for a in ti.static(range(3)):
        if ti.abs(ld[a]) < 1e-9:
            if ti.abs(lo[a]) > h[a]:
                t0 = 1e30
        else:
            ta = (-h[a] - lo[a]) / ld[a]; tb = (h[a] - lo[a]) / ld[a]
            t0 = ti.max(t0, ti.min(ta, tb)); t1 = ti.min(t1, ti.max(ta, tb))
    t = 1e30
    if t0 <= t1 and t1 > 0.0:
        t = ti.max(t0, 0.0)
    return t


@ti.kernel
def _k_pick(n: ti.i32, o: ti.types.vector(3, ti.f32), d: ti.types.vector(3, ti.f32),
            k: ti.i32, L: ti.f32):
    pick_t[None] = _BIG; pick_i[None] = -1
    for i in range(n):
        if pos_f[i][1] > -1e5:
            best = 1e30
            for kk in range(-k, k + 1):
                best = ti.min(best, _hit(i, o, d, ti.Vector([0.0, 0.0, kk * L])))
            if best < 1e29:
                ti.atomic_min(pick_t[None], int(best * 64.0))
    for i in range(n):
        if pos_f[i][1] > -1e5:
            for kk in range(-k, k + 1):
                t = _hit(i, o, d, ti.Vector([0.0, 0.0, kk * L]))
                if t < 1e29 and int(t * 64.0) == pick_t[None]:
                    pick_i[None] = i


@ti.kernel
def _k_drop(i: ti.i32):
    pos_f[i][1] = -1e6


def pick(o, d, copies=0, period=0.0):
    """Index of the nearest block hit by the ray o + t d (unit d), or -1."""
    n = n_f[None]
    if n == 0:
        return -1
    _k_pick(n, ti.Vector(list(map(float, o))), ti.Vector(list(map(float, d))), int(copies), float(period))
    return int(pick_i[None])


def drop(i):
    """Send block i far below the world; the scene's next cull removes it."""
    _k_drop(int(i))


def ray_from_click(renderer, x, y, w, h):
    """(origin, unit direction) of the ray through window point (x, y) (top-left origin)."""
    import glm
    inv = glm.inverse(renderer._mvp(w, h))
    nx = 2.0 * x / max(w, 1) - 1.0; ny = 1.0 - 2.0 * y / max(h, 1)
    a = inv * glm.vec4(nx, ny, -1.0, 1.0); b = inv * glm.vec4(nx, ny, 1.0, 1.0)
    a = np.array([a.x, a.y, a.z]) / a.w; b = np.array([b.x, b.y, b.z]) / b.w
    return a, (b - a) / np.linalg.norm(b - a)

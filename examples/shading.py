"""shading.py — the LUT-shaded look shared by wall.py and pile.py.

make_lut: grey as a function of face orientation, the table render.py's LUT mode samples.
k_motion: per-block velocity + omega, for render.py's motion blur.
"""
import numpy as np
import taichi as ti

from ceaulver import fields
from ceaulver.fields import vel_f, omega_f

LIGHT_LEFT  = (1.0, 1.0, 1.0)      # white, from screen-left
LIGHT_RIGHT = (1.0, 1.0, 1.0)      # white, from screen-right
AMBIENT     = (0.06, 0.06, 0.06)   # grey (monochrome)
LIGHT_GAIN  = 0.94                 # brightest orientation (between the lights) -> 1.0


def make_lut(kind, eye, top_n=(0.0, 1.0, 0.0), w=256, h=128):
    """Grey as a function of face orientation, on a lat-long grid (u = azimuth
    atan2(z, x), v = polar angle from +y) -- the table render.py's LUT mode samples.
    `eye`: direction from the scene towards the camera. `top_n`: the modal normal of
    up-facing faces ("lights" only).
    A face seen square-on (normal along the view direction, towards the camera) is
    WHITE; grey falls off as cos^2 of the angle away from it. Upward-facing faces are
    near-white too (cos^4 of the angle from +y): the row treads of the stepped face are
    seen almost edge-on, and a dark tread popping in and out with every jitter of a
    cube reads as flickering lines -- in the face's own tone, it disappears. Sides,
    bottoms and faces turned away stay dark. bands: the same, posterised into 6 greys."""
    u = (np.arange(w) + 0.5) / w; v = (np.arange(h) + 0.5) / h
    az = (u - 0.5) * 2.0 * np.pi; po = v * np.pi
    U, V = np.meshgrid(az, po)
    nx = np.sin(V) * np.cos(U); ny = np.cos(V); nz = np.sin(V) * np.sin(U)
    if kind == "lights":
        # Two directional lights at 45 deg elevation, 45 deg either side of head-on (white;
        # LIGHT_LEFT / LIGHT_RIGHT). Ambient + Lambert per light, normalised so the
        # brightest orientation is white. No shadows -- a LUT only knows orientation.
        e = np.asarray(eye, float).copy(); e[1] = 0.0; e /= np.linalg.norm(e)   # horizontal head-on
        side = np.cross([0.0, 1.0, 0.0], e)                              # screen right
        el, az = np.radians(45.0), np.radians(45.0)
        Ls = [np.cos(el) * (np.cos(az) * e + sgn * np.sin(az) * side) + np.sin(el) * np.array([0, 1.0, 0])
              for sgn in (1.0, -1.0)]
        cols = [np.array(LIGHT_RIGHT), np.array(LIGHT_LEFT)]             # Ls[0] is the screen-right light
        amb = np.array(AMBIENT)
        peak = np.linalg.norm(Ls[0] + Ls[1])
        def shade(nxs, nys, nzs):
            out = np.zeros(np.shape(nxs) + (3,)) + amb
            for L, c in zip(Ls, cols):
                lam = np.clip(nxs * L[0] + nys * L[1] + nzs * L[2], 0.0, 1.0)
                out = out + np.asarray(lam)[..., None] * c / peak * LIGHT_GAIN
            return out
        g = shade(nx, ny, nz)
        # Up-facing faces get ONE colour: that of a quasi-horizontal slab, lit at the modal
        # top normal -- a top's colour cannot shimmer with every small tilt or jitter.
        # Blended in over the upper hemisphere (n_y 0.5 -> 0.8).
        tn = np.asarray(top_n, float)
        g_top = shade(tn[0], tn[1], tn[2])
        w = np.clip((ny - 0.5) / 0.3, 0.0, 1.0)
        w = (w * w * (3.0 - 2.0 * w))[..., None]
        g = (1.0 - w) * g + w * g_top
        return np.clip(g, 0.0, 1.0).astype(np.float32)
    e = np.asarray(eye, float); e /= np.linalg.norm(e)    # from the scene towards the camera
    facing = np.clip(nx * e[0] + ny * e[1] + nz * e[2], 0.0, 1.0)
    up = np.clip(ny, 0.0, 1.0)
    g = 0.06 + 0.94 * np.maximum(facing ** 2, 0.97 * up ** 4)
    if kind == "bands":
        g = np.floor(np.clip(g, 0, 0.999) * 6) / 5.0 * 0.94 + 0.06
    return np.clip(g, 0.0, 1.0).astype(np.float32)


@ti.kernel
def k_motion(out: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        v = vel_f[i]; w = omega_f[i]
        out[i, 0] = v[0]; out[i, 1] = v[1]; out[i, 2] = v[2]
        out[i, 3] = w[0]; out[i, 4] = w[1]; out[i, 5] = w[2]


def motion_data(n):
    """(n, 6) float32: velocity, omega per block -- the `motion` render.py's blur takes."""
    mot = np.empty((n, 6), dtype=np.float32)
    if n:
        k_motion(mot, n)
    return mot


# ---- block tint from a cyclic colormap (hue only; the LUT keeps the brightness) ----
def cyclic_palette(name="twilight_r", n=256):
    """(n,3) f32: matplotlib colormap `name` sampled around its cycle, each entry normalised
    to brightest channel 1 -- the colormap gives the hue, the LUT the brightness."""
    import matplotlib
    cols = matplotlib.colormaps[name](np.linspace(0.0, 1.0, n, endpoint=False))[:, :3]
    return (cols / cols.max(axis=1, keepdims=True)).astype(np.float32)


_TINT_N = 256
_tint_pal = ti.Vector.field(3, ti.f32, _TINT_N)
_tint_ready = [False]


@ti.kernel
def _k_time_tint(out: ti.types.ndarray(), n: ti.i32, period: ti.f32, awake_only: ti.i32):
    for i in range(n):
        u = fields.birth_f[i] / period
        col = _tint_pal[int((u - ti.floor(u)) * _TINT_N) % _TINT_N]
        if awake_only == 1 and fields.sleep_f[i] == 1:
            col = ti.Vector([1.0, 1.0, 1.0])                         # asleep: grey (the LUT as is)
        out[i, 0] = col[0]; out[i, 1] = col[1]; out[i, 2] = col[2]


def time_tint(n, period, awake_only=False, cmap="twilight_r"):
    """(n,3) tint per block: hue from its spawn time (fields.birth_f), one turn of the cyclic
    colormap every `period` seconds; awake_only: sleeping blocks grey."""
    if not _tint_ready[0]:
        _tint_pal.from_numpy(cyclic_palette(cmap, _TINT_N)); _tint_ready[0] = True
    out = np.empty((n, 3), dtype=np.float32)
    if n:
        _k_time_tint(out, n, float(period), 1 if awake_only else 0)
    return out

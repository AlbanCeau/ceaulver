"""pile.py — a single growing pile, to judge visually how well static friction holds it.

Blocks (production size distribution, production block-block contact) are dropped
one at a time from a GRID x GRID box of cells centred on (x, z) = (0, 0) (GRID=1: one point), bottom
at a fixed height above the ground -- onto an infinite flat ground with mu = 2 (not drawn). The pile can only grow by blocks landing
on blocks, so its shape is the friction model's answer. Spawning never stops (SPC
pauses the whole scene).

Run:
    uv run examples/pile.py
    uv run examples/pile.py --check      # headless, prints stats
    uv run examples/pile.py --capture    # record until the pile
        reaches the spawn height: runs/<date>_pile_capture/pile.mp4 (native framebuffer,
        60 fps, video time = wall time since the first frame) + pile.csv (a row per frame
        and per printed line, wallclock-stamped). The window is not
        resizable. PILE_CAP_MAX=7200: give up after this many wall seconds.
Env:
    PILE_RATE=7      blocks per second of sim time
    PILE_SPAWN_Y=100 height of the new block's bottom above the ground (fixed)
    PILE_G=1         gravity, as a multiple of config.GRAVITY
    PILE_GRID=1      spawn box: GRID x GRID cells (1 = a single column)
    PILE_PITCH=5.1   spawn box cell pitch (default: the largest block's edge + margin)
    PILE_SUB=2       physics substeps per frame (default config.PHYSICS_SUBSTEPS; use 2x at PILE_G=3)
    PILE_SLEEP_IMB=0.1   sleep only if |net force| < this x sqrt(sum |F_k|^2), torque likewise (EXPERIMENTAL, T-16; 0 = off)
    PILE_VIEW=100    fixed ortho half-height of the view (larger = wider)
    PILE_MU=2        ground mu
    PILE_T=60        --check only: seconds to simulate
    PILE_LUT=lights  shaded mode's LUT (shading.make_lut: lights | soft | bands)
    PILE_SHUTTER=1/120  motion blur, seconds

Controls:
    F      static friction on/off, live (same pile) -- the A/B this scene exists for
    W      LUT-shaded (default) / wireframe, as wall.py
    B      motion blur on/off (shaded mode only)
    S      sleep on/off (off: every block wakes and none may fall asleep)
    SPC    pause/resume
    R      reset
    ESC    quit

Console, every ~1 s: blocks, awake, pile height, base radius, and the apparent repose
angle atan(height / base radius).
"""
from console import info, onoff, STATUS   # first: before taichi is imported
import os
import sys
import time as _time
_T_LAUNCH = _time.time()       # --capture: the CSV's t_wall is measured from here

import numpy as np
import taichi as ti

from ceaulver import config
import scene_config
from ceaulver import params, fields, broadphase
from ceaulver.fields import n_f, sleep_f, pos_f, vel_f, R_f, he_f, gnd_f, cntct_f, nosleep_f, grid_cnt, grid_data, grid_cell_f, MAX_PER_CELL
from ceaulver.broadphase import _grid_hash
from ceaulver.solver import Solver, _rodrigues

SPAWN_RATE = float(os.environ.get("PILE_RATE", "7"))      # blocks / sim-second
SPAWN_Y    = float(os.environ.get("PILE_SPAWN_Y", "100")) # new block's bottom, above ground
GRAVITY    = float(os.environ.get("PILE_G", "1")) * config.GRAVITY
SUBSTEPS   = int(os.environ.get("PILE_SUB", str(config.PHYSICS_SUBSTEPS)))
VIEW_SCALE = float(os.environ.get("PILE_VIEW", "100"))     # ortho half-height
LOOK_AT    = (0.0, 18.0, 0.0)
GROUND_MU  = float(os.environ.get("PILE_MU", "2"))
GROUND_HALF = 1e6     # ground force applies only inside ledge_bounds: make them unbounded
SPAWN_MARGIN = 0.1    # free-space check: required gap between the new block and any other
# Spawn box: the queued block goes to the first cell whose space is free, searching from
# the cell after the last one used (from cell 0 every time, the same few cells would take
# every block). A cell is reused once its last block fell clear.
GRID       = int(os.environ.get("PILE_GRID", "1"))
PITCH      = float(os.environ.get("PILE_PITCH", str(2.0 * config.SPAWN_HE_MAX + SPAWN_MARGIN)))
_g         = (np.arange(GRID) - 0.5 * (GRID - 1)) * PITCH
CELLS      = np.array([[x, 0.0, z] for z in _g for x in _g])      # (x, 0, z) offsets
# Initial downward speed, so a cell's previous block has fallen clear of it by the time
# the cell comes round again, tau = GRID^2 / rate later: v0*tau + g*tau^2/2 >= CLEAR.
# Orientation is fully random, so CLEAR covers a block standing on a corner -- height =
# its full diagonal, 2*sqrt(3)*SPAWN_HE_MAX ~ 8.7 -- plus margin. GRID=1 at 7/s needs
# ~60 down; a 4x4 box at 14/s needs none (tau ~1.1 s).
CLEAR      = 2.0 * np.sqrt(3.0) * config.SPAWN_HE_MAX + 0.5
_tau       = GRID * GRID / SPAWN_RATE
SPAWN_V0   = max(0.0, (CLEAR - 0.5 * abs(GRAVITY) * _tau ** 2) / _tau)
SEED       = 3
DT         = 1.0 / 60.0
REST_SPEED = 2.0      # a block counts as part of the pile (for the spawn height) below this speed
SHUTTER    = float(os.environ.get("PILE_SHUTTER", str(1.0 / 120.0)))   # motion blur, seconds

params.SOIL_MU = GROUND_MU                                # this scene's ground only
# EXPERIMENTAL sleep gate (T-16). Without a force gate a block tipping off an edge is
# re-frozen every step (its spin gain per substep, ~g/he*dt, is under SLEEP_OMEGA) ->
# asleep in impossible positions.
params.SLEEP_IMBALANCE = float(os.environ.get("PILE_SLEEP_IMB", "0.1"))
params.GRAVITY = GRAVITY                                  # read live by sync_params
# Hertz stiffness + yield ceiling are DERIVED from g and the substep dt (config.py,
# derive_contact): re-derive for this scene's values, or the contact stays sized for
# config's g / substeps.
params.FENG_KHERTZ, params.VMAX_CONTACT = config.derive_contact(
    config.CONTACT_SAFETY, config.PILE_DEPTH, GRAVITY, config.SPAWN_HE_MAX, SUBSTEPS,
    config.YIELD_STRAIN)

_SIGNS = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                   for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])

# the static-friction switches, so F can toggle them live (all runtime parameters)
_FRICTION_ON = {k: getattr(params, k) for k in
                ("SHEAR_K_FRAC", "PLANE_SHEAR_K", "CHUNK_MU_S", "ZETA_T", "ZETA_N")}
_FRICTION_OFF = dict(SHEAR_K_FRAC=0.0, PLANE_SHEAR_K=0.0, CHUNK_MU_S=params.CHUNK_MU,
                     ZETA_T=0.0, ZETA_N=0.0)


@ti.kernel
def _k_wake_all():
    for i in range(n_f[None]):
        sleep_f[i] = 0


@ti.kernel
def _k_cells_free(cand: ti.types.ndarray(), half: ti.types.ndarray(), out: ti.types.ndarray(),
                  m: ti.i32, r: ti.i32, margin: ti.f32):
    """out[c] = 1 if the box centred at cand[c] with axis-aligned half-size `half` clears
    every block's exact axis-aligned bounds (|R| he) by `margin`. Broadphase: the
    (2r+1)^3 grid cells around cand[c] (the caller sizes r so this cannot miss). A bucket
    that overflowed MAX_PER_CELL hides blocks -> the cell counts as occupied."""
    for c in range(m):
        p = ti.Vector([cand[c, 0], cand[c, 1], cand[c, 2]])
        h = ti.Vector([half[0], half[1], half[2]]) + margin
        cs = grid_cell_f[None]
        cx = int(ti.floor(p[0] / cs)); cy = int(ti.floor(p[1] / cs)); cz = int(ti.floor(p[2] / cs))
        free = 1
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    b = _grid_hash(cx + dx, cy + dy, cz + dz)
                    cnt = grid_cnt[b]
                    if cnt > MAX_PER_CELL:
                        free = 0; cnt = MAX_PER_CELL
                    for s in range(cnt):
                        j = grid_data[b, s]
                        d = ti.abs(pos_f[j] - p)
                        hj = ti.abs(R_f[j]) @ he_f[j]
                        if d[0] < h[0] + hj[0] and d[1] < h[1] + hj[1] and d[2] < h[2] + hj[2]:
                            free = 0
        out[c] = free


def set_friction(on):
    for k, v in (_FRICTION_ON if on else _FRICTION_OFF).items():
        setattr(params, k, v)


class PileScene:
    """Spawner + stats, shared by the window and the headless check."""

    def __init__(self):
        self.sleep = True
        self.rng = np.random.default_rng(SEED)
        s2 = scene_config.SPAWN_LOG_SIGMA ** 2; rho = scene_config.SPAWN_SHAPE_RHO
        self.cov = s2 * (rho * np.ones((3, 3)) + (1 - rho) * np.eye(3))
        self.mu = np.full(3, scene_config.SPAWN_LOG_MU)
        self.cm = Solver(max_he=config.SPAWN_HE_MAX,
                               ledge_bounds=(-GROUND_HALF, GROUND_HALF, -GROUND_HALF, GROUND_HALF))
        # compile the kernels before the clock starts
        self.cm.spawn_box(pos=(0, 1000, 0), half_extents=(1, 1, 1))
        self.cm.snapshot_ke(); self.cm.step(1e-4)
        self.cm.get_instance_data(); self.cm.cull(despawn_y=500.0)
        self.reset()

    def reset(self):
        self.cm.clear()               # also clears nosleep_f
        self.set_sleep(self.sleep)
        self.sim_time = 0.0
        self.next_spawn = 0.0
        self.pending = None      # sampled block waiting for its space to clear
        self.waits = 0           # frames a spawn was held back because every cell was occupied
        self.next_cell = 0       # where the first-free-cell search starts

    def set_sleep(self, on):
        """Sleep on/off. The only way into sleep is the solver's gate on nosleep_f == 0, so off
        = nosleep_f all 1 (every slot, so blocks spawned later are covered) + wake all."""
        self.sleep = on
        nosleep_f.fill(0 if on else 1)
        if not on:
            _k_wake_all()

    def corners(self):
        n = n_f[None]
        if n == 0:
            return np.zeros((0, 8, 3))
        pos = pos_f.to_numpy()[:n]; R = R_f.to_numpy()[:n]; he = he_f.to_numpy()[:n]
        return np.einsum("nij,nkj->nki", R, _SIGNS[None] * he[:, None, :]) + pos[:, None, :]

    def pile_top(self):
        """Top of the PILE: highest corner over blocks that are part of it -- asleep, or
        slow AND touching the ground or another block. Speed alone is not enough: a block
        is spawned AT REST, so for its first instant in free fall it read as 'resting',
        which (a) ran the old relative spawn height away and (b) stopped the fixed-height
        spawner early on a phantom 'pile top' at the spawn point."""
        c = self.corners()
        if len(c) == 0:
            return float(config.GROUND_Y)
        n = len(c)
        slow = np.linalg.norm(vel_f.to_numpy()[:n], axis=1) < REST_SPEED
        touching = (gnd_f.to_numpy()[:n] == 1) | (cntct_f.to_numpy()[:n] == 1)
        rest = (sleep_f.to_numpy()[:n] == 1) | (slow & touching)
        return float(c[rest][:, :, 1].max()) if rest.any() else float(config.GROUND_Y)

    def sample_block(self):
        he = np.clip(np.exp(self.rng.multivariate_normal(self.mu, self.cov)),
                     scene_config.SPAWN_HE_MIN, config.SPAWN_HE_MAX)
        # random tip, tilt AND yaw: uniform over all rotations (Shoemake's random unit
        # quaternion). Three independent uniform Euler angles would not be uniform --
        # they bunch orientations near the poles.
        u1, u2, u3 = self.rng.uniform(size=3)
        qx, qy = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        qz, qw = np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)
        R = np.array([[1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                      [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                      [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]],
                     dtype=np.float32)
        half_h = float(np.abs(R[1, :]) @ he)              # vertical half-extent after rotation
        pos = np.array([0.0, config.GROUND_Y + SPAWN_Y + half_h, 0.0])
        return pos, he, R

    def cells_free(self, pos, he, R):
        """(len(CELLS),) bool: True where the new block, placed at pos + CELLS[k], clears
        every existing block's box by SPAWN_MARGIN. Axis-aligned bounds of the real
        (rotated) boxes: conservative, never misses an overlap, may refuse a near-miss --
        which only delays the spawn. On the GPU, through the solver's broadphase grid
        (current after step(); rebuilt here if a spawn or reset invalidated it): one
        kernel for all cells, one 16-int readback, no other host<->GPU traffic (an empty
        world has an empty grid, so needs no special case)."""
        if not self.cm._grid_valid:
            broadphase._build_grid()
            self.cm._grid_valid = True
        half = (np.abs(R.astype(np.float64)) @ he).astype(np.float32)
        # neighbour rings so no overlapping block can sit outside the scan: overlap needs
        # |dp| < half + SPAWN_MARGIN + (an existing block's half <= max_he*sqrt(3)) per axis
        cs = 2.0 * self.cm._grid_max_he * np.sqrt(3.0)          # = grid_cell_f, host-side
        reach = float(half.max()) + SPAWN_MARGIN + self.cm._grid_max_he * np.sqrt(3.0)
        r = max(1, int(np.ceil(reach / cs)))
        cand = np.ascontiguousarray(pos + CELLS, dtype=np.float32)
        out = np.zeros(len(CELLS), dtype=np.int32)
        _k_cells_free(cand, half, out, len(CELLS), r, SPAWN_MARGIN)
        return out == 1

    def spawn_one(self):
        """Commit the pending block to the first spawn cell whose space is free; if none
        is, keep it and retry next frame. Returns True if spawned."""
        if self.pending is None:
            self.pending = self.sample_block()
        pos, he, R = self.pending
        free = self.cells_free(pos, he, R)
        for j in range(len(CELLS)):
            cell = (self.next_cell + j) % len(CELLS)
            p = pos + CELLS[cell]
            if free[cell]:
                self.cm.spawn_box(pos=tuple(p.tolist()), half_extents=tuple(he.tolist()),
                                  velocity=(0.0, -SPAWN_V0, 0.0), R=R)
                self.next_cell = (cell + 1) % len(CELLS)
                self.pending = None
                return True
        self.waits += 1
        return False

    def advance(self, dt):
        self.spawn_tick(dt)
        self.physics(dt)

    def spawn_tick(self, dt):
        self.sim_time += dt
        # sim-time schedule, as scree.py. A blocked spawn does not consume its slot: it is
        # retried every frame until the space is free, then the schedule resumes from now
        # (no catch-up burst of queued blocks).
        if self.sim_time >= self.next_spawn and self.spawn_one():
            self.next_spawn = max(self.next_spawn + 1.0 / SPAWN_RATE, self.sim_time)

    def physics(self, dt):
        self.cm.snapshot_ke()
        for _ in range(SUBSTEPS):
            self.cm.step(dt / SUBSTEPS)

    def stats(self):
        n = n_f[None]
        if n == 0:
            return dict(n=0, awake=0, height=0.0, radius=0.0, angle=0.0, on_ground=0,
                        waits=self.waits)
        c = self.corners()
        top = self.pile_top() - config.GROUND_Y
        r = np.linalg.norm(c[:, :, [0, 2]], axis=2).max(axis=1)      # outermost corner radius
        base = float(np.percentile(r, 95))
        on_ground = int((c[:, :, 1].min(axis=1) < config.GROUND_Y + 0.35).sum())
        return dict(n=n, awake=int((sleep_f.to_numpy()[:n] == 0).sum()), height=top,
                    radius=base, angle=float(np.degrees(np.arctan2(top, max(base, 1e-6)))),
                    on_ground=on_ground, waits=self.waits)


def _fmt(st, on):
    return (f"t-sim blocks={st['n']:4d} awake={st['awake']:4d} on-ground={st['on_ground']:4d}  "
            f"height={st['height']:5.1f} base r95={st['radius']:5.1f}  "
            f"apparent repose={st['angle']:4.1f} deg  spawn waits={st['waits']}  "
            f"static friction {'ON' if on else 'OFF'}")


# The friction-side parameters a saved run needs to be reproducible, read LIVE from
# `params` (what set_friction / sync_params hand the kernels), not from config.
_PARAM_KEYS = ("CHUNK_MU", "CHUNK_MU_S", "SOIL_MU", "SHEAR_K_FRAC", "PLANE_SHEAR_K",
               "ZETA_T", "ZETA_N", "CHUNK_ROLL", "SLEEP_IMBALANCE")


def _params():
    return "  ".join(f"{k}={getattr(params, k)}" for k in _PARAM_KEYS if hasattr(params, k))


def check():
    """Headless: run the scene, print stats every 5 s of sim time."""
    T = float(os.environ.get("PILE_T", "60"))
    on = "--off" not in sys.argv
    set_friction(on)
    sc = PileScene()
    print(f"[pile --check] rate={SPAWN_RATE}/s spawn y={SPAWN_Y} ground mu={GROUND_MU} "
          f"sub={SUBSTEPS} T={T}s friction={'ON' if on else 'OFF'} "
          f"(CHUNK_MU={params.CHUNK_MU}, MU_S={params.CHUNK_MU_S})")
    t0 = _time.perf_counter(); k = 0
    while sc.sim_time < T:
        sc.advance(DT); k += 1
        if k % int(5 / DT) == 0:
            st = sc.stats()
            print(f"t={sc.sim_time:5.1f}  " + _fmt(st, on))
            print("  params: " + _params())
    print(f"wall {(_time.perf_counter() - t0):.1f}s for {T:.0f}s sim")


def run_window():
    import moderngl_window as mglw
    import taichi as ti
    from render import Renderer
    from shading import make_lut, motion_data

    CAPTURE = "--capture" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--capture"]  # moderngl-window parses argv too
    cap = {}                                               # recorder / log, if capturing
    if CAPTURE:
        from pathlib import Path
        import capture
        out = Path(__file__).resolve().parent / "runs" / (
            _time.strftime("%Y-%m-%d_%H%M%S", _time.localtime(_T_LAUNCH)) + "_pile_capture")
        out.mkdir(parents=True, exist_ok=True)
        cap["dir"] = out
        cap["log"] = capture.Log(out / "pile.csv", _T_LAUNCH)
        cap["stdout"] = sys.stdout
        sys.stdout = capture.Tee(cap["log"], sys.stdout)
        cap["max"] = float(os.environ.get("PILE_CAP_MAX", "7200"))
        STATUS.say(f"[capture] -> {out}")

    def finish_capture(reason):
        if "log" not in cap or cap.get("done"):
            return
        cap["done"] = True
        STATUS.say(f"[capture] stop: {reason}")
        if "rec" in cap:
            STATUS.say(f"[capture] video {cap['rec'].video_time():.1f} s, closing ffmpeg ...")
            cap["rec"].close()
        sys.stdout = cap["stdout"]
        cap["log"].close()
        STATUS.say(f"[capture] wrote {cap['dir']}")

    EYE_DIR = (1.0, 0.7, 1.0)                              # 3/4 iso, looking down ~30 deg

    class PileWindow(mglw.WindowConfig):
        title = "pile — static friction"
        gl_version = (3, 3)
        window_size = (scene_config.WINDOW_W, scene_config.WINDOW_H)
        aspect_ratio = None
        resizable = not CAPTURE        # a capture's frame size is fixed at the first frame

        def __init__(self, **kw):
            super().__init__(**kw)
            self.friction = True
            set_friction(True)
            self.scene = PileScene()
            scene_config.ISO_SCALE = VIEW_SCALE
            self.renderer = Renderer(self.ctx, LOOK_AT, slope_normal=(1.0, 0.0, 0.0),
                                     eye_dir=EYE_DIR)
            self.renderer.enable_lut_shading(make_lut(os.environ.get("PILE_LUT", "lights"),
                                                      eye=EYE_DIR))
            self.renderer.enable_motion_blur(shutter=SHUTTER)
            self.paused = False
            # scree.py's profile readout: same phases, same 120-frame cadence. This scene
            # never culls, so "cull" stays 0 -- kept so the two readouts line up.
            self._prof = {k: 0.0 for k in ("physics", "cull", "spawn", "geom", "render", "total")}
            self._prof_n = 0
            self._frame = 0; self._t_prev = None; self._t_check = 0.0
            info(f"[contact] FENG_FACECULL={params.FENG_FACECULL}  island_stag={fields.island_stag[None]}")
            info(f"Pile: spawn y={SPAWN_Y} at {SPAWN_RATE}/s, {GRID}x{GRID} box pitch {PITCH:.1f}, v0={SPAWN_V0:.1f} down, "
                  f"ground mu={GROUND_MU} (infinite), "
                  f"view half-height={VIEW_SCALE}")
            STATUS.show_controls(lambda: (
                f"F   = static friction {onoff(self.friction)}\n"
                f"B   = motion blur {onoff(self.renderer.blur)}\n"
                f"S   = enable sleep {onoff(self.scene.sleep)}\n"
                "W   = wireframe/shaded\n"
                "SPC = pause\n"
                "R   = reset\n"
                "ESC = quit"))

        def on_render(self, time, frametime):
            STATUS.poll()
            t_frame = _time.perf_counter()
            if not self.paused:
                dt = min(frametime, DT)
                t0 = _time.perf_counter()
                self.scene.physics(dt)
                ti.sync()
                t1 = _time.perf_counter()
                t2 = t1                                   # no cull in this scene
                self.scene.spawn_tick(dt)
                t3 = _time.perf_counter()
            inst, count = self.scene.cm.get_instance_data()
            # no ground slab: the ground is infinite and not drawn. count == 0 (after R)
            # must still be passed, so the renderer drops the old instances.
            self.renderer.update_instances(inst[:count], count, motion=motion_data(count))
            t4 = _time.perf_counter()
            self.renderer.draw(self.wnd.width, self.wnd.height)
            t5 = _time.perf_counter()
            if CAPTURE and not cap.get("done"):
                self.capture_frame(t_frame)
                if cap.get("done"):
                    return
            if self.paused:
                return
            p = self._prof
            STATUS.tick(t1 - t0)
            p["physics"] += t1 - t0
            p["cull"]    += t2 - t1
            p["spawn"]   += t3 - t2
            p["geom"]    += t4 - t3
            p["render"]  += t5 - t4
            p["total"]   += t5 - t_frame
            self._prof_n += 1
            if self._prof_n >= 120:
                n = self._prof_n
                nc = n_f[None]
                n_awake = int((sleep_f.to_numpy()[:nc] == 0).sum()) if nc else 0
                info(f"--- Profile ({nc} blocks, {n_awake} awake, {n} frames) ---")
                for k in ["physics", "cull", "spawn", "geom", "render", "total"]:
                    info(f"  {k:8s}: {p[k]/n*1000:6.2f} ms/frame")
                if nc:
                    slp = sleep_f.to_numpy()[:nc]
                    act = int(fields.active.to_numpy()[:nc].sum())
                    aw = slp == 0
                    v = vel_f.to_numpy()[:nc][aw]
                    w = fields.omega_f.to_numpy()[:nc][aw]
                    sp = np.linalg.norm(v, axis=1) if len(v) else np.zeros(1)
                    om = np.linalg.norm(w, axis=1) if len(w) else np.zeros(1)
                    slow = int(((sp < params.SLEEP_SPEED) & (om < params.SLEEP_OMEGA)).sum())
                    info(f"  awake-set: active(clip)={act}/{nc}  "
                          f"speed med={np.median(sp):.2f} p90={np.percentile(sp,90):.2f}  "
                          f"slow(should-sleep)={slow}/{n_awake}  "
                          f"[g={params.GRAVITY:.0f} sub={SUBSTEPS} kHz={params.FENG_KHERTZ:.0f}]")
                info(f"  pile: t={self.scene.sim_time:6.1f}  " + _fmt(self.scene.stats(), self.friction))
                info("  params: " + _params())
                for k in p: p[k] = 0.0
                self._prof_n = 0

        def capture_frame(self, t_frame):
            log = cap["log"]
            if "rec" not in cap:
                import capture
                w, h = self.wnd.buffer_size
                cap["rec"] = capture.Recorder(str(cap["dir"] / "pile.mp4"), w, h, fps=60)
                STATUS.say(f"[capture] recording {w}x{h} @ 60 fps")
            rec = cap["rec"]
            now = _time.time()
            log.state = dict(frame=self._frame, t_video=f"{rec.video_time():.4f}",
                             sim_t=f"{self.scene.sim_time:.4f}")
            reps, read_ms, q_ms = rec.add(self.wnd.fbo, now)
            dt_ms = (t_frame - self._t_prev) * 1e3 if self._t_prev is not None else ""
            self._t_prev = t_frame
            log.row("frame", frame_ms=f"{dt_ms:.3f}" if dt_ms != "" else "",
                    fps=f"{1e3 / dt_ms:.2f}" if dt_ms != "" and dt_ms > 0 else "",
                    repeats=reps, read_ms=f"{read_ms:.3f}", queue_ms=f"{q_ms:.3f}",
                    blocks=n_f[None])
            self._frame += 1
            # stop when the pile reaches the spawn height (pile_top reads the GPU: 2x/s)
            if now - self._t_check >= 0.5:
                self._t_check = now
                top = self.scene.pile_top() - config.GROUND_Y
                if top >= SPAWN_Y:
                    finish_capture(f"pile top {top:.1f} >= spawn height {SPAWN_Y}")
                elif now - _T_LAUNCH > cap["max"]:
                    finish_capture(f"PILE_CAP_MAX={cap['max']:.0f} s reached (pile top {top:.1f})")
                if cap.get("done"):
                    self.wnd.close()

        def on_key_event(self, key, action, modifiers):
            k = self.wnd.keys
            if action != k.ACTION_PRESS:
                return
            if key == k.ESCAPE:
                self.wnd.close()
            elif key == k.SPACE:
                self.paused = not self.paused
                info("[PAUSED]" if self.paused else "[RUNNING]")
            elif key == k.R:
                self.scene.reset()
                info("[RESET]")
            elif key == k.F:
                self.friction = not self.friction
                set_friction(self.friction)
                info(f"[static friction {'ON' if self.friction else 'OFF'}]")
            elif key == k.W:
                self.renderer.shaded = not self.renderer.shaded
                info(f"[{'LUT-shaded' if self.renderer.shaded else 'wireframe'}]")
            elif key == k.S:
                self.scene.set_sleep(not self.scene.sleep)
                info(f"[sleep {'ON' if self.scene.sleep else 'OFF'}]")
            elif key == k.B:
                self.renderer.blur = not self.renderer.blur
                info(f"[motion blur {'ON' if self.renderer.blur else 'OFF'}]")

    import traceback
    code = 0
    try:
        mglw.run_window_config(PileWindow)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(); code = 1
    finally:
        STATUS.end()
        try:
            finish_capture("window closed")       # ESC / crash: still a valid mp4 + csv
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

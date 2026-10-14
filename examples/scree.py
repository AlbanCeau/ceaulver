from console import info, onoff, STATUS   # first: before taichi is imported
import time as _time
import os

import numpy as np
import moderngl_window as mglw
import taichi as ti

from ceaulver import config
import scene_config
from render import Renderer

COLOR_PERIOD = 120.0     # s: one full turn of the cyclic colormap, by block spawn time
COLOR_MODE   = 1         # at start: 0 grey, 1 all blocks, 2 awake blocks only; C cycles

# Live tuning panel (dev tool). NOGUI=1 skips it entirely -- the piece runs unchanged
# without imgui installed, and without the panel's per-frame stats readback.
USE_GUI = os.environ.get("NOGUI", "0") != "1"
if USE_GUI:
    try:
        from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer
        from imgui_bundle import imgui
        from tuning import TuningPanel
    except ImportError as e:
        print(f"[tuning] disabled ({e}); run `uv add \"moderngl-window[imgui]\"` to enable")
        USE_GUI = False

# PROF_GPU=1: wait for the GL frame to finish inside the "render" timer. Without it, "render" is
# only the CPU-side command submission and the GPU's render work is billed to the NEXT frame's
# "physics" (its ti.sync() waits for the whole GPU queue, GL included).
PROF_GPU = os.environ.get("PROF_GPU", "0") == "1"

from ceaulver import params, fields
from ceaulver.solver import Solver, _rodrigues
# island-sleeping recheck mode: 1 (DEFAULT) -> staggered (immune to the one-ring
# active-blanket). ISLAND_STAG=0 -> original one-ring.
fields.island_stag[None] = int(os.environ.get("ISLAND_STAG", "1"))
info(f"[contact] FENG_FACECULL={params.FENG_FACECULL}  island_stag={fields.island_stag[None]}")


class BlockCascade(mglw.WindowConfig):
    title        = "Block Cascade"
    gl_version   = (3, 3)
    window_size  = (scene_config.WINDOW_W, scene_config.WINDOW_H)
    aspect_ratio = None
    resizable    = True

    # class attributes so the event forwarders below are safe if a window event
    # arrives before __init__ has finished building the panel
    panel          = None
    imgui_renderer = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        angle_rad = np.radians(scene_config.SLOPE_ANGLE)
        sin_a = np.sin(angle_rad)
        cos_a = np.cos(angle_rad)

        self._slope_normal = np.array([sin_a, cos_a, 0.0])
        self._slope_d      = cos_a * scene_config.SLOPE_TOP_Y
        self._slope_down   = np.array([cos_a, -sin_a, 0.0])
        self._x_base       = scene_config.SLOPE_TOP_Y * cos_a / sin_a
        self._slope_length = scene_config.SLOPE_TOP_Y / sin_a
        self._ledge_width  = scene_config.SLOPE_Z_MAX - scene_config.SLOPE_Z_MIN
        self._ledge_depth  = 24.0

        z_span = scene_config.SLOPE_Z_MAX - scene_config.SLOPE_Z_MIN
        self._spawn_z_min = scene_config.SLOPE_Z_MIN + z_span / 3.0
        self._spawn_z_max = scene_config.SLOPE_Z_MAX - z_span / 3.0

        s2 = scene_config.SPAWN_LOG_SIGMA ** 2
        rho = scene_config.SPAWN_SHAPE_RHO
        self._spawn_cov = s2 * (rho * np.ones((3, 3)) + (1 - rho) * np.eye(3))
        self._spawn_mu = np.full(3, scene_config.SPAWN_LOG_MU)

        z_center = (scene_config.SLOPE_Z_MIN + scene_config.SLOPE_Z_MAX) / 2
        slope_base = np.array([self._x_base, 0.0, z_center])
        look_center = tuple(slope_base - 46.0 * self._slope_down)

        view_y_top = look_center[1] + scene_config.ISO_SCALE
        view_y_bot = look_center[1] - scene_config.ISO_SCALE
        margin = scene_config.VIEW_MARGIN

        self._despawn_y = view_y_bot - margin
        spawn_y_min = view_y_top + margin
        self._spawn_t_max = (scene_config.SLOPE_TOP_Y - spawn_y_min) / sin_a
        self._spawn_t_min = -30.0

        # slope-mu transition band: view fractions from config (live-tunable, see
        # _apply_mu_band); the two y values go to the solver and to the guide lines.
        self._view_y_bot, self._view_y_top = view_y_bot, view_y_top
        self._mu_band = None
        mu_y_lo, mu_y_hi = self._mu_band_y()

        # Optional invisible vertical wall at the bottom of the slope.
        # Wall normal points uphill (-X); plane eq is n.x >= d, so
        # blocks at X <= X_wall satisfy the constraint, blocks past X_wall
        # are pushed back.
        wall = None
        if getattr(config, "ENABLE_BOTTOM_WALL", False):
            x_wall = self._x_base + float(scene_config.WALL_X_OFFSET)
            wall = ((-1.0, 0.0, 0.0), -x_wall)
            info(f"Wall: vertical, normal=-X, at X={x_wall:.1f}")

        self.chunks = Solver(
            slope_normal=self._slope_normal,
            slope_d=self._slope_d,
            slope_mu_y_range=(mu_y_lo, mu_y_hi),
            wall=wall,
            max_he=config.SPAWN_HE_MAX,   # size broadphase cell to actual blocks, not he=5 (perf)
        )

        self.renderer = Renderer(self.ctx, look_center,
                                 slope_normal=tuple(self._slope_normal))
        # LUT-shaded look + motion blur, as wall.py / pile.py; wireframe by default (W, B).
        self._motion_data = None
        self._time_tint = None
        self.color_mode = COLOR_MODE
        from shading import make_lut, motion_data, time_tint
        sn = np.asarray(self._slope_normal, float)
        eye = (sn[0], 0.0, sn[2])                 # the renderer's horizontal view direction
        self.renderer.enable_lut_shading(make_lut(os.environ.get("MAIN_LUT", "lights"), eye=eye))
        self.renderer.enable_motion_blur()
        self.renderer.shaded = False
        self._motion_data = motion_data
        self._time_tint = time_tint
        self._apply_mu_band()

        self._gui_ok = False
        if USE_GUI:
            # The import guard above only sees a MISSING imgui. A present-but-incompatible
            # one fails here instead -- e.g. imgui_bundle 1.92 dropped
            # ImFontAtlas.get_tex_data_as_rgba32, which moderngl-window 3.1.1's integration
            # calls (hence the imgui-bundle<1.92 pin). Fall back to no panel, not a crash.
            try:
                imgui.create_context()
                self.imgui_renderer = ModernglWindowRenderer(self.wnd)
                self._gui_ok = True
            except Exception as e:
                print(f"[tuning] disabled: imgui renderer failed to start "
                      f"({type(e).__name__}: {e}); running without the panel")
                self.imgui_renderer = None

        info(f"View Y: [{view_y_bot:.1f}, {view_y_top:.1f}], "
              f"spawn above {spawn_y_min:.1f}, despawn below {self._despawn_y:.1f}")

        self._spawning    = True
        self._last_spawn  = 0.0    # in SIM time (accumulated capped dt), not wall-clock
        self._sim_time    = 0.0
        self._prof = {"spawn": 0.0, "cull": 0.0, "physics": 0.0,
                      "geom": 0.0, "render": 0.0, "total": 0.0}
        self._prof_n = 0

        if self._gui_ok:
            self.panel = TuningPanel(self)   # needs _prof/_prof_n and _spawn_cov/_spawn_mu

        info("Warming up Taichi kernels…")
        self.chunks.spawn_box(pos=(0, 1000, 0), half_extents=(1, 1, 1))
        self.chunks.snapshot_ke()
        self.chunks.step(1e-4)
        self.chunks.compute_support()
        self.chunks.get_instance_data()
        self.chunks.cull(despawn_y=2000.0)
        self.chunks.clear()

        info(f"Slope: {scene_config.SLOPE_ANGLE}°, "
              f"top_y={scene_config.SLOPE_TOP_Y}, base_x={self._x_base:.1f}")
        STATUS.show_controls(lambda: (
            "SPACE      = toggle spawn\n"
            "R          = clear\n"
            "left click = delete a block\n"
            "W          = wireframe/shaded\n"
            f"B          = motion blur {onoff(self.renderer.blur)}\n"
            "G          = guides\n"
            "C          = colours (all / awake only / grey)\n"
            "ESC        = quit"))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------


    def on_render(self, time: float, frametime: float):
        STATUS.poll()
        t_frame = _time.perf_counter()


        t0 = _time.perf_counter()
        self.chunks.snapshot_ke()
        # CAP the physics dt: with raw frametime, a fps dip grows sub_dt -> omega*dt blows past 2
        # -> ringing -> everything stays awake -> fps dips further (death spiral). Capping keeps
        # sub_dt <= ~1/240 (stable at FENG_KHERTZ), so a heavy moment runs in mild slow-mo instead
        # of exploding; once it settles & sleeps, real-time resumes.
        dt = min(frametime, 1.0 / 60.0)
        self._sim_time += dt          # accumulate the CAPPED sim time (see spawn note below)
        sub_dt = dt / config.PHYSICS_SUBSTEPS
        for _ in range(config.PHYSICS_SUBSTEPS):
            self.chunks.step(sub_dt)
        ti.sync()
        t1 = _time.perf_counter()

        self.chunks.cull(despawn_y=self._despawn_y)
        self._apply_mu_band()      # no-op unless the panel moved the band
        t2 = _time.perf_counter()

        # Spawn on SIM time, not wall-clock `time`. dt is capped to 1/60, so on a slow frame the
        # sim runs in slow-mo -- but wall-clock keeps advancing, so a wall-clock spawn schedule
        # fires every frame and injects ~(60/fps)x too many blocks PER SIM-SECOND. That floods the
        # slope, which keeps fps low, which over-spawns more = a death spiral (measured: ~6x over-
        # spawn at 11 fps -> pile stays ~80% awake). Sim-time scheduling keeps the rate at the
        # intended SPAWN_BATCH/SPAWN_INTERVAL per sim-second at any fps (spawning also slows in a
        # heavy moment), so the slope can settle and real-time recovers. sim_time grows <=1/60 per
        # frame < SPAWN_INTERVAL, so at most one batch per frame -- no catch-up burst either.
        if self._spawning and (self._sim_time - self._last_spawn) >= scene_config.SPAWN_INTERVAL:
            self._last_spawn = self._sim_time
            batch_pos, batch_he, batch_R = [], [], []
            for _ in range(scene_config.SPAWN_BATCH):
                result = self._spawn_block()
                if result is not None:
                    batch_pos.append(result[0])
                    batch_he.append(result[1])
                    batch_R.append(result[2])
            if batch_pos:
                if self._time_tint is not None:
                    fields.birth_now[None] = self._sim_time      # stamped on these blocks (colour)
                self.chunks.spawn_boxes(batch_pos, batch_he, batch_R)
        t3 = _time.perf_counter()

        inst_data, inst_count = self.chunks.get_instance_data()
        tint = (self._time_tint(inst_count, COLOR_PERIOD, self.color_mode == 2)
                if (self._time_tint is not None and self.color_mode and inst_count) else None)
        self.renderer.update_instances(
            inst_data, inst_count,
            motion=self._motion_data(inst_count) if self._motion_data is not None else None,
            tint=tint)
        t4 = _time.perf_counter()

        self.renderer.draw(self.wnd.width, self.wnd.height)
        if self.panel is not None:
            imgui.new_frame()
            self.panel.draw()
            imgui.render()
            self.imgui_renderer.render(imgui.get_draw_data())
        if PROF_GPU:
            self.ctx.finish()
        t5 = _time.perf_counter()

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
            nc = fields.n_f[None]
            n_awake = int((fields.sleep_f.to_numpy()[:nc] == 0).sum()) if nc else 0
            info(f"--- Profile ({nc} blocks, {n_awake} awake, {n} frames) ---")
            for k in ["physics", "cull", "spawn", "geom", "render", "total"]:
                info(f"  {k:8s}: {p[k]/n*1000:6.2f} ms/frame")
            # --- awake-set diagnostic: is the cost a genuinely-moving set, or jitter? ---
            # active = #blocks that actually run the contact clip (the cost driver). If
            # active ~ nc, the one-ring is blanketing the pile -> B (island-skip) is dead.
            # If active << nc, B is engaging and the cost is the true awake set.
            # Among awake blocks: median/p90 speed and the slow-fraction (speed < SLEEP_SPEED
            # AND omega < SLEEP_OMEGA = "should-be-asleep jitter" -> sleep-path fix is the lever).
            if nc:
                slp = fields.sleep_f.to_numpy()[:nc]
                act = int(fields.active.to_numpy()[:nc].sum())
                aw = slp == 0
                v = fields.vel_f.to_numpy()[:nc][aw]
                w = fields.omega_f.to_numpy()[:nc][aw]
                sp = np.linalg.norm(v, axis=1) if len(v) else np.zeros(1)
                om = np.linalg.norm(w, axis=1) if len(w) else np.zeros(1)
                slow = int(((sp < params.SLEEP_SPEED) & (om < params.SLEEP_OMEGA)).sum())
                info(f"  awake-set: active(clip)={act}/{nc}  "
                      f"speed med={np.median(sp):.2f} p90={np.percentile(sp,90):.2f}  "
                      f"slow(should-sleep)={slow}/{n_awake}  "
                      f"[g={params.GRAVITY:.0f} sub={config.PHYSICS_SUBSTEPS} kHz={params.FENG_KHERTZ:.0f}]")
            for k in p: p[k] = 0.0
            self._prof_n = 0

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def on_key_event(self, key, action, modifiers):
        if self.panel is not None:
            self.imgui_renderer.key_event(key, action, modifiers)
            if action == self.wnd.keys.ACTION_PRESS and key == self.wnd.keys.TAB:
                self.panel.visible = not self.panel.visible
                return
            # a focused widget owns the keyboard -- otherwise ESC closes the app
            # mid-edit and R wipes the scene while you are typing a value
            if imgui.get_io().want_capture_keyboard:
                return
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key == self.wnd.keys.ESCAPE:
            self.wnd.close()
        if key == self.wnd.keys.G:
            self.renderer.show_guides = not self.renderer.show_guides
        if key == self.wnd.keys.W and self._motion_data is not None:
            self.renderer.shaded = not self.renderer.shaded
            info(f"[{'LUT-shaded' if self.renderer.shaded else 'wireframe'}]")
        if key == self.wnd.keys.C and self._time_tint is not None:
            self.color_mode = (self.color_mode + 1) % 3     # all -> awake only -> grey
            info(f"[colours: {('grey', 'all blocks', 'awake blocks only')[self.color_mode]}]")
        if key == self.wnd.keys.B and self._motion_data is not None:
            self.renderer.blur = not self.renderer.blur
            info(f"[motion blur {'ON' if self.renderer.blur else 'OFF'}]")
        if key == self.wnd.keys.SPACE:
            self._spawning = not self._spawning
            info(f"Spawning {'ON' if self._spawning else 'OFF'}")
        if key == self.wnd.keys.R:
            self.chunks.clear()
            info("Cleared")

    # imgui event forwarding. Mirrors ModernglWindowRenderer's mixin methods.

    def on_resize(self, width: int, height: int):
        if self.panel is not None:
            self.imgui_renderer.resize(width, height)

    def on_mouse_position_event(self, x, y, dx, dy):
        if self.panel is not None:
            self.imgui_renderer.mouse_position_event(x, y, dx, dy)

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.panel is not None:
            self.imgui_renderer.mouse_drag_event(x, y, dx, dy)

    def on_mouse_press_event(self, x, y, button):
        if self.panel is not None:
            self.imgui_renderer.mouse_press_event(x, y, button)
            if imgui.get_io().want_capture_mouse:      # a click on the panel is the panel's
                return
        if button == 1 and self._time_tint is not None:    # left click: delete the block under it
            import picking
            o, d = picking.ray_from_click(self.renderer, x, y, self.wnd.width, self.wnd.height)
            i = picking.pick(o, d)
            if i >= 0:
                picking.drop(i)                         # culled next frame; what it held wakes
            info(f"[click] deleted block {i}" if i >= 0 else "[click] no block under the cursor")

    def on_mouse_release_event(self, x, y, button):
        if self.panel is not None:
            self.imgui_renderer.mouse_release_event(x, y, button)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        if self.panel is not None:
            self.imgui_renderer.mouse_scroll_event(x_offset, y_offset)

    def on_unicode_char_entered(self, char):
        if self.panel is not None:
            self.imgui_renderer.unicode_char_entered(char)

    # ------------------------------------------------------------------
    # Spawner
    # ------------------------------------------------------------------

    def _slope_rest_rotation(self, he):
        """Rotation that puts the block's largest face against the slope."""
        k = int(np.argmin(he))
        n = self._slope_normal
        t1 = self._slope_down
        t2 = np.cross(n, t1)
        t2 /= np.linalg.norm(t2) + 1e-12
        cols = [None, None, None]
        cols[k] = n
        rem = [i for i in range(3) if i != k]
        cols[rem[0]] = t1
        cols[rem[1]] = t2
        R = np.column_stack(cols)
        if np.linalg.det(R) < 0:
            cols[rem[0]], cols[rem[1]] = cols[rem[1]], cols[rem[0]]
            R = np.column_stack(cols)
        return R

    def _mu_band_y(self):
        span = self._view_y_top - self._view_y_bot
        return (self._view_y_bot + span * scene_config.SLOPE_MU_BAND_LO,
                self._view_y_bot + span * scene_config.SLOPE_MU_BAND_HI)

    def _apply_mu_band(self):
        """Push the slope-mu band to the solver and redraw its guide lines -- only when
        the config fractions changed (the tuning panel writes them)."""
        band = (scene_config.SLOPE_MU_BAND_LO, scene_config.SLOPE_MU_BAND_HI)
        if band == self._mu_band:
            return
        self._mu_band = band
        y_lo, y_hi = self._mu_band_y()
        if hasattr(self.chunks, "set_slope_mu_band"):
            self.chunks.set_slope_mu_band(y_lo, y_hi)
        # a line across the slope's full width at each height, lifted off the surface
        # along the normal so it does not z-fight (drawn without depth test anyway)
        n = self._slope_normal; s, c = n[0], n[1]
        segs = []
        for y in (y_lo, y_hi):
            x = (self._slope_d - c * y) / s
            p = np.array([x, y, 0.0]) + 0.1 * n
            segs.append([[p[0], p[1], scene_config.SLOPE_Z_MIN], [p[0], p[1], scene_config.SLOPE_Z_MAX]])
        self.renderer.set_guides(segs)
        info(f"[slope mu] band y = {y_lo:.1f} .. {y_hi:.1f} "
              f"(mu {params.SLOPE_MU_HI} below, {params.SLOPE_MU_LO} above)")

    def _spawn_block(self):
        log_he = np.random.multivariate_normal(self._spawn_mu, self._spawn_cov)
        he = np.exp(log_he)
        he = np.clip(he, scene_config.SPAWN_HE_MIN, config.SPAWN_HE_MAX)

        R = self._slope_rest_rotation(he)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis) + 1e-8
        R = _rodrigues(axis * np.random.uniform(0, np.radians(8))) @ R

        z = np.random.uniform(self._spawn_z_min + he[2],
                              self._spawn_z_max - he[2])
        t = np.random.uniform(self._spawn_t_min, self._spawn_t_max)
        surface_pt = (np.array([0.0, scene_config.SLOPE_TOP_Y, z])
                      + t * self._slope_down)

        sn = self._slope_normal
        support = float(np.abs(R.T @ sn) @ he)
        pos = surface_pt + (support + 0.05) * sn

        return pos, he, R



if __name__ == "__main__":
    # mglw.run_window_config returns when the window closes, but several
    # non-daemon threads can keep the Python process hanging: Taichi
    # runtime (Metal/CUDA dispatcher), pyglet's NSApp on macOS, etc.
    # Tear them down explicitly and then os._exit to guarantee process
    # termination. atexit handlers are skipped — acceptable for an
    # interactive piece with no critical cleanup.
    import os
    import sys
    import traceback
    exit_code = 0
    try:
        mglw.run_window_config(BlockCascade)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        STATUS.end()
        try:
            import taichi as _ti
            _ti.reset()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

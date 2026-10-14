"""minimal.py — the smallest feng3d piece: a grid of randomly sized and rotated boxes dropped
onto the ground, drawn in wireframe. Start here to build your own.

    uv run examples/minimal.py

The loop, once per frame:
    world.snapshot_ke()                 once, before the substeps
    world.step(dt / SUBSTEPS)           SUBSTEPS times
    rows, n = world.get_instance_data() (n, 15) float32: centre(3), half-extents(3), rotation(9)
Hand the rows to any renderer; here, examples/render.py.
"""
import console                      # first: quiets Taichi's and moderngl-window's start-up output
import os
import sys

import numpy as np
import moderngl_window as mglw

from feng3d import config
from feng3d.solver import Solver
import scene_config
from render import Renderer

DT = 1.0 / 60.0
SUBSTEPS = config.PHYSICS_SUBSTEPS  # config.py sizes the contact stiffness for this many
HE_MAX = 2.0                        # largest half-extent spawned; must not exceed max_he below


def make_world():
    # The ground (y = config.GROUND_Y) acts only inside ledge_bounds (x0, x1, z0, z1).
    world = Solver(max_he=HE_MAX, ledge_bounds=(-1e6, 1e6, -1e6, 1e6))
    rng = np.random.default_rng(0)
    pitch = 2.0 * np.sqrt(3.0) * HE_MAX + 0.5      # a block's diagonal + margin: no overlaps
    for layer in range(6):
        for ix in range(4):
            for iz in range(4):
                pos = ((ix - 1.5) * pitch, pitch * (layer + 1), (iz - 1.5) * pitch)
                he = rng.uniform(0.6, HE_MAX, 3)
                q, r = np.linalg.qr(rng.normal(size=(3, 3)))
                R = q * np.sign(np.diag(r))          # uniformly random rotation
                if np.linalg.det(R) < 0:
                    R[:, 0] = -R[:, 0]
                world.spawn_box(pos=pos, half_extents=tuple(he), R=R)
    return world


class MinimalWindow(mglw.WindowConfig):
    title = "feng3d — minimal"
    gl_version = (3, 3)
    window_size = (scene_config.WINDOW_W, scene_config.WINDOW_H)
    aspect_ratio = None

    def __init__(self, **kw):
        super().__init__(**kw)
        self.world = make_world()
        scene_config.ISO_SCALE = 40.0                # half-height of the view, world units
        self.renderer = Renderer(self.ctx, look_center=(0.0, 10.0, 0.0),
                                 slope_normal=(1.0, 0.0, 0.0), eye_dir=(1.0, 0.7, 1.0))
        console.STATUS.show_controls(lambda: "ESC = quit")

    def on_render(self, time, frametime):
        self.world.snapshot_ke()
        for _ in range(SUBSTEPS):
            self.world.step(DT / SUBSTEPS)
        rows, n = self.world.get_instance_data()
        self.renderer.update_instances(rows, n)
        self.renderer.draw(self.wnd.width, self.wnd.height)


if __name__ == "__main__":
    try:
        mglw.run_window_config(MinimalWindow)
    finally:
        console.STATUS.end()
        # Taichi's and the window's threads can keep the process alive after the window closes
        sys.stdout.flush(); os._exit(0)

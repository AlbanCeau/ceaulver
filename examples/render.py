from pathlib import Path

import glm
import moderngl
import numpy as np

import scene_config

_SHADER_DIR = Path(__file__).parent / "shaders"

# ------------------------------------------------------------------
# Unit cube geometry (static, uploaded once)
# ------------------------------------------------------------------

_CORNERS = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype="f4")

_TRI_IDX = np.array([0,1,2,0,2,3, 4,6,5,4,7,6, 0,4,5,0,5,1,
                      3,2,6,3,6,7, 0,3,7,0,7,4, 1,5,6,1,6,2])
_EDGE_IDX = np.array([0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4,
                       0,4,1,5,2,6,3,7])

_CUBE_TRI_VERTS  = np.ascontiguousarray(_CORNERS[_TRI_IDX],  dtype="f4")  # (36, 3)
# per-vertex outward face normal for the 6 faces (6 vertices each): the axis every
# corner of the face shares, with its sign
_FACE_NRM = np.zeros((36, 3), dtype="f4")
for _f in range(6):
    _c = _CUBE_TRI_VERTS[6 * _f:6 * _f + 6]
    _ax = int(np.argmax(np.abs(_c.mean(axis=0))))
    _FACE_NRM[6 * _f:6 * _f + 6, _ax] = np.sign(_c[0, _ax])
_CUBE_FACE_VERTS = np.ascontiguousarray(np.hstack([_CUBE_TRI_VERTS, _FACE_NRM]), dtype="f4")
_CUBE_EDGE_VERTS = np.ascontiguousarray(_CORNERS[_EDGE_IDX], dtype="f4")  # (24, 3)

# ------------------------------------------------------------------

class Renderer:
    def __init__(self, ctx: moderngl.Context, look_center: tuple,
                 slope_normal: tuple, eye_dir: tuple = None):
        self.ctx = ctx
        self._look_center = look_center
        self._slope_normal = slope_normal
        self._eye_dir = eye_dir          # if set, full 3D view direction (overrides horizontal)

        # Instanced shaders for block geometry
        self._depth_inst_prog = ctx.program(
            vertex_shader=(_SHADER_DIR / "depth_inst.vert").read_text(),
            fragment_shader=(_SHADER_DIR / "depth.frag").read_text(),
        )
        self._edge_inst_prog = ctx.program(
            vertex_shader=(_SHADER_DIR / "edge_inst.vert").read_text(),
            fragment_shader=(_SHADER_DIR / "edge.frag").read_text(),
        )

        # Non-instanced line shader (guide lines, flash overlay)
        self._edge_prog = ctx.program(
            vertex_shader=(_SHADER_DIR / "edge.vert").read_text(),
            fragment_shader=(_SHADER_DIR / "edge.frag").read_text(),
        )

        r, g, b = scene_config.BG_COLOR
        self._depth_inst_prog["u_bg"].value = (r, g, b, 1.0)
        self._edge_inst_prog["u_color"].value = scene_config.EDGE_COLOR
        self._edge_inst_prog["u_bias"].value = scene_config.EDGE_BIAS
        self._edge_prog["u_color"].value = (1.0, 1.0, 1.0)

        # Static unit cube VBOs
        self._cube_solid_vbo = ctx.buffer(_CUBE_TRI_VERTS.tobytes())
        self._cube_edge_vbo  = ctx.buffer(_CUBE_EDGE_VERTS.tobytes())

        # LUT-shaded faces (opt-in via enable_lut_shading)
        self.shaded = False
        # velocity-buffer motion blur over the shaded faces (opt-in via enable_motion_blur)
        self.blur = False
        self._mb_prog = None
        self._motion_vbo = None
        self._tint_vbo = None              # per-instance colour (3f); white when not given
        self._tint_white = -1              # instance count the buffer holds white for (-1: not white)
        self._mb_vao = None
        self._mb_fbo = None
        self._face_prog = None
        self._face_vao = None
        self._lut_tex = None

        # Instance data buffer (allocated on first update)
        self._inst_vbo = None
        self._solid_vao = None
        self._edge_vao = None
        self._n_instances = 0

        self._identity_bytes = np.eye(4, dtype="f4").tobytes()

        # Guide lines (world-space segments, e.g. the slope-mu transition band)
        self._guide_vbo = None
        self._guide_vao = None
        self._n_guide = 0
        self.show_guides = True
        self.copies = [(0.0, 0.0, 0.0)]    # instanced passes are drawn once per offset (set_copies)
        self.persp_fov = None              # deg: perspective camera (None = orthographic, the default)

        # Highlight overlay (diagnostic: mark "hot" blocks)
        self._hot_vbo = None
        self._hot_vao = None
        # Flash overlay (diagnostic: sync circle)
        self._circ_vbo = None
        self._circ_vao = None

    # ------------------------------------------------------------------
    # Instance data (called every frame)
    # ------------------------------------------------------------------

    def update_instances(self, inst_data: np.ndarray, count: int, motion: np.ndarray = None,
                         tint: np.ndarray = None):
        self._n_instances = count
        if count == 0:
            return
        # per-instance colour multiplier for the LUT-shaded faces; white (no change) if None
        if tint is not None:
            tdata = np.ascontiguousarray(tint, dtype="f4"); self._tint_white = -1
        elif self._tint_white != count:
            tdata = np.ones((count, 3), dtype="f4"); self._tint_white = count
        else:
            tdata = None
        if tdata is not None:
            if self._tint_vbo is None:
                self._tint_vbo = self.ctx.buffer(tdata.tobytes())
            else:
                self._tint_vbo.orphan(tdata.nbytes)
                self._tint_vbo.write(tdata.tobytes())
        if motion is not None:                       # per instance: velocity(3), omega(3)
            mdata = np.ascontiguousarray(motion, dtype="f4")
            if self._motion_vbo is None:
                self._motion_vbo = self.ctx.buffer(mdata.tobytes())
            else:
                self._motion_vbo.orphan(mdata.nbytes)
                self._motion_vbo.write(mdata.tobytes())
            self._has_motion = True
        else:
            self._has_motion = False

        data = np.ascontiguousarray(inst_data, dtype="f4")
        nbytes = data.nbytes

        if self._inst_vbo is None:
            self._inst_vbo = self.ctx.buffer(data.tobytes())
            # per-instance: pos(3f) he(3f) R0(3f) R1(3f) R2(3f) = 15f, 60 bytes
            inst_fmt = "3f 3f 3f 3f 3f /i"
            inst_attrs = ("in_pos", "in_he", "in_R0", "in_R1", "in_R2")

            self._solid_vao = self.ctx.vertex_array(
                self._depth_inst_prog,
                [(self._cube_solid_vbo, "3f", "in_vert"),
                 (self._inst_vbo, inst_fmt, *inst_attrs)],
            )
            self._edge_vao = self.ctx.vertex_array(
                self._edge_inst_prog,
                [(self._cube_edge_vbo, "3f", "in_vert"),
                 (self._inst_vbo, inst_fmt, *inst_attrs)],
            )
        else:
            self._inst_vbo.orphan(nbytes)
            self._inst_vbo.write(data.tobytes())

    def draw_highlight(self, hot_data: np.ndarray, count: int, width: int, height: int,
                       color=(1.0, 0.25, 0.1)):
        """Diagnostic overlay: re-draw the edges of 'hot' blocks in a bright colour, on top.
        hot_data is the (count, 15) instance rows of the blocks to highlight."""
        if count == 0:
            return
        data = np.ascontiguousarray(hot_data, dtype="f4")
        inst_fmt = "3f 3f 3f 3f 3f /i"
        inst_attrs = ("in_pos", "in_he", "in_R0", "in_R1", "in_R2")
        if self._hot_vbo is None:
            self._hot_vbo = self.ctx.buffer(data.tobytes())
            self._hot_vao = self.ctx.vertex_array(
                self._edge_inst_prog,
                [(self._cube_edge_vbo, "3f", "in_vert"),
                 (self._hot_vbo, inst_fmt, *inst_attrs)],
            )
        else:
            self._hot_vbo.orphan(data.nbytes)
            self._hot_vbo.write(data.tobytes())
        mvp_bytes = np.array(self._mvp(width, height), dtype="f4").T.tobytes()
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self._edge_inst_prog["u_mvp"].write(mvp_bytes)
        self._edge_inst_prog["u_color"].value = color
        self._edge_inst_prog["u_bias"].value = scene_config.EDGE_BIAS * 4.0   # bias forward -> draws on top
        self._hot_vao.render(moderngl.LINES, instances=count)
        self._edge_inst_prog["u_color"].value = scene_config.EDGE_COLOR       # restore
        self._edge_inst_prog["u_bias"].value = scene_config.EDGE_BIAS

    def draw_circle(self, width: int, height: int, radius: float = 0.42, color=(1.0, 1.0, 1.0)):
        """Diagnostic sync overlay: a filled circle in screen centre (NDC), aspect-corrected."""
        aspect = width / height
        k = 64
        ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=True)
        rim = np.stack([np.cos(ang) * radius / aspect, np.sin(ang) * radius,
                        np.full(k, -1.0)], axis=1)
        verts = np.vstack([[0.0, 0.0, -1.0], rim]).astype("f4")
        if self._circ_vbo is None:
            self._circ_vbo = self.ctx.buffer(verts.tobytes())
            self._circ_vao = self.ctx.vertex_array(
                self._edge_prog, [(self._circ_vbo, "3f", "in_vert")])
        else:
            self._circ_vbo.orphan(verts.nbytes)
            self._circ_vbo.write(verts.tobytes())
        self.ctx.enable_only(0)
        self._edge_prog["u_mvp"].write(self._identity_bytes)
        self._edge_prog["u_bias"].value = 0.0
        self._edge_prog["u_color"].value = color
        self._circ_vao.render(moderngl.TRIANGLE_FAN, vertices=len(verts))

    def set_guides(self, segments):
        """World-space line segments, shape (n, 2, 3). Drawn over the blocks."""
        verts = np.asarray(segments, dtype="f4").reshape(-1, 3)
        self._n_guide = len(verts)
        if self._n_guide == 0:
            return
        if self._guide_vbo is None or self._guide_vbo.size != verts.nbytes:
            self._guide_vbo = self.ctx.buffer(verts.tobytes())
            self._guide_vao = self.ctx.vertex_array(
                self._edge_prog, [(self._guide_vbo, "3f", "in_vert")]
            )
        else:
            self._guide_vbo.write(verts.tobytes())

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def enable_lut_shading(self, lut: np.ndarray):
        """Opt in to the LUT-shaded mode: every cube face drawn flat, its grey looked up
        from `lut` (H x W, values 0..1) by the face's world orientation (lat-long: u =
        azimuth atan2(z, x), v = polar angle from +y). One instanced pass, no edges."""
        self._face_prog = self.ctx.program(
            vertex_shader=(_SHADER_DIR / "face_inst.vert").read_text(),
            fragment_shader=(_SHADER_DIR / "face_lut.frag").read_text(),
        )
        lut = np.clip(np.asarray(lut, dtype="f4"), 0.0, 1.0)
        if lut.ndim == 2:                    # grey LUT: replicate into RGB
            lut = np.repeat(lut[:, :, None], 3, axis=2)
        lut = np.ascontiguousarray(lut, dtype="f4")
        self._lut_tex = self.ctx.texture((lut.shape[1], lut.shape[0]), 3, lut.tobytes(), dtype="f4")
        self._lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._lut_tex.repeat_x = True; self._lut_tex.repeat_y = False
        self._face_vbo = self.ctx.buffer(_CUBE_FACE_VERTS.tobytes())
        self._face_vao = None                # built once the instance buffer exists
        self.shaded = True

    def enable_motion_blur(self, shutter=1.0 / 120.0, samples=12, max_px=64.0, dilate_px=10.0):
        """Opt in to velocity-buffer motion blur (needs enable_lut_shading, and motion
        passed to update_instances). Faces render off-screen with a per-pixel screen
        motion over `shutter` seconds (exact rigid-body v + omega x r); a full-screen
        pass averages colour along the largest motion in a small neighbourhood."""
        self._mb_prog = self.ctx.program(
            vertex_shader=(_SHADER_DIR / "face_inst_mb.vert").read_text(),
            fragment_shader=(_SHADER_DIR / "face_lut_mb.frag").read_text(),
        )
        self._post_prog = self.ctx.program(
            vertex_shader=(_SHADER_DIR / "fullscreen.vert").read_text(),
            fragment_shader=(_SHADER_DIR / "mblur.frag").read_text(),
        )
        self._post_vao = self.ctx.vertex_array(self._post_prog, [])
        self.mb_shutter = float(shutter); self.mb_samples = int(samples)
        self.mb_max_px = float(max_px); self.mb_dilate_px = float(dilate_px)
        self.blur = True

    def _ensure_mb_fbo(self):
        size = self.ctx.fbo.size
        if self._mb_fbo is not None and self._mb_size == size:
            return
        for o in (getattr(self, "_mb_col", None), getattr(self, "_mb_velt", None),
                  getattr(self, "_mb_depth", None), self._mb_fbo):
            if o is not None:
                o.release()
        self._mb_col = self.ctx.texture(size, 4, dtype="f1")
        self._mb_velt = self.ctx.texture(size, 2, dtype="f2")
        for t in (self._mb_col, self._mb_velt):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False; t.repeat_y = False
        self._mb_depth = self.ctx.depth_renderbuffer(size)
        self._mb_fbo = self.ctx.framebuffer([self._mb_col, self._mb_velt], self._mb_depth)
        self._mb_size = size

    def set_copies(self, offsets):
        """Draw every instanced pass once per world offset -- periodic tiling on the GPU,
        instead of uploading copies of the instance buffer. Default: one copy at 0."""
        self.copies = [tuple(map(float, o)) for o in offsets] or [(0.0, 0.0, 0.0)]

    def _each_copy(self, prog):
        for o in self.copies:
            prog["u_offset"].value = o
            yield

    def draw(self, width: int, height: int):
        mvp_bytes = np.array(self._mvp(width, height), dtype="f4").T.tobytes()

        r, g, b = scene_config.BG_COLOR
        self.ctx.fbo.clear(r, g, b, 1.0)
        self.ctx.enable_only(moderngl.DEPTH_TEST)

        if (self._n_instances > 0 and self.shaded and self.blur and self._mb_prog is not None
                and getattr(self, "_has_motion", False)):
            # motion blur: faces + velocity off-screen, then the blur pass to the screen
            self._ensure_mb_fbo()
            if self._mb_vao is None:
                self._mb_vao = self.ctx.vertex_array(
                    self._mb_prog,
                    [(self._face_vbo, "3f 3f", "in_vert", "in_nrm"),
                     (self._inst_vbo, "3f 3f 3f 3f 3f /i",
                      "in_pos", "in_he", "in_R0", "in_R1", "in_R2"),
                     (self._motion_vbo, "3f 3f /i", "in_vel", "in_omg"),
                     (self._tint_vbo, "3f /i", "in_tint")],
                )
            screen = self.ctx.fbo
            self._mb_fbo.use()
            self._mb_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)   # velocity 0, alpha 0 = empty
            self.ctx.enable_only(moderngl.DEPTH_TEST)
            self._mb_prog["u_mvp"].write(mvp_bytes)
            self._mb_prog["u_shutter"].value = self.mb_shutter
            self._lut_tex.use(0)
            self._mb_prog["u_lut"].value = 0
            for _ in self._each_copy(self._mb_prog):
                self._mb_vao.render(moderngl.TRIANGLES, instances=self._n_instances)
            screen.use()
            self.ctx.enable_only(0)
            w_px, h_px = self._mb_size
            self._mb_col.use(1); self._mb_velt.use(2)
            pp = self._post_prog
            pp["u_color"].value = 1; pp["u_vel"].value = 2
            pp["u_texel"].value = (1.0 / w_px, 1.0 / h_px)
            pp["u_max_uv"].value = self.mb_max_px / max(w_px, h_px)
            pp["u_dilate_px"].value = self.mb_dilate_px
            pp["u_bg"].value = (r, g, b)
            pp["u_samples"].value = self.mb_samples
            self._post_vao.render(moderngl.TRIANGLES, vertices=3)
            self.ctx.enable_only(moderngl.DEPTH_TEST)
        elif self._n_instances > 0 and self.shaded and self._face_prog is not None:
            # LUT-shaded faces: a single instanced pass, depth-tested, no edges
            if self._face_vao is None:
                self._face_vao = self.ctx.vertex_array(
                    self._face_prog,
                    [(self._face_vbo, "3f 3f", "in_vert", "in_nrm"),
                     (self._inst_vbo, "3f 3f 3f 3f 3f /i",
                      "in_pos", "in_he", "in_R0", "in_R1", "in_R2"),
                     (self._tint_vbo, "3f /i", "in_tint")],
                )
            self._face_prog["u_mvp"].write(mvp_bytes)
            self._lut_tex.use(0)
            self._face_prog["u_lut"].value = 0
            for _ in self._each_copy(self._face_prog):
                self._face_vao.render(moderngl.TRIANGLES, instances=self._n_instances)
        elif self._n_instances > 0:
            # Pass 1: depth pre-pass (instanced solid cubes)
            self._depth_inst_prog["u_mvp"].write(mvp_bytes)
            for _ in self._each_copy(self._depth_inst_prog):
                self._solid_vao.render(moderngl.TRIANGLES, instances=self._n_instances)

            # Pass 2: block edges (instanced edge lines)
            self._edge_inst_prog["u_mvp"].write(mvp_bytes)
            self._edge_inst_prog["u_bias"].value = scene_config.EDGE_BIAS
            self._edge_inst_prog["u_color"].value = scene_config.EDGE_COLOR
            for _ in self._each_copy(self._edge_inst_prog):
                self._edge_vao.render(moderngl.LINES, instances=self._n_instances)

        # Pass 2b: guide lines (world space, no depth test so blocks never hide them)
        if self.show_guides and self._n_guide > 0:
            self.ctx.enable_only(0)
            self._edge_prog["u_mvp"].write(mvp_bytes)
            self._edge_prog["u_bias"].value = 0.0
            self._edge_prog["u_color"].value = scene_config.GUIDE_COLOR
            self._guide_vao.render(moderngl.LINES, vertices=self._n_guide)

    # ------------------------------------------------------------------
    # Camera — horizontal view
    # ------------------------------------------------------------------

    def _mvp(self, width: int, height: int) -> glm.mat4:
        center = glm.vec3(*self._look_center)
        if self._eye_dir is not None:
            eye_dir = glm.normalize(glm.vec3(*self._eye_dir))   # full 3D (e.g. iso, off-vertical)
        else:
            sn = self._slope_normal
            eye_dir = glm.normalize(glm.vec3(sn[0], 0.0, sn[2] if sn[2] != 0 else 0.0))
        aspect = width / height
        s = scene_config.ISO_SCALE
        if self.persp_fov:
            # perspective: back off until the half-height visible at the look centre is the
            # ortho view's s -- same framing, but depth reads (near = bigger)
            half = np.radians(self.persp_fov) * 0.5
            dist = s / np.tan(half)
            eye = center + eye_dir * float(dist)
            view = glm.lookAt(eye, center, glm.vec3(0.0, 1.0, 0.0))
            proj = glm.perspective(float(2 * half), aspect, float(max(0.5, dist * 0.02)), float(dist * 20.0))
            return proj * view
        eye = center + eye_dir * scene_config.ISO_EYE_DIST

        view = glm.lookAt(eye, center, glm.vec3(0.0, 1.0, 0.0))

        proj = glm.ortho(-s * aspect, s * aspect, -s, s,
                         scene_config.ISO_NEAR, scene_config.ISO_FAR)
        return proj * view

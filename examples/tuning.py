"""Live parameter panel for the block cascade — a dev tuning tool.

Why this works at all: params.sync_params() runs at the top of every
step(), builds a tuple from *params'* own module globals via _PARAM_NAMES, early-returns
when nothing changed, and otherwise uploads to the PRM Taichi field. So 31 solver
parameters are hot-swappable at the cost of one attribute write. Nothing in the solver
had to change to make these sliders bite.

Two traps this module exists to handle:

  * params binds config into its OWN globals once at import.
    Writing config.GRAVITY at runtime is a silent no-op -- solver params must be written
    to params.<NAME>. Params that scree.py re-reads each frame (the spawner, substeps)
    are the opposite case and must be written to config.<NAME>. The `target` field on
    each Param says which.
  * Several values are derived. Moving a driver without re-deriving its dependents leaves
    the sim quietly inconsistent -- e.g. dragging GRAVITY without re-deriving VMAX_CONTACT
    moves the pile load but not the yield ceiling. _push() handles all four chains.

Save writes back into config.py by replacing only the numeric token of assignments whose
right-hand side is already a numeric literal. Comments, layout and derived expressions are
untouched by construction.
"""

import ast
import shutil
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from feng3d import params, fields
from feng3d import config
import scene_config


def _cfg(name):
    """The module a name is defined in: the examples' scene_config, else the solver's config."""
    return scene_config if hasattr(scene_config, name) else config

# Restitution feeds _damping_coeff = -2 ln(e) / sqrt(pi^2 + ln^2 e); e=0 is a log domain
# error and e=1 is zero damping. Keep sliders strictly inside.
_E_MIN, _E_MAX = 0.001, 0.999

# Changing any of these re-derives FENG_KHERTZ and VMAX_CONTACT together.
_CONTACT_DRIVERS = {"CONTACT_SAFETY", "PILE_DEPTH", "YIELD_STRAIN",
                    "GRAVITY", "SPAWN_HE_MAX", "PHYSICS_SUBSTEPS"}

LOG = int(imgui.SliderFlags_.logarithmic)


class Param:
    __slots__ = ("name", "lo", "hi", "kind", "flags", "target", "note", "ro")

    # target: "chunks"     -> a live PRM/solver global (params.<NAME>)
    #         "config"      -> re-read from config each frame by scree.py
    #         "derived_d"   -> restitution; re-derives params._SLOPE_D
    #         "spawn_dist"  -> rebuilds the spawner's cached mu/covariance
    #         "driver"      -> not a solver param; only feeds derive_contact()
    def __init__(self, name, lo, hi, kind="f", log=False,
                 target="chunks", note="", ro=False):
        self.name, self.lo, self.hi, self.kind = name, lo, hi, kind
        self.flags = LOG if log else 0
        self.target, self.note, self.ro = target, note, ro


# group title -> params. Ranges bracket the values this piece has been measured at.
GROUPS = [
    ("World", [
        Param("GRAVITY", -200.0, -1.0),
        Param("PHYSICS_SUBSTEPS", 1, 16, kind="i", target="config",
              note="main.py re-reads this every frame; also moves VMAX_CONTACT."),
    ]),
    ("Contact sizing (drives FENG_KHERTZ / VMAX_CONTACT)", [
        Param("CONTACT_SAFETY", 0.5, 12.0, target="driver",
              note="measured ceiling ~3; at 10 the scene sheds ~20% of its blocks."),
        Param("PILE_DEPTH", 2, 20, kind="i", target="driver"),
        Param("YIELD_STRAIN", 1e-4, 3e-2, log=True, target="driver",
              note="the look knob. 0.26% settles 3/3 runs, 0.165% 2/3, 0.101% 1/3."),
        Param("D_HERTZ", 1.0, 5000.0, log=True),
        Param("CHUNK_MU", 0.0, 2.0),
    ]),
    ("Boundary", [
        Param("SLOPE_K", 100.0, 300000.0, log=True),
        Param("SOIL_MU", 0.0, 20.0),
        Param("SLOPE_MU", 0.0, 20.0),
        Param("SLOPE_MU_LO", 0.0, 20.0),
        Param("SLOPE_MU_HI", 0.0, 20.0),
        Param("SLOPE_MU_BAND_LO", -0.5, 1.5, target="config",
              note="view fraction from the bottom where the slope is fully SLOPE_MU_HI. "
                   "G toggles the guide lines."),
        Param("SLOPE_MU_BAND_HI", -0.5, 1.5, target="config",
              note="view fraction above which the slope is fully SLOPE_MU_LO."),
        Param("SLOPE_RESTITUTION", _E_MIN, _E_MAX, target="derived_d"),
        Param("STIFFNESS_ALPHA", 0.01, 1.0,
              note="K_eff = min(K, alpha*m_eff/dt^2); pins boundary omega*dt at sqrt(alpha)."),
        Param("PEN_MAX", 0.1, 5.0),
    ]),
    ("Rolling", [
        Param("ROLLING_RESIST", 0.0, 20.0),
        Param("CHUNK_ROLL", 0.0, 20.0),
        Param("SLOPE_ROLL", 0.0, 20.0),
        Param("ROLLING_M_REF", 1.0, 500.0,
              note="WARNING: baked into rs_f at spawn (chunks.py:1747) but live in the "
                   "sleep scale (chunks.py:1454). Existing blocks desync from new ones "
                   "until you clear (R)."),
    ]),
    ("Sleep / freeze", [
        Param("SLEEP_SPEED", 0.0, 10.0),
        Param("SLEEP_OMEGA", 0.0, 10.0),
        Param("FREEZE_ABOVE", 0, 10, kind="i"),
        Param("FREEZE_CONTACTS", 0, 20, kind="i"),
        Param("WAKE_VEL", 0.0, 20.0),
        Param("WAKE_FORCE", 0.0, 5.0),
        Param("ISLAND_RECHECK", 1, 120, kind="i"),
    ]),
    ("Clamps", [
        Param("MAX_SPEED", 10.0, 1000.0, log=True),
        Param("MAX_OMEGA", 1.0, 200.0, log=True),
        Param("CONTACT_TOL", 0.0, 5.0),
        Param("FRAC_BUDGET", 0.1, 500.0, log=True),
        Param("FRICTION_VISC", 0.0, 10000.0, log=True),
    ]),
    ("Spawner", [
        Param("SPAWN_INTERVAL", 0.05, 3.0, target="config"),
        Param("SPAWN_BATCH", 1, 50, kind="i", target="config"),
        Param("SPAWN_LOG_MU", -1.0, 2.0, target="spawn_dist"),
        Param("SPAWN_LOG_SIGMA", 0.01, 1.5, target="spawn_dist"),
        Param("SPAWN_SHAPE_RHO", 0.0, 0.99, target="spawn_dist"),
        Param("SPAWN_HE_MIN", 0.1, 2.5, target="config"),
        Param("SPAWN_HE_MAX", 0.5, 5.0, target="config", ro=True,
              note="read-only: also sizes the broadphase cell (main.py:114 -> "
                   "chunks.py:1653). The invariant is cell >= 2*max_he*sqrt(3); raising "
                   "it live silently drops contacts. Restart to change."),
    ]),
]

ALL = [p for _, ps in GROUPS for p in ps]
BY_NAME = {p.name: p for p in ALL}


def _fmt(value, kind):
    """Render a value as a Python numeric literal for config.py."""
    if kind == "i":
        return str(int(round(value)))
    s = f"{value:.6g}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s


class TuningPanel:
    def __init__(self, app):
        self.app = app
        self.visible = True
        self.status = ""
        self.state = {p.name: getattr(_cfg(p.name), p.name) for p in ALL}
        self._stats = {}
        self._stats_age = 0
        self.push_all()

    # ---------------------------------------------------------------- runtime push

    def push_all(self):
        for p in ALL:
            self._push(p.name)

    def _push(self, name):
        """Write one panel value to wherever the running sim actually reads it."""
        p = BY_NAME[name]
        v = self.state[name]
        if p.kind == "i":
            v = int(round(v))

        if p.target == "chunks":
            setattr(params, name, v)
        elif p.target == "config":
            setattr(_cfg(name), name, v)
        elif p.target == "derived_d":
            # The restitution is consumed inline in params.py and never rebound;
            # only the derived damping coefficient is in _PARAM_NAMES.
            setattr(config, name, v)
            d = params._damping_coeff(min(max(v, _E_MIN), _E_MAX))
            params._SLOPE_D = d
        elif p.target == "spawn_dist":
            setattr(_cfg(name), name, v)
            self._rebuild_spawn_dist()
        elif p.target == "driver":
            # not a solver param -- it only acts through _rederive_contact() below
            setattr(config, name, v)

        if name in _CONTACT_DRIVERS:
            self._rederive_contact()

    def _rederive_contact(self):
        kh, vmax = config.derive_contact(
            self.state["CONTACT_SAFETY"], int(round(self.state["PILE_DEPTH"])),
            self.state["GRAVITY"], self.state["SPAWN_HE_MAX"],
            int(round(self.state["PHYSICS_SUBSTEPS"])), self.state["YIELD_STRAIN"])
        params.FENG_KHERTZ = kh
        params.VMAX_CONTACT = vmax
        config.FENG_KHERTZ = kh
        config.VMAX_CONTACT = vmax

    def _rebuild_spawn_dist(self):
        """Mirror of scree.py — the spawner caches these in __init__."""
        s2 = self.state["SPAWN_LOG_SIGMA"] ** 2
        rho = self.state["SPAWN_SHAPE_RHO"]
        self.app._spawn_cov = s2 * (rho * np.ones((3, 3)) + (1 - rho) * np.eye(3))
        self.app._spawn_mu = np.full(3, self.state["SPAWN_LOG_MU"])

    # ---------------------------------------------------------------- save / revert

    def save(self):
        """Rewrite only the numeric token of literal assignments, in each config file."""
        written, skipped, files = [], [], []
        for mod in (config, scene_config):
            path = Path(mod.__file__)
            n = self._save_file(path, [p for p in ALL if _cfg(p.name) is mod], written, skipped)
            if n:
                files.append(f"{n} to {path.name}")
        if not files:
            self.status = "nothing changed"
            return
        self.status = f"saved {', '.join(files)} (backup .py.bak)"
        print(f"[tuning] {self.status}")
        for w in written:
            print(f"[tuning]   {w}")
        if skipped:
            print(f"[tuning]   not written (derived/non-literal): {', '.join(skipped)}")

    def _save_file(self, path, plist, written, skipped):
        src = path.read_text()
        tree = ast.parse(src)
        spans = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            val = node.value
            ok = (isinstance(val, ast.Constant)
                  and isinstance(val.value, (int, float))
                  and not isinstance(val.value, bool))
            neg = (isinstance(val, ast.UnaryOp) and isinstance(val.op, ast.USub)
                   and isinstance(val.operand, ast.Constant)
                   and isinstance(val.operand.value, (int, float)))
            if (ok or neg) and val.lineno == val.end_lineno:
                spans[tgt.id] = (val.lineno, val.col_offset, val.end_col_offset,
                                 val.value if ok else -val.operand.value)

        lines = src.splitlines(keepends=True)
        edits = []
        for p in plist:
            span = spans.get(p.name)
            if span is None:
                skipped.append(p.name)      # derived or non-literal: never overwrite
                continue
            ln, c0, c1, old = span
            new = self.state[p.name]
            if p.kind == "i":
                new = int(round(new))
            if _fmt(new, p.kind) == _fmt(old, p.kind):
                continue
            edits.append((ln, c0, c1, _fmt(new, p.kind)))
            written.append(f"{p.name} {_fmt(old, p.kind)} -> {_fmt(new, p.kind)}")

        if not edits:
            return 0

        # apply right-to-left within each line so earlier column offsets stay valid
        for ln, c0, c1, text in sorted(edits, key=lambda e: (-e[0], -e[1])):
            line = lines[ln - 1]
            lines[ln - 1] = line[:c0] + text + line[c1:]

        shutil.copyfile(path, path.with_suffix(".py.bak"))
        path.write_text("".join(lines))
        return len(edits)

    def revert(self):
        """Reload both config files from disk and re-apply everything."""
        import importlib
        importlib.reload(config); importlib.reload(scene_config)
        self.state = {p.name: getattr(_cfg(p.name), p.name) for p in ALL}
        self.push_all()
        self.status = "reverted to the config files"
        print(f"[tuning] {self.status}")

    # ---------------------------------------------------------------- readouts

    def _refresh_stats(self):
        """to_numpy() is a GPU->CPU copy (~ms at 16K blocks) — sample, don't poll."""
        n = fields.n_f[None]
        if n == 0:
            self._stats = {"n": 0}
            return
        slp = fields.sleep_f.to_numpy()[:n]
        act = int(fields.active.to_numpy()[:n].sum())
        v = np.linalg.norm(fields.vel_f.to_numpy()[:n][slp == 0], axis=1)
        self._stats = {
            "n": int(n), "awake": int((slp == 0).sum()), "active": act,
            "v_med": float(np.median(v)) if v.size else 0.0,
            "v_p90": float(np.percentile(v, 90)) if v.size else 0.0,
        }

    # ---------------------------------------------------------------- draw

    def draw(self):
        if not self.visible:
            return
        imgui.set_next_window_pos(imgui.ImVec2(12, 12), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(400, 720), imgui.Cond_.first_use_ever)
        imgui.begin("Tuning  [TAB hides]")

        self._stats_age += 1
        if self._stats_age >= 15:
            self._stats_age = 0
            self._refresh_stats()
        s = self._stats
        if s.get("n"):
            imgui.text(f"{s['n']} blocks   {s['awake']} awake   {s['active']} active(clip)")
            imgui.text(f"awake speed  med {s['v_med']:.2f}   p90 {s['v_p90']:.2f}")
        prof, pn = self.app._prof, max(self.app._prof_n, 1)
        imgui.text(f"physics {prof['physics']/pn*1000:5.2f} ms   "
                   f"total {prof['total']/pn*1000:5.2f} ms")

        imgui.separator()
        imgui.text(f"FENG_KHERTZ   {params.FENG_KHERTZ:11.4g}   (derived)")
        imgui.text(f"VMAX_CONTACT  {params.VMAX_CONTACT:11.4g}   (derived)")
        imgui.separator()

        if imgui.button("Save to config"):
            self.save()
        imgui.same_line()
        if imgui.button("Revert"):
            self.revert()
        if self.status:
            imgui.text_wrapped(self.status)
        imgui.separator()

        for title, plist in GROUPS:
            if not imgui.collapsing_header(title):
                continue
            for p in plist:
                if p.ro:
                    imgui.begin_disabled()
                if p.kind == "i":
                    changed, val = imgui.slider_int(
                        p.name, int(round(self.state[p.name])),
                        int(p.lo), int(p.hi), "%d", p.flags)
                else:
                    fmt = "%.5f" if p.hi <= 1.0 else "%.4g"
                    changed, val = imgui.slider_float(
                        p.name, float(self.state[p.name]),
                        float(p.lo), float(p.hi), fmt, p.flags)
                if p.ro:
                    imgui.end_disabled()
                    changed = False
                if changed:
                    self.state[p.name] = val
                    self._push(p.name)
                if p.note and imgui.is_item_hovered():
                    imgui.set_tooltip(p.note)
            imgui.spacing()

        imgui.end()

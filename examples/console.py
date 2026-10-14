"""console.py — what the examples print.

Import this FIRST in an example (before taichi): it turns off Taichi's import banner and
moderngl-window's start-up log. A window then shows one status line, redrawn in place
every STATUS_EVERY seconds: physics time per frame, frames per second, and the number
of blocks awake and asleep. Above it, the controls, with an [on] / [off] telltale next to
the toggles that have one.

CEAULVER_VERBOSE=1 brings the full output back instead (scene setup, the
120-frame profile blocks, key feedback, Taichi's and moderngl-window's own lines).
The headless modes (--check, --diag) always print their stats.
"""
import logging
import os
import re
import shutil
import sys
import time
import warnings

VERBOSE = os.environ.get("CEAULVER_VERBOSE", "0") == "1"
STATUS_EVERY = 2.0      # seconds between redraws of the status line

if not VERBOSE:
    os.environ.setdefault("ENABLE_TAICHI_HEADER_PRINT", "0")      # read when taichi is imported
    warnings.filterwarnings("ignore", category=SyntaxWarning)    # taichi/tools/image.py, on first import
    import moderngl_window as _mglw
    _mglw.WindowConfig.log_level = logging.WARNING


def info(*args, **kwargs):
    """print(), only under CEAULVER_VERBOSE=1."""
    if VERBOSE:
        print(*args, **kwargs)


def _block_counts():
    """(awake, asleep) blocks: one GPU readback, done only when the line is redrawn."""
    from ceaulver import fields
    n = int(fields.n_f[None])
    awake = int((fields.sleep_f.to_numpy()[:n] == 0).sum()) if n else 0
    return awake, n - awake


_COLOUR = sys.stdout.isatty() and not VERBOSE     # ANSI colours only in the live block
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def onoff(flag):
    """The telltale shown next to a toggle in the controls: [on] green, [off] red."""
    text = "[on]" if flag else "[off]"
    if not _COLOUR:
        return text
    # 256-colour palette entries 46 and 196 are exactly #00ff00 and #ff0000
    return f"\x1b[38;5;{46 if flag else 196}m{text}\x1b[0m"


class _Status:
    """The live block at the bottom of the terminal: the controls line(s), then the status line."""

    def __init__(self):
        self.t0 = None; self.frames = 0; self.physics = 0.0
        self.controls = None        # callable -> the controls text (may hold telltales, may be 2 lines)
        self.line = ""              # the status line
        self._shown = None          # controls text as last drawn
        self._rows = 0              # terminal rows the block occupies (0: not drawn yet)
        self._tty = sys.stdout.isatty()

    def show_controls(self, fn):
        """Print the controls. fn() is re-read every frame (poll), so telltales follow the keys."""
        self.controls = fn
        if VERBOSE or not self._tty:
            print(fn())
        else:
            self.redraw()

    def redraw(self):
        if VERBOSE or not self._tty:
            return
        self._shown = self.controls() if self.controls else None
        lines = (self._shown.split("\n") if self._shown else []) + [self.line]
        cols = max(1, shutil.get_terminal_size().columns)
        out = "\r"
        if self._rows > 1:
            out += f"\x1b[{self._rows - 1}A"          # up to the block's first row
        out += "\x1b[J" + "\n".join(lines)            # clear to the end of the screen, redraw
        sys.stdout.write(out); sys.stdout.flush()
        self._rows = sum(max(1, -(-len(_ANSI.sub('', l)) // cols)) for l in lines)    # long lines wrap

    def poll(self):
        """Call once per frame, paused or not: redraws at once when a telltale changed."""
        if self.controls and not VERBOSE and self._tty and self.controls() != self._shown:
            self.redraw()

    def say(self, msg):
        """Print a message above the live block."""
        if VERBOSE or not self._tty or self._rows == 0:
            print(msg); return
        out = "\r" + (f"\x1b[{self._rows - 1}A" if self._rows > 1 else "") + "\x1b[J"
        sys.stdout.write(out + msg + "\n"); self._rows = 0
        self.redraw()

    def tick(self, physics_s):
        """Call once per simulated frame with the seconds that frame spent in physics."""
        if VERBOSE:
            return
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
            return
        self.frames += 1; self.physics += physics_s
        if now - self.t0 < STATUS_EVERY:
            return
        awake, asleep = _block_counts()
        self.line = (f"physics {self.physics / self.frames * 1e3:6.2f} ms    "
                     f"{self.frames / (now - self.t0):5.1f} fps    "
                     f"{awake} awake    {asleep} asleep")
        if self._tty:
            self.redraw()
        else:
            print(self.line, flush=True)
        self.t0 = now; self.frames = 0; self.physics = 0.0

    def end(self):
        """Leave the block on screen and move to a fresh line."""
        if not VERBOSE and self._tty and self._rows:
            sys.stdout.write("\n"); sys.stdout.flush()


STATUS = _Status()

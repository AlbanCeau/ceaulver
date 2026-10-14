"""capture.py — window capture to H.264 + a wallclock-stamped CSV log.

Recorder: frames read from the default framebuffer, handed to a background thread that
pipes them to ffmpeg (rawvideo rgb24 -> libx264), so the encode does not run in the
render loop; the glReadPixels copy still does. Wallclock-synced constant frame rate:
each rendered frame is written as many times as needed for video time to track the
wall time since the first frame (0 times if the window renders faster than `fps`).
GL framebuffers are bottom-up -> ffmpeg `vflip`; odd sizes are cropped to even
(yuv420p). Adapted from pieces/003_waves/shared/capture.py (pieces do not import
each other).

Log: one CSV, long format -- a `frame` row per rendered frame, a `print` row per line
written to stdout (the Tee keeps printing to the terminal too).
"""
from console import STATUS
import csv
import queue
import shutil
import subprocess
import sys
import threading
import time


class Recorder:
    def __init__(self, out_path, w, h, fps=60, crf=18, preset="veryfast", max_queue=120):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg not found on PATH")
        self.w, self.h, self.fps = int(w), int(h), int(fps)
        cmd = [exe, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{self.w}x{self.h}", "-framerate", str(self.fps),
               "-i", "pipe:0",
               "-vf", "vflip,crop=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.q = queue.Queue(maxsize=max_queue)      # full -> the render loop waits
        self.err = None
        self.t0 = None                               # wall time of the first frame
        self.n_video = 0                             # video frames written so far
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            buf, n = item
            try:
                for _ in range(n):
                    self.proc.stdin.write(buf)
            except BrokenPipeError as e:
                self.err = e
                break

    def add(self, fbo, now):
        """Read `fbo` (w x h, rgb) and queue it for as many video frames as wall time
        `now` calls for. Returns (repeats, read_ms, queue_wait_ms)."""
        if self.err is not None or self.proc.poll() is not None:
            raise RuntimeError(f"ffmpeg stopped (code {self.proc.poll()}, {self.err})")
        if self.t0 is None:
            self.t0 = now
        target = int((now - self.t0) * self.fps) + 1    # video frames that should exist
        n = target - self.n_video
        if n <= 0:
            return 0, 0.0, 0.0
        t0 = time.perf_counter()
        buf = fbo.read(viewport=(0, 0, self.w, self.h), components=3, alignment=1)
        t1 = time.perf_counter()
        self.q.put((buf, n))
        t2 = time.perf_counter()
        self.n_video = target
        return n, (t1 - t0) * 1e3, (t2 - t1) * 1e3

    def video_time(self):
        return self.n_video / self.fps

    def close(self):
        self.q.put(None)
        self.thread.join()
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed (code {code})")


class Log:
    """The capture CSV. `state` is updated by the window each frame, so print rows carry
    the frame / sim time they happened at."""
    FIELDS = ("wall_unix", "t_wall", "kind", "frame", "t_video", "sim_t", "frame_ms",
              "fps", "repeats", "read_ms", "queue_ms", "blocks", "text")

    def __init__(self, path, t_launch):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.w.writeheader()
        self.t_launch = t_launch
        self.state = dict(frame=-1, t_video="", sim_t="")
        self._n = 0

    def row(self, kind, **kw):
        now = time.time()
        r = dict(wall_unix=f"{now:.6f}", t_wall=f"{now - self.t_launch:.6f}", kind=kind)
        r.update(self.state)
        r.update(kw)
        self.w.writerow(r)
        self._n += 1
        if self._n % 600 == 0:
            self.f.flush()

    def close(self):
        self.f.flush(); self.f.close()


class Tee:
    """stdout -> terminal AND a `print` row per complete line."""

    def __init__(self, log, stream):
        self.log, self.stream, self.buf = log, stream, ""

    def write(self, s):
        self.stream.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.log.row("print", text=line)
        return len(s)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class WindowCapture:
    """One --capture session for a moderngl-window scene: runs/<stamp>_<name>_capture/
    with <name>.mp4 + <name>.csv, stdout tee'd into the CSV from construction on.
    Call frame() once per on_render after drawing; finish() is idempotent (call it from
    the window's exit path too, so ESC / a crash still leaves a valid mp4 + csv)."""

    def __init__(self, name, piece_dir, t_launch, fps=60):
        from pathlib import Path
        self.name, self.fps = name, fps
        self.dir = Path(piece_dir) / "runs" / (
            time.strftime("%Y-%m-%d_%H%M%S", time.localtime(t_launch)) + f"_{name}_capture")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.dir / f"{name}.csv", t_launch)
        self.stdout = sys.stdout
        sys.stdout = Tee(self.log, sys.stdout)
        self.rec = None
        self.done = False
        self.n = 0
        self._t_prev = None
        STATUS.say(f"[capture] -> {self.dir}")

    def video_time(self):
        return self.rec.video_time() if self.rec is not None else 0.0

    def frame(self, wnd, sim_t, blocks, t_frame):
        """Record the frame just drawn. t_frame: perf_counter at the frame's start."""
        if self.done:
            return
        if self.rec is None:
            w, h = wnd.buffer_size
            self.rec = Recorder(str(self.dir / f"{self.name}.mp4"), w, h, fps=self.fps)
            STATUS.say(f"[capture] recording {w}x{h} @ {self.fps} fps")
        self.log.state = dict(frame=self.n, t_video=f"{self.rec.video_time():.4f}",
                              sim_t=f"{sim_t:.4f}")
        reps, read_ms, q_ms = self.rec.add(wnd.fbo, time.time())
        dt_ms = (t_frame - self._t_prev) * 1e3 if self._t_prev is not None else None
        self._t_prev = t_frame
        self.log.row("frame", frame_ms=f"{dt_ms:.3f}" if dt_ms is not None else "",
                     fps=f"{1e3 / dt_ms:.2f}" if dt_ms else "",
                     repeats=reps, read_ms=f"{read_ms:.3f}", queue_ms=f"{q_ms:.3f}",
                     blocks=blocks)
        self.n += 1

    def finish(self, reason):
        if self.done:
            return
        self.done = True
        STATUS.say(f"[capture] stop: {reason}")
        if self.rec is not None:
            STATUS.say(f"[capture] video {self.rec.video_time():.1f} s, closing ffmpeg ...")
            self.rec.close()
        sys.stdout = self.stdout
        self.log.close()
        STATUS.say(f"[capture] wrote {self.dir}")

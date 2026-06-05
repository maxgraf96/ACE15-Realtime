"""Ableton Link integration — the live grid ground truth (tempo/beat/phase), shared with Ableton.

aalink runs in a SEPARATE process (`link_proc.py`), NOT here: aalink's threads + a torch-MPS
workload in the same process hang/crash (reproducible headless; it took down the live app). This
module spawns that process, reads its ~60 Hz state stream, and EXTRAPOLATES the beat against the
shared monotonic clock between samples (beat advances linearly at the tempo, so this is
sample-accurate). The MPS sidecar process therefore contains NO aalink -> stable, and the same
readonly API (beat/phase/tempo/peers/connected) the engine already uses is preserved.

Graceful no-op if aalink/link_proc isn't available: `available` stays False and the app runs its
onset-anchored fallback."""
import json
import os
import subprocess
import sys
import threading
import time


class LinkSync:
    def __init__(self, bpm=120.0, quantum=4.0):
        self._quantum = float(quantum)
        self._snap = None              # (t_mono, beat, tempo, phase, peers, playing) — newest sample
        self._lock = threading.Lock()
        self._proc = None
        self._thread = threading.Thread(target=self._run, name="ableton-link", daemon=True)
        self._thread.start()
        for _ in range(50):            # block briefly so callers can check .available right away
            if self._snap is not None:
                break
            time.sleep(0.1)

    def _run(self):
        proc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "link_proc.py")
        try:
            self._proc = subprocess.Popen(
                [sys.executable, proc_path, str(self._quantum)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1, env=dict(os.environ))
        except Exception:
            return
        try:
            for line in self._proc.stdout:      # one JSON sample per line (~60 Hz)
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                    s = (m["t"], m["beat"], m["tempo"], m["phase"], m["peers"], m["playing"])
                except Exception:
                    continue
                with self._lock:
                    self._snap = s
        except Exception:
            pass

    def _cur(self):
        """Latest Link state with the beat/phase extrapolated to NOW (beat += elapsed*tempo/60)."""
        with self._lock:
            s = self._snap
        if s is None:
            return None
        t0, beat0, tempo, phase0, peers, playing = s
        dbeats = (time.monotonic() - t0) * (tempo / 60.0)
        beat = beat0 + dbeats
        phase = (phase0 + dbeats) % self._quantum if self._quantum > 0 else phase0
        return beat, tempo, phase, peers, playing

    # ---- status (any thread) ----
    @property
    def available(self):
        """The Link reader process is up and has produced at least one sample."""
        return self._snap is not None

    @property
    def connected(self):
        """At least one OTHER Link peer is present (e.g. Ableton with Link enabled)."""
        c = self._cur()
        return bool(c and c[3] > 0)

    @property
    def peers(self):
        c = self._cur()
        return int(c[3]) if c else 0

    # ---- live grid (thread-safe; the producer polls these) ----
    @property
    def beat(self):
        c = self._cur()
        return float(c[0]) if c else None

    @property
    def phase(self):
        c = self._cur()
        return float(c[2]) if c else None

    @property
    def tempo(self):
        c = self._cur()
        return float(c[1]) if c else None

    @property
    def playing(self):
        c = self._cur()
        return bool(c[4]) if c else False

    def snapshot(self):
        c = self._cur()
        if not c:
            return None
        return {"peers": int(c[3]), "tempo": float(c[1]), "beat": float(c[0]),
                "phase": float(c[2]), "quantum": self._quantum, "playing": bool(c[4])}

    def set_playing(self, playing):
        """Drive the shared Link transport: play -> Ableton starts, stop -> Ableton stops (if its Link
        Start/Stop Sync is on). Sent to link_proc over stdin; safe no-op if the reader isn't up yet."""
        p = self._proc
        if p is None or p.stdin is None:
            return
        try:
            p.stdin.write(("play" if playing else "stop") + "\n")
            p.stdin.flush()
        except Exception:
            pass

    def set_quantum(self, quantum):
        self._quantum = float(quantum)   # used by phase extrapolation (the subprocess set 4/4 at spawn)

    def close(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass


# ---- process-wide singleton (one Link reader per sidecar) ----
_LINK = None
_LINK_LOCK = threading.Lock()


def get_link(bpm=120.0, quantum=4.0):
    """Lazily create the single shared LinkSync. Returns it even if Link is unavailable
    (then `.available` is False)."""
    global _LINK
    with _LINK_LOCK:
        if _LINK is None:
            _LINK = LinkSync(bpm=bpm, quantum=quantum)
        return _LINK

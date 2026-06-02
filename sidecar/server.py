"""Streaming cover sidecar — local TCP server wrapping RealtimeCover (Phase 2b).

This is the engine process the Standalone app bundles and spawns internally.
The JUCE app connects over a local socket and speaks one framed protocol:

  frame = [4-byte big-endian length][1-byte type][payload]
    type 0x01 CONTROL (client->server, JSON utf8)
    type 0x02 AUDIO   (server->client, float32 interleaved stereo PCM @ 48k)
    type 0x03 EVENT   (server->client, JSON utf8: ready / stats / error)

CONTROL commands (JSON):
  {"cmd":"load","path":"<wav>","seconds":N}      # or {"cmd":"load","pcm_b64":...,"sr":...}
  {"cmd":"style","tags":"...","denoise":0.8,"timbre":"none"}
  {"cmd":"prompt","tags":"..."}     {"cmd":"denoise","value":0.7}
  {"cmd":"play"}                    {"cmd":"stop"}

The audio sender paces at 1x; the client buffers into its own jitter ring and
drains from the audio callback. Single client (the plugin).
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import mps_compat  # noqa: E402
mps_compat.force_bf16_on_mps()  # bf16 for the XL "Quality" model (no-op for 2B)
from engine.realtime import RealtimeCover, SR  # noqa: E402

MODELS = {"fast": "acestep-v15-turbo", "quality": "acestep-v15-xl-turbo"}

T_CONTROL, T_AUDIO, T_EVENT = 0x01, 0x02, 0x03
HDR = struct.Struct(">IB")


def _send(sock, type_, payload: bytes):
    sock.sendall(HDR.pack(len(payload), type_) + payload)


def _send_json(sock, type_, obj):
    _send(sock, type_, json.dumps(obj).encode("utf-8"))


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _recv_frame(sock):
    hdr = _recv_exact(sock, HDR.size)
    if hdr is None:
        return None
    length, type_ = HDR.unpack(hdr)
    payload = _recv_exact(sock, length) if length else b""
    return type_, payload


class Connection:
    def __init__(self, sock, addr, block=2048):
        self.sock = sock
        self.addr = addr
        self.block = block
        self.rc: RealtimeCover | None = None
        self.playing = False
        self.alive = True
        self._loaded = False

    def handle(self):
        threading.Thread(target=self._sender, daemon=True).start()
        try:
            while self.alive:
                frame = _recv_frame(self.sock)
                if frame is None:
                    break
                type_, payload = frame
                if type_ == T_CONTROL:
                    self._on_control(json.loads(payload.decode("utf-8")))
        finally:
            self.close()

    def _send_event(self, obj):
        try:
            _send_json(self.sock, T_EVENT, obj)
        except OSError:
            pass

    def _stats_loop(self):
        """Push engine telemetry (buffer / regens / playhead) ~5 Hz for the meter."""
        while self.alive and self.playing and self.rc is not None:
            ev = self.rc.stats(); ev["event"] = "stats"
            self._send_event(ev)
            time.sleep(0.1)

    def _download_with_progress(self, repo, local_dir, allow_patterns, target_bytes, label):
        """snapshot_download with a dir-size progress monitor -> download_progress EVENTs."""
        from huggingface_hub import snapshot_download
        from pathlib import Path
        local_dir = Path(local_dir); local_dir.mkdir(parents=True, exist_ok=True)
        self._send_event({"event": "download_progress", "pct": 0, "label": label})
        done = threading.Event()

        def monitor():
            while not done.is_set():
                sz = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
                self._send_event({"event": "download_progress", "pct": min(99.0, 100.0 * sz / target_bytes), "label": label})
                time.sleep(0.5)
        mon = threading.Thread(target=monitor, daemon=True); mon.start()
        try:
            snapshot_download(repo, local_dir=str(local_dir), allow_patterns=allow_patterns, max_workers=8)
        finally:
            done.set(); mon.join(timeout=1)
        self._send_event({"event": "download_progress", "pct": 100, "label": label})

    def _ensure_models(self, model="fast"):
        """Ensure the cover-path models for the chosen model exist; download w/ progress."""
        from acestep.paths import checkpoints_dir
        ck = checkpoints_dir()
        base = ["acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B"]   # vae+qwen shared; turbo = Fast model
        if not (ck.exists() and all((ck / c).is_dir() and any((ck / c).iterdir()) for c in base)):
            self._download_with_progress("ACE-Step/Ace-Step1.5", ck,
                                         ["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*"],
                                         6.05e9, "models (Fast)")
            (ck / "acestep-5Hz-lm-1.7B").mkdir(exist_ok=True)  # satisfy main-model check w/o the LM
        if model in ("quality", "xl"):
            xl = ck / "acestep-v15-xl-turbo"
            if not (xl.is_dir() and any(xl.iterdir())):
                self._download_with_progress("ACE-Step/acestep-v15-xl-turbo", xl, None, 20.0e9, "Quality model (XL, 20 GB)")

    def _on_control(self, msg):
        cmd = msg.get("cmd")
        try:
            if cmd == "load":
                model = msg.get("model", "fast")
                cfg = MODELS.get(model, MODELS["fast"])
                self._ensure_models(model)
                if self.rc is not None:           # free the previous model before loading another
                    try: self.rc.close()
                    except Exception: pass
                    self.rc = None
                self.playing = False
                self.rc = RealtimeCover(device="mps", steps=msg.get("steps", 8),
                                        window_s=msg.get("window", 20.0),
                                        lookahead_s=msg.get("lookahead", 2.0 if model in ("quality", "xl") else 1.0),
                                        config_path=cfg)
                if msg.get("path"):
                    self.rc.load_track(msg["path"], seconds=msg.get("seconds"))
                elif msg.get("file_b64"):           # drag-drop: raw encoded audio bytes
                    import base64, tempfile
                    raw = base64.b64decode(msg["file_b64"])
                    p = tempfile.mktemp(suffix=msg.get("ext", ".wav"))
                    with open(p, "wb") as f:
                        f.write(raw)
                    self.rc.load_track(p, seconds=msg.get("seconds"))
                else:
                    import base64, numpy as np
                    pcm = np.frombuffer(base64.b64decode(msg["pcm_b64"]), dtype=np.float32)
                    self._load_pcm(pcm.reshape(-1, msg.get("channels", 2)), msg.get("sr", SR), msg.get("seconds"))
                self._loaded = True
                self._send_event({"event": "loaded", "duration": self.rc.full_T / 25.0,
                                  "peaks": self.rc.peaks(), "model": model})
            elif cmd == "style":
                self.rc.set_style(msg["tags"], denoise=msg.get("denoise", 0.8),
                                  character=msg.get("character", 0.0),
                                  send_bpm=msg.get("send_bpm", True), send_key=msg.get("send_key", True))
                self._send_event({"event": "styled", "bpm": self.rc.jit.bpm, "key": self.rc.jit.key})
            elif cmd == "play":
                if not self.playing and self.rc is not None:
                    self.rc.start(); self.playing = True
                    threading.Thread(target=self._stats_loop, daemon=True).start()
                    self._send_event({"event": "playing"})
            elif cmd == "prompt":
                self.rc.set_prompt(msg["tags"])
            elif cmd == "denoise":
                self.rc.set_denoise(msg["value"])
            elif cmd == "character":
                self.rc.set_character(msg["value"])
            elif cmd == "metas":
                self.rc.set_metas(send_bpm=msg.get("send_bpm"), send_key=msg.get("send_key"))
            elif cmd == "evolve":
                self.rc.set_evolve(bool(msg["value"]))
            elif cmd == "reconfigure":
                if self.rc and self.playing:
                    self.rc.reconfigure(steps=msg.get("steps"), window_s=msg.get("window"))
            elif cmd == "stop":
                self.playing = False
                if self.rc:
                    self.rc.stop()
        except Exception as e:
            _send_json(self.sock, T_EVENT, {"event": "error", "cmd": cmd, "msg": str(e)})

    def _load_pcm(self, pcm_nx2, sr, seconds):
        # Upload path (plugin drag-drop). Write a temp wav and load via the engine.
        import soundfile as sf, tempfile
        p = tempfile.mktemp(suffix=".wav")
        sf.write(p, pcm_nx2, sr)
        self.rc.load_track(p, seconds=seconds)

    def _sender(self):
        period = self.block / SR
        next_t = time.perf_counter()
        while self.alive:
            if not self.playing or self.rc is None:
                time.sleep(0.01); next_t = time.perf_counter(); continue
            pcm = self.rc.read(self.block)             # [n,2] float32
            try:
                _send(self.sock, T_AUDIO, pcm.tobytes())
            except OSError:
                break
            next_t += period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            elif dt < -0.5:
                next_t = time.perf_counter()           # fell behind; resync
        # done

    def close(self):
        self.alive = False
        try:
            if self.rc:
                self.rc.close()
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[sidecar] listening on {args.host}:{args.port}", flush=True)
    while True:
        sock, addr = srv.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[sidecar] client {addr}", flush=True)
        Connection(sock, addr).handle()
        print(f"[sidecar] client {addr} disconnected", flush=True)


if __name__ == "__main__":
    main()

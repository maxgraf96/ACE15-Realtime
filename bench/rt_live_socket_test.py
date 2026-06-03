"""Phase B integration: the full sidecar live path over the socket.
Spawns the sidecar, sends live_start (2B), streams a file as 0x04 AUDIO-IN @48k at 1x,
and verifies live_started + that non-silent cover frames (0x02) stream back. Validates
live_start -> live engine -> 0x04 feed -> rolling cover -> 0x02 output end-to-end.
"""
import os, sys, socket, struct, subprocess, threading, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import loader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8801
HDR = struct.Struct(">IB")
SR = 48000
SRC = os.path.join(ROOT, "assets", "source.wav")

proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "sidecar", "server.py"), "--port", str(PORT)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
ready = threading.Event()


def pump():
    for line in proc.stdout:
        if "listening" in line.lower(): ready.set()
        if any(k in line for k in ("[sidecar]", "Error", "Traceback", "error")): print("  " + line.rstrip())


threading.Thread(target=pump, daemon=True).start()
if not ready.wait(120):
    print("FAIL: sidecar didn't start"); proc.kill(); sys.exit(1)

sock = socket.create_connection(("127.0.0.1", PORT), timeout=10)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
state = {"cover_frames": 0, "cover_peak": 0.0, "live_started": False, "alive": True}


def reader():
    buf = b""
    def rd(n):
        nonlocal buf
        while len(buf) < n:
            try: c = sock.recv(65536)
            except OSError: return None
            if not c: return None
            buf += c
        out, buf = buf[:n], buf[n:]
        return out
    while state["alive"]:
        h = rd(5)
        if h is None: break
        length, type_ = HDR.unpack(h)
        p = rd(length) if length else b""
        if p is None: break
        if type_ == 0x02:
            a = np.frombuffer(p, dtype=np.float32).reshape(-1, 4)
            state["cover_frames"] += a.shape[0]
            if a.size: state["cover_peak"] = max(state["cover_peak"], float(np.abs(a[:, :2]).max()))
        elif type_ == 0x03:
            ev = json.loads(p.decode())
            if ev.get("event") == "live_started": state["live_started"] = True
            if ev.get("event") in ("loading", "live_started", "error"): print(f"  event: {ev}")


threading.Thread(target=reader, daemon=True).start()


def send(type_, payload): sock.sendall(HDR.pack(len(payload), type_) + payload)


try:
    send(0x01, json.dumps({"cmd": "live_start", "model": "fast", "tags": "warm lo-fi hip hop, dusty drums",
                           "denoise": 0.8, "bpm": 120, "key": "C minor", "sr": SR,
                           "window": 8.0, "pin": 3.0,
                           "stems": (os.environ.get("ACE15_LIVE_STEMS", "").split(",") if os.environ.get("ACE15_LIVE_STEMS") else None)}).encode())
    t0 = time.time()
    while not state["live_started"] and time.time() - t0 < 180:
        time.sleep(0.2)
    assert state["live_started"], "no live_started (model load failed?)"
    print(f"  live_started in {time.time()-t0:.1f}s; streaming input…")

    src = loader.load_audio(SRC, duration=18).waveform.float().numpy().T  # [N,2] @48k
    blk = 2048; period = blk / SR; nxt = time.perf_counter()
    for i in range(0, src.shape[0], blk):
        send(0x04, src[i:i + blk].astype(np.float32).tobytes())
        nxt += period; dt = nxt - time.perf_counter()
        if dt > 0: time.sleep(dt)
    time.sleep(2.0)   # let the tail stream out
    send(0x01, json.dumps({"cmd": "live_stop"}).encode())
    time.sleep(0.5)
    state["alive"] = False; sock.close()

    cov_secs = state["cover_frames"] / SR
    print(f"[result] live_started={state['live_started']}  cover_out={cov_secs:.1f}s  cover_peak={state['cover_peak']:.3f}")
    ok = state["live_started"] and cov_secs > 2.0 and state["cover_peak"] > 0.02
    print("PASS: sidecar live path streams a non-silent cover from live input" if ok else "FAIL")
    sys.exit(0 if ok else 1)
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()

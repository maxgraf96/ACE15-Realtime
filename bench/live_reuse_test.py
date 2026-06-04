"""Issue 1: Stop in live mode must NOT unload the model -> the 2nd Play reuses the loaded
engine (no 'loading model' event, fast restart) and still generates + locks the loop.

Cycle: live_start -> feed loop -> live_stop -> live_start (REUSE) -> feed loop -> check.
PASS = the model 'loading' event fires ONCE (first start only), the 2nd start is fast, and
the 2nd run produces non-silent audio + re-locks."""
import os, sys, socket, struct, subprocess, threading, time, json
ROOT = "/Users/max/Code/ACE15-Realtime"; sys.path.insert(0, ROOT)
import numpy as np
from engine import loader
PORT = 8819; HDR = struct.Struct(">IB"); SR = 48000
PIANO = "/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "sidecar", "server.py"), "--port", str(PORT)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
ready = threading.Event()
def pump():
    for line in proc.stdout:
        if "listening" in line.lower(): ready.set()
        if any(k in line for k in ("Error", "Traceback")): print("  " + line.rstrip())
threading.Thread(target=pump, daemon=True).start()
if not ready.wait(120): print("FAIL: no start"); proc.kill(); sys.exit(1)
sock = socket.create_connection(("127.0.0.1", PORT), timeout=10); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
st = {"cover": 0, "peak": 0.0, "live": 0, "locked": 0, "model_loads": 0, "alive": True}
def reader():
    buf = b""
    def rd(n):
        nonlocal buf
        while len(buf) < n:
            try: c = sock.recv(65536)
            except OSError: return None
            if not c: return None
            buf += c
        out, buf = buf[:n], buf[n:]; return out
    while st["alive"]:
        h = rd(5)
        if h is None: break
        length, t = HDR.unpack(h); p = rd(length) if length else b""
        if p is None: break
        if t == 0x02:
            a = np.frombuffer(p, dtype=np.float32).reshape(-1, 4); st["cover"] += a.shape[0]
            if a.size: st["peak"] = max(st["peak"], float(np.abs(a[:, :2]).max()))
        elif t == 0x03:
            ev = json.loads(p.decode())
            if ev.get("event") == "loading" and ev.get("stage") == "model": st["model_loads"] += 1
            if ev.get("event") == "live_started": st["live"] += 1
            if ev.get("event") == "stats" and ev.get("loop_locked"): st["locked"] += 1
threading.Thread(target=reader, daemon=True).start()
def send(t, p): sock.sendall(HDR.pack(len(p), t) + p)
full = loader.load_audio(PIANO).waveform.float().numpy().T
bar = int(60.0 / 90 * 4 * SR); loop4 = full[:4 * bar]; N = loop4.shape[0]

def start_and_feed(secs, label):
    send(0x01, json.dumps({"cmd": "live_start", "model": "fast", "tags": "jazz trio brushed drums upright bass",
        "denoise": 0.7, "bpm": 90, "key": "C minor", "loop_bars": 4, "sr": SR, "window": 8.0, "pin": 3.0}).encode())
    t0 = time.time(); n0 = st["live"]
    while st["live"] == n0 and time.time() - t0 < 180: time.sleep(0.1)
    dt = time.time() - t0
    print(f"  [{label}] live_started in {dt:.1f}s (model_loads so far={st['model_loads']})", flush=True)
    nxt = time.perf_counter(); fed = 0; period = 2048 / SR
    while time.time() - t0 < secs:
        s = fed % N; ch = loop4[s:s + 2048]
        if ch.shape[0] < 2048: ch = np.concatenate([ch, loop4[:2048 - ch.shape[0]]], 0)
        send(0x04, ch.astype(np.float32).tobytes()); fed += 2048
        nxt += period; d = nxt - time.perf_counter()
        if d > 0: time.sleep(d)
    return dt

try:
    d1 = start_and_feed(40, "START 1")
    cov1, lk1 = st["cover"], st["locked"]
    send(0x01, json.dumps({"cmd": "live_stop"}).encode()); time.sleep(1.0)
    print(f"  STOP sent; model_loads={st['model_loads']} (should be 1)", flush=True)
    st["peak"] = 0.0  # reset peak to measure the 2nd run's audio fresh
    d2 = start_and_feed(40, "START 2 (reuse)")
    time.sleep(0.5); st["alive"] = False; sock.close()
    cov2, lk2 = st["cover"] - cov1, st["locked"] - lk1
    print(f"[result] model_loads={st['model_loads']} start1={d1:.1f}s start2={d2:.1f}s "
          f"reuse_peak={st['peak']:.3f} relock_stats={lk2}", flush=True)
    ok = (st["model_loads"] == 1 and st["peak"] > 0.02 and lk2 > 0 and d2 < d1)
    print("PASS: model stayed loaded; reuse generated + re-locked, faster restart" if ok else "FAIL", flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()

"""Minimal: does the SIDECAR reach live_started (model load + out-of-process link) or hang/crash?
Spawns the sidecar, connects, sends live_start, waits for live_started. Reports timing + the
sidecar's exit code if it dies. No audio feed (isolates the live_start path)."""
import os, sys, socket, struct, subprocess, threading, time, json
ROOT = "/Users/max/Code/ACE15-Realtime"; sys.path.insert(0, ROOT)
import numpy as np
PORT = 8841; HDR = struct.Struct(">IB"); SR = 48000
sclog = open("/tmp/sidecar_out.log", "w")
proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "sidecar", "server.py"), "--port", str(PORT)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
ready = threading.Event()
def pump():
    for line in proc.stdout:
        sclog.write(line); sclog.flush()
        if "listening" in line.lower(): ready.set()
threading.Thread(target=pump, daemon=True).start()
if not ready.wait(60): print("FAIL: sidecar never listened"); proc.kill(); sys.exit(1)
print("sidecar listening", flush=True)
sock = socket.create_connection(("127.0.0.1", PORT), timeout=60)   # tolerate a long first-lock render stall
st = {"link": False, "live": False, "alive": True, "cover": 0, "peak": 0.0, "locked": False, "link_active": False}
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
            if ev.get("event") == "link" and not st["link"]:
                print(f"  link event: connected={ev.get('connected')} peers={ev.get('peers')} tempo={ev.get('tempo')}", flush=True); st["link"] = True
            if ev.get("event") == "live_started": st["live"] = True
            if ev.get("event") == "stats":
                if ev.get("loop_locked"): st["locked"] = True
                if ev.get("link_active"): st["link_active"] = True
            if ev.get("event") == "error": print(f"  ENGINE ERROR: {ev}", flush=True)
threading.Thread(target=reader, daemon=True).start()
def send(t, p): sock.sendall(HDR.pack(len(p), t) + p)
try:
    time.sleep(6)  # let link_proc discover peers
    t0 = time.time()
    print(f"sending live_start (t=0)…", flush=True)
    send(0x01, json.dumps({"cmd": "live_start", "model": "fast", "tags": "jazz trio",
        "denoise": 0.7, "bpm": "90", "key": "C minor", "loop_bars": 4, "sr": SR, "window": 20.0, "pin": 4.0}).encode())
    while not st["live"] and time.time() - t0 < 90:
        if proc.poll() is not None:
            print(f"  SIDECAR DIED during live_start, exit code = {proc.poll()}", flush=True); break
        time.sleep(0.5)
    dt = time.time() - t0
    if not st["live"]:
        if proc.poll() is not None: print(f"FAIL: sidecar crashed (exit {proc.poll()})", flush=True)
        else: print(f"FAIL: live_started never arrived in {dt:.0f}s (sidecar HUNG)", flush=True)
        st["alive"] = False; raise SystemExit
    print(f"  live_started in {dt:.1f}s; feeding a loop for 45s…", flush=True)
    # feed a cyclic loop (interpolate the piano to the 140-bpm Link period; linear, no AF.resample hang)
    from engine import loader
    import torch
    full = loader.load_audio("/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav").waveform.float()
    N = int(round(4 * 60.0 / 90.0 * 4 * SR))   # 512000 — raw piano loop, no interpolation (match standalone)
    base = full[:, :N].t().contiguous().numpy().astype(np.float32)
    blk = 2048; period = blk / SR; nxt = time.perf_counter(); fed = 0; t1 = time.time()
    while time.time() - t1 < 45:
        if proc.poll() is not None: print(f"  SIDECAR DIED while feeding, exit {proc.poll()}", flush=True); break
        s = fed % N; ch = base[s:s + blk]
        if ch.shape[0] < blk: ch = np.concatenate([ch, base[:blk - ch.shape[0]]], 0)
        send(0x04, ch.tobytes()); fed += blk
        nxt += period; d = nxt - time.perf_counter()
        if d > 0: time.sleep(d)
    st["alive"] = False
    crashed = proc.poll() is not None
    print(f"[result] locked={st['locked']} link_active={st['link_active']} cover={st['cover']/SR:.1f}s "
          f"peak={st['peak']:.3f} crashed={crashed}", flush=True)
    ok = (st["locked"] and st["peak"] > 0.02 and not crashed)   # default (onset) path: link_active optional
    print("PASS: full sidecar path — locked, non-silent, stable 45s" if ok else "FAIL", flush=True)
finally:
    try: proc.terminate(); proc.wait(timeout=5)
    except Exception: proc.kill()
    sclog.close()

"""Reproduce the runaway memory leak. Feed NON-looping audio (white noise) so the engine never
locks -> after loop_listen_s (28s) it gives up to CONTINUOUS mode (model runs every tick). Monitor
the sidecar process RSS over time. If it climbs unboundedly -> continuous-mode MPS leak confirmed."""
import os, sys, socket, struct, subprocess, threading, time, json
ROOT = "/Users/max/Code/ACE15-Realtime"; sys.path.insert(0, ROOT)
import numpy as np
PORT = 8851; HDR = struct.Struct(">IB"); SR = 48000

def rss_mb(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out) / 1024 if out else 0
    except Exception:
        return -1

proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "sidecar", "server.py"), "--port", str(PORT)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
ready = threading.Event()
def pump():
    for line in proc.stdout:
        if "listening" in line.lower(): ready.set()
        if any(k in line for k in ("Error", "Traceback", "memdbg")): print("  " + line.rstrip())
threading.Thread(target=pump, daemon=True).start()
if not ready.wait(60): print("FAIL: no start"); proc.kill(); sys.exit(1)
sock = socket.create_connection(("127.0.0.1", PORT), timeout=60)
live = {"v": False}
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
    while True:
        h = rd(5)
        if h is None: break
        length, t = HDR.unpack(h); p = rd(length) if length else b""
        if p is None: break
        if t == 0x03 and json.loads(p.decode()).get("event") == "live_started": live["v"] = True
threading.Thread(target=reader, daemon=True).start()
def send(t, p): sock.sendall(HDR.pack(len(p), t) + p)
print(f"sidecar pid={proc.pid} baseline RSS={rss_mb(proc.pid):.0f}MB", flush=True)
# NO loop_bars (auto) -> won't lock on noise -> give_up -> continuous
send(0x01, json.dumps({"cmd": "live_start", "model": "fast", "tags": "ambient pad",
    "denoise": 0.7, "bpm": "120", "key": "C minor", "sr": SR, "window": 8.0}).encode())
t0 = time.time()
while not live["v"] and time.time() - t0 < 120: time.sleep(0.2)
print(f"live_started; RSS={rss_mb(proc.pid):.0f}MB; feeding noise 75s, give_up at ~28s -> continuous", flush=True)
rng = np.random.default_rng(0)
blk = 2048; period = blk / SR; nxt = time.perf_counter(); t1 = time.time(); last = 0
while time.time() - t1 < 75:
    if proc.poll() is not None: print(f"  SIDECAR DIED, exit {proc.poll()}", flush=True); break
    ch = (rng.standard_normal((blk, 2)).astype(np.float32) * 0.2)   # loud noise (onset sets, won't lock)
    try: send(0x04, ch.tobytes())
    except Exception as e: print(f"  send err {e}", flush=True); break
    el = time.time() - t1
    if el - last >= 5:
        last = el; print(f"  t={el:4.0f}s  RSS={rss_mb(proc.pid):7.0f}MB", flush=True)
    nxt += period; d = nxt - time.perf_counter()
    if d > 0: time.sleep(d)
print(f"[result] final RSS={rss_mb(proc.pid):.0f}MB (baseline ~ model size; leak if it climbed GB/s)", flush=True)
try: proc.terminate(); proc.wait(timeout=5)
except Exception: proc.kill()

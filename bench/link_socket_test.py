"""Full sidecar path with Ableton Link: verify the sidecar emits 'link' status events, the engine
takes the Link grid (stats link_active=True), locks, and streams non-silent audio. A conductor Link
peer (in this process) stands in for Ableton so the sidecar (separate process) discovers a peer."""
import os, sys, socket, struct, subprocess, threading, time, json, asyncio
ROOT = "/Users/max/Code/ACE15-Realtime"; sys.path.insert(0, ROOT)
import numpy as np
from engine import loader
PORT = 8831; HDR = struct.Struct(">IB"); SR = 48000
PIANO = "/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"

# conductor peer (fake Ableton) so the sidecar process sees a Link peer
def conductor():
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    async def s():
        from aalink import Link
        lk = Link(96.0); lk.enabled = True; lk.quantum = 4.0; conductor.lk = lk
    loop.run_until_complete(s()); loop.run_forever()
threading.Thread(target=conductor, daemon=True).start(); time.sleep(2.0)

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
st = {"cover": 0, "peak": 0.0, "live": False, "locked": False, "bars": 0,
      "link_seen": False, "link_connected": False, "link_tempo": 0, "link_active": False, "alive": True}
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
            e = ev.get("event")
            if e == "link":
                st["link_seen"] = True
                st["link_connected"] = ev.get("connected", False); st["link_tempo"] = ev.get("tempo", 0)
            if e == "live_started": st["live"] = True
            if e == "stats":
                if ev.get("loop_locked"): st["locked"] = True; st["bars"] = ev.get("loop_bars")
                if ev.get("link_active"): st["link_active"] = True
threading.Thread(target=reader, daemon=True).start()
def send(t, p): sock.sendall(HDR.pack(len(p), t) + p)
try:
    t0 = time.time()
    while not st["link_seen"] and time.time() - t0 < 10: time.sleep(0.1)
    print(f"  link event: seen={st['link_seen']} connected={st['link_connected']} tempo={st['link_tempo']}", flush=True)
    send(0x01, json.dumps({"cmd": "live_start", "model": "fast", "tags": "jazz trio brushed drums upright bass",
        "denoise": 0.7, "key": "C minor", "loop_bars": 4, "sr": SR, "window": 8.0, "pin": 3.0}).encode())
    t0 = time.time()
    while not st["live"] and time.time() - t0 < 180: time.sleep(0.2)
    full = loader.load_audio(PIANO).waveform.float().numpy().T
    bar = int(60.0 / 96 * 4 * SR); loop4 = full[:4 * bar]; Nn = loop4.shape[0]
    blk = 2048; period = blk / SR; nxt = time.perf_counter(); fed = 0
    while time.time() - t0 < 50:
        s = fed % Nn; ch = loop4[s:s + blk]
        if ch.shape[0] < blk: ch = np.concatenate([ch, loop4[:blk - ch.shape[0]]], 0)
        send(0x04, ch.astype(np.float32).tobytes()); fed += blk
        nxt += period; dt = nxt - time.perf_counter()
        if dt > 0: time.sleep(dt)
    send(0x01, json.dumps({"cmd": "live_stop"}).encode()); time.sleep(0.5); st["alive"] = False; sock.close()
    print(f"[result] link_seen={st['link_seen']} link_connected={st['link_connected']} link_active={st['link_active']} "
          f"locked={st['locked']} bars={st['bars']} cover={st['cover']/SR:.1f}s peak={st['peak']:.3f}", flush=True)
    ok = (st["link_seen"] and st["link_connected"] and st["link_active"] and st["locked"] and st["peak"] > 0.02)
    print("PASS: sidecar Link path — connected, link_active, locked, non-silent" if ok else "FAIL", flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()

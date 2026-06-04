import os, sys, socket, struct, subprocess, threading, time, json
ROOT="/Users/max/Code/ACE15-Realtime"; sys.path.insert(0, ROOT)
import numpy as np
from engine import loader
PORT=8807; HDR=struct.Struct(">IB"); SR=48000
PIANO="/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
proc=subprocess.Popen([sys.executable, os.path.join(ROOT,"sidecar","server.py"),"--port",str(PORT)],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
ready=threading.Event()
def pump():
    for line in proc.stdout:
        if "listening" in line.lower(): ready.set()
        if any(k in line for k in ("Error","Traceback","error")): print("  "+line.rstrip())
threading.Thread(target=pump,daemon=True).start()
if not ready.wait(120): print("FAIL: no start"); proc.kill(); sys.exit(1)
sock=socket.create_connection(("127.0.0.1",PORT),timeout=10); sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
st={"cover":0,"peak":0.0,"live":False,"locked":False,"bars":0,"alive":True}
def reader():
    buf=b""
    def rd(n):
        nonlocal buf
        while len(buf)<n:
            try: c=sock.recv(65536)
            except OSError: return None
            if not c: return None
            buf+=c
        out,buf=buf[:n],buf[n:]; return out
    while st["alive"]:
        h=rd(5)
        if h is None: break
        length,t=HDR.unpack(h); p=rd(length) if length else b""
        if p is None: break
        if t==0x02:
            a=np.frombuffer(p,dtype=np.float32).reshape(-1,4); st["cover"]+=a.shape[0]
            if a.size: st["peak"]=max(st["peak"],float(np.abs(a[:,:2]).max()))
        elif t==0x03:
            ev=json.loads(p.decode())
            if ev.get("event")=="live_started": st["live"]=True
            if ev.get("event")=="stats" and ev.get("loop_locked"):
                if not st["locked"]: print(f"  LOOP LOCKED: bars={ev.get('loop_bars')}",flush=True)
                st["locked"]=True; st["bars"]=ev.get("loop_bars")
threading.Thread(target=reader,daemon=True).start()
def send(t,p): sock.sendall(HDR.pack(len(p),t)+p)
try:
    send(0x01, json.dumps({"cmd":"live_start","model":"fast","tags":"jazz trio brushed drums upright bass",
        "denoise":0.7,"bpm":90,"key":"C minor","sr":SR,"window":8.0,"pin":3.0}).encode())
    t0=time.time()
    while not st["live"] and time.time()-t0<180: time.sleep(0.2)
    print(f"  live_started in {time.time()-t0:.1f}s; streaming 4-bar loop…",flush=True)
    full=loader.load_audio(PIANO).waveform.float().numpy().T
    bar=int(60.0/90*4*SR); loop4=full[:4*bar]; N=loop4.shape[0]
    blk=2048; period=blk/SR; nxt=time.perf_counter(); fed=0
    while time.time()-t0<55:
        s=fed%N; ch=loop4[s:s+blk]
        if ch.shape[0]<blk: ch=np.concatenate([ch,loop4[:blk-ch.shape[0]]],0)
        send(0x04, ch.astype(np.float32).tobytes()); fed+=blk
        nxt+=period; dt=nxt-time.perf_counter()
        if dt>0: time.sleep(dt)
    send(0x01, json.dumps({"cmd":"live_stop"}).encode()); time.sleep(0.5); st["alive"]=False; sock.close()
    print(f"[result] live={st['live']} cover_out={st['cover']/SR:.1f}s peak={st['peak']:.3f} LOCKED={st['locked']} bars={st['bars']}",flush=True)
    print("PASS: full sidecar path locked the loop" if (st['locked'] and st['bars']==4 and st['peak']>0.02) else "FAIL",flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()

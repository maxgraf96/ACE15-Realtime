"""Phase A validation: the real-time live-input path (client -> sidecar -> resample -> WAV).

Spawns the sidecar on a test port, streams a 1 kHz stereo tone at 44.1 kHz as 0x04
AUDIO-IN frames (the same framing the JUCE app uses), then checks the sidecar's
captured 48 k WAV preserves pitch (resample correct), level, and duration. This
exercises everything except the JUCE audio-thread capture (validated by build + ears).
"""
import os, sys, socket, struct, subprocess, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8799
HDR = struct.Struct(">IB")
HOST_SR, TONE_HZ, DUR = 44100, 1000.0, 2.0
CAP = os.path.join(ROOT, "test_output", "diag", "rt_input_capture.wav")


def send(sock, type_, payload: bytes):
    sock.sendall(HDR.pack(len(payload), type_) + payload)


def recv_frame(sock):
    hdr = b""
    while len(hdr) < 5:
        c = sock.recv(5 - len(hdr))
        if not c: return None
        hdr += c
    length, type_ = HDR.unpack(hdr)
    buf = b""
    while len(buf) < length:
        c = sock.recv(length - len(buf))
        if not c: return None
        buf += c
    return type_, buf


# ---- spawn the sidecar on a test port ----
if os.path.exists(CAP):
    os.remove(CAP)
proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "sidecar", "server.py"), "--port", str(PORT)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
ready = threading.Event()


def pump():
    for line in proc.stdout:
        if "listening" in line.lower():
            ready.set()
        if "[sidecar]" in line or "Error" in line or "Traceback" in line:
            print("  " + line.rstrip())


threading.Thread(target=pump, daemon=True).start()
if not ready.wait(120):
    print("FAIL: sidecar did not start"); proc.kill(); sys.exit(1)

try:
    sock = socket.create_connection(("127.0.0.1", PORT), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    import json
    send(sock, 0x01, json.dumps({"cmd": "input_config", "sr": HOST_SR, "channels": 2}).encode())
    send(sock, 0x01, json.dumps({"cmd": "input_start"}).encode())

    # stream a 1 kHz stereo tone at host SR in ~2048-frame chunks
    n = int(HOST_SR * DUR)
    t = np.arange(n) / HOST_SR
    tone = (0.5 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)  # [n,2]
    blk = 2048
    for i in range(0, n, blk):
        send(sock, 0x04, stereo[i:i + blk].tobytes())
        time.sleep(0.002)
    send(sock, 0x01, json.dumps({"cmd": "input_stop"}).encode())

    # wait for input_captured event
    captured = None
    sock.settimeout(15)
    levels = 0
    while True:
        fr = recv_frame(sock)
        if fr is None: break
        type_, payload = fr
        if type_ == 0x03:
            ev = json.loads(payload.decode())
            if ev.get("event") == "input_level": levels += 1
            if ev.get("event") == "input_captured":
                captured = ev; break
    sock.close()

    assert captured is not None, "no input_captured event"
    assert levels > 0, "no input_level events received"
    import soundfile as sf
    y, sr = sf.read(CAP, dtype="float32")
    assert sr == 48000, f"captured SR {sr} != 48000"
    dur = len(y) / sr
    m = y.mean(1) if y.ndim > 1 else y
    X = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    freq = np.fft.rfftfreq(len(m), 1 / sr)[np.argmax(X)]
    peak = float(np.abs(y).max())
    print(f"[result] level_events={levels} captured_secs={captured['secs']} "
          f"wav: {dur:.2f}s @{sr}  dom_freq={freq:.1f}Hz  peak={peak:.3f}")
    ok = (abs(dur - DUR) < 0.15) and (abs(freq - TONE_HZ) < 5.0) and (abs(peak - 0.5) < 0.05)
    print("PASS: input path round-trips (pitch/level/duration preserved through resample)" if ok
          else "FAIL: capture mismatch")
    sys.exit(0 if ok else 1)
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()

"""Python test client for the sidecar — validates the IPC protocol end-to-end
(the contract the JUCE app will implement). Connects, loads a track, sets a
style, plays, receives PCM at ~1x, swaps the prompt mid-stream, writes the WAV."""
from __future__ import annotations
import json, os, socket, struct, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T_CONTROL, T_AUDIO, T_EVENT = 0x01, 0x02, 0x03
HDR = struct.Struct(">IB")
SR = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output", "audition", "sidecar_stream.wav")


def send_json(sock, obj):
    p = json.dumps(obj).encode(); sock.sendall(HDR.pack(len(p), T_CONTROL) + p)


def recv_exact(sock, n):
    b = b""
    while len(b) < n:
        c = sock.recv(n - len(b))
        if not c:
            return None
        b += c
    return b


def main():
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "source.wav")
    sock = socket.create_connection(("127.0.0.1", 8765))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    audio = []
    events = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            hdr = recv_exact(sock, HDR.size)
            if hdr is None:
                break
            ln, t = HDR.unpack(hdr)
            payload = recv_exact(sock, ln) if ln else b""
            if payload is None:
                break
            if t == T_AUDIO:
                audio.append(np.frombuffer(payload, dtype=np.float32).reshape(-1, 2))
            elif t == T_EVENT:
                ev = json.loads(payload.decode()); events.append(ev); print("[client] event:", ev, flush=True)

    th = threading.Thread(target=reader, daemon=True); th.start()

    print("[client] load + style + play")
    send_json(sock, {"cmd": "load", "path": src, "seconds": 30, "steps": 8, "window": 20, "lookahead": 1.5})
    time.sleep(0.5)
    send_json(sock, {"cmd": "style", "tags": "8-bit chiptune, square wave synth lead", "denoise": 0.8})
    time.sleep(0.5)
    send_json(sock, {"cmd": "play"})

    # receive ~30s at 1x; swap prompt at ~10s and ~20s of received audio
    t0 = time.time(); swapped = [False, False]
    target = 30.0
    while time.time() - t0 < target + 8:
        recvd = sum(a.shape[0] for a in audio) / SR
        if recvd >= 10 and not swapped[0]:
            send_json(sock, {"cmd": "prompt", "tags": "lo-fi hip hop, jazzy piano, vinyl, boom-bap drums"}); swapped[0] = True; print(f"[client] t={recvd:.1f}s prompt->lofi", flush=True)
        if recvd >= 20 and not swapped[1]:
            send_json(sock, {"cmd": "prompt", "tags": "aggressive heavy metal, distorted guitars, double kick"}); swapped[1] = True; print(f"[client] t={recvd:.1f}s prompt->metal", flush=True)
        if recvd >= target:
            break
        time.sleep(0.1)

    send_json(sock, {"cmd": "stop"}); stop.set(); time.sleep(0.2); sock.close()
    pcm = np.concatenate(audio, axis=0) if audio else np.zeros((1, 2), np.float32)
    import soundfile as sf
    m = np.abs(pcm).max(); pcm = pcm * (0.97 / m) if m > 1e-6 else pcm
    sf.write(OUT, pcm, SR)
    print(f"[client] received {len(pcm)/SR:.1f}s over IPC -> {OUT}")
    print(f"[client] events: {[e.get('event') for e in events]}")


if __name__ == "__main__":
    main()

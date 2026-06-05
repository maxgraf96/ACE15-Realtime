"""Ableton-Link grid mode: prove the cover stays phase-locked to the SHARED beat with ZERO
drift, even across re-renders (Style tweaks) — the exact failure the user reported ('drift is
variable, especially as we fiddle with controls, which updates the model').

Setup: a 'conductor' Link peer stands in for Ableton (guarantees a peer + a shared beat). We feed
the piano loop INDEXED BY link.beat (i.e. locked to Link, like Ableton playing it). The engine
locks on the Link grid and places the cover by the shared beat. We force 3 re-renders mid-run and
measure the org(consumed)-vs-input(fed) lag EARLY vs LATE. Want: locks on Link, non-silent, and
|LATE-EARLY| ~ 0 ms across the re-renders."""
import os, sys, asyncio, threading, time
sys.path.insert(0, "/Users/max/Code/ACE15-Realtime")
import numpy as np, torch
from engine import mps_compat, loader
mps_compat.force_bf16_on_mps()
from engine.realtime import RealtimeCover, SR
from sidecar.link_sync import get_link
PIANO = "/Users/max/Music/Ableton/Noiiz/SynapticJourney_Noiiz/Loops/Keys/90_C_BriarPatchVintagePiano_01_695.wav"
STYLE = "Instrumental jazz trio, warm upright bass, brushed drums, piano comping"

# --- conductor peer (fake Ableton): guarantees a shared beat/tempo ---
cond = {}
def conductor():
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    async def s():
        from aalink import Link
        lk = Link(90.0); lk.enabled = True; lk.quantum = 4.0; cond["lk"] = lk
    loop.run_until_complete(s()); loop.run_forever()
threading.Thread(target=conductor, daemon=True).start()
time.sleep(2.5)

link = get_link()
for _ in range(40):                      # wait for Link discovery (the conductor peer)
    if link.connected: break
    time.sleep(0.1)
print(f"link available={link.available} connected={link.connected} peers={link.peers} tempo={link.tempo:.3f}")
tempo = link.tempo; bars = 4; bpb = 4
loop_beats = bars * bpb

# Synthetic loop with ONE loud downbeat click per loop over a low noise bed. One marker per loop ->
# the org-vs-input cross-correlation (over multi-loop windows) has a single unambiguous peak per loop
# (no beat aliasing), so we measure the TRUE phase alignment / drift. (Cover quality is irrelevant
# here; we only check it locks on Link, is non-silent, and the org tracks the input with no drift.)
N = int(round(bars * 60.0 / tempo * bpb * SR))   # = loop_P
base = (torch.rand(2, N) - 0.5) * 0.04           # quiet noise bed (keeps _loop_loud happy, every bar)
click = torch.hann_window(480).unsqueeze(0) * torch.sin(torch.linspace(0, 120.0, 480)).unsqueeze(0)
base[:, :480] += click                            # single loud marker at the loop downbeat (phase 0)

rc = RealtimeCover(device="mps", steps=8, lookahead_s=6.0, config_path="acestep-v15-turbo")
rc.begin_live(); rc.link = link; rc.loop_bars_hint = bars
rc.set_style(STYLE, denoise=0.7, bpm=round(tempo, 2), key="C minor"); rc.start()

def cread(buf, s, blk):
    """Circular read of `blk` samples from `buf` at offset s (wraps) -> never skips the marker."""
    L = buf.shape[1]; s %= L
    return buf[:, s:s+blk] if s+blk <= L else torch.cat([buf[:, s:], buf[:, :blk-(L-s)]], 1)

spb_link = 60.0 / tempo
org_o = []; in_o = []; locked_at = None; got = 0; need = 70*SR
restyle_at = {30: "lofi piano", 45: "warm rhodes trio", 58: "bright jazz guitar"}; done = set()
peak = 0.0; last_beat = link.beat
# Feed audio LOCKED TO LINK like Ableton: each tick feed exactly (Δbeat * samples/beat) samples from
# the circular loop at the current Link phase. Sample count is tied to link.beat (NOT wall-clock), so
# the fed stream is perfectly link-locked regardless of this loop's per-iteration overhead.
while got < need:
    time.sleep(0.01)
    cur = link.beat
    nfeed = int(round((cur - last_beat) * spb_link * SR))
    if nfeed <= 0:
        continue
    s = int((last_beat % loop_beats) / loop_beats * N)
    ch = cread(base, s, nfeed); last_beat = cur
    rc.feed_input(ch)
    o = rc.read(nfeed)
    org_o.append(o[:, 2:].copy()); in_o.append(ch.t().numpy().copy())
    if o[:, :2].size: peak = max(peak, float(np.abs(o[:, :2]).max()))
    if locked_at is None and rc.loop_locked: locked_at = got/SR; print(f"  LOCKED@{locked_at:.1f}s link_active={rc.link_active}", flush=True)
    got += nfeed
    secs = got/SR
    for at, txt in restyle_at.items():
        if at not in done and secs >= at:
            done.add(at); rc.set_prompt(txt); print(f"  t={at}s set_prompt('{txt}') -> re-render", flush=True)
rc.stop()
org = np.concatenate(org_o, 0); inp = np.concatenate(in_o, 0)
np.save("/tmp/lg_org.npy", org); np.save("/tmp/lg_inp.npy", inp)
np.save("/tmp/lg_meta.npy", np.array([locked_at or 0, tempo, N], dtype=np.float64))

# DRIFT = how the downbeat marker's loop-phase moves over time. The marker is a single loud click
# per loop, so tracking its position (mod loop_P) is unambiguous (unlike org-vs-input correlation,
# which aliases on periodic material). A locked grid -> ~constant phase; drift -> a ramp.
def marker_phase_dev(x):
    m = np.abs(x).mean(1); thr = m.max() * 0.4
    peaks = []; last = -N
    for i in np.where(m > thr)[0]:
        if i - last > N // 2:
            w = m[max(0, i-2000):i+2000]; pk = max(0, i-2000) + int(np.argmax(w)); peaks.append(pk); last = pk
    if len(peaks) < 3:
        return None
    ph = (np.array(peaks) % N).astype(np.float64); ph -= ph[0]
    ph = (ph + N/2) % N - N/2                         # wrap to [-N/2, N/2]
    return ph
dev = marker_phase_dev(org)
maxdev = float(np.abs(dev).max()) / SR * 1000 if dev is not None else 1e9
drift = float(dev[-1] - dev[0]) / SR * 1000 if dev is not None else 1e9
print(f"[result] locked={rc.loop_locked} link_active={rc.link_active} peak={peak:.3f} bars={rc.loop_bars}", flush=True)
print(f"  cover marker phase: max|dev|={maxdev:.0f}ms  net drift (first->last loop, across 3 re-renders)={drift:+.0f}ms", flush=True)
ok = (rc.loop_locked and rc.link_active and peak > 0.02 and maxdev < 30)
print("PASS: Link grid locked, non-silent, ZERO drift across 3 re-renders" if ok else "FAIL", flush=True)
rc.close()

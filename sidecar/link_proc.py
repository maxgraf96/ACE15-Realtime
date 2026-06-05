"""Ableton Link reader — runs in a SEPARATE process (NO torch / NO MPS).

aalink spins up background threads (its scheduler + Link's own networking); running them in the
same process as the MPS model is unstable here — aalink + a torch-MPS workload hangs/crashes
(reproducible headless, and it took down the live app). So we isolate Link in this tiny process
(like the prompt-LM enhancer) and stream its state to the sidecar over stdout; the sidecar reads
the latest sample and extrapolates the beat between samples (beat advances linearly at the tempo).

Emits one JSON line per tick (~60 Hz): {t, beat, tempo, phase, peers, playing}. `t` is
time.monotonic() — system-wide on macOS/Linux, so the sidecar can extrapolate against its own
monotonic clock. Arg: quantum (beats per bar, default 4). Exits cleanly if aalink is missing or
the pipe closes."""
import asyncio
import json
import sys
import time


async def _run(quantum):
    from aalink import Link
    link = Link(120.0)
    link.enabled = True
    link.quantum = quantum
    link.start_stop_sync_enabled = True
    out = sys.stdout
    period = 1.0 / 60.0
    while True:
        msg = {"t": time.monotonic(), "beat": link.beat, "tempo": link.tempo,
               "phase": link.phase, "peers": link.num_peers, "playing": link.playing}
        try:
            out.write(json.dumps(msg) + "\n")
            out.flush()
        except (BrokenPipeError, ValueError):
            return                              # sidecar gone -> exit
        await asyncio.sleep(period)


if __name__ == "__main__":
    q = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    try:
        asyncio.run(_run(q))
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception as e:
        sys.stderr.write(f"link_proc: {e}\n")
        sys.exit(1)

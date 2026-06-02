# Phase 1 — headless streaming cover, operating points (M4 Max, MPS)

- 30 s source, style='8-bit chiptune, retro video game, square-wave synth lead', depth-1 drain, RCFG/CFG off, DCW off, fp32.
- RTF = compute wall / audio seconds (<1 = real-time). max-chunk = worst gen+decode for one chunk (= min producer lookahead). Overlapping windows (0.5 s) for exact timeline.

| window | steps | denoise | RTF | chunks | gen ms | dec ms | max-chunk ms | chroma | onset |
|---|---|---|---|---|---|---|---|---|---|
| 10s | 8 | 0.8 | 0.115 | 4 | 431 | 364 | 982 | 0.537 | 0.158 |
| 20s | 8 | 0.8 | 0.102 | 2 | 762 | 694 | 1768 | 0.686 | 0.469 |
| 30s | 8 | 0.8 | 0.089 | 1 | 1225 | 1405 | 2630 | 0.642 | 0.007 |
| 10s | 4 | 0.8 | 0.085 | 4 | 212 | 361 | 716 | 0.530 | 0.166 |
| 10s | 8 | 0.6 | 0.113 | 4 | 417 | 361 | 947 | 0.821 | 0.770 |

- Live prompt swap re-encode ~121 ms; applies at next chunk boundary (v0 control granularity = window size). 1-tick control needs the continuous depth pipeline (next).
- Worst per-chunk gen+decode sets the minimum producer lookahead buffer for gapless playback.

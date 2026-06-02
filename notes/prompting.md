# Prompting strategy for the ACE15 cover engine

Synthesis of ACE-Step 1.5's official prompting guide + our own measured findings
(CLAP style-sim + chroma/onset). How the engine builds the conditioning and why.

## The prompt ACE-Step actually sees (cover task)

The model is **instruction-tuned**; the text it consumes is a structured doc, not
one sentence. Our engine (`JITCover._encode`) builds exactly:

```
# Instruction
Generate audio semantic tokens based on the given conditions:   <- TASK_INSTRUCTIONS["cover"], a LEARNED task token; do NOT change

# Caption
<your Style text>                                               <- style/instruments/timbre/era ONLY

# Metas
- bpm: <detected>          (omitted if "Match: Tempo" off)
- timesignature: 4
- keyscale: <detected>     (omitted if "Match: Key" off)
- duration: <track secs>
<|endoftext|>
```
plus a lyrics doc: `# Languages\nen\n\n# Lyric\n<empty><|endoftext|><|endoftext|>`.

## Rules we follow (from the guide)

1. **Caption = global control only**: genre/vibe, instruments, timbre texture,
   era/production feel. Comma-separated keywords beat poetry. Combine several
   dimensions (style + instruments + timbre + era) to anchor direction.
2. **NEVER put bpm/key/tempo in the Caption.** They go in `# Metas` as dedicated
   params. (We verified ours does this; the plugin_morph approach of appending
   "<n> BPM" to the caption is the wrong pattern per the official guide.)
3. **Metas are soft anchors, optional.** bpm 60–180 / common keys (C,G,D,Am,Em) /
   4/4 are reliable; extremes are unstable. For a *cover* we default to injecting
   the SOURCE's detected bpm/key (keeps the remix in time/key with the structure),
   but expose **Match: Tempo / Key** toggles to omit them (e.g. when BPM detection
   octave-errors, or to give the model freedom).
4. **Avoid conflicts.** Don't mix contradictory styles; the model fuses poorly.
   Don't fight the source: a slow ballad won't become *upbeat* (the cover keeps
   the source tempo/structure) — it'll get the genre's *timbre*, not its energy.
   For conflicting goals, the guide suggests temporal evolution ("starts X, becomes
   Y") via lyrics tags — not relevant to our instrumental cover use.
5. **Specificity = control; omission = freedom.** Detailed caption constrains;
   sparse caption lets the model roam (more random surprises).

## Our measured deviations / findings (engine-specific)

- **Lyrics: EMPTY, not `[Instrumental]`.** The guide says use `[Instrumental]`
  for instrumental music, but for the COVER task we measured EMPTY lyrics gives
  *stronger* style (away→funk: empty CLAP **0.26** vs `[Instrumental]` **0.14**) —
  the source structure already implies instrumental, and the lyric token competes.
  So the engine sends empty lyrics. (`self.lyrics` is overridable if vocals are
  ever wanted.)
- **Amount (denoise) is song-dependent.** Tight-beat sources tolerate 0.8; real
  songs lose their groove above ~0.7 (away×funk onset: 0.85@0.5 → 0.14@0.8).
  Default 0.7; real songs usually want ~0.5–0.65. This is the structure↔style dial.
- **Character (timbre ref).** 0 = full restyle (no source timbre) = strongest
  style; 1 = keep source character. Blends the refer latent silence↔source.
- **Model.** XL ("Quality") is markedly better at hard/real-song style transfer
  (away→funk: 2B CLAP −0.06 vs XL +0.22) and is the default.
- **Cover quality is source-dependent**, not just prompt-dependent: a source with
  a clear groove covers far better than a sparse/quiet one, for every style.

## Practical recipe for a good cover

1. Pick a source that already has the energy you want (the cover keeps its tempo).
2. Caption: 4–8 comma-separated descriptors — genre + key instruments + timbre +
   era/production. e.g. `lo-fi hip hop, jazzy Rhodes, dusty vinyl, boom-bap drums,
   warm tape saturation`.
3. Amount ~0.55–0.7 (real songs lower); Character 0 for max style; Quality model.
4. Leave Match: Tempo/Key on unless the detected BPM looks wrong.

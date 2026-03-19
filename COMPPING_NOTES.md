# Compping Branch — Session Notes

Branch: `compping` (renamed from `anticipateaccompniment`)

---

## Goal

Detect whether the user is **comping** (chords) or **improvising** (melody), and generate the complementary role:

- **User comps** → chords as controls (instrument 0 / piano) → model generates melody
- **User improvises** → melody as controls (instrument 52 / Choir Aahs) → model generates chords

Both modes use **anticipate** generation — user input is always passed as controls.

## Pop909 Instrument Mapping

From `prepare_pop909.py` on `AMT-fintuning` branch:

| Channel | Role | MIDI Program | Instrument |
|---------|------|-------------|------------|
| 0 | MELODY | 52 | Choir Aahs |
| 1 | BRIDGE | 24 | Acoustic Guitar Nylon |
| 2 | PIANO | 0 | Acoustic Grand Piano |

Training uses 3 augmentation modes:
- `k%3 == 0` → melody (52) as control, generate chords + guitar
- `k%3 == 1` → chords (0) as control, generate melody + guitar
- `k%3 == 2` → no control, pure autoregressive

## Role Detection

5 signals combined into an improv_score (higher = more likely improvising):

| Signal | Weight | Comping | Improv |
|--------|--------|---------|--------|
| Polyphony | 0.30 | Multiple simultaneous notes | Single notes |
| Repetition | 0.20 | Repeating patterns (within-window motif) | Varied phrases |
| Rhythmic regularity | 0.20 | Steady IOI | Irregular timing |
| Pitch range | 0.15 | Wide (chord voicings) | Narrow |
| Stepwise motion | 0.15 | Large intervals (chord jumps) | Small intervals (scales) |

- EMA smoothing (alpha=0.4) + hysteresis (switch high=0.65, low=0.35)
- Minimum 2 phrases before allowing mode switch
- Monophonic input → immediately classified as improv

### Known issues with role detection

- **Onset tolerance too tight** (30ms) — arpeggiated chords get split into separate onsets, making polyphonic comping look monophonic (poly=0.90 improv when actually comping)
- **Repetition within-window**: motif detection (`pitches[i] == pitches[i % motif_len]`) has false positives — first motif occurrence trivially matches itself. Excluding first occurrence was proposed but rejected (too many rules)

## Anticipation Timing — Current Bug

**The model generates 0 events in anticipate mode.**

### Root cause

The generate function (sample.py line 161) feeds controls to the model when:
```
current_time >= anticipated_time - delta
```

With `DELTA=5`, `delta=500` bins, controls at time 500+, and `current_time=0`:
- `0 >= 500 - 500` → True
- ALL controls are consumed immediately at time 0
- Model has no future to anticipate → generates time >= end_time → 0 events

### What we tried

1. **`end_time = generation_length + DELTA`** (10s instead of 5s) — extends generation window so controls at 5-10s are fed progressively. Output clipped to first 5s for playback. Not yet confirmed working.

2. **Controls looped to `generation_length + DELTA`** (10s) — after DELTA shift, controls land at 5-15s. Should cover the extended generation window.

### Timing diagram (current setup)

```
Controls looped:     [0 ──────────── 10s]
After DELTA shift:        [5s ──────────── 15s]
After clip(DELTA):        [5s ──────────── 15s]
Generation window:   [0 ──────────── 10s]  (gen_length + DELTA)
Output clipped to:   [0 ───── 5s]          (gen_length only)
```

### Next steps

- Debug whether `end_time = generation_length + DELTA` actually produces events
- If not, may need to adjust how controls are timed relative to the generation window
- Consider whether the DELTA shift in `notes_to_controls` is even needed since the generate function already handles anticipation offsets internally

# jam_server.py — Session Notes

Branch: `main`

---

## What We Built

- **Musical phrase detection** — replaced fixed time windows with silence gap (1.5s) + max duration cap (5s) to detect phrase boundaries. The cap enables co-playing — robot joins during long continuous phrases rather than only responding after the human stops.
- **Dynamic window sizing** — phrase duration drives generation length 1:1. Human plays 4s → model generates 4s continuation.
- **OSC feedback** — `/gen/params [temperature, top_p]` and `/gen/scale [root, mode, oos_ratio]` sent to Max after every generation.

---

## Adaptive Sampling: Temperature Mapping (WIP)

### Three factors

| Signal | Direction | Why |
|--------|-----------|-----|
| **Out-of-scale notes** | more out-of-scale → higher temp | Chromatic playing = user being adventurous, model should match |
| **Note density** | more notes → higher temp | Dense playing = energetic, model should match energy |
| **Repetition** | repeating phrases → higher temp | User is looping, model should break out and introduce variation |

### Current approach (threshold-based)

Density is the base, with hard overrides:

```
temperature = 0.5 + density_norm * 0.7          base range [0.5, 1.2]
if rep_score > 0.7:   temperature = 1.9          override: high repetition
elif oos_ratio > 0.3: temperature = 1.7          override: chromatic playing
```

### What we tried and why it didn't work

| Approach | How | Why it failed |
|----------|-----|---------------|
| **Weighted average** | `0.4*oos + 0.3*density + 0.3*rep` | All three average out to ~0.3-0.4 → temp always 0.9-1.1, no dynamic range |
| **Max wins** | `max(oos, density, rep)` | At least one factor always elevated from noise → temp stays high all the time |
| **Repetition only** | `adventurousness = rep_score` | False positives in motif detection (chromatic scale scores 0.5), single factor misses other signals |
| **OOS ratio only** | `adventurousness = oos_ratio` | Scale detection keeps switching mode to fit whatever is played → oos stays near zero |
| **Interval-based** | avg semitone distance between notes | Reverted — didn't address the core combination problem |
| **Threshold-based** | density base + rep/oos overrides | Current — still not producing good results |

### Known sub-component issues

- **Scale detection**: mode switches too often even with sticky scale (15% hysteresis). Reduced to 5 templates (major, minor, dorian, mixolydian, lydian) but still too reactive per-phrase.
- **Repetition detection**: within-phrase motif matching (`pitches[i] == pitches[i % motif_len]`) has false positives — first motif occurrence trivially matches itself, inflating score.

### Ideas not yet tried

- Each factor controls its own parameter (rep → temp, oos → top_p, density → gen length)
- Median of three factors (needs 2/3 to agree)
- User-set key via OSC (makes oos_ratio reliable)
- Accumulate histogram across multiple phrases instead of per-phrase
- Velocity as a signal

### OSC interface

| Message | Direction | Content |
|---------|-----------|---------|
| `/control/adaptive 0\|1` | Max → server | Toggle adaptive mode on/off (default on) |
| `/control/temperature float` | Max → server | Manual temperature (used when adaptive off) |
| `/control/top_p float` | Max → server | Manual top_p (used when adaptive off) |
| `/gen/params [temp, top_p]` | server → Max | Current sampling params (every phrase) |
| `/gen/scale [root, mode, oos_ratio]` | server → Max | Detected scale e.g. "C" "dorian" 0.15 |

---

## Open Questions / Future Work

### 1. Phrase boundary detection

Current silence gap rule is purely timing-based with no musical awareness. Better approaches:
- **Melodic contour** — descending line or arrival on a long held note as landing signal
- **Rhythmic deceleration** — notes getting further apart toward phrase end
- **Energy envelope** — velocity dropping off signals wind-down

### 2. Tempo-aware generation (leader-follower)

From Gil Weinberg's leader-follower framework: when human leads, Shimon should track and adapt to the human's tempo. Approach:
- Run a real-time beat tracker (e.g. BTrack or madmom) on the input note stream
- At output scheduling, quantize generated notes to snap to the detected beat grid
- Generation itself stays unchanged — tempo alignment happens at playback only

### 3. Velocity as a signal

Velocity is currently thrown away in the NoteBuffer. Could be used as an additional factor:
- Loud = intensity → higher top_p
- Soft = restraint → lower top_p

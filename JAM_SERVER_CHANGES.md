# jam_server.py — Session Notes

Branch: `main`

---

## What We Built

- **Musical phrase detection** — replaced fixed time windows with silence gap (1.5s) + max duration cap (5s) to detect phrase boundaries. The cap enables co-playing — robot joins during long continuous phrases rather than only responding after the human stops.
- **Dynamic window sizing** — phrase duration drives generation length 1:1. Human plays 4s → model generates 4s continuation.
- **OSC feedback** — `/gen/params [temperature, top_p]` and `/gen/scale [root, mode, oos_ratio]` sent to Max after every generation.

---

## Adaptive Sampling: How Temperature Is Controlled

Replaced the old density-only mapping with a music-theory-driven approach using three signals. Parameters **persist across phrases** and only recompute on key shift or high repetition — no jarring changes within a single key.

### Three factors

| Signal | Direction | Why |
|--------|-----------|-----|
| **Out-of-scale notes** | more out-of-scale → higher temp | Chromatic playing = user being adventurous, model should match |
| **Note density** | more notes → lower temp | Dense playing = user is leading, model follows conservatively |
| **Repetition** | repeating phrases → higher temp | User is looping, model should break out and introduce variation |

### Formula

```
adventurousness = 0.4 * oos_ratio + 0.3 * (1 - density_norm) + 0.3 * repetition_score

temperature = 0.5 + adventurousness * 1.5    → range [0.5, 2.0]
top_p       = 0.85 + adventurousness * 0.15   → range [0.85, 1.0]
```

### Scale detection

- Builds **duration-weighted** pitch class histogram from the phrase
- Matches against 9 scale templates x 12 roots = 108 candidates
- Templates: major, natural minor, dorian, mixolydian, harmonic minor, pentatonic major/minor, blues, whole tone
- Best match by: coverage → smallest scale on ties → root weight → priority
- If best coverage < 0.6 → "chromatic" with forced oos_ratio = 0.5
- Out-of-scale ratio by **note count** (how often the user reaches for chromatic notes)

### Repetition detection

- Stores previous phrase's pitch class sequence
- Computes longest common subsequence (LCS) ratio with current phrase
- Score 0 = completely different, 1 = exact repeat
- Only triggers param recomputation when score >= 0.5

### When params update

- **Key shift**: detected scale changes (e.g. C major → Eb dorian) → recompute
- **High repetition**: repetition score >= 0.5 → recompute
- **Otherwise**: params stay the same — stable behavior within a key

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

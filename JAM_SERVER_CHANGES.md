# jam_server.py — Session Notes

Branch: `anticipateaccompniment`

---

## What We Built

- **Musical phrase detection** — replaced fixed time windows with silence gap (1.5s) + max duration cap (8s) to detect phrase boundaries. The 8s cap enables co-playing — robot joins during long continuous phrases rather than only responding after the human stops.
- **Dynamic window sizing** — phrase duration drives generation length 1:1. Human plays 4s → model generates 4s continuation.
- **Density-adaptive sampling** — note density mapped linearly to `temperature` and `top_p`. Dense playing → more adventurous output, sparse playing → focused conservative output.
- **OSC feedback** — `/gen/params [temperature, top_p]` sent to Max after every generation so current generation character can be visualized.

---

## Open Questions / Future Work

### 1. Density → sampling parameter mapping

Note density (notes/sec) is a weak proxy — fast scales are dense but predictable. Better signals:
- **Velocity** — currently thrown away in the buffer. Most direct musical expression signal. Loud = intensity, soft = restraint. Maps naturally to `top_p`.
- **Average pitch interval** — mean semitones between consecutive notes. Large leaps = angular/adventurous. Maps naturally to `temperature`.
- **Pitch range** — span between lowest and highest note. Wide range = exploratory playing.
- **IOI variance** — irregular rhythm = more expressive/free playing.

Best candidate: velocity → `top_p`, average interval → `temperature`. Two independent axes that capture genuinely different musical gestures.

### 2. Phrase boundary detection

Current silence gap rule is purely timing-based with no musical awareness. Free jazz improvisation does not follow Western cadence rules — tonic resolution does not reliably signal phrase end. Better approaches:
- **Melodic contour** — descending line or arrival on a long held note as landing signal
- **Rhythmic deceleration** — notes getting further apart toward phrase end
- **Energy envelope** — velocity dropping off signals wind-down
- **ML phrase segmentation** — lightweight model trained on jazz (longer term)

### 3. Tempo-aware generation (leader-follower)

From Gil Weinberg's leader-follower framework: when human leads, Shimon should track and adapt to the human's tempo. Approach:
- Run a real-time beat tracker (e.g. BTrack or madmom) on the input note stream to detect master tempo
- At the output scheduling stage, quantize/retime generated notes to snap to the detected beat grid
- Generation itself stays unchanged — tempo alignment happens at playback scheduling only

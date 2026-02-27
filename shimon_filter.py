"""
Shimon note-processing pipeline.

Transforms a block of decoded AMT notes — represented as
(t, dur, pitch, instrument) tuples — to respect the physical
constraints of the Shimon marimba robot.

Pipeline order (applied in this sequence):
    octave_fold → expand_tremolo → stagger_chords → nudge_runs → filter_notes
"""


def filter_notes(notes: list[tuple], min_note_dist_ms: float = 50, max_notes_per_onset: int = 4) -> list[tuple]:
    """Filter generated notes: drop notes too close in time, cap polyphony per onset.

    Mirrors Performer.filter_phrase() from demos.py but operates on
    (t, dur, pitch, instrument) tuples instead of pretty_midi Note objects.

    min_note_dist_ms   : minimum gap between consecutive note onsets (ms)
    max_notes_per_onset: maximum simultaneous notes at the same onset (<10 ms apart)
    """
    if not notes:
        return notes

    min_note_dist = min_note_dist_ms / 1000.0
    filtered = [notes[0]]
    same_onset_count = 0

    for i in range(1, len(notes)):
        t_curr = notes[i][0]
        t_prev = filtered[-1][0]
        d_time = abs(t_curr - t_prev)

        if d_time < 1e-2:                              # same onset (within 10 ms)
            if same_onset_count >= max_notes_per_onset - 1:
                continue
            same_onset_count += 1
        elif d_time < min_note_dist:                   # too close – skip
            continue
        else:
            same_onset_count = 0

        filtered.append(notes[i])

    return filtered


def octave_fold(notes: list[tuple], lo: int = 48, hi: int = 95) -> list[tuple]:
    """Transpose out-of-range notes by octaves until they fall within [lo, hi]."""
    result = []
    for (t, dur, pitch, instr) in notes:
        p = pitch
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        result.append((t, dur, p, instr))
    return result


def expand_tremolo(notes: list[tuple], max_dur_s: float = 1.0,
                   tremolo_rate: float = 10.0, strike_dur_ms: float = 50.0) -> list[tuple]:
    """Replace notes longer than max_dur_s with repeated rapid strikes (tremolo).

    tremolo_rate  : strikes per second (10 → one strike every 100 ms)
    strike_dur_ms : duration of each individual strike
    The expanded notes are re-sorted so they interleave correctly with other notes.
    """
    interval   = 1.0 / tremolo_rate
    strike_dur = strike_dur_ms / 1000.0
    result     = []

    for (t, dur, pitch, instr) in notes:
        if dur <= max_dur_s:
            result.append((t, dur, pitch, instr))
        else:
            n_strikes = int(dur * tremolo_rate)
            for i in range(n_strikes):
                result.append((t + i * interval, strike_dur, pitch, instr))

    result.sort(key=lambda x: x[0])
    return result


def nudge_runs(notes: list[tuple], max_interval_ms: float = 150.0,
               max_semitones: int = 3) -> list[tuple]:
    """Snap fast close-pitched sequential notes to the previous pitch.

    For consecutive notes that are both:
      - sequential (not same-onset, gap > 10 ms)
      - close in time  (gap < max_interval_ms)
      - close in pitch (interval <= max_semitones)
    the note's pitch is replaced with the previous pitch so the arm stays on
    the same bar instead of travelling to an adjacent one.

    Comparison is against the already-transformed previous note, so a whole
    run collapses to repeated strikes on the run's starting pitch.
    Notes with interval >= 4 semitones are left untouched (different arm).
    """
    if len(notes) < 2:
        return notes

    max_interval = max_interval_ms / 1000.0
    result = [notes[0]]

    for (t, dur, pitch, instr) in notes[1:]:
        prev_t, _, prev_pitch, _ = result[-1]
        d_time  = t - prev_t
        d_pitch = abs(pitch - prev_pitch)

        if d_time > 1e-2 and d_time < max_interval and 0 < d_pitch <= max_semitones:
            result.append((t, dur, prev_pitch, instr))   # repeat previous pitch
        else:
            result.append((t, dur, pitch, instr))

    return result


def stagger_chords(notes: list[tuple], stagger_ms: float = 10.0) -> list[tuple]:
    """Spread simultaneous notes (same onset) apart by stagger_ms each.

    Notes within 10 ms of each other are treated as a chord. The first note
    plays at its original time; each subsequent note in the chord is delayed
    by an additional stagger_ms so arms do not all strike at the same instant.
    """
    if not notes:
        return notes

    stagger  = stagger_ms / 1000.0
    result   = []
    group_t  = notes[0][0]
    count    = 0

    for (t, dur, pitch, instrument) in notes:
        if abs(t - group_t) < 1e-2:          # same onset group
            result.append((t + count * stagger, dur, pitch, instrument))
            count += 1
        else:                                  # new onset group
            group_t = t
            count   = 1
            result.append((t, dur, pitch, instrument))

    return result

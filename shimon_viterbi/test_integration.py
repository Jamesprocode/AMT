"""
End-to-end integration test: plan a short melody on a Shimon-flavored
state space with real cost functions.

We use a restricted pitch range (MIDI 60-71, one octave) to keep the
state space small enough for pure-Python Viterbi to run in seconds.
The full [48,95] range works identically but is ~100x slower.
"""

import time

from costs import CostModel
from state_space import StateSpace, State, home_state, pitch_to_mm
from viterbi import viterbi_decode


def plan_melody(melody: list[tuple[float, int]], pitch_range=(60, 71),
                allow_skip: bool = True, octave_range: int = 2):
    """melody: list of (onset_s, pitch). Returns ViterbiResult."""
    ss = StateSpace(pitch_range=pitch_range)
    cm = CostModel()

    # Build a start state inside the reduced space. Just pick the first
    # state whose striker happens to play the range's low pitch.
    start = ss.beam_for_pitch[pitch_range[0]][0]

    pitches = [p for _, p in melody]
    onsets = [t for t, _ in melody]

    def beam_fn(pitch):
        return ss.extended_beam_fn(pitch, octave_range=octave_range)

    def emission_fn(state, pitch):
        return cm.emission(state, pitch)

    def transition_fn(prev, curr, step_idx):
        if step_idx == 0:
            dt = max(onsets[0], 0.1)  # assume at least 100ms to reach first note
        else:
            dt = onsets[step_idx] - onsets[step_idx - 1]
        return cm.transition(prev, curr, dt)

    skip = (lambda state, pitch: cm.skip_score(state, pitch)) if allow_skip else None

    return ss, viterbi_decode(
        observations=pitches,
        start_state=start,
        beam_fn=beam_fn,
        emission_fn=emission_fn,
        transition_fn=transition_fn,
        skip_score_fn=skip,
    )


def test_ascending_triad():
    """C major triad ascending: C4 -> E4 -> G4."""
    melody = [(0.0, 60), (0.3, 64), (0.6, 67)]
    t0 = time.time()
    ss, result = plan_melody(melody, pitch_range=(60, 71))
    elapsed = time.time() - t0

    print(f"state space: {len(ss):,} states, planned in {elapsed:.2f}s")
    print(f"beam sizes: C4={len(ss.beam_fn(60))}, E4={len(ss.beam_fn(64))}, G4={len(ss.beam_fn(67))}")
    print(f"path ({len(result.path)} states):")
    for t, (state, pitch) in enumerate(zip(result.path, [60, 64, 67])):
        match = "OK" if pitch in state.struck_pitches else "MISS"
        print(f"  step {t}: {state} -> reaches {state.struck_pitches} (target {pitch}) [{match}]")
    print(f"total score: {result.total_score:.3f}")
    print(f"skipped: {result.skipped}")

    # Correctness: each returned state must reach the target pitch (no skips, no octave shift).
    for state, pitch, was_skip in zip(result.path, [60, 64, 67], result.is_skip):
        assert not was_skip, f"step should not be skipped"
        assert pitch in state.struck_pitches, \
            f"state reaches {state.struck_pitches}, expected to include {pitch}"
    # Total score should be positive (3 emission hits dominate small efficiency penalty).
    assert result.total_score > 0
    assert result.skipped == []


def test_repeated_note_no_movement():
    """Two C4s in a row: second state should equal the first (no motion)."""
    melody = [(0.0, 60), (0.5, 60)]
    ss, result = plan_melody(melody, pitch_range=(60, 71))
    assert result.path[0] == result.path[1], \
        f"arms shouldn't move for repeat: {result.path}"
    assert result.is_skip == [False, False], "neither step should be a skip"


def test_octave_fallback_when_pitch_unreachable():
    """Pitch 71 isn't reachable inside [60,67] beam — Viterbi should fall back
    to the same pitch class one octave down (59) — wait, also outside range.
    With range (60,71) and target 60, no octave shift is in range; we use
    a wider range here to actually exercise octave fallback."""
    # Restrict pitch range so that target 72 isn't directly reachable.
    # We allow 60..71 in the state space, so 72 must fall back to 60 (-1 oct).
    # The test asserts the path strikes pitch 60 (octave-shifted hit) instead of skipping.
    melody = [(0.0, 60), (0.4, 72)]
    ss, result = plan_melody(melody, pitch_range=(60, 71), octave_range=2)
    print(f"octave fallback path: {[(s.struck_pitches, sk) for s, sk in zip(result.path, result.is_skip)]}")
    # Step 1's reachable pitches must include 60 (the only pitch class reachable
    # that's an octave-multiple of 72 in [60,71]). Crucially, NOT skipped.
    assert result.is_skip == [False, False], f"got is_skip={result.is_skip}"
    assert 60 in result.path[1].struck_pitches, \
        f"expected fallback reachable pitches to include 60, got {result.path[1].struck_pitches}"


def test_skip_chosen_when_unreachable():
    """Pitch 100 is outside the entire 88-key range. With no octave fallback in
    range, the only option is skip. Path should remain at start_state."""
    melody = [(0.0, 60), (0.3, 100)]
    ss, result = plan_melody(melody, pitch_range=(60, 71), octave_range=0)
    print(f"skip path: {[(s.struck_pitches if not sk else 'SKIP', sk) for s, sk in zip(result.path, result.is_skip)]}")
    assert result.is_skip[0] is False
    assert result.is_skip[1] is True
    assert result.path[1] == result.path[0], "skip should keep arms at previous state"


if __name__ == "__main__":
    test_ascending_triad()
    print()
    test_repeated_note_no_movement()
    print()
    test_octave_fallback_when_pitch_unreachable()
    print()
    test_skip_chosen_when_unreachable()
    print("All integration tests passed.")

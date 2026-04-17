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


def plan_melody(melody: list[tuple[float, int]], pitch_range=(60, 71)):
    """melody: list of (onset_s, pitch). Returns ViterbiResult."""
    ss = StateSpace(pitch_range=pitch_range)
    cm = CostModel()

    # Build a start state inside the reduced space. Just pick the first
    # state whose striker happens to play the range's low pitch.
    start = ss.beam_for_pitch[pitch_range[0]][0]

    # Convert melody to (obs_index, pitch) observations and a timing array.
    pitches = [p for _, p in melody]
    onsets = [t for t, _ in melody]

    def beam_fn(pitch):
        return ss.beam_fn(pitch)

    def emission_fn(state, pitch):
        return cm.emission(state, pitch)

    def transition_fn(prev, curr, step_idx):
        if step_idx == 0:
            dt = max(onsets[0], 0.1)  # assume at least 100ms to reach first note
        else:
            dt = onsets[step_idx] - onsets[step_idx - 1]
        return cm.transition(prev, curr, dt)

    return ss, viterbi_decode(
        observations=pitches,
        start_state=start,
        beam_fn=beam_fn,
        emission_fn=emission_fn,
        transition_fn=transition_fn,
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
        match = "OK" if state.struck_pitch == pitch else "MISS"
        print(f"  step {t}: {state} -> plays MIDI {state.struck_pitch} (target {pitch}) [{match}]")
    print(f"total score: {result.total_score:.3f}")
    print(f"dropped: {result.dropped}")

    # Correctness: each returned state must play the target pitch.
    for state, pitch in zip(result.path, [60, 64, 67]):
        assert state.struck_pitch == pitch, \
            f"state plays {state.struck_pitch}, expected {pitch}"
    # Total score should be positive (3 emission hits dominate small efficiency penalty).
    assert result.total_score > 0
    assert result.dropped == []


def test_repeated_note_no_movement():
    """Two C4s in a row: second state should equal the first (no motion)."""
    melody = [(0.0, 60), (0.5, 60)]
    ss, result = plan_melody(melody, pitch_range=(60, 71))
    assert result.path[0] == result.path[1], \
        f"arms shouldn't move for repeat: {result.path}"


if __name__ == "__main__":
    test_ascending_triad()
    print()
    test_repeated_note_no_movement()
    print("All integration tests passed.")

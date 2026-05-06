"""
Monophonic Shimon state space.

A state captures: four arm mm-positions, strictly increasing, respecting
inter-arm gap constraints. The striker for a given target pitch is
implicit — whichever arm sits at that pitch's bar position. Matches the
representation in Bretan 2017 thesis Ch. III ("about 73,815 possible arm
configurations").

Constants mirror pi-shimon/Include/Def.h:
  - kNotePositionTable: mm positions for MIDI notes 48..95 (48 entries).
  - kBoundaries: per-arm {left_gap, right_gap} in mm.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


# MIDI notes 48..95 -> slider mm position. The C++ table has 49 entries
# (last one is a duplicate sentinel); we take the first 48.
NOTE_POSITIONS_MM: tuple[int, ...] = (
    0, 10, 44, 73, 102, 157, 184, 212, 240, 267, 294, 324,
    377, 406, 434, 463, 490, 546, 574, 599, 624, 651, 673, 698,
    749, 771, 798, 820, 846, 894, 919, 945, 969, 993, 1018, 1044,
    1092, 1118, 1142, 1167, 1193, 1240, 1266, 1291, 1315, 1339, 1364, 1385,
)

MIN_NOTE = 48
MAX_NOTE = 95
NUM_ARMS = 4

# kBoundaries[i] = (left_gap_mm, right_gap_mm) required for arm i.
# Inter-arm gaps are widened by 20mm over the C++ kBoundaries values to
# absorb overshoot/settle slop from the IAI controller. Wall gaps stay at
# 0 — the rail extends past the extreme bars on hardware, so we don't need
# to forbid arm 0 from mm=0 or arm 3 from mm=1385.
ARM_GAPS: tuple[tuple[int, int], ...] = (
    (0, 60),
    (60, 180),
    (180, 60),
    (60, 0),
)

HOME_POSITIONS_MM: tuple[int, int, int, int] = (44, 434, 894, 1240)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def pitch_to_mm(pitch: int) -> int:
    """MIDI pitch -> slider mm position. Raises if out of range."""
    if pitch < MIN_NOTE or pitch > MAX_NOTE:
        raise ValueError(f"pitch {pitch} outside [{MIN_NOTE}, {MAX_NOTE}]")
    return NOTE_POSITIONS_MM[pitch - MIN_NOTE]


def mm_to_pitch(mm: int) -> int:
    """Slider mm position -> MIDI pitch. Raises if mm is not on a bar."""
    idx = NOTE_POSITIONS_MM.index(mm)   # exact match required
    return MIN_NOTE + idx


def gap_ok(left_arm: int, right_arm: int, left_pos: int, right_pos: int) -> bool:
    """True iff two adjacent arms' positions respect the gap constraint."""
    required = max(ARM_GAPS[left_arm][1], ARM_GAPS[right_arm][0])
    return (right_pos - left_pos) >= required


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    positions_mm: tuple[int, int, int, int]  # arm 0..3, strictly increasing

    @property
    def struck_pitches(self) -> tuple[int, int, int, int]:
        """The four pitches reachable from this configuration (one per arm)."""
        return tuple(mm_to_pitch(p) for p in self.positions_mm)

    def striker_for(self, pitch: int) -> int | None:
        """Index of the arm sitting at `pitch`, or None if no arm is there.

        Because positions_mm is strictly increasing, at most one arm can
        match — so the answer (if it exists) is unique.
        """
        target_mm = pitch_to_mm(pitch)
        for i, p in enumerate(self.positions_mm):
            if p == target_mm:
                return i
        return None


# --------------------------------------------------------------------------
# State space
# --------------------------------------------------------------------------

class StateSpace:
    """All valid Shimon states + a beam index from pitch -> states.

    pitch_range optionally restricts the set of bars considered. Useful
    for keeping the state space small during testing.
    """

    def __init__(
        self,
        pitch_range: tuple[int, int] = (MIN_NOTE, MAX_NOTE),
        max_states_per_pitch: int = 0,
    ):
        """
        max_states_per_pitch : if > 0, retain only this many configurations
        per pitch in beam_for_pitch[p]. Selected to span the rail evenly so
        the planner still has flexibility in placing passive arms. 0 = no cap.
        """
        lo, hi = pitch_range
        assert MIN_NOTE <= lo <= hi <= MAX_NOTE
        self.pitch_range = (lo, hi)
        self.max_states_per_pitch = max_states_per_pitch
        self._bar_indices = tuple(range(lo - MIN_NOTE, hi - MIN_NOTE + 1))
        self.states: list[State] = []
        self.beam_for_pitch: dict[int, list[State]] = {
            p: [] for p in range(lo, hi + 1)
        }
        self._enumerate()
        if max_states_per_pitch > 0:
            self._prune_beams()
        self._build_numpy_index()

    # ------------------------------------------------------------------
    # Numpy index — built once after enumeration. Used by viterbi_decode_np
    # and CostModel.{emission_np, transition_np} for vectorized scoring.
    # ------------------------------------------------------------------
    def _build_numpy_index(self) -> None:
        """Populate positions_array, state_to_idx, beam_idx_for_pitch.

        positions_array : (N, 4) int64 array of arm positions, row-aligned
                          with self.states.
        state_to_idx    : State -> row index.
        beam_idx_for_pitch[p] : np.ndarray (int64) of state indices whose
                          configuration places an arm at pitch p — the
                          numpy mirror of beam_for_pitch[p].
        """
        if not self.states:
            self.positions_array = np.zeros((0, NUM_ARMS), dtype=np.int64)
            self.state_pitches = np.zeros((0, NUM_ARMS), dtype=np.int64)
            self.state_to_idx = {}
            self.beam_idx_for_pitch = {
                p: np.zeros((0,), dtype=np.int64)
                for p in self.beam_for_pitch
            }
            return

        self.positions_array = np.asarray(
            [s.positions_mm for s in self.states], dtype=np.int64
        )
        # Precompute the MIDI pitch each arm strikes for each state.
        self.state_pitches = np.asarray(
            [s.struck_pitches for s in self.states], dtype=np.int64
        )
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.beam_idx_for_pitch = {
            p: np.asarray([self.state_to_idx[s] for s in beam], dtype=np.int64)
            for p, beam in self.beam_for_pitch.items()
        }

    def extended_beam_idx(self, pitch: int, octave_range: int = 2) -> np.ndarray:
        """Numpy index version of extended_beam_fn — returns state indices."""
        parts: list[np.ndarray] = []
        for k in range(-octave_range, octave_range + 1):
            shifted = pitch + 12 * k
            if shifted < self.pitch_range[0] or shifted > self.pitch_range[1]:
                continue
            parts.append(self.beam_idx_for_pitch.get(
                shifted, np.zeros((0,), dtype=np.int64)
            ))
        if not parts:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(parts)

    def _prune_beams(self) -> None:
        """For each pitch, keep at most `max_states_per_pitch` configurations.

        Selection criterion: prefer configurations where passive arms are
        spread as evenly as possible across the rail (i.e., minimize the
        variance of inter-arm gaps). This keeps the planner's options open
        because passive arms aren't all clustered at one end.
        """
        K = self.max_states_per_pitch
        for pitch in list(self.beam_for_pitch.keys()):
            states = self.beam_for_pitch[pitch]
            if len(states) <= K:
                continue
            # Score each by how evenly spread the arms are. Lower = more even.
            def spread_score(s: State) -> float:
                gaps = [
                    s.positions_mm[i + 1] - s.positions_mm[i]
                    for i in range(NUM_ARMS - 1)
                ]
                mean = sum(gaps) / len(gaps)
                return sum((g - mean) ** 2 for g in gaps)
            sorted_states = sorted(states, key=spread_score)
            self.beam_for_pitch[pitch] = sorted_states[:K]

    def _enumerate(self) -> None:
        positions = NOTE_POSITIONS_MM

        # Choose 4 bar positions (strictly increasing via combinations) that
        # satisfy the three pairwise gap constraints. Each configuration is
        # registered under all 4 of its arm pitches so beam_for_pitch[P]
        # returns every config in which some arm sits at P.
        for combo in combinations(self._bar_indices, NUM_ARMS):
            p0, p1, p2, p3 = (positions[i] for i in combo)
            if not gap_ok(0, 1, p0, p1):
                continue
            if not gap_ok(1, 2, p1, p2):
                continue
            if not gap_ok(2, 3, p2, p3):
                continue

            pos_tuple = (p0, p1, p2, p3)
            state = State(positions_mm=pos_tuple)
            self.states.append(state)
            for arm_pos in pos_tuple:
                pitch = mm_to_pitch(arm_pos)
                if pitch in self.beam_for_pitch:
                    self.beam_for_pitch[pitch].append(state)

    def __len__(self) -> int:
        return len(self.states)

    def beam_fn(self, pitch: int) -> list[State]:
        """Exact-pitch beam: states whose striker hits exactly `pitch`."""
        return self.beam_for_pitch.get(pitch, [])

    def extended_beam_fn(self, pitch: int, octave_range: int = 2) -> list[State]:
        """Beam including octave-shifted candidates up to ±octave_range.

        Out-of-range octaves (outside [MIN_NOTE, MAX_NOTE]) are silently
        dropped. The cost model handles the per-octave penalty via emission().
        """
        beam: list[State] = []
        for k in range(-octave_range, octave_range + 1):
            shifted = pitch + 12 * k
            if shifted < self.pitch_range[0] or shifted > self.pitch_range[1]:
                continue
            beam.extend(self.beam_for_pitch.get(shifted, []))
        return beam


def home_state() -> State:
    """Canonical start state — arms spread across the rail at MIDI pitches
    50/62/77/86 (D3/D4/F5/D6). Bars all on exact positions so no snap is
    needed. The startup routine should physically move the arms here via
    raw /arm before the first phrase plays."""
    assert (HOME_POSITIONS_MM[0] < HOME_POSITIONS_MM[1]
            < HOME_POSITIONS_MM[2] < HOME_POSITIONS_MM[3]), HOME_POSITIONS_MM
    return State(positions_mm=HOME_POSITIONS_MM)

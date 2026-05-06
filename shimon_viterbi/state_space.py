"""
Monophonic Shimon state space.

A state captures: four arm mm-positions (strictly increasing, respecting
inter-arm gap constraints) plus a striker id in {0,1,2,3} identifying
which arm strikes the target pitch.

Constants mirror ShimonController/Include/Def.h:
  - kNotePositionTable: mm positions for MIDI notes 48..95 (48 entries).
  - kBoundaries: per-arm {left_gap, right_gap} in mm.
"""

from dataclasses import dataclass
from itertools import combinations


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
# Mirrors kBoundaries in pi-shimon/Include/Def.h (arm 1/2 inner gap = 160mm).
ARM_GAPS: tuple[tuple[int, int], ...] = (
    (0, 40),
    (40, 160),
    (160, 40),
    (40, 0),
)

HOME_POSITIONS_MM: tuple[int, int, int, int] = (0, 50, 1345, 1385)


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
    striker: int                              # which arm strikes (0..3)

    @property
    def struck_pitch(self) -> int:
        return mm_to_pitch(self.positions_mm[self.striker])


# --------------------------------------------------------------------------
# State space
# --------------------------------------------------------------------------

class StateSpace:
    """All valid Shimon states + a beam index from pitch -> states.

    pitch_range optionally restricts the set of bars considered. Useful
    for keeping the state space small during testing.
    """

    def __init__(self, pitch_range: tuple[int, int] = (MIN_NOTE, MAX_NOTE)):
        lo, hi = pitch_range
        assert MIN_NOTE <= lo <= hi <= MAX_NOTE
        self.pitch_range = (lo, hi)
        self._bar_indices = tuple(range(lo - MIN_NOTE, hi - MIN_NOTE + 1))
        self.states: list[State] = []
        self.beam_for_pitch: dict[int, list[State]] = {
            p: [] for p in range(lo, hi + 1)
        }
        self._enumerate()

    def _enumerate(self) -> None:
        positions = NOTE_POSITIONS_MM

        # Choose 4 bar positions (strictly increasing via combinations) that
        # satisfy the three pairwise gap constraints.
        for combo in combinations(self._bar_indices, NUM_ARMS):
            p0, p1, p2, p3 = (positions[i] for i in combo)
            if not gap_ok(0, 1, p0, p1):
                continue
            if not gap_ok(1, 2, p1, p2):
                continue
            if not gap_ok(2, 3, p2, p3):
                continue

            pos_tuple = (p0, p1, p2, p3)
            for striker in range(NUM_ARMS):
                state = State(positions_mm=pos_tuple, striker=striker)
                self.states.append(state)
                self.beam_for_pitch[state.struck_pitch].append(state)

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
    """A canonical start state at the C++ home positions. Striker 0 by default."""
    # Home positions 0 and 1385 are exact bar positions (MIDI 48 and 95).
    # 50 and 1345 are NOT bar positions; snap to nearest bars for a valid state.
    def snap_to_bar(mm: int) -> int:
        return min(NOTE_POSITIONS_MM, key=lambda b: abs(b - mm))

    snapped = tuple(snap_to_bar(p) for p in HOME_POSITIONS_MM)
    # Enforce strictly increasing after snapping (should already be true).
    assert snapped[0] < snapped[1] < snapped[2] < snapped[3], snapped
    return State(positions_mm=snapped, striker=0)

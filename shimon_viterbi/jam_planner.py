"""
Bridge between AMT-generated note sequences and the Shimon robot's OSC
interface, using Viterbi path planning.

Replaces the per-note greedy planPath in C++ with a phrase-level plan:
  1. Run Viterbi over the AMT-generated notes (sorted by onset).
  2. Emit raw `/arm armId pos acc v_max` so the C++ moves arms directly
     (bypassing planPath).
  3. Emit `/striker strikerIds pos time_ms` (choreo format) so C++ fires
     each strike at the right time (also bypassing planPath).
  4. Keep the existing `/gen/noteon` / `/gen/noteoff` for Max-side
     monitoring (audio, visualization).

Cross-window continuity: the planner remembers the last state from the
previous phrase so each new phrase plans from the actual current arm
positions, not from home.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# The other modules in this package use absolute imports (so test scripts
# can run with this dir on sys.path). Match that here so jam_planner is
# importable both as `shimon_viterbi.jam_planner` and as a sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from costs import CostModel
from motion import compute_motion_params
from state_space import State, StateSpace, home_state
from viterbi import viterbi_decode, viterbi_decode_np

log = logging.getLogger(__name__)


# Lead-in slack so /arm messages reach the robot and arms physically arrive
# before the strike fires. Replaces the C++ DLY=465ms (which only applied
# to the bypassed greedy planPath).
PHRASE_LEAD_IN_S: float = 0.50

# Time between sending the strike command and the audible sound (per
# pi-shimon/Include/Def.h STRIKE_TIME).
STRIKE_TIME_S: float = 0.05

# Black-key MIDI pitch classes (per pi-shimon/Include/Def.h kBlackKeys).
_BLACK_KEY_PCS = frozenset({1, 3, 6, 8, 10})


def is_white_key(midi_note: int) -> bool:
    return (midi_note % 12) not in _BLACK_KEY_PCS


def striker_bitmask(arm_id: int, midi_note: int) -> int:
    """Mirrors `1 << ((armId * 2) + Util::isWhiteKey(note))` from the C++."""
    return 1 << (arm_id * 2 + (1 if is_white_key(midi_note) else 0))


def _striker_position_mm(midi_velocity: int) -> int:
    """Striker target position from MIDI velocity (Strike mode formula in
    StrikerController.h). Linearly maps velocity to mm of striker travel."""
    v = max(0, min(127, midi_velocity))
    return int(round(v * 0.317 + 9.683))


class JamPlanner:
    """Stateful Viterbi planner across AMT windows. Exposes
    `plan_and_schedule(notes, play_start, win_start)` which returns the
    same `(target_time, osc_address, args)` shape the existing
    `_playback_thread` already consumes.
    """

    def __init__(
        self,
        pitch_range: tuple[int, int] = (48, 95),
        max_states_per_pitch: int = 50,
        beam_width: int = 100,
        octave_range: int = 2,
        cost_model: CostModel | None = None,
        midi_velocity: int = 80,
        use_numpy: bool = True,
    ):
        self.state_space = StateSpace(
            pitch_range=pitch_range,
            max_states_per_pitch=max_states_per_pitch,
        )
        self.cost_model = cost_model or CostModel()
        self.beam_width = beam_width
        self.octave_range = octave_range
        self.midi_velocity = midi_velocity
        self.use_numpy = use_numpy
        self.last_state: State = home_state()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset planner state to home — call on /control/start."""
        self.last_state = home_state()

    def plan_and_schedule(
        self,
        notes: list[tuple[float, float, int, int]],   # (t, dur, pitch, instr)
        play_start: float,
        win_start: float = 0.0,
        lead_in_s: float = PHRASE_LEAD_IN_S,
    ) -> tuple[list[tuple[float, str, list]], dict]:
        """Plan the phrase, return (schedule, stats).

        schedule entries: (target_wallclock_time, osc_address, args)
        stats: dict with keys n_notes, n_skipped, n_octave_shifts,
               viterbi_ms, total_score, last_state_repr.
        """
        if not notes:
            return [], {
                "n_notes": 0, "n_skipped": 0, "n_octave_shifts": 0,
                "viterbi_ms": 0.0, "total_score": 0.0,
            }

        notes_sorted = sorted(notes, key=lambda x: x[0])
        pitches = [p for (_, _, p, _) in notes_sorted]
        onsets  = [t for (t, _, _, _) in notes_sorted]

        # Run Viterbi.
        t0 = time.perf_counter()
        result = self._run_viterbi(pitches, onsets)
        viterbi_ms = (time.perf_counter() - t0) * 1000.0

        if not result.path:
            log.warning("Viterbi returned empty path on %d notes — falling "
                        "back to /gen/noteon-only schedule.", len(notes_sorted))
            return self._fallback_noteon_only(notes_sorted, play_start, win_start, lead_in_s), {
                "n_notes": len(notes_sorted), "n_skipped": len(notes_sorted),
                "n_octave_shifts": 0, "viterbi_ms": viterbi_ms,
                "total_score": float("-inf"),
            }

        schedule = self._build_schedule(
            notes_sorted, result.path, result.is_skip,
            play_start=play_start, win_start=win_start, lead_in_s=lead_in_s,
        )

        # Persist for next phrase. Use the LAST non-skip state as the
        # carry-forward (skips don't move arms; the carrier underneath is
        # what the robot is actually at).
        for state, was_skip in zip(reversed(result.path), reversed(result.is_skip)):
            if not was_skip:
                self.last_state = state
                break

        n_oct = sum(
            1 for state, was_skip, p in zip(result.path, result.is_skip, pitches)
            if not was_skip and p not in state.struck_pitches
        )

        stats = {
            "n_notes": len(notes_sorted),
            "n_skipped": sum(result.is_skip),
            "n_octave_shifts": n_oct,
            "viterbi_ms": viterbi_ms,
            "total_score": float(result.total_score),
        }
        return schedule, stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_viterbi(self, pitches, onsets):
        ss = self.state_space
        cm = self.cost_model

        if self.use_numpy:
            # Numpy decoder takes (onset_s, pitch) tuples and computes the
            # first dt as max(onsets[0], 0.1). To give the arms enough time
            # to move from home_state on the very first note, shift every
            # onset by PHRASE_LEAD_IN_S — this preserves all gap durations
            # and effectively makes dt0 = onsets[0] + lead_in.
            observations = [(o + PHRASE_LEAD_IN_S, p) for o, p in zip(onsets, pitches)]
            return viterbi_decode_np(
                observations=observations,
                start_state=self.last_state,
                state_space=ss,
                cost_model=cm,
                beam_width=self.beam_width,
                octave_range=self.octave_range,
                allow_skip=True,
            )

        # Pure-Python fallback (kept for debugging / parity checks).
        def beam_fn(pitch):
            return ss.extended_beam_fn(pitch, octave_range=self.octave_range)

        def transition_fn(prev, curr, t):
            if t == 0:
                dt = max(onsets[0], PHRASE_LEAD_IN_S)
            else:
                dt = onsets[t] - onsets[t - 1]
            return cm.transition(prev, curr, dt)

        return viterbi_decode(
            observations=pitches,
            start_state=self.last_state,
            beam_fn=beam_fn,
            emission_fn=lambda s, p: cm.emission(s, p),
            transition_fn=transition_fn,
            skip_score_fn=lambda s, p: cm.skip_score(s, p),
            beam_width=self.beam_width,
        )

    def _build_schedule(
        self,
        notes: list[tuple[float, float, int, int]],
        path: list[State],
        is_skip: list[bool],
        play_start: float,
        win_start: float,
        lead_in_s: float,
    ) -> list[tuple[float, str, list]]:
        schedule: list[tuple[float, str, list]] = []
        prev_state = self.last_state

        for i, ((t_sec, d_sec, pitch, instr), state, was_skip) in enumerate(
            zip(notes, path, is_skip)
        ):
            audible_t = play_start + lead_in_s + (t_sec - win_start)

            # Always emit /gen/noteon for Max-side monitoring (info-only).
            channel = (instr % 15) + 2
            schedule.append((audible_t,             "/gen/noteon",  [int(pitch), 80, channel]))
            schedule.append((audible_t + d_sec,     "/gen/noteoff", [int(pitch), channel]))

            if was_skip:
                # Silent on the robot — no /arm, no /striker.
                continue

            # /arm: one message per arm whose position changed.
            if i == 0:
                dt_s = max(notes[0][0] - win_start + lead_in_s, lead_in_s)
            else:
                dt_s = max(notes[i][0] - notes[i - 1][0], 0.001)
            dt_ms = max(1, int(round(dt_s * 1000)))

            for arm_id in range(4):
                if state.positions_mm[arm_id] == prev_state.positions_mm[arm_id]:
                    continue
                dist = abs(state.positions_mm[arm_id] - prev_state.positions_mm[arm_id])
                try:
                    acc_g, v_max = compute_motion_params(dist, dt_ms)
                except ValueError as e:
                    # Should be rare since Viterbi already gated feasibility,
                    # but timing rounding could push us over. Skip the move
                    # rather than send an invalid command.
                    log.warning("Motion params infeasible for arm %d (%dmm in %dms): %s",
                                arm_id, dist, dt_ms, e)
                    continue
                arm_send_t = audible_t - lead_in_s
                schedule.append((
                    arm_send_t,
                    "/arm",
                    [arm_id + 1,                            # 1-indexed on the wire
                     int(state.positions_mm[arm_id]),       # mm
                     float(acc_g),                          # g
                     int(round(v_max))],                    # mm/s
                ))

            # /striker: choreo format. Find which arm strikes the (possibly
            # octave-shifted) pitch.
            striker_arm = state.striker_for(pitch)
            if striker_arm is None:
                for k in (1, -1, 2, -2):
                    striker_arm = state.striker_for(pitch + 12 * k)
                    if striker_arm is not None:
                        break
            if striker_arm is None:
                log.warning("No striker arm found for pitch %d; skipping strike.", pitch)
                continue

            # The striker_arm's actual struck pitch (might be octave-shifted).
            actual_pitch = state.struck_pitches[striker_arm]
            mask = striker_bitmask(striker_arm, actual_pitch)
            pos = _striker_position_mm(self.midi_velocity)
            striker_send_t = audible_t - STRIKE_TIME_S
            schedule.append((
                striker_send_t,
                "/striker",
                [int(mask), int(pos), 0],   # time_ms=0 → fire on receipt
            ))

            prev_state = state

        schedule.sort(key=lambda x: x[0])
        return schedule

    def _fallback_noteon_only(self, notes, play_start, win_start, lead_in_s):
        """If Viterbi fails entirely, at least let Max hear the notes so the
        AMT continuation isn't completely silent."""
        sched: list[tuple[float, str, list]] = []
        for (t_sec, d_sec, pitch, instr) in notes:
            channel = (instr % 15) + 2
            t_on = play_start + lead_in_s + (t_sec - win_start)
            sched.append((t_on,         "/gen/noteon",  [int(pitch), 80, channel]))
            sched.append((t_on + d_sec, "/gen/noteoff", [int(pitch), channel]))
        sched.sort(key=lambda x: x[0])
        return sched

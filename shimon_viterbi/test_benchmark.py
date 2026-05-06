"""
Benchmark: greedy (current C++ logic, ported to Python) vs Viterbi.

For each melody, runs both planners, measures planning time, and scores
the resulting path with the same CostModel so we can compare apples to
apples. Greedy is allowed octave fallback (matching kOctavesToTry from
Def.h: {0, +1, -1, +2}); both planners use the same hard feasibility
gate (safety-derated velocity + acceleration limits).

Run:  python test_benchmark.py
"""

import time
from math import inf

from costs import CostModel
from state_space import StateSpace, State, NUM_ARMS, NOTE_POSITIONS_MM, MIN_NOTE, MAX_NOTE
from viterbi import viterbi_decode


# --------------------------------------------------------------------------
# Greedy planner (Python port of C++ ArmController::planPath)
# --------------------------------------------------------------------------

OCTAVES_TO_TRY = (0, 12, -12, 24)


def greedy_plan(
    melody: list[tuple[float, int]],
    state_space: StateSpace,
    cost_model: CostModel,
    start_state: State,
) -> tuple[list[State], list[bool], float]:
    """Run a per-note greedy planner mirroring C++ planPath.

    For each note in arrival order, picks the arm assignment with the lowest
    acceleration (i.e., easiest move) among feasible candidates. Allows
    octave fallback. Returns (path, is_skip, total_score) using the same
    CostModel that scores Viterbi, so scores are directly comparable.
    """
    pitches = [p for _, p in melody]
    onsets = [t for t, _ in melody]

    path: list[State] = []
    is_skip: list[bool] = []
    total_score = 0.0
    current = start_state

    for step, target_pitch in enumerate(pitches):
        if step == 0:
            dt = max(onsets[0], 0.1)
        else:
            dt = onsets[step] - onsets[step - 1]

        chosen_state = None
        chosen_emit = -inf
        chosen_acc = inf  # greedy picks the smallest acceleration

        # Try each octave fallback in priority order.
        for oct_shift in OCTAVES_TO_TRY:
            shifted_pitch = target_pitch + oct_shift
            if shifted_pitch < MIN_NOTE or shifted_pitch > MAX_NOTE:
                continue

            # Candidates that strike this exact (possibly-shifted) pitch.
            candidates = state_space.beam_for_pitch.get(shifted_pitch, [])
            if not candidates:
                continue

            for cand in candidates:
                # Hard feasibility (same gate as Viterbi).
                if not cost_model.is_feasible(current, cand, dt):
                    continue

                # Score the move using the cost model so we can pick the
                # "easiest" — greedy in C++ used acceleration; here we use
                # cost transition + emission, which monotonically tracks
                # "how good is this single move".
                step_score = (
                    cost_model.transition(current, cand, dt)
                    + cost_model.emission(cand, target_pitch)
                )

                # Approximate "acceleration cost" = -transition score; greedy
                # prefers least-cost moves. Tie-break by picking first found
                # (mirrors C++ list.sort(by acc) on equal accelerations).
                acc_proxy = -cost_model.transition(current, cand, dt)
                if acc_proxy < chosen_acc:
                    chosen_acc = acc_proxy
                    chosen_state = cand
                    chosen_emit = step_score

            if chosen_state is not None:
                break  # found a feasible octave; greedy stops searching

        if chosen_state is None:
            # All octaves failed — fall back to a "skip" (same semantics as Viterbi's skip).
            chosen_state = current
            chosen_emit = cost_model.skip_score(current, target_pitch)
            is_skip.append(True)
        else:
            is_skip.append(False)

        path.append(chosen_state)
        total_score += chosen_emit
        current = chosen_state

    return path, is_skip, total_score


# --------------------------------------------------------------------------
# Viterbi planner (using the same beam + skip semantics)
# --------------------------------------------------------------------------

def viterbi_plan(
    melody: list[tuple[float, int]],
    state_space: StateSpace,
    cost_model: CostModel,
    start_state: State,
    octave_range: int = 2,
):
    pitches = [p for _, p in melody]
    onsets = [t for t, _ in melody]

    def beam_fn(pitch):
        return state_space.extended_beam_fn(pitch, octave_range=octave_range)

    def transition_fn(prev, curr, step_idx):
        if step_idx == 0:
            dt = max(onsets[0], 0.1)
        else:
            dt = onsets[step_idx] - onsets[step_idx - 1]
        return cost_model.transition(prev, curr, dt)

    return viterbi_decode(
        observations=pitches,
        start_state=start_state,
        beam_fn=beam_fn,
        emission_fn=lambda s, p: cost_model.emission(s, p),
        transition_fn=transition_fn,
        skip_score_fn=lambda s, p: cost_model.skip_score(s, p),
    )


# --------------------------------------------------------------------------
# Test melodies
# --------------------------------------------------------------------------

def melody_arpeggio_simple():
    """C major triad ascending — easy."""
    return [(0.0, 60), (0.3, 64), (0.6, 67), (0.9, 72)]


def melody_alternating_octaves():
    """Pitches that swing back and forth across the full range — designed
    to expose greedy's myopia: a wrong arm choice early forces costly
    repositioning later."""
    return [
        (0.00, 60), (0.25, 84), (0.50, 62), (0.75, 86),
        (1.00, 64), (1.25, 88), (1.50, 65), (1.75, 89),
    ]


def melody_chromatic_run():
    """Stepwise chromatic line — many notes, small intervals."""
    return [(0.1 * i, 60 + i) for i in range(16)]


def melody_wide_leap_then_recover():
    """Wide leap that traps greedy: best arm for low notes blocks the high note."""
    return [
        (0.00, 50),  # low — greedy will pick whichever arm is closest
        (0.30, 90),  # huge leap
        (0.60, 52),
        (0.90, 88),
        (1.20, 54),
        (1.50, 86),
    ]


def melody_with_repeats():
    """Repeated notes don't move arms — both planners should agree."""
    return [
        (0.00, 60), (0.20, 60), (0.40, 60),
        (0.60, 64), (0.80, 64),
        (1.00, 67), (1.20, 60),
    ]


# --------------------------------------------------------------------------
# Benchmark runner
# --------------------------------------------------------------------------

def total_arm_distance(path: list[State]) -> int:
    total = 0
    for prev, curr in zip(path, path[1:]):
        total += sum(
            abs(prev.positions_mm[n] - curr.positions_mm[n])
            for n in range(NUM_ARMS)
        )
    return total


def num_octave_shifts(path: list[State], pitches: list[int]) -> int:
    shifts = 0
    for state, target in zip(path, pitches):
        if state.struck_pitch != target:
            shifts += 1
    return shifts


def benchmark_one(name: str, melody: list[tuple[float, int]],
                  state_space: StateSpace, cost_model: CostModel,
                  start_state: State):
    pitches = [p for _, p in melody]

    # Greedy.
    t0 = time.perf_counter()
    g_path, g_skip, g_score = greedy_plan(melody, state_space, cost_model, start_state)
    g_time = time.perf_counter() - t0

    # Viterbi.
    t0 = time.perf_counter()
    v_result = viterbi_plan(melody, state_space, cost_model, start_state)
    v_time = time.perf_counter() - t0

    # Apples-to-apples scoring uses the SAME CostModel for both paths.
    print(f"\n=== {name} ({len(melody)} notes) ===")
    print(f"             {'GREEDY':>10s}   {'VITERBI':>10s}")
    print(f"  plan time  {g_time*1000:>8.2f}ms   {v_time*1000:>8.2f}ms")
    print(f"  score      {g_score:>10.3f}   {v_result.total_score:>10.3f}   "
          f"(viterbi - greedy = {v_result.total_score - g_score:+.3f})")
    print(f"  skipped    {sum(g_skip):>10d}   {sum(v_result.is_skip):>10d}")
    print(f"  oct shifts {num_octave_shifts(g_path, pitches):>10d}   "
          f"{num_octave_shifts(v_result.path, pitches):>10d}")
    print(f"  total mm   {total_arm_distance(g_path):>10d}   "
          f"{total_arm_distance(v_result.path):>10d}")

    return {
        "name": name,
        "n_notes": len(melody),
        "g_time_ms": g_time * 1000,
        "v_time_ms": v_time * 1000,
        "g_score": g_score,
        "v_score": v_result.total_score,
    }


def main():
    # Use the FULL Shimon range so the comparison is realistic.
    print("Building full-range state space …")
    t0 = time.perf_counter()
    ss = StateSpace(pitch_range=(MIN_NOTE, MAX_NOTE))
    print(f"  {len(ss):,} states in {(time.perf_counter()-t0)*1000:.0f}ms")

    cm = CostModel()

    # Pick a start state: lowest-pitch beam, first entry.
    start = ss.beam_for_pitch[MIN_NOTE][0] if ss.beam_for_pitch.get(MIN_NOTE) else ss.states[0]

    melodies = [
        ("simple arpeggio",        melody_arpeggio_simple()),
        ("repeats",                melody_with_repeats()),
        ("chromatic run",          melody_chromatic_run()),
        ("alternating octaves",    melody_alternating_octaves()),
        ("wide leap recovery",     melody_wide_leap_then_recover()),
    ]

    results = []
    for name, mel in melodies:
        results.append(benchmark_one(name, mel, ss, cm, start))

    print("\n=== summary ===")
    print(f"{'melody':<22s} {'g_ms':>7s} {'v_ms':>8s} {'Δscore':>8s}  {'verdict'}")
    for r in results:
        delta = r["v_score"] - r["g_score"]
        better = "viterbi" if delta > 1e-6 else ("equal" if abs(delta) < 1e-6 else "greedy")
        print(f"{r['name']:<22s} {r['g_time_ms']:>7.2f} {r['v_time_ms']:>8.2f} "
              f"{delta:>+8.3f}  {better}")


if __name__ == "__main__":
    main()

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
    beam_width: int = 200,
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
        beam_width=beam_width,
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


def melody_out_of_range():
    """Pitches outside the Shimon range [48,95]. Both planners MUST shift octaves."""
    return [
        (0.00, 60),  # in range
        (0.30, 100), # 5 semitones above max → must shift -12 to 88
        (0.60, 38),  # 10 below min → must shift +12 to 50
        (0.90, 102), # again above max → -12 to 90
        (1.20, 36),  # again below min → +12 to 48
    ]


def melody_infeasible_fast_leap():
    """Wide leaps in tiny time windows force shifts: no arm can travel
    600mm in 50ms (would need >12 m/s, far above the 2.5 m/s limit)."""
    return [
        (0.00, 50),
        (0.05, 90),  # 50ms to leap ~30 semitones — likely infeasible
        (0.10, 50),
        (0.15, 90),
        (0.20, 50),
    ]


def melody_high_register_run():
    """Tight chromatic in the top of the range, with 500ms lead-in to allow
    arms to pre-position from home (so the test isn't dominated by the
    first-leap infeasibility). Greedy can't pre-plan; Viterbi can."""
    return [(0.5 + 0.10 * i, 88 + i) for i in range(8)]  # 88..95


def melody_octave_only_when_needed():
    """Notes inside range with comfortable timing — exact pitch should
    always be reachable. Octave shifts here would be a planning bug."""
    return [
        (0.0, 60), (0.4, 64), (0.8, 67), (1.2, 72), (1.6, 76), (2.0, 79),
    ]


# --------------------------------------------------------------------------
# AMT-scale melodies — simulate what the model actually generates
# --------------------------------------------------------------------------

def _amt_like(n_notes: int, density_hz: float = 5.0, seed: int = 42) -> list[tuple[float, int]]:
    """Generate a pseudo-random melody resembling AMT output.

    density_hz : average notes per second (5 Hz ≈ AMT cadence)
    Pitches drawn from a typical jazz/blues range with stepwise tendency.
    """
    import random
    rng = random.Random(seed)
    melody = []
    t = 0.0
    pitch = 65  # mid-range start
    for _ in range(n_notes):
        # Rhythm: roughly 1/density_hz spacing with jitter (a beat-driven melody).
        gap = max(0.04, rng.gauss(1.0 / density_hz, 0.04))
        t += gap
        # Pitch: stepwise with occasional leap. Stay in [50, 90] mostly.
        if rng.random() < 0.85:
            pitch += rng.choice([-2, -1, 1, 2])
        else:
            pitch += rng.choice([-7, -5, 5, 7, 12, -12])
        pitch = max(50, min(90, pitch))
        melody.append((t, pitch))
    return melody


def melody_amt_window(n_notes: int, window_s: float = 4.0, seed: int = 42):
    """N notes spread across a 4-second AMT window.

    Realistic AMT density is ~5 Hz (= 20 notes / 4s). Stress: ~10 Hz upper
    bound. The window length matches jam_server.py's actual window_size=4.0.
    """
    return _amt_like(n_notes, density_hz=n_notes / window_s, seed=seed)


def melody_mixed_extremes():
    """Mixes in-range and out-of-range with tight timing — the kind of
    thing AMT might generate when it goes off the rails."""
    return [
        (0.00, 60),
        (0.20, 96),    # just above max (96 → 84)
        (0.40, 47),    # just below min (47 → 59)
        (0.60, 105),   # way above (105 → 81)
        (0.80, 60),
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


def num_octave_shifts(path: list[State], is_skip: list[bool], pitches: list[int]) -> int:
    """Count notes where the configuration octave-shifted (excluding skips)."""
    shifts = 0
    for state, skip, target in zip(path, is_skip, pitches):
        if skip:
            continue  # skip is its own category, not an octave shift
        if target not in state.struck_pitches:
            shifts += 1
    return shifts


def benchmark_one(name: str, melody: list[tuple[float, int]],
                  state_space: StateSpace, cost_model: CostModel,
                  start_state: State, beam_width: int = 200):
    pitches = [p for _, p in melody]

    # Greedy.
    t0 = time.perf_counter()
    g_path, g_skip, g_score = greedy_plan(melody, state_space, cost_model, start_state)
    g_time = time.perf_counter() - t0

    # Viterbi.
    t0 = time.perf_counter()
    v_result = viterbi_plan(melody, state_space, cost_model, start_state, beam_width=beam_width)
    v_time = time.perf_counter() - t0

    # Apples-to-apples scoring uses the SAME CostModel for both paths.
    print(f"\n=== {name} ({len(melody)} notes) ===")
    print(f"             {'GREEDY':>10s}   {'VITERBI':>10s}")
    print(f"  plan time  {g_time*1000:>8.2f}ms   {v_time*1000:>8.2f}ms")
    print(f"  score      {g_score:>10.3f}   {v_result.total_score:>10.3f}   "
          f"(viterbi - greedy = {v_result.total_score - g_score:+.3f})")
    print(f"  skipped    {sum(g_skip):>10d}   {sum(v_result.is_skip):>10d}   (drops)")
    print(f"  oct shifts {num_octave_shifts(g_path, g_skip, pitches):>10d}   "
          f"{num_octave_shifts(v_result.path, v_result.is_skip, pitches):>10d}   (excludes drops)")
    print(f"  exact hits {len(pitches) - sum(g_skip) - num_octave_shifts(g_path, g_skip, pitches):>10d}   "
          f"{len(pitches) - sum(v_result.is_skip) - num_octave_shifts(v_result.path, v_result.is_skip, pitches):>10d}")
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
    K_PER_PITCH = 50    # halved from 100 — see if quality drops at K=50
    BEAM_WIDTH  = 100   # halved from 200
    print(f"Building full-range state space (max_states_per_pitch={K_PER_PITCH}) …")
    t0 = time.perf_counter()
    ss = StateSpace(pitch_range=(MIN_NOTE, MAX_NOTE), max_states_per_pitch=K_PER_PITCH)
    print(f"  {len(ss):,} states total, built in {(time.perf_counter()-t0)*1000:.0f}ms")
    print(f"  beam[60] size: {len(ss.beam_for_pitch[60])}")
    print(f"  beam[80] size: {len(ss.beam_for_pitch[80])}")
    print(f"  beam_width={BEAM_WIDTH}")

    cm = CostModel()

    # Pick a start state: lowest-pitch beam, first entry.
    start = ss.beam_for_pitch[MIN_NOTE][0] if ss.beam_for_pitch.get(MIN_NOTE) else ss.states[0]

    melodies = [
        ("simple arpeggio",        melody_arpeggio_simple()),
        ("repeats",                melody_with_repeats()),
        ("chromatic run",          melody_chromatic_run()),
        ("alternating octaves",    melody_alternating_octaves()),
        ("wide leap recovery",     melody_wide_leap_then_recover()),
        # Adversarial cases: force octave shifts and challenge feasibility.
        ("OUT-OF-RANGE pitches",   melody_out_of_range()),
        ("infeasible fast leap",   melody_infeasible_fast_leap()),
        ("high-register run",      melody_high_register_run()),
        ("mixed extremes",         melody_mixed_extremes()),
        ("octave only when needed", melody_octave_only_when_needed()),
        # Realistic AMT phrases — all fit inside a 4-second window.
        ("AMT 20n / 4s win (5Hz)",  melody_amt_window(20, window_s=4.0, seed=42)),
        ("AMT 30n / 4s win (7.5Hz)",melody_amt_window(30, window_s=4.0, seed=43)),
        ("AMT 40n / 4s win (10Hz)", melody_amt_window(40, window_s=4.0, seed=44)),
        ("AMT 50n / 4s win (12.5Hz, upper)", melody_amt_window(50, window_s=4.0, seed=45)),
    ]

    results = []
    for name, mel in melodies:
        results.append(benchmark_one(name, mel, ss, cm, start, beam_width=BEAM_WIDTH))

    print("\n=== summary ===")
    print(f"{'melody':<22s} {'g_ms':>7s} {'v_ms':>8s} {'Δscore':>8s}  {'verdict'}")
    for r in results:
        delta = r["v_score"] - r["g_score"]
        better = "viterbi" if delta > 1e-6 else ("equal" if abs(delta) < 1e-6 else "greedy")
        print(f"{r['name']:<22s} {r['g_time_ms']:>7.2f} {r['v_time_ms']:>8.2f} "
              f"{delta:>+8.3f}  {better}")


if __name__ == "__main__":
    main()

"""
Tests for the generic Viterbi decoder.

The key correctness test is the "greedy trap" scenario from
shimon_viterbi_planning.md section 5.1: a 3-step problem where a locally
optimal first choice leads to an unrecoverable second step. Greedy picks
the trap; Viterbi must see past it.
"""

from math import inf

from viterbi import viterbi_decode


# --------------------------------------------------------------------------
# Toy scenario
# --------------------------------------------------------------------------
# Three time steps. Each step has two candidate states: "a" and "b".
# Greedy path:  start -> 1a (cheap) -> 2a (forced, expensive) -> 3a -> total 11
# Viterbi path: start -> 1b (costly) -> 2b (cheap) -> 3b        -> total  4
#
# Emission is constant (0) — all states are compatible with the observation.
# The whole story lives in transition scores.

TRANSITIONS = {
    # (prev, curr): score   (higher = better; we're maximizing)
    ("start", "1a"): 0,
    ("start", "1b"): -2,
    ("1a", "2a"): -10,  # 1a's only continuation is bad
    ("1a", "2b"): -inf,
    ("1b", "2a"): -inf,
    ("1b", "2b"): -1,
    ("2a", "3a"): -1,
    ("2a", "3b"): -inf,
    ("2b", "3a"): -inf,
    ("2b", "3b"): -1,
}

BEAM = {
    0: ["1a", "1b"],
    1: ["2a", "2b"],
    2: ["3a", "3b"],
}


def beam_fn(obs):
    return BEAM[obs]


def emission_fn(state, obs):
    return 0.0


def transition_fn(prev, curr, t):
    return TRANSITIONS.get((prev, curr), -inf)


def test_viterbi_escapes_greedy_trap():
    """Viterbi must pick the 1b->2b->3b path, total score -4."""
    result = viterbi_decode(
        observations=[0, 1, 2],
        start_state="start",
        beam_fn=beam_fn,
        emission_fn=emission_fn,
        transition_fn=transition_fn,
    )
    assert result.path == ["1b", "2b", "3b"], f"got {result.path}"
    assert result.total_score == -4, f"got {result.total_score}"
    assert result.skipped == []
    assert result.is_skip == [False, False, False]


def test_viterbi_single_observation():
    """Degenerate: single observation -> pick best state from start."""
    result = viterbi_decode(
        observations=[0],
        start_state="start",
        beam_fn=beam_fn,
        emission_fn=emission_fn,
        transition_fn=transition_fn,
    )
    # "1a" has transition cost 0, "1b" has -2. 1a wins.
    assert result.path == ["1a"]
    assert result.total_score == 0


def test_viterbi_empty():
    """Empty observation sequence -> empty path."""
    result = viterbi_decode(
        observations=[],
        start_state="start",
        beam_fn=beam_fn,
        emission_fn=emission_fn,
        transition_fn=transition_fn,
    )
    assert result.path == []
    assert result.total_score == 0.0


def test_viterbi_beats_greedy_on_same_problem():
    """Sanity: the scores we claim above actually hold."""
    # Simulate greedy manually.
    current = "start"
    greedy_path = []
    greedy_score = 0.0
    for t in [0, 1, 2]:
        best_state, best_score = None, -inf
        for s in BEAM[t]:
            score = TRANSITIONS.get((current, s), -inf)
            if score > best_score:
                best_score, best_state = score, s
        greedy_path.append(best_state)
        greedy_score += best_score
        current = best_state

    viterbi_result = viterbi_decode(
        observations=[0, 1, 2],
        start_state="start",
        beam_fn=beam_fn,
        emission_fn=emission_fn,
        transition_fn=transition_fn,
    )
    # Greedy path: 1a -> 2a -> 3a. Score: 0 + -10 + -1 = -11.
    assert greedy_path == ["1a", "2a", "3a"]
    assert greedy_score == -11
    # Viterbi strictly better.
    assert viterbi_result.total_score > greedy_score


def test_viterbi_skip_chosen_when_beam_costs_more():
    """When a feasible hit costs -3 but skip costs -2, Viterbi must pick skip."""
    transitions = {
        ("start", "a"): -3,    # only state available, expensive
    }
    beam = {0: ["a"]}
    skip_score = -2.0  # less bad than the beam transition

    result = viterbi_decode(
        observations=[0],
        start_state="start",
        beam_fn=lambda o: beam[o],
        emission_fn=lambda s, o: 0.0,
        transition_fn=lambda p, c, t: transitions.get((p, c), -inf),
        skip_score_fn=lambda s, o: skip_score,
    )
    assert result.path == ["start"], f"got {result.path}"
    assert result.is_skip == [True]
    assert result.skipped == [0]
    assert result.total_score == skip_score


def test_viterbi_real_hit_beats_skip_when_cheaper():
    """When the beam transition costs -1 and skip costs -2, real hit wins."""
    transitions = {("start", "a"): -1}
    beam = {0: ["a"]}

    result = viterbi_decode(
        observations=[0],
        start_state="start",
        beam_fn=lambda o: beam[o],
        emission_fn=lambda s, o: 0.0,
        transition_fn=lambda p, c, t: transitions.get((p, c), -inf),
        skip_score_fn=lambda s, o: -2.0,
    )
    assert result.path == ["a"]
    assert result.is_skip == [False]


if __name__ == "__main__":
    test_viterbi_escapes_greedy_trap()
    test_viterbi_single_observation()
    test_viterbi_empty()
    test_viterbi_beats_greedy_on_same_problem()
    test_viterbi_skip_chosen_when_beam_costs_more()
    test_viterbi_real_hit_beats_skip_when_cheaper()
    print("All Viterbi tests passed.")

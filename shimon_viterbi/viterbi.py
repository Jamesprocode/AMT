"""
Generic Viterbi decoder.

Not Shimon-specific. Given a state space, an observation sequence, and
emission/transition score functions, returns the best state sequence
(maximum total additive score).

Implements thesis Algorithm 2 (Bretan 2017, Ch. III) in log/score domain:
scores are additive, -inf means infeasible.
"""

from dataclasses import dataclass
from math import inf
from typing import Callable, Hashable, Sequence, TypeVar

State = TypeVar("State", bound=Hashable)
Obs = TypeVar("Obs")


@dataclass
class ViterbiResult:
    path: list           # best state sequence, one per observation
    total_score: float   # sum of emission + transition scores along path
    dropped: list[int]   # indices where no feasible predecessor existed


def viterbi_decode(
    observations: Sequence[Obs],
    start_state: State,
    beam_fn: Callable[[Obs], Sequence[State]],
    emission_fn: Callable[[State, Obs], float],
    transition_fn: Callable[[State, State, int], float],
) -> ViterbiResult:
    """Decode the best state sequence for a given observation sequence.

    beam_fn(obs)                       -> candidate states for this step
    emission_fn(state, obs)            -> score for state emitting obs
    transition_fn(prev, curr, step_idx)-> score for prev -> curr at step t

    step_idx is passed to transition_fn so the caller can look up dt from
    its own observation timing if needed.
    """
    if not observations:
        return ViterbiResult(path=[], total_score=0.0, dropped=[])

    # Step 0: initialize from start_state.
    obs0 = observations[0]
    V: dict = {}   # state -> best score to reach state at current step
    P: list[dict] = []  # back-pointers per step (P[t][state] -> predecessor)

    first_P: dict = {}
    for s in beam_fn(obs0):
        trans = transition_fn(start_state, s, 0)
        emit = emission_fn(s, obs0)
        score = trans + emit
        if score > -inf:
            V[s] = score
            first_P[s] = start_state
    P.append(first_P)

    dropped: list[int] = []

    # Forward pass.
    for t in range(1, len(observations)):
        obs = observations[t]
        V_new: dict = {}
        P_new: dict = {}

        for j in beam_fn(obs):
            emit = emission_fn(j, obs)
            if emit <= -inf:
                continue

            best_score = -inf
            best_prev = None
            for k, v_k in V.items():
                trans = transition_fn(k, j, t)
                if trans <= -inf:
                    continue
                score = v_k + trans
                if score > best_score:
                    best_score = score
                    best_prev = k

            if best_prev is not None:
                V_new[j] = best_score + emit
                P_new[j] = best_prev

        if not V_new:
            # No feasible extension. Drop this observation and restart the
            # trellis from the current beam using only emission scores.
            dropped.append(t)
            for j in beam_fn(obs):
                emit = emission_fn(j, obs)
                if emit > -inf:
                    V_new[j] = emit
                    P_new[j] = None  # broken chain

        V = V_new
        P.append(P_new)

    # Backtrace.
    if not V:
        return ViterbiResult(path=[], total_score=-inf, dropped=dropped)

    last_state = max(V, key=V.get)
    total_score = V[last_state]

    path = [last_state]
    for t in range(len(observations) - 1, 0, -1):
        prev = P[t].get(path[-1])
        if prev is None:
            # Broken chain from a drop; fall back to best state of prior step.
            if not P[t - 1]:
                break
            prev = max(P[t - 1].keys(), key=lambda s: s.__hash__())
        path.append(prev)
    path.reverse()

    return ViterbiResult(path=path, total_score=total_score, dropped=dropped)

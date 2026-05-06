"""
Generic Viterbi decoder.

Not Shimon-specific. Given a state space, an observation sequence, and
emission/transition score functions, returns the best state sequence
(maximum total additive score).

Implements thesis Algorithm 2 (Bretan 2017, Ch. III) in log/score domain:
scores are additive, -inf means infeasible.

Optional skip mechanism: if `skip_score_fn` is provided, every step also
considers a "stay at previous state and silence this observation" path
with the given score added. The skipped step's predecessor remains in the
trellis with the new score, so future transitions correctly start from
the unmoved state. Used by Shimon to fall back to a rest when no feasible
beam exists rather than break the trellis.
"""

from dataclasses import dataclass
from math import inf
from typing import Callable, Hashable, Optional, Sequence, TypeVar

State = TypeVar("State", bound=Hashable)
Obs = TypeVar("Obs")


@dataclass
class ViterbiResult:
    path: list                        # best state sequence, one per observation
    total_score: float                # sum of emission + transition scores along path
    skipped: list[int]                # indices where the skip option was chosen
    is_skip: list[bool]               # per-step flag: True if this step's state was a skip


def _prune_to_top_k(V: dict, k: int) -> dict:
    """Keep only the top-k entries by score. Used for beamed Viterbi pruning."""
    top = sorted(V.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return dict(top)


def viterbi_decode(
    observations: Sequence[Obs],
    start_state: State,
    beam_fn: Callable[[Obs], Sequence[State]],
    emission_fn: Callable[[State, Obs], float],
    transition_fn: Callable[[State, State, int], float],
    skip_score_fn: Optional[Callable[[State, Obs], float]] = None,
    beam_width: int = 0,
) -> ViterbiResult:
    """Decode the best state sequence for a given observation sequence.

    beam_fn(obs)                       -> candidate states for this step
    emission_fn(state, obs)            -> score for state emitting obs
    transition_fn(prev, curr, step_idx)-> score for prev -> curr at step t
    skip_score_fn(state, obs)          -> score added when the path "stays at
                                           state" instead of striking obs.
                                           If None, skipping is not allowed.
    beam_width                         -> if > 0, after each step retain only
                                           the top-K states in V by score.
                                           Implements beamed Viterbi (Bretan
                                           2017 Fig. 9). 0 = no pruning.

    step_idx is passed to transition_fn so the caller can look up dt from
    its own observation timing if needed.
    """
    if not observations:
        return ViterbiResult(path=[], total_score=0.0, skipped=[], is_skip=[])

    # Step 0: initialize from start_state.
    obs0 = observations[0]
    V: dict = {}                  # state -> best score to reach state at current step
    V_skip: dict = {}             # state -> True if best path here is a skip
    P: list[dict] = []            # back-pointers per step (P[t][state] -> predecessor)
    S: list[dict] = []            # per-step skip flag: S[t][state] = was step t a skip?

    first_P: dict = {}
    first_S: dict = {}
    for s in beam_fn(obs0):
        trans = transition_fn(start_state, s, 0)
        emit = emission_fn(s, obs0)
        score = trans + emit
        if score > -inf:
            if s not in V or score > V[s]:
                V[s] = score
                first_P[s] = start_state
                first_S[s] = False

    if skip_score_fn is not None:
        # Skip option at step 0 keeps us at start_state.
        skip = skip_score_fn(start_state, obs0)
        if skip > -inf:
            if start_state not in V or skip > V[start_state]:
                V[start_state] = skip
                first_P[start_state] = start_state
                first_S[start_state] = True
    P.append(first_P)
    S.append(first_S)

    if beam_width > 0 and len(V) > beam_width:
        V = _prune_to_top_k(V, beam_width)

    skipped: list[int] = []

    # Forward pass.
    for t in range(1, len(observations)):
        obs = observations[t]
        V_new: dict = {}
        P_new: dict = {}
        S_new: dict = {}

        # Beam transitions: real strikes.
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
                cand_score = best_score + emit
                if j not in V_new or cand_score > V_new[j]:
                    V_new[j] = cand_score
                    P_new[j] = best_prev
                    S_new[j] = False

        # Skip option: "stay at predecessor" with skip penalty.
        if skip_score_fn is not None:
            for k, v_k in V.items():
                skip = skip_score_fn(k, obs)
                if skip <= -inf:
                    continue
                cand_score = v_k + skip
                if k not in V_new or cand_score > V_new[k]:
                    V_new[k] = cand_score
                    P_new[k] = k
                    S_new[k] = True

        V = V_new
        P.append(P_new)
        S.append(S_new)

        if beam_width > 0 and len(V) > beam_width:
            V = _prune_to_top_k(V, beam_width)

        if V and all(S_new.get(s, False) for s in V):
            skipped.append(t)

    # Backtrace.
    if not V:
        return ViterbiResult(path=[], total_score=-inf, skipped=skipped, is_skip=[])

    last_state = max(V, key=V.get)
    total_score = V[last_state]

    path = [last_state]
    is_skip_path = [S[-1].get(last_state, False)]
    for t in range(len(observations) - 1, 0, -1):
        prev = P[t].get(path[-1])
        if prev is None:
            break
        path.append(prev)
        is_skip_path.append(S[t - 1].get(prev, False))
    path.reverse()
    is_skip_path.reverse()

    # Recompute skipped indices from is_skip_path (chosen path, not just availability).
    skipped = [i for i, s in enumerate(is_skip_path) if s]

    return ViterbiResult(
        path=path,
        total_score=total_score,
        skipped=skipped,
        is_skip=is_skip_path,
    )

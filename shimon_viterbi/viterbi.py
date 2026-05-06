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

import numpy as np

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


# ==========================================================================
# Numpy-vectorized variant
# ==========================================================================

def viterbi_decode_np(
    observations: list[tuple[float, int]],   # (onset_s, pitch) per note
    start_state,                             # State (must be in state_space)
    state_space,                             # StateSpace with positions_array etc.
    cost_model,                              # CostModel with emission_np / transition_np
    beam_width: int = 100,
    octave_range: int = 2,
    allow_skip: bool = True,
) -> ViterbiResult:
    """Numpy-vectorized monophonic Viterbi for Shimon planning.

    Equivalent to the pure-Python `viterbi_decode` when configured for
    Shimon (skip via skip_score, octave fallback via extended beam,
    beam-width pruning) but does the (|V| x |beam|) score computation in
    a single numpy operation per step.

    Differences from `viterbi_decode`:
      - Takes (onset_s, pitch) tuples directly so dt is computed inside.
      - Takes a StateSpace instead of beam_fn (uses state_space.extended_beam_idx).
      - Takes a CostModel directly (uses emission_np / transition_np).
      - start_state must be a State already present in state_space.

    Returns the same `ViterbiResult` shape (path, total_score, skipped, is_skip).
    """
    if not observations:
        return ViterbiResult(path=[], total_score=0.0, skipped=[], is_skip=[])

    states = state_space.states
    pos_arr = state_space.positions_array          # (N, 4)
    pitch_arr = state_space.state_pitches          # (N, 4)
    state_to_idx = state_space.state_to_idx
    if start_state not in state_to_idx:
        raise ValueError(
            "start_state must be a State present in state_space.state_to_idx"
        )
    start_idx = state_to_idx[start_state]

    onsets = [t for t, _ in observations]
    pitches = [p for _, p in observations]

    skip_score = -cost_model.gamma_drop  # constant per cost_model.skip_score

    # Per-step back-pointers, stored as dict[curr_idx -> prev_idx], plus
    # is_skip flag per (step, curr_idx).
    P: list[dict[int, int]] = []
    S: list[dict[int, bool]] = []

    # ---- Step 0 ----
    obs0 = pitches[0]
    dt0 = max(onsets[0], 0.1)

    beam0_idx = state_space.extended_beam_idx(obs0, octave_range=octave_range)
    if beam0_idx.size == 0 and not allow_skip:
        return ViterbiResult(path=[], total_score=-inf, skipped=[], is_skip=[])

    # Score the (start) -> j transitions in one shot.
    V_idx_to_score: dict[int, float] = {}
    first_P: dict[int, int] = {}
    first_S: dict[int, bool] = {}

    if beam0_idx.size > 0:
        start_pos = pos_arr[start_idx:start_idx + 1]            # (1, 4)
        beam0_pos = pos_arr[beam0_idx]                          # (B0, 4)
        beam0_pitches = pitch_arr[beam0_idx]                    # (B0, 4)
        trans0 = cost_model.transition_np(start_pos, beam0_pos, dt0)[0]  # (B0,)
        emit0 = cost_model.emission_np(beam0_pitches, obs0)               # (B0,)
        cand0 = trans0 + emit0                                            # (B0,)
        feasible_mask = np.isfinite(cand0)
        if feasible_mask.any():
            for j_local in np.flatnonzero(feasible_mask):
                j_idx = int(beam0_idx[j_local])
                score = float(cand0[j_local])
                if j_idx not in V_idx_to_score or score > V_idx_to_score[j_idx]:
                    V_idx_to_score[j_idx] = score
                    first_P[j_idx] = start_idx
                    first_S[j_idx] = False

    if allow_skip:
        # Skip at step 0 keeps us at start_state with skip_score.
        skip0 = skip_score
        if start_idx not in V_idx_to_score or skip0 > V_idx_to_score[start_idx]:
            V_idx_to_score[start_idx] = skip0
            first_P[start_idx] = start_idx
            first_S[start_idx] = True

    P.append(first_P)
    S.append(first_S)

    # Prune V to top-K.
    if beam_width > 0 and len(V_idx_to_score) > beam_width:
        V_idx_to_score = dict(
            sorted(V_idx_to_score.items(), key=lambda kv: kv[1], reverse=True)[:beam_width]
        )

    # ---- Steps t >= 1 ----
    for t in range(1, len(observations)):
        if not V_idx_to_score:
            break
        obs = pitches[t]
        dt = onsets[t] - onsets[t - 1]

        V_idx_arr = np.fromiter(V_idx_to_score.keys(), dtype=np.int64,
                                count=len(V_idx_to_score))
        V_score_arr = np.fromiter((V_idx_to_score[i] for i in V_idx_arr.tolist()),
                                   dtype=np.float64, count=V_idx_arr.size)

        beam_idx = state_space.extended_beam_idx(obs, octave_range=octave_range)

        new_score: dict[int, float] = {}
        new_P: dict[int, int] = {}
        new_S: dict[int, bool] = {}

        if beam_idx.size > 0:
            V_pos = pos_arr[V_idx_arr]                     # (V, 4)
            beam_pos = pos_arr[beam_idx]                   # (B, 4)
            beam_pitches = pitch_arr[beam_idx]             # (B, 4)
            trans = cost_model.transition_np(V_pos, beam_pos, dt)   # (V, B)
            emit = cost_model.emission_np(beam_pitches, obs)        # (B,)

            # Filter beam columns where emit is -inf (unreachable pitch class).
            emit_finite = np.isfinite(emit)
            if emit_finite.any():
                # scores[k, j] = V_score[k] + trans[k, j] + emit[j]
                scores = V_score_arr[:, None] + trans + emit[None, :]   # (V, B)
                # Restrict to columns with finite emission for argmax efficiency,
                # but easier: leave -inf in place; argmax of all -inf columns will
                # still produce some k, but we skip that column via best_score.
                best_score_per_j = scores.max(axis=0)        # (B,)
                best_prev_per_j = scores.argmax(axis=0)      # (B,)

                feasible_j = np.isfinite(best_score_per_j)
                for j_local in np.flatnonzero(feasible_j):
                    j_idx = int(beam_idx[j_local])
                    score = float(best_score_per_j[j_local])
                    prev_idx = int(V_idx_arr[best_prev_per_j[j_local]])
                    if j_idx not in new_score or score > new_score[j_idx]:
                        new_score[j_idx] = score
                        new_P[j_idx] = prev_idx
                        new_S[j_idx] = False

        # Skip option: each k in V stays at k with skip penalty.
        if allow_skip:
            skip_cands = V_score_arr + skip_score
            for local_k, k_idx in enumerate(V_idx_arr.tolist()):
                cand = float(skip_cands[local_k])
                if k_idx not in new_score or cand > new_score[k_idx]:
                    new_score[k_idx] = cand
                    new_P[k_idx] = k_idx
                    new_S[k_idx] = True

        V_idx_to_score = new_score
        P.append(new_P)
        S.append(new_S)

        if beam_width > 0 and len(V_idx_to_score) > beam_width:
            V_idx_to_score = dict(
                sorted(V_idx_to_score.items(), key=lambda kv: kv[1], reverse=True)[:beam_width]
            )

    # ---- Backtrace ----
    if not V_idx_to_score:
        return ViterbiResult(path=[], total_score=-inf, skipped=[], is_skip=[])

    last_idx = max(V_idx_to_score, key=V_idx_to_score.get)
    total_score = V_idx_to_score[last_idx]

    path_idx = [last_idx]
    is_skip_path = [S[-1].get(last_idx, False)]
    for t in range(len(P) - 1, 0, -1):
        prev = P[t].get(path_idx[-1])
        if prev is None:
            break
        path_idx.append(prev)
        is_skip_path.append(S[t - 1].get(prev, False))
    path_idx.reverse()
    is_skip_path.reverse()

    path = [states[i] for i in path_idx]
    skipped = [i for i, s in enumerate(is_skip_path) if s]

    return ViterbiResult(
        path=path,
        total_score=float(total_score),
        skipped=skipped,
        is_skip=is_skip_path,
    )

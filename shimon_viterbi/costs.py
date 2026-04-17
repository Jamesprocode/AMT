"""
Shimon cost functions (monophonic).

Thesis Eqs 1-4 (Bretan 2017, Ch. III), in additive score form:

  B(state, pitch)   = alpha  if state.struck_pitch == pitch, else 0
  h(k, j, dt)       = 0      if transition feasible, else -inf (hard prune)
  p(k, j)           = 0      if arms_moved <= 1, else -lambda  (monophonic)
  d(k, j)           = -omega * total_mm_distance_moved / bar_width_scale

Higher score = better. Infeasible transitions return -inf and are pruned
by the Viterbi loop.
"""

from dataclasses import dataclass
from math import inf

from state_space import ARM_GAPS, NUM_ARMS, State


# Typical spacing between adjacent bars (mm). Used only to normalize the
# efficiency term so omega stays near the thesis's ~0.1 scale.
MEAN_BAR_WIDTH_MM = 29.0  # mean of diffs in NOTE_POSITIONS_MM

VELOCITY_MM_PER_S = 2500.0  # from Def.h VELOCITY_LIMIT = 2.5 m/s


@dataclass
class CostModel:
    alpha: float = 1.0    # emission weight
    lambda_: float = 0.5  # perceptual weight
    omega: float = 0.1    # efficiency weight
    velocity_mm_per_s: float = VELOCITY_MM_PER_S

    # ------------------------------------------------------------------
    # Eq 1 — emission
    # ------------------------------------------------------------------
    def emission(self, state: State, pitch: int) -> float:
        return self.alpha if state.struck_pitch == pitch else 0.0

    # ------------------------------------------------------------------
    # Eq 2 — harm (feasibility gate)
    # ------------------------------------------------------------------
    def is_feasible(self, k: State, j: State, dt_s: float) -> bool:
        # Gap check: kBoundaries already holds at both endpoints because
        # every State in the space was enumerated to satisfy it. The only
        # *motion* risk would be arms crossing mid-flight, but since arms
        # move monotonically on a shared rail and endpoints are non-crossing,
        # they cannot cross mid-flight either. Gaps are fine by construction.

        # Speed check: for each arm, distance_mm / velocity <= dt_s.
        if dt_s <= 0:
            return all(k.positions_mm[n] == j.positions_mm[n] for n in range(NUM_ARMS))

        max_time_needed = 0.0
        for n in range(NUM_ARMS):
            dist = abs(k.positions_mm[n] - j.positions_mm[n])
            t_needed = dist / self.velocity_mm_per_s
            if t_needed > max_time_needed:
                max_time_needed = t_needed
        return max_time_needed <= dt_s

    # ------------------------------------------------------------------
    # Eq 3 — perceptual
    # ------------------------------------------------------------------
    def perceptual(self, k: State, j: State) -> float:
        # Monophonic: one note struck, so penalise any time more than one
        # arm moved.
        arms_moved = sum(
            1 for n in range(NUM_ARMS)
            if k.positions_mm[n] != j.positions_mm[n]
        )
        return 0.0 if arms_moved <= 1 else -self.lambda_

    # ------------------------------------------------------------------
    # Eq 4 — efficiency
    # ------------------------------------------------------------------
    def efficiency(self, k: State, j: State) -> float:
        total_mm = sum(
            abs(k.positions_mm[n] - j.positions_mm[n])
            for n in range(NUM_ARMS)
        )
        return -self.omega * (total_mm / MEAN_BAR_WIDTH_MM)

    # ------------------------------------------------------------------
    # Combined transition (Eq 5)
    # ------------------------------------------------------------------
    def transition(self, k: State, j: State, dt_s: float) -> float:
        if not self.is_feasible(k, j, dt_s):
            return -inf
        return self.perceptual(k, j) + self.efficiency(k, j)

"""
Shimon cost functions (monophonic).

Thesis Eqs 1-4 (Bretan 2017, Ch. III), in additive score form:

  B(state, pitch)   = alpha            if struck_pitch == pitch
                    = alpha - k*beta_octave  if struck_pitch == pitch ± 12k
                    = -inf              otherwise
  h(k, j, dt)       = 0      if transition feasible (within safety-derated limits)
                    = -inf   otherwise (hard prune — never violated)
  p(k, j)           = 0      if arms_moved <= 1, else -lambda  (monophonic)
  d(k, j)           = -omega * total_mm_distance_moved / bar_width_scale

Higher score = better. Infeasible transitions return -inf and are pruned
by the Viterbi loop. Safety: hard limits are derated by SAFETY_FACTOR so
Viterbi can never schedule a move at the absolute hardware limit.
"""

from dataclasses import dataclass
from math import inf, sqrt

from state_space import ARM_GAPS, NUM_ARMS, State


# Typical spacing between adjacent bars (mm). Used only to normalize the
# efficiency term so omega stays near the thesis's ~0.1 scale.
MEAN_BAR_WIDTH_MM = 29.0  # mean of diffs in NOTE_POSITIONS_MM

# Hardware limits from pi-shimon/Include/Def.h
VELOCITY_LIMIT_MM_PER_S = 2500.0   # VELOCITY_LIMIT = 2.5 m/s
ACC_LIMIT_G = 3.0                  # ACC_LIMIT = 3 g
G_MM_PER_S2 = 9800.0               # 9.8 m/s^2 in mm/s^2

# Safety margin: only schedule moves that fit within SAFETY_FACTOR of the
# hardware limit. Anything tighter is treated as infeasible. Crash-avoidance
# is always the highest priority in the cost ordering.
SAFETY_FACTOR = 0.85


@dataclass
class CostModel:
    alpha: float = 1.0           # emission weight (exact-pitch hit)
    lambda_: float = 0.5         # perceptual penalty (>1 arm moved monophonically)
    omega: float = 0.1           # efficiency penalty per bar-width moved
    beta_octave: float = 0.3     # penalty per |octave| of fallback shift
    gamma_drop: float = 2.0      # penalty for skipping a note (rest fallback)
    velocity_mm_per_s: float = VELOCITY_LIMIT_MM_PER_S * SAFETY_FACTOR
    acc_g: float = ACC_LIMIT_G * SAFETY_FACTOR

    # ------------------------------------------------------------------
    # Eq 1 — emission (with octave fallback)
    # ------------------------------------------------------------------
    def emission(self, state: State, pitch: int) -> float:
        struck = state.struck_pitch
        diff = struck - pitch
        if diff == 0:
            return self.alpha
        if diff % 12 != 0:
            return -inf
        octaves = abs(diff) // 12
        return self.alpha - octaves * self.beta_octave

    # ------------------------------------------------------------------
    # Skip score — chosen by Viterbi when no feasible hit is reachable.
    # Stays at the previous state's positions (no arm motion); the note is
    # silenced on the robot but Max still emits /gen/noteon for monitoring.
    # ------------------------------------------------------------------
    def skip_score(self, state: State, pitch: int) -> float:
        return -self.gamma_drop

    # ------------------------------------------------------------------
    # Eq 2 — harm (feasibility gate, derated for safety)
    # ------------------------------------------------------------------
    def is_feasible(self, k: State, j: State, dt_s: float) -> bool:
        # Gap check: kBoundaries already holds at both endpoints because
        # every State in the space was enumerated to satisfy it. The only
        # *motion* risk would be arms crossing mid-flight, but since arms
        # move monotonically on a shared rail and endpoints are non-crossing,
        # they cannot cross mid-flight either. Gaps are fine by construction.

        if dt_s <= 0:
            return all(k.positions_mm[n] == j.positions_mm[n] for n in range(NUM_ARMS))

        # For each arm, check that the trapezoidal motion profile fits within
        # the safety-derated velocity AND acceleration limits. Mirrors the
        # formula in pi-shimon/Shimon/Src/ArmController/Include/Arm.h:181.
        v_lim_mm_s = self.velocity_mm_per_s
        a_lim_mm_s2 = self.acc_g * G_MM_PER_S2

        for n in range(NUM_ARMS):
            dist = abs(k.positions_mm[n] - j.positions_mm[n])
            if dist == 0:
                continue

            # Required peak velocity for a triangular profile in dt_s:
            #   v_peak_tri = 2 * dist / dt_s
            v_peak_tri = 2.0 * dist / dt_s
            v_peak = min(v_lim_mm_s, v_peak_tri)

            # Time-at-velocity-limit must remain positive: dt_s * v_peak > dist
            denom = dt_s * v_peak - dist
            if denom <= 0.0:
                # Move requires impossible peak velocity (above derated limit).
                return False

            # Acceleration in mm/s^2.
            a_required = (v_peak * v_peak) / denom
            if a_required > a_lim_mm_s2:
                return False
        return True

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

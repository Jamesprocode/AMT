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

import numpy as np

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
    # Priority (strongest preference first):
    #   1. exact pitch         : +alpha       = +1.0
    #   2. ±1 octave shift     : alpha - β    =  -1.0
    #   3. ±2 octave shift     : alpha - 2β   =  -3.0
    #   4. drop note           : -γ           = -10.0   ← never unless infeasible
    # Travel cost (omega) is small — only acts as a tie-breaker between
    # otherwise-equivalent paths. The robot is allowed to move more if that
    # means hitting more exact notes.
    alpha: float = 1.0           # emission weight (exact-pitch hit)
    lambda_: float = 3.0         # perceptual penalty (>1 arm moved monophonically)
    omega: float = 0.02          # efficiency penalty per bar-width moved (tie-breaker only)
    beta_octave: float = 2.0     # penalty per |octave| of fallback shift
    gamma_drop: float = 10.0     # penalty for skipping a note (rest fallback)
    velocity_mm_per_s: float = VELOCITY_LIMIT_MM_PER_S * SAFETY_FACTOR
    acc_g: float = ACC_LIMIT_G * SAFETY_FACTOR

    # ------------------------------------------------------------------
    # Eq 1 — emission (with octave fallback)
    # ------------------------------------------------------------------
    def emission(self, state: State, pitch: int) -> float:
        # Each configuration reaches up to 4 pitches (one per arm). Pick the
        # best-matching one: exact > 1-octave shift > 2-octave shift > etc.
        best = -inf
        for struck in state.struck_pitches:
            diff = struck - pitch
            if diff == 0:
                return self.alpha  # exact match wins immediately
            if diff % 12 != 0:
                continue
            octaves = abs(diff) // 12
            score = self.alpha - octaves * self.beta_octave
            if score > best:
                best = score
        return best

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

    # ==================================================================
    # Vectorized variants — numpy implementations of the scalar methods
    # above. Used by viterbi_decode_np for ~30-100x speedups on AMT loads.
    # Equivalent semantics; tested for parity with scalar methods.
    # ==================================================================

    def emission_np(
        self,
        states_pitches: np.ndarray,   # (B, 4) int — pitch each arm reaches
        pitch: int,
    ) -> np.ndarray:
        """Vectorized emission. For each state, score the best-matching arm.

        Mirrors the scalar `emission()` exactly:
          exact match -> alpha
          k-octave shift (|diff| % 12 == 0) -> alpha - k * beta_octave
          otherwise -> -inf
        Per-state result is the max across all 4 arms.
        """
        if states_pitches.size == 0:
            return np.zeros((0,), dtype=np.float64)

        diff = states_pitches - int(pitch)              # (B, 4)
        abs_diff = np.abs(diff)
        on_octave = (abs_diff % 12) == 0                # (B, 4) bool
        octaves = abs_diff // 12                        # (B, 4)
        # Per-arm score: alpha - octaves*beta where on-octave, else -inf.
        per_arm = np.where(
            on_octave,
            self.alpha - octaves.astype(np.float64) * self.beta_octave,
            -np.inf,
        )
        # Best across 4 arms.
        return per_arm.max(axis=1)

    def transition_np(
        self,
        V_pos: np.ndarray,    # (V, 4) int — predecessor positions
        beam_pos: np.ndarray, # (B, 4) int — successor positions
        dt_s: float,
    ) -> np.ndarray:
        """Vectorized transition. Returns (V, B) float matrix.

        Element [k, j] = perceptual(V_pos[k], beam_pos[j])
                       + efficiency(V_pos[k], beam_pos[j])
                       if feasible else -inf.

        Feasibility uses the same trapezoidal-profile gate as is_feasible:
        for every arm with non-zero displacement d,
          v_peak = min(velocity_mm_per_s, 2*d/dt_s)
          denom  = dt_s * v_peak - d
          a_req  = v_peak**2 / denom
        Infeasible iff any arm has denom <= 0 OR a_req > acc limit.
        """
        V = V_pos.shape[0]
        B = beam_pos.shape[0]
        if V == 0 or B == 0:
            return np.zeros((V, B), dtype=np.float64)

        # Pairwise per-arm displacement: (V, B, 4)
        diff = V_pos[:, None, :].astype(np.float64) - beam_pos[None, :, :].astype(np.float64)
        dist = np.abs(diff)
        moved = dist > 0  # (V, B, 4)

        v_lim = float(self.velocity_mm_per_s)
        a_lim = float(self.acc_g) * G_MM_PER_S2

        if dt_s <= 0:
            # Special case mirroring is_feasible: dt<=0 only feasible if no arm moves.
            any_move = moved.any(axis=2)            # (V, B)
            arms_moved = moved.sum(axis=2)
            total_mm = dist.sum(axis=2)
            perceptual = np.where(arms_moved <= 1, 0.0, -self.lambda_)
            efficiency = -self.omega * (total_mm / MEAN_BAR_WIDTH_MM)
            cost = perceptual + efficiency
            return np.where(any_move, -np.inf, cost)

        # Trapezoidal profile per arm.
        # v_peak = min(v_lim, 2*d/dt_s).
        v_peak_tri = (2.0 * dist) / dt_s             # (V, B, 4)
        v_peak = np.minimum(v_lim, v_peak_tri)       # (V, B, 4)
        denom = dt_s * v_peak - dist                 # (V, B, 4)

        # Mask: arms that didn't move can't be infeasible regardless.
        # For moved arms, denom must be > 0 and a_required <= a_lim.
        # Use safe arithmetic: where denom <= 0, mark infeasible directly.
        with np.errstate(divide="ignore", invalid="ignore"):
            a_required = np.where(
                moved & (denom > 0.0),
                (v_peak * v_peak) / np.where(denom > 0.0, denom, 1.0),
                0.0,
            )

        arm_infeasible = moved & ((denom <= 0.0) | (a_required > a_lim))
        pair_infeasible = arm_infeasible.any(axis=2)  # (V, B)

        arms_moved = moved.sum(axis=2)                # (V, B)
        total_mm = dist.sum(axis=2)                   # (V, B)
        perceptual = np.where(arms_moved <= 1, 0.0, -self.lambda_)
        efficiency = -self.omega * (total_mm / MEAN_BAR_WIDTH_MM)
        cost = perceptual + efficiency

        return np.where(pair_infeasible, -np.inf, cost)

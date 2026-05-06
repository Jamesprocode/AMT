"""
Motion parameter helper — computes (acceleration_g, v_max_mm_s) for an arm
move of a given distance in a given time.

Mirrors Arm::computeMotionParam() in
pi-shimon/Shimon/Src/ArmController/Include/Arm.h:181 so that the values we
emit on /arm raw match the trapezoidal profile the IAI controller expects.
Apply the same SAFETY_FACTOR derating used by the cost model so the values
fit comfortably inside the hardware envelope.
"""

from costs import (
    VELOCITY_LIMIT_MM_PER_S,
    ACC_LIMIT_G,
    G_MM_PER_S2,
    SAFETY_FACTOR,
)


def compute_motion_params(
    dist_mm: int,
    time_ms: int,
    velocity_mm_per_s: float = VELOCITY_LIMIT_MM_PER_S * SAFETY_FACTOR,
) -> tuple[float, float]:
    """Return (acceleration_g, v_max_mm_per_s) for the requested move.

    Returns (0.0, 0.0) if dist_mm is zero (no motion needed).
    Raises ValueError if the move is infeasible (not enough time at the
    given velocity limit, or required acceleration exceeds the hardware
    limit).
    """
    if dist_mm == 0:
        return 0.0, 0.0
    if time_ms < 1:
        raise ValueError(f"time_ms must be >= 1, got {time_ms}")

    fPos = dist_mm / 1000.0       # metres
    fTime = time_ms / 1000.0      # seconds
    v_lim_m_s = velocity_mm_per_s / 1000.0

    # Pick the minimum velocity that completes the move in fTime — capped at
    # the (derated) hardware limit. This matches the C++ formula exactly:
    #   v = min(VELOCITY_LIMIT, 2 * dist / time)
    v_peak = min(v_lim_m_s, 2.0 * fPos / fTime)
    v_max_mm_s = 1000.0 * v_peak

    denom = fTime * v_peak - fPos
    if denom <= 0:
        raise ValueError(
            f"infeasible move: {dist_mm}mm in {time_ms}ms exceeds velocity limit "
            f"{velocity_mm_per_s} mm/s"
        )

    acc_g = (v_peak * v_peak) / denom / 9.8
    safety_limit = ACC_LIMIT_G * SAFETY_FACTOR

    # Clip to the safety-derated limit. The Viterbi feasibility gate uses
    # the same formula but with raw float dt_s; here we operate on integer
    # ms after rounding, so a fraction-of-a-percent drift can push us past
    # the limit on an otherwise-accepted move. Clipping caps at the safe
    # acceleration; the arm moves a hair slower (≤1ms late), which is
    # negligible inside bar-strike tolerance.
    if acc_g > safety_limit * 1.05:
        # Genuinely infeasible by any reasonable margin — should not happen
        # if Viterbi gated correctly. Surface as an error.
        raise ValueError(
            f"infeasible move: {dist_mm}mm in {time_ms}ms requires {acc_g:.2f}g, "
            f"more than 5% above safety-derated limit {safety_limit:.2f}g"
        )
    if acc_g > safety_limit:
        acc_g = safety_limit

    return acc_g, v_max_mm_s

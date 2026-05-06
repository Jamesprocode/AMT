"""Shimon Viterbi path planner (first-pass, monophonic)."""

import sys as _sys
from pathlib import Path as _Path

# Absolute imports inside the package files (e.g., `from state_space import ...`)
# need this directory on sys.path. Adding here once means downstream callers
# can simply `from shimon_viterbi import JamPlanner`.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from viterbi import viterbi_decode, ViterbiResult
from jam_planner import JamPlanner, PHRASE_LEAD_IN_S, STRIKE_TIME_S

__all__ = [
    "viterbi_decode",
    "ViterbiResult",
    "JamPlanner",
    "PHRASE_LEAD_IN_S",
    "STRIKE_TIME_S",
]

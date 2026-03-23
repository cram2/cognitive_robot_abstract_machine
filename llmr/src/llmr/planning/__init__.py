"""Planning subpackage: motion preconditions for task actions.

Computes the preparatory actions (navigate, move torso, park arms) that must
precede each task action, and carries cross-instruction state.
"""

from llmr.planning.motion_precondition_planner import (
    ExecutionState,
    MotionPreconditionPlanner,
    PreconditionProvider,
    PreconditionResult,
)

__all__ = [
    "ExecutionState",
    "MotionPreconditionPlanner",
    "PreconditionProvider",
    "PreconditionResult",
]

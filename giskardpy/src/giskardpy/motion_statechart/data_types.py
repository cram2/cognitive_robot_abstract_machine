from __future__ import annotations

from enum import IntEnum, Enum
from typing import Union

from krrood.symbolic_math.symbolic_math import Scalar

goal_parameter = Union[str, float, bool, dict, list, IntEnum, None]


class LifeCycleValues(IntEnum):
    NOT_STARTED = 0
    RUNNING = 1
    PAUSED = 2
    DONE = 3
    FAILED = 4


class FloatEnum(float, Enum):
    """Enum where members are also (and must be) floats"""


class ObservationStateValues(FloatEnum):
    FALSE = float(Scalar.const_false())
    UNKNOWN = float(Scalar.const_trinary_unknown())
    TRUE = float(Scalar.const_true())


class DefaultWeights(FloatEnum):
    WEIGHT_MAX = 10000.0
    WEIGHT_ABOVE_CA = 2500.0
    WEIGHT_COLLISION_AVOIDANCE = 50.0
    WEIGHT_BELOW_CA = 1.0
    WEIGHT_MIN = 0.0


class TransitionKind(Enum):
    START = 1
    """
    Transitions nodes from NOT_STARTED to RUNNING.
    """
    PAUSE = 2
    """
    Transitions nodes from RUNNING to PAUSED if True, or back if False.
    """
    END = 3
    """
    Transitions nodes from RUNNING or PAUSED to DONE.
    """
    RESET = 4
    """
    Transitions nodes from any state to NOT_STARTED.
    """

from __future__ import annotations

from enum import IntEnum, Enum
from typing import Union

import semantic_digital_twin.spatial_types.spatial_types as cas

goal_parameter = Union[str, float, bool, dict, list, IntEnum, None]


class LifeCycleValues(IntEnum):
    NOT_STARTED = 0
    RUNNING = 1
    PAUSED = 2
    DONE = 3
    FAILED = 4


class FloatEnum(float, Enum):
    """Enum where members are also (and must be) floats"""

    pass


class ObservationState(FloatEnum):
    false = cas.TrinaryFalse.to_np()[0]
    unknown = cas.TrinaryUnknown.to_np()[0]
    true = cas.TrinaryTrue.to_np()[0]

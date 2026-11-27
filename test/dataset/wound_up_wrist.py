"""
Where the last wrist joint of Tracy's right arm starts, for tests of motions blocked by
its position limits.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

# %% starts of the wrist


class WristWinding(float, Enum):
    """
    Where the last wrist joint of Tracy's right arm starts, in radians.

    Both members put the arm in the same geometric configuration; they differ by a full
    turn of the joint, whose position limits are a full turn either side of zero.
    """

    UNWOUND = -0.5
    """
    Close to zero, a full turn away from either position limit.
    """

    WOUND_UP = 2 * np.pi - 0.5
    """
    Just short of the upper position limit.
    """

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


# %% a wrist whose range is shorter than a full turn


class ShortWrist(float, Enum):
    """
    Positions of the last wrist joint of Tracy's right arm, in radians, once its
    position limits are narrowed to less than a full turn.
    """

    LOWER_LIMIT = -1.2
    """
    The narrowed lower position limit.
    """

    UPPER_LIMIT = 4.9
    """
    The narrowed upper position limit.
    """

    START = -0.3
    """
    Where the wrist starts.
    """

    NEAR_UPPER_LIMIT = 4.6
    """
    Where an earlier joint move takes the wrist, just short of the upper limit.
    """

    GOAL = 5.2
    """
    Where a tool turn leads from :attr:`NEAR_UPPER_LIMIT` the short way, past the upper
    limit; from :attr:`START` the same turn ends a full turn lower, within the limits.
    """

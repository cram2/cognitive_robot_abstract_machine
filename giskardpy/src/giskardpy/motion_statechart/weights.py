from __future__ import annotations

from cramph.data_types import FloatEnum

# %% weights


class DefaultWeights(FloatEnum):
    """
    The weights goals and tasks give their constraints, relative to the weight of
    collision avoidance.
    """

    WEIGHT_MAXIMUM = 10000.0
    """
    The highest weight a constraint is given.
    """

    WEIGHT_ABOVE_COLLISION_AVOIDANCE = 2500.0
    """
    A weight that wins against collision avoidance.
    """

    WEIGHT_COLLISION_AVOIDANCE = 50.0
    """
    The weight of collision avoidance.
    """

    WEIGHT_BELOW_COLLISION_AVOIDANCE = 1.0
    """
    A weight that collision avoidance wins against.
    """

    WEIGHT_MINIMUM = 0.0
    """
    The lowest weight a constraint is given.
    """

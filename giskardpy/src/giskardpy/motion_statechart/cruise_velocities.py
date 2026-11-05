from __future__ import annotations

from cramph.data_types import FloatEnum

# %% default cruise velocities


class DefaultCruiseVelocity(FloatEnum):
    """
    The speeds Cartesian tasks hold on the way to their goals, until they slow down
    close to them, unless told otherwise.
    """

    TRANSLATION = 0.4
    """
    Unit: m/s.
    """

    ROTATION = 1.2
    """
    Unit: rad/s.
    """

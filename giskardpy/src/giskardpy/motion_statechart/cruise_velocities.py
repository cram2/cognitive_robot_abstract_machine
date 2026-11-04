from __future__ import annotations

from cramph.data_types import FloatEnum

# %% default cruise velocities


class DefaultCruiseVelocity(FloatEnum):
    """
    The speeds at which tasks approach their Cartesian goals unless told otherwise.
    """

    TRANSLATION = 0.4
    """
    Unit: m/s.
    """

    ROTATION = 1.2
    """
    Unit: rad/s.
    """

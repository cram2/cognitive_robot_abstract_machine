"""
Jerk-limited braking of a single degree of freedom on the controller's grid of equal time
steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

BRAKING_RELATIVE_TOLERANCE = 1e-9
"""
Relative tolerance with which a braking counts as removing a velocity, so that a jerk
limit derived from a number of steps is not pushed to the next step by rounding.
"""


@dataclass(frozen=True)
class JerkLimitedBraking:
    """
    Braking of a degree of freedom from its velocity limit to rest with bounded jerk, on a
    grid of equal time steps, starting and ending with zero acceleration.
    """

    velocity_limit: float
    """
    Velocity the degree of freedom brakes from.
    """

    jerk_limit: float
    """
    Largest jerk magnitude allowed while braking.
    """

    time_step: float
    """
    Duration of a single step of the grid.
    """

    @classmethod
    def from_braking_time(
        cls, velocity_limit: float, braking_time: float, time_step: float
    ) -> JerkLimitedBraking:
        """
        Creates the braking whose jerk limit brings a continuous, jerk-limited braking
        from the velocity limit to rest in exactly ``braking_time``.

        The jerk limit does not depend on ``time_step``.

        :param velocity_limit: Velocity the degree of freedom brakes from.
        :param braking_time: Duration of the continuous braking, in seconds.
        :param time_step: Duration of a single step of the grid.
        """
        return cls(
            velocity_limit=velocity_limit,
            jerk_limit=4 * velocity_limit / braking_time**2,
            time_step=time_step,
        )

    @classmethod
    def number_of_steps_for_braking_time(
        cls, braking_time: float, time_step: float
    ) -> int:
        """
        Returns the number of steps a braking created by :meth:`from_braking_time` needs.

        The jerk limit scales with the velocity limit, so every degree of freedom needs the
        same number of steps.

        :param braking_time: Duration of the continuous braking, in seconds.
        :param time_step: Duration of a single step of the grid.
        """
        return cls._smallest_number_of_steps(braking_time**2 / (4 * time_step**2))

    def limited_to_acceleration(self, acceleration_limit: float) -> JerkLimitedBraking:
        """
        Returns this braking with its jerk limit lowered, where needed, so that its
        acceleration stays within ``acceleration_limit``.

        Starting and ending with zero acceleration, a jerk limit J changes the velocity
        by at most 2 * velocity_limit, from one velocity limit to the other, with an
        acceleration of at most sqrt(2 * velocity_limit * J). The acceleration of that
        largest change, and so of every braking, stays within the limit.

        :param acceleration_limit: Largest acceleration magnitude allowed.
        """
        return replace(
            self,
            jerk_limit=min(
                self.jerk_limit, acceleration_limit**2 / (2 * self.velocity_limit)
            ),
        )

    @property
    def number_of_steps(self) -> int:
        """
        Smallest number of steps in which the braking reaches rest.
        """
        return self._smallest_number_of_steps(
            self.velocity_limit / (self.jerk_limit * self.time_step**2)
        )

    @staticmethod
    def _smallest_number_of_steps(velocity_in_jerk_steps: float) -> int:
        """
        Returns the smallest number of steps that removes the given velocity, expressed in
        units of ``jerk_limit * time_step**2``.

        :param velocity_in_jerk_steps: Velocity to remove, in units of
            ``jerk_limit * time_step**2``.
        """
        required = velocity_in_jerk_steps * (1 - BRAKING_RELATIVE_TOLERANCE)
        number_of_steps = 0
        while (
            JerkLimitedBraking._removable_velocity_in_jerk_steps(number_of_steps)
            < required
        ):
            number_of_steps += 1
        return number_of_steps

    @staticmethod
    def _removable_velocity_in_jerk_steps(number_of_steps: int) -> int:
        """
        Returns the largest velocity that ``number_of_steps`` steps remove, in units of
        ``jerk_limit * time_step**2``.

        The acceleration ramps up and back down by one jerk step per time step, which
        removes ⌈m/2⌉·⌈(m+1)/2⌉ units in m steps.

        :param number_of_steps: Number of steps the braking may take.
        """
        return math.ceil(number_of_steps / 2) * math.ceil((number_of_steps + 1) / 2)

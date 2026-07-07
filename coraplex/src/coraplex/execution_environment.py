from __future__ import annotations

import logging
from dataclasses import dataclass, field

from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable

logger = logging.getLogger(__name__)


@dataclass
class ExecutionEnvironment:
    """
    Base class for managing execution context of all actions within.

    Instances of this class is to be used with a "with" context block

    Example:

        >>> with ExecutionEnvironment(ExecutionType.SIMULATED):
        >>>     SequentialPlan(context, NavigateActionDescription, ...)
    """

    execution_type: ExecutionType
    """
    The type of the execution environment.
    """

    collision_avoidance: bool = False
    """
    Whether an :class:`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCollisionAvoidance`
    is added to every motion state chart created within this environment.
    """

    previous_type: ExecutionType = field(init=False, default=None)
    """
    Type of the execution environment before setting it, used for nested
    environments.
    """

    previous_collision_avoidance: bool = field(init=False, default=False)
    """
    Collision avoidance setting before entering this environment, used for nested
    environments.
    """

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type` and
        :py:attr:`~pycram.plans.executables.GiskardExecutable.collision_avoidance`
        and sets them to the values of this environment.
        """
        self.previous_type = GiskardExecutable.execution_type
        self.previous_collision_avoidance = GiskardExecutable.collision_avoidance
        GiskardExecutable.execution_type = self.execution_type
        GiskardExecutable.collision_avoidance = self.collision_avoidance

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, restores the
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type` and
        :py:attr:`~pycram.plans.executables.GiskardExecutable.collision_avoidance`
        to the previously used values.
        """
        GiskardExecutable.execution_type = self.previous_type
        GiskardExecutable.collision_avoidance = self.previous_collision_avoidance

    def __call__(self, collision_avoidance: bool = False):
        """
        Configure the environment for use as a context manager, allowing
        ``with simulated_robot(collision_avoidance=True):``.
        """
        self.collision_avoidance = collision_avoidance
        return self


# These are imported, so they don't have to be initialized when executing with
simulated_robot = ExecutionEnvironment(ExecutionType.SIMULATED)
real_robot = ExecutionEnvironment(ExecutionType.REAL)
semi_real_robot = ExecutionEnvironment(ExecutionType.SEMI_REAL)
no_execution = ExecutionEnvironment(ExecutionType.NO_EXECUTION)

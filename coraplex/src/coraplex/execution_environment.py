from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

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
    Whether an :class:`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCo
    llisionAvoidance` is added to every motion state chart created within this
    environment.
    """

    real_time_pacing: bool = False
    """
    Whether the simulated tick loop is paced to wall-clock time instead of
    running as fast as the QP solve allows. Needed whenever any DOF the motion
    drives is physically simulated rather than kinematically teleported, so
    Giskard's belief of that DOF's position does not race ahead of where it has
    actually, physically settled.
    """

    previous_type: ExecutionType = field(init=False, default=None)
    """
    Type of the execution environment before setting it, used for nested environments.
    """

    previous_collision_avoidance: bool = field(init=False, default=False)
    """
    Collision avoidance setting before entering this environment, used for nested
    environments.
    """

    max_ticks_per_motion_mapping: Optional[int] = None
    """
    Per-motion tick budget applied to every motion state chart created within
    this environment. ``None`` leaves
    :py:attr:`~coraplex.plans.executables.GiskardExecutable.max_ticks_per_motion_mapping`
    unchanged.

    Worth setting whenever ``real_time_pacing`` is on: a paced tick sleeps for
    a full control period, so the default budget takes tens of seconds per
    motion mapping to give up on a stuck motion.
    """

    previous_real_time_pacing: bool = field(init=False, default=False)
    """
    Real-time pacing setting before entering this environment, used for nested
    environments.
    """

    previous_max_ticks_per_motion_mapping: int = field(init=False, default=0)
    """
    Tick budget before entering this environment, used for nested
    environments.
    """

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type` and
        :py:attr:`~pycram.plans.executables.GiskardExecutable.collision_avoidance` and
        sets them to the values of this environment.
        """
        self.previous_type = GiskardExecutable.execution_type
        self.previous_collision_avoidance = GiskardExecutable.collision_avoidance
        self.previous_real_time_pacing = GiskardExecutable.real_time_pacing
        self.previous_max_ticks_per_motion_mapping = (
            GiskardExecutable.max_ticks_per_motion_mapping
        )
        GiskardExecutable.execution_type = self.execution_type
        GiskardExecutable.collision_avoidance = self.collision_avoidance
        GiskardExecutable.real_time_pacing = self.real_time_pacing
        if self.max_ticks_per_motion_mapping is not None:
            GiskardExecutable.max_ticks_per_motion_mapping = (
                self.max_ticks_per_motion_mapping
            )

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, restores the
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type` and
        :py:attr:`~pycram.plans.executables.GiskardExecutable.collision_avoidance` to
        the previously used values.
        """
        GiskardExecutable.execution_type = self.previous_type
        GiskardExecutable.collision_avoidance = self.previous_collision_avoidance
        GiskardExecutable.real_time_pacing = self.previous_real_time_pacing
        GiskardExecutable.max_ticks_per_motion_mapping = (
            self.previous_max_ticks_per_motion_mapping
        )

    def __call__(self, collision_avoidance: bool = False):
        """
        Configure the environment for use as a context manager, allowing ``with
        simulated_robot(collision_avoidance=True):``.
        """
        self.collision_avoidance = collision_avoidance
        return self


# These are imported, so they don't have to be initialized when executing with
simulated_robot = ExecutionEnvironment(ExecutionType.SIMULATED)
real_robot = ExecutionEnvironment(ExecutionType.REAL)
semi_real_robot = ExecutionEnvironment(ExecutionType.SEMI_REAL)
no_execution = ExecutionEnvironment(ExecutionType.NO_EXECUTION)

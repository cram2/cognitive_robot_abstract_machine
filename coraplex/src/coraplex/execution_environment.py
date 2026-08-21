from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import Optional

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

    real_time_factor: Optional[float] = None
    """
    Multiple of real (wall-clock) time to pace :meth:`GiskardExecutable
    ._execute_simulation`'s tick loop to. ``None`` (the default) ticks as fast as the
    QP solver allows.
    """

    prediction_horizon: Optional[int] = None
    """
    Overrides :py:attr:`~pycram.plans.executables.GiskardExecutable
    .prediction_horizon` for this environment. ``None`` (the default) leaves it at
    whatever it already was, i.e. every existing robot's tuned value.
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

    previous_real_time_factor: Optional[float] = field(init=False, default=None)
    """
    Real time factor before entering this environment, used for nested environments.
    """

    previous_prediction_horizon: int = field(init=False, default=None)
    """
    Prediction horizon before entering this environment, used for nested environments.
    """

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type`,
        :py:attr:`~pycram.plans.executables.GiskardExecutable.collision_avoidance`,
        :py:attr:`~pycram.plans.executables.GiskardExecutable.real_time_factor`, and
        :py:attr:`~pycram.plans.executables.GiskardExecutable.prediction_horizon` and
        sets them to the values of this environment.
        """
        self.previous_type = GiskardExecutable.execution_type
        self.previous_collision_avoidance = GiskardExecutable.collision_avoidance
        self.previous_real_time_factor = GiskardExecutable.real_time_factor
        self.previous_prediction_horizon = GiskardExecutable.prediction_horizon
        GiskardExecutable.execution_type = self.execution_type
        GiskardExecutable.collision_avoidance = self.collision_avoidance
        GiskardExecutable.real_time_factor = self.real_time_factor
        if self.prediction_horizon is not None:
            GiskardExecutable.prediction_horizon = self.prediction_horizon

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, restores the
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type`,
        :py:attr:`~pycram.plans.executables.GiskardExecutable.collision_avoidance`,
        :py:attr:`~pycram.plans.executables.GiskardExecutable.real_time_factor`, and
        :py:attr:`~pycram.plans.executables.GiskardExecutable.prediction_horizon` to
        the previously used values.
        """
        GiskardExecutable.execution_type = self.previous_type
        GiskardExecutable.collision_avoidance = self.previous_collision_avoidance
        GiskardExecutable.real_time_factor = self.previous_real_time_factor
        GiskardExecutable.prediction_horizon = self.previous_prediction_horizon

    def __call__(
        self,
        collision_avoidance: bool = False,
        real_time_factor: Optional[float] = None,
        prediction_horizon: Optional[int] = None,
    ):
        """
        Configure the environment for use as a context manager, allowing ``with
        simulated_robot(collision_avoidance=True, real_time_factor=1.0,
        prediction_horizon=20):``.
        """
        self.collision_avoidance = collision_avoidance
        self.real_time_factor = real_time_factor
        self.prediction_horizon = prediction_horizon
        return self


# These are imported, so they don't have to be initialized when executing with
simulated_robot = ExecutionEnvironment(ExecutionType.SIMULATED)
real_robot = ExecutionEnvironment(ExecutionType.REAL)
semi_real_robot = ExecutionEnvironment(ExecutionType.SEMI_REAL)
no_execution = ExecutionEnvironment(ExecutionType.NO_EXECUTION)

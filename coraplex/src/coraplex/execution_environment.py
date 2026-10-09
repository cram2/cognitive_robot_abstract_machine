from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import Optional, TYPE_CHECKING

from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.multi_sim import MujocoSim

logger = logging.getLogger(__name__)


@dataclass
class ExecutionEnvironment:
    """
    Base class for managing execution context of all actions within.

    Instances of this class is to be used with a "with" context block

    Example:

        >>> with ExecutionEnvironment(ExecutionType.KINEMATICALLY_SIMULATED):
        >>>     SequentialPlan(context, NavigateActionDescription, ...)
    """

    execution_type: ExecutionType
    """
    The type of the execution environment.
    """

    collision_avoidance: bool = False
    """
    Whether the robot avoids colliding with its surroundings and with itself in every
    motion state chart created within this environment.
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

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type` and
        :py:attr:`~pycram.plans.executables.GiskardExecutable.collision_avoidance` and
        sets them to the values of this environment.
        """
        self.previous_type = GiskardExecutable.execution_type
        self.previous_collision_avoidance = GiskardExecutable.collision_avoidance
        GiskardExecutable.execution_type = self.execution_type
        GiskardExecutable.collision_avoidance = self.collision_avoidance

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, restores the
        :py:attr:`~pycram.plans.executables.GiskardExecutable.execution_type` and
        :py:attr:`~pycram.plans.executables.GiskardExecutable.collision_avoidance` to
        the previously used values.
        """
        GiskardExecutable.execution_type = self.previous_type
        GiskardExecutable.collision_avoidance = self.previous_collision_avoidance

    def __call__(self, collision_avoidance: bool = False):
        """
        Configure the environment for use as a context manager, allowing ``with
        kinematically_simulated_robot(collision_avoidance=True):``.
        """
        self.collision_avoidance = collision_avoidance
        return self


@dataclass
class PhysicallySimulatedRobot(ExecutionEnvironment):
    """
    Performs every action within it against a physically simulated world.

    The robot's joints are driven to where the controller sends them by the
    simulation's own servos, so an action that depends on contact behaves the way it
    would on hardware.

    Example:

        >>> with PhysicallySimulatedRobot(simulation=simulation):
        >>>     SequentialPlan(context, PickUpActionDescription, ...)
    """

    execution_type: ExecutionType = field(
        default=ExecutionType.PHYSICALLY_SIMULATED, init=False
    )
    """
    Always a physically simulated robot.
    """

    simulation: MujocoSim = field(kw_only=True)
    """
    The world to perform in; it has to be started with
    :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoSim.start_stepped_simulation`
    already.
    """

    previous_simulation: Optional[MujocoSim] = field(init=False, default=None)
    """
    The simulation that was being performed in before this environment was entered, used
    for nested environments.
    """

    def __enter__(self):
        """
        Hands the simulation to the executables alongside the execution type.
        """
        super().__enter__()
        self.previous_simulation = GiskardExecutable.simulation
        GiskardExecutable.simulation = self.simulation

    def __exit__(self, _type, value, traceback):
        """
        Restores the simulation that was being performed in before.
        """
        GiskardExecutable.simulation = self.previous_simulation
        super().__exit__(_type, value, traceback)


# These are imported, so they don't have to be initialized when executing with
kinematically_simulated_robot = ExecutionEnvironment(
    ExecutionType.KINEMATICALLY_SIMULATED
)
real_robot = ExecutionEnvironment(ExecutionType.REAL)
semi_real_robot = ExecutionEnvironment(ExecutionType.SEMI_REAL)
no_execution = ExecutionEnvironment(ExecutionType.NO_EXECUTION)

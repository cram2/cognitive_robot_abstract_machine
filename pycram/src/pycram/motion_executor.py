import logging
from dataclasses import dataclass, field
from typing import List, Any, ClassVar

from typing_extensions import Callable

from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.graph_node import Task
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from pycram.datastructures.enums import ExecutionType
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)


@dataclass
class MotionExecutor:
    motions: List[Task]
    """
    The motions to execute
    """

    world: World
    """
    The world in which the motions should be executed.
    """

    motion_state_chart: MotionStatechart = field(init=False)
    """
    Giskard's motion state chart that is created from the motions.
    """

    ros_node: Any = field(kw_only=True, default=None)
    """
    ROS node that should be used for communication. Only relevant for real execution.
    """

    execution_type: ClassVar[ExecutionType] = None

    def construct_msc(self):
        self.motion_state_chart = MotionStatechart()
        sequence_node = Sequence(nodes=self.motions)
        self.motion_state_chart.add_node(sequence_node)

        self.motion_state_chart.add_node(EndMotion.when_true(sequence_node))

    def execute(self):
        """
        Executes the constructed motion state chart in the given world.
        """
        # If there are no motions to construct an msc, return
        if len(self.motions) == 0:
            return
        match MotionExecutor.execution_type:
            case ExecutionType.SIMULATED:
                self._execute_for_simulation()
            case ExecutionType.REAL:
                self._execute_for_real()
            case ExecutionType.NO_EXECUTION:
                return
            case _:
                logger.error(f"Unknown execution type: {MotionExecutor.execution_type}")

    def _execute_for_simulation(self):
        """
        Creates an executor and executes the motion state chart until it is done.
        """
        logger.debug(f"Executing {self.motions} motions in simulation")
        executor = Ros2Executor(
            self.world,
            controller_config=QPControllerConfig(
                target_frequency=50, prediction_horizon=4, verbose=False
            ),
            ros_node=self.ros_node,
        )
        executor.compile(self.motion_state_chart)
        try:
            executor.tick_until_end(timeout=2000)
        except TimeoutError as e:
            failed_nodes = [
                (
                    node
                    if node.life_cycle_state
                    not in [LifeCycleValues.DONE, LifeCycleValues.NOT_STARTED]
                    else None
                )
                for node in self.motion_state_chart.nodes
            ]
            failed_nodes = list(filter(None, failed_nodes))
            logger.error(f"Failed Nodes: {failed_nodes}")
            raise e

    def _execute_for_real(self):
        from giskardpy_ros.python_interface.python_interface import GiskardWrapper

        giskard = GiskardWrapper(self.ros_node)
        giskard.execute(self.motion_state_chart)


class RealRobot:
    """
    Management class for executing designators on the real robot. This is intended to be used in a with environment.
    When importing this class an instance is imported instead.

    Example:

    .. code-block:: python

        with real_robot:
            some designators
    """

    def __init__(self):
        self.pre: ExecutionType = ExecutionType.REAL

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set :py:attr:`~MotionExecutor.execution_type` and
        sets it to 'real'
        """
        self.pre = MotionExecutor.execution_type
        MotionExecutor.execution_type = ExecutionType.REAL

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, sets the :py:attr:`~MotionExecutor.execution_type` to the previously
        used one.
        """
        MotionExecutor.execution_type = self.pre

    def __call__(self):
        return self


class SimulatedRobot:
    """
    Management class for executing designators on the simulated robot. This is intended to be used in
    a with environment. When importing this class an instance is imported instead.

    Example:

    .. code-block:: python

        with simulated_robot:
            some designators
    """

    def __init__(self):
        self.pre: ExecutionType = ExecutionType.SIMULATED

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set :py:attr:`~MotionExecutor.execution_type` and
        sets it to 'simulated'
        """
        self.pre = MotionExecutor.execution_type
        MotionExecutor.execution_type = ExecutionType.SIMULATED

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, sets the :py:attr:`~MotionExecutor.execution_type` to the previously
        used one.
        """
        MotionExecutor.execution_type = self.pre

    def __call__(self):
        return self


class SemiRealRobot:
    """
    Management class for executing designators on the semi-real robot. This is intended to be used in a with environment.
    When importing this class an instance is imported instead.

    Example:

    .. code-block:: python

        with semi_real_robot:
            some designators
    """

    def __init__(self):
        self.pre: ExecutionType = ExecutionType.SEMI_REAL

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set :py:attr:`~MotionExecutor.execution_type` and
        sets it to 'semi_real'
        """
        self.pre = MotionExecutor.execution_type
        MotionExecutor.execution_type = ExecutionType.SEMI_REAL

    def __exit__(self, type, value, traceback):
        """
        Exit method for the 'with' scope, sets the :py:attr:`~MotionExecutor.execution_type` to the previously
        used one.
        """
        MotionExecutor.execution_type = self.pre

    def __call__(self):
        return self


class NoExecution:

    def __init__(self):
        self.pre: ExecutionType = ExecutionType.SEMI_REAL

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set :py:attr:`~MotionExecutor.execution_type` and
        sets it to 'semi_real'
        """
        self.pre = MotionExecutor.execution_type
        MotionExecutor.execution_type = ExecutionType.NO_EXECUTION

    def __exit__(self, type, value, traceback):
        """
        Exit method for the 'with' scope, sets the :py:attr:`~MotionExecutor.execution_type` to the previously
        used one.
        """
        MotionExecutor.execution_type = self.pre

    def __call__(self):
        return self


def with_real_robot(func: Callable) -> Callable:
    """
    Decorator to execute designators in the decorated class on the real robot.

    Example:

    .. code-block:: python

        @with_real_robot
        def plan():
            some designators

    :param func: Function this decorator is annotating
    :return: The decorated function wrapped into the decorator
    """

    def wrapper(*args, **kwargs):
        pre = MotionExecutor.execution_type
        MotionExecutor.execution_type = ExecutionType.REAL
        ret = func(*args, **kwargs)
        MotionExecutor.execution_type = pre
        return ret

    return wrapper


def with_simulated_robot(func: Callable) -> Callable:
    """
    Decorator to execute designators in the decorated class on the simulated robot.

    Example:

    .. code-block:: python

        @with_simulated_robot
        def plan():
            some designators

    :param func: Function this decorator is annotating
    :return: The decorated function wrapped into the decorator
    """

    def wrapper(*args, **kwargs):
        pre = MotionExecutor.execution_type
        MotionExecutor.execution_type = ExecutionType.SIMULATED
        ret = func(*args, **kwargs)
        MotionExecutor.execution_type = pre
        return ret

    return wrapper


# These are imported, so they don't have to be initialized when executing with
simulated_robot = SimulatedRobot()
real_robot = RealRobot()
semi_real_robot = SemiRealRobot()
no_execution = NoExecution()

from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Any, ClassVar

from typing_extensions import TYPE_CHECKING

from giskardpy.motion_statechart.context import MotionStatechartContext
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
from pycram.exceptions import MotionDidNotFinish

from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from pycram.plans.plan_node import PlanNode

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

    plan_node: PlanNode = field(kw_only=True)
    """
    The plan node that created this executor.
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
            context=MotionStatechartContext(
                world=self.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=50, prediction_horizon=4, verbose=False
                ),
            ),
            ros_node=self.ros_node,
        )
        executor.compile(self.motion_state_chart)
        # execute the motion state chart until it is done
        counter = 0
        while counter < 2000:
            if self.plan_node.is_interrupted:
                return
            elif self.plan_node.is_paused:
                time.sleep(0.01)
                continue

            executor.tick()
            counter += 1
            time.sleep(0.01)
            if executor.motion_statechart.is_end_motion():
                break

        executor._set_velocity_acceleration_jerk_to_zero()
        executor.motion_statechart.cleanup_nodes(context=executor.context)
        executor.context.cleanup()

        if not executor.motion_statechart.is_end_motion():
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
            raise MotionDidNotFinish(failed_nodes)

    def _monitor_interrupt(self, giskard_wrapper, kill_event: threading.Event):
        while True:
            if self.plan_node.is_paused:
                raise NotImplementedError("Pause not implemented for real execution")
            elif self.plan_node.is_interrupted or kill_event.is_set():
                giskard_wrapper.cancel_goal_async()
            time.sleep(0.01)

    def _execute_for_real(self):
        from giskardpy.middleware.ros2.python_interface import GiskardWrapper

        giskard = GiskardWrapper(self.ros_node)

        kill_event = threading.Event()
        interrupt_thread = threading.Thread(
            target=self._monitor_interrupt, args=(giskard, kill_event)
        )
        interrupt_thread.start()

        giskard.execute(self.motion_state_chart)

        kill_event.set()
        interrupt_thread.join()


@dataclass
class ExecutionEnvironment:
    """
    Base class for managing execution context of all actions within. Instances of this class is to be used with a
    "with" context block

    Example:

        >>> with ExecutionEnvironment(ExecutionType.SIMULATED):
        >>>     SequentialPlan(context, NavigateActionDescription, ...)

    """

    execution_type: ExecutionType
    """
    The type of the execution environment 
    """

    previous_type: ExecutionType = field(init=False, default=None)
    """
    Type of the execution environment before setting it, used for nested environments
    """

    def __enter__(self):
        """
        Entering function for 'with' scope, saves the previously set :py:attr:`~MotionExecutor.execution_type` and
        sets it to 'real'
        """
        self.pre = MotionExecutor.execution_type
        MotionExecutor.execution_type = self.execution_type

    def __exit__(self, _type, value, traceback):
        """
        Exit method for the 'with' scope, sets the :py:attr:`~MotionExecutor.execution_type` to the previously
        used one.
        """
        MotionExecutor.execution_type = self.pre

    def __call__(self):
        return self


# These are imported, so they don't have to be initialized when executing with
simulated_robot = ExecutionEnvironment(ExecutionType.SIMULATED)
real_robot = ExecutionEnvironment(ExecutionType.REAL)
semi_real_robot = ExecutionEnvironment(ExecutionType.SEMI_REAL)
no_execution = ExecutionEnvironment(ExecutionType.NO_EXECUTION)

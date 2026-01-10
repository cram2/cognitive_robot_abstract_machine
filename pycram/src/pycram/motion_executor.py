import logging
from dataclasses import dataclass, field
from typing import List, Any

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
from pycram.process_module import ProcessModuleManager
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
        match ProcessModuleManager.execution_type:
            case ExecutionType.SIMULATED:
                self._execute_for_simulation()
            case ExecutionType.REAL:
                self._execute_for_real()
            case ExecutionType.NO_EXECUTION:
                return
            case _:
                logger.error(
                    f"Unknown execution type: {ProcessModuleManager.execution_type}"
                )

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

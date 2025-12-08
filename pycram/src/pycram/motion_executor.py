from dataclasses import dataclass, field
from typing import List, Any

from giskardpy.executor import Executor
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    LifeCycleState,
)
from giskardpy.motion_statechart.graph_node import Task
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World

from pycram.datastructures.enums import ExecutionType
from pycram.process_module import ProcessModuleManager
import logging

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
        if ProcessModuleManager.execution_type == ExecutionType.SIMULATED:
            self._execute_for_simulation()
        elif ProcessModuleManager.execution_type == ExecutionType.REAL:
            self._execute_for_real()

    def _execute_for_simulation(self):
        """
        Creates an executor and executes the motion state chart until it is done.
        """
        logger.debug(f"Executing {self.motions} motions in simulation")
        executor = Executor(
            self.world,
            controller_config=QPControllerConfig(
                target_frequency=50, prediction_horizon=4, verbose=False
            ),
        )
        executor.compile(self.motion_state_chart)
        try:
            executor.tick_until_end(timeout=2000)
        except TimeoutError as e:
            failed_nodes = [
                node if node.life_cycle_state != LifeCycleValues.DONE else None
                for node in self.motion_state_chart.nodes
            ]
            logger.error(f"Failed Nodes: {failed_nodes}")
            raise e

    def _execute_for_real(self):
        from giskardpy_ros.python_interface.python_interface import GiskardWrapper

        giskard = GiskardWrapper(self.ros_node)
        giskard.execute(self.motion_state_chart)

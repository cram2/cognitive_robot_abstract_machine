"""
Driving Robotiq grippers through the ``control_msgs/ParallelGripperCommand`` action
their drivers serve.

.. note:: Importing this module needs ``control_msgs``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from control_msgs.action import ParallelGripperCommand
from rclpy.task import Future

from cramph.data_types import ObservationStateValues
from cramph.node import StatechartNode
from krrood.patterns.called_positionally import called_positionally
from giskardpy.motion_statechart.tasks.gripper import (
    MoveFingers,
    MovesGripperToConfiguration,
)
from giskardpy.motion_statechart.ros2_nodes.action_servers import ActionServerTask
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.robots.robotiq_85_gripper import (
    HasParallelGripperCommandServer,
)

logger = logging.getLogger(__name__)

# %% the parallel gripper command action server task


@dataclass(eq=False, repr=False, kw_only=True)
class RobotiqGripperActionServerTask(
    ActionServerTask[
        ParallelGripperCommand,
        ParallelGripperCommand.Goal,
        ParallelGripperCommand.Result,
        ParallelGripperCommand.Feedback,
    ],
    MovesGripperToConfiguration,
):
    """
    Node for calling a Robotiq ROS2 action server using the
    control_msgs/ParallelGripperCommand interface.
    """

    message_type: type[ParallelGripperCommand] = field(
        default=ParallelGripperCommand, kw_only=True
    )
    """
    The ROS2 action interface type.
    """

    target_position: float
    """
    The position to bring the connection the server's controller commands to, in that
    connection's own units: radians for the knuckle joint of a Robotiq 2F-85.
    """

    target_velocity: float = 10.0
    """
    Desired gripper joint velocity in radians per second.
    """

    target_effort: float = 50.0
    """
    Maximum gripping force or effort applied by the fingers (in Newtons / effort units,
    up to 125.0).
    """

    _goal_accepted: bool = field(init=False, default=False)
    """
    Whether the server accepted the goal last sent.
    """

    _done: bool = field(init=False, default=False)
    """
    Whether the server answered the goal last sent, by rejecting it or with a result.
    """

    @classmethod
    def create_gripper_goal(
        cls,
        *,
        configuration: GripperConfiguration,
        stall_minimum_time: float | None = None,
        tolerate_stall: bool = False,
    ) -> StatechartNode:
        """
        Differs from :meth:`MovesGripperToConfiguration.create_gripper_goal` in driving
        a :class:`HasParallelGripperCommandServer` through its server, to the position
        `configuration` gives its
        :attr:`~HasParallelGripperCommandServer.commanded_connection`; a stall always
        counts as done.

        Any other configuration is reached as :meth:`MoveFingers.create_gripper_goal`
        reaches it.

        :raises ConnectionNotInJointStateError: If `configuration` sets no position for
            the commanded connection.
        """
        end_effector = configuration.end_effector
        if not isinstance(end_effector, HasParallelGripperCommandServer):
            return MoveFingers.create_gripper_goal(
                configuration=configuration,
                stall_minimum_time=stall_minimum_time,
                tolerate_stall=tolerate_stall,
            )
        return cls(
            action_topic=end_effector.action_topic_serving(
                configuration.joint_state.state_type
            ),
            target_position=configuration.joint_state.target_value(
                end_effector.commanded_connection
            ),
        )

    def build_action_goal(self) -> None:
        self._action_goal = ParallelGripperCommand.Goal()
        self._action_goal.command.position = [float(self.target_position)]
        self._action_goal.command.velocity = [float(self.target_velocity)]
        self._action_goal.command.effort = [float(self.target_effort)]
        self._goal_accepted = False
        self._done = False

    @called_positionally
    def goal_response_callback(self, future: Future) -> None:
        """
        Differs from :meth:`ActionServerTask.goal_response_callback` in recording
        whether the server accepted the goal.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.error("Gripper goal rejected by action server")
            self._goal_accepted = False
            self._done = True
            return

        self._goal_accepted = True
        logger.info("Gripper goal accepted, waiting for result...")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    @called_positionally
    def result_callback(self, future: Future) -> None:
        """
        Differs from :meth:`ActionServerTask.result_callback` in marking the goal as
        answered.
        """
        super().result_callback(future=future)
        self._done = True
        logger.info(
            f"Gripper action finished. "
            f"Reached goal: {self._result.result.reached_goal}, "
            f"Stalled: {self._result.result.stalled}"
        )

    def on_tick(self) -> ObservationStateValues:
        """
        :return: ``TRUE`` once the server reported the fingers at the goal or stalled
            on an object or their limit, ``FALSE`` once it rejected the goal or
            reported neither, and ``UNKNOWN`` until it answered.
        """
        if not self._done:
            return ObservationStateValues.UNKNOWN
        if not self._goal_accepted:
            return ObservationStateValues.FALSE
        action_result = self._result.result
        if action_result.reached_goal or action_result.stalled:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

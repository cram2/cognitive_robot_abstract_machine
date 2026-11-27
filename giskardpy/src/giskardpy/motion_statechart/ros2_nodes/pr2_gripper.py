"""
Driving the PR2's grippers through the ``pr2_controllers_msgs/Pr2GripperCommand`` action
their drivers serve.

.. note:: Importing this module needs ``pr2_controllers_msgs``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pr2_controllers_msgs.action import Pr2GripperCommand
from rclpy.task import Future

from cramph.data_types import ObservationStateValue
from cramph.node import StatechartNode
from giskardpy.motion_statechart.ros2_nodes.action_servers import ActionServerTask
from giskardpy.motion_statechart.tasks.gripper import (
    MoveFingers,
    MovesGripperToConfiguration,
)
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.robots.pr2 import PR2Gripper

logger = logging.getLogger(__name__)

# %% the PR2 gripper command action server task


@dataclass(eq=False, repr=False, kw_only=True)
class PR2GripperActionServerTask(
    ActionServerTask[
        Pr2GripperCommand,
        Pr2GripperCommand.Goal,
        Pr2GripperCommand.Result,
        Pr2GripperCommand.Feedback,
    ],
    MovesGripperToConfiguration,
):
    """
    Moves a PR2 gripper to a gap between its fingertips through the action its driver
    serves, ending once the driver reports the gap reached or the fingers stalled.
    """

    message_type: type[Pr2GripperCommand] = field(
        default=Pr2GripperCommand, kw_only=True
    )
    """
    The ROS2 action interface type.
    """

    target_gap: float
    """
    The gap between the fingertips to move to, in metres.
    """

    max_effort: float = 50.0
    """
    The highest effort the fingers squeeze with, in Newtons; a negative value leaves it
    unlimited.
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
        finger_stall_minimum_time: float | None = None,
        tolerate_finger_stall: bool = False,
    ) -> StatechartNode:
        """
        Differs from :meth:`MovesGripperToConfiguration.create_gripper_goal` in driving
        a :class:`PR2Gripper` through its action, to the gap
        :meth:`PR2Gripper.target_gap` gives for `configuration`; a stall always counts
        as done.

        Any other configuration is reached as :meth:`MoveFingers.create_gripper_goal`
        reaches it.
        """
        end_effector = configuration.end_effector
        if not isinstance(end_effector, PR2Gripper):
            return MoveFingers.create_gripper_goal(
                configuration=configuration,
                finger_stall_minimum_time=finger_stall_minimum_time,
                tolerate_finger_stall=tolerate_finger_stall,
            )
        return cls(
            action_topic=end_effector.action_topic_serving(
                configuration.joint_state.state_type
            ),
            target_gap=end_effector.target_gap(joint_state=configuration.joint_state),
        )

    def build_action_goal(self) -> None:
        self._action_goal = Pr2GripperCommand.Goal()
        self._action_goal.command.position = float(self.target_gap)
        self._action_goal.command.max_effort = float(self.max_effort)

    def send_goal(self) -> None:
        """
        Differs from :meth:`ActionServerTask.send_goal` in forgetting whether the server
        accepted and answered the goal sent before.
        """
        self._goal_accepted = False
        self._done = False
        super().send_goal()

    def record_goal_response(self, *, accepted: bool) -> None:
        """
        Differs from :meth:`ActionServerTask.record_goal_response` in keeping whether
        the server accepted the goal, and counting a rejected goal as answered.
        """
        super().record_goal_response(accepted=accepted)
        self._goal_accepted = accepted
        self._done = not accepted

    def result_callback(self, *, future: Future) -> None:
        """
        Differs from :meth:`ActionServerTask.result_callback` in marking the goal as
        answered.
        """
        super().result_callback(future=future)
        self._done = True
        logger.info(
            f"Gripper action finished at a gap of {self._result.result.position} m. "
            f"Reached goal: {self._result.result.reached_goal}, "
            f"Stalled: {self._result.result.stalled}"
        )

    def on_tick(self) -> ObservationStateValue:
        """
        :return: ``TRUE`` once the server reported the gap reached or the fingers
            stalled on an object or their limit, ``FALSE`` once it rejected the goal,
            aborted it or reported neither, and ``UNKNOWN`` until it answered.
        """
        if not self._done:
            return ObservationStateValue.UNKNOWN
        if not self._goal_accepted:
            return ObservationStateValue.FALSE
        action_result = self._result.result
        if action_result.reached_goal or action_result.stalled:
            return ObservationStateValue.TRUE
        return ObservationStateValue.FALSE

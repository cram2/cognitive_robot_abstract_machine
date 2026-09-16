from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict

from json_msgs.action import JsonAction

from giskardpy.executor import Executor
from giskardpy.middleware.ros2.action_server import ActionServerHandler


class MotionStatechartPayloadKey(StrEnum):
    """
    The keys of the motion statechart feedback and result sent to the action client.
    """

    LIFE_CYCLE_STATE = "life_cycle_state"
    """
    The life cycle state of every node.
    """

    OBSERVATION_STATE = "observation_state"
    """
    The observation state of every node.
    """

    LAST_OBSERVATION_STATE = "last_observation_state"
    """
    The observation every node took most recently.
    """

    MOTION_STATECHART = "motion_statechart"
    """
    The structure of the motion statechart, sent once per goal.
    """

    GOAL_ID = "goal_id"
    """
    The goal the feedback belongs to.
    """


@dataclass
class ActionFeedbackPublisher:
    """
    Reports the state of the running motion statechart to the action client.
    """

    executor: Executor
    """
    The executor holding the motion statechart that is reported on.
    """

    action_server: ActionServerHandler
    """
    The action server the feedback is sent through.
    """

    last_history_length: int = field(init=False, default=-1)
    """
    Length of the statechart history at the most recent feedback, used to detect state
    changes.
    """

    def publish_structure(self) -> None:
        """
        Send the state of the motion statechart together with its structure.

        Serializing the structure is expensive, so it is sent once while the goal is
        compiled instead of from inside a control cycle.
        """
        if self.executor.motion_statechart is None:
            return
        data = self.create_states()
        data[MotionStatechartPayloadKey.MOTION_STATECHART] = (
            self.executor.motion_statechart.create_structure_copy().to_json()
        )
        data[MotionStatechartPayloadKey.GOAL_ID] = self.action_server.goal_id
        self.last_history_length = len(self.executor.motion_statechart.history)
        self.send(data)

    def publish_if_changed(self) -> None:
        """
        Send feedback only when the state of the motion statechart changed.
        """
        if self.executor.motion_statechart is None:
            return
        if not self.has_state_changed():
            return
        data = self.create_states()
        data[MotionStatechartPayloadKey.GOAL_ID] = self.action_server.goal_id
        self.send(data)

    def publish(self) -> None:
        """
        Send feedback regardless of whether anything changed.
        """
        if self.executor.motion_statechart is None:
            return
        data = self.create_states()
        data[MotionStatechartPayloadKey.GOAL_ID] = self.action_server.goal_id
        self.send(data)

    def create_states(self) -> Dict[str, Any]:
        """
        Collect the life cycle, observation and last observation state of the motion
        statechart.
        """
        motion_statechart = self.executor.motion_statechart
        return {
            MotionStatechartPayloadKey.LIFE_CYCLE_STATE: motion_statechart.life_cycle_state.to_json(),
            MotionStatechartPayloadKey.OBSERVATION_STATE: motion_statechart.observation_state.to_json(),
            MotionStatechartPayloadKey.LAST_OBSERVATION_STATE: motion_statechart.last_observation_state.to_json(),
        }

    def has_state_changed(self) -> bool:
        """
        Whether the statechart history grew since the last feedback.
        """
        history_length = len(self.executor.motion_statechart.history)
        has_changed = self.last_history_length != history_length
        if has_changed:
            self.last_history_length = history_length
        return has_changed

    def send(self, data: Dict[str, Any]) -> None:
        """
        Publish the given data as action feedback.
        """
        message = JsonAction.Feedback()
        message.feedback = json.dumps(data)
        self.action_server.send_feedback(message)

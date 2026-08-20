from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

from json_msgs.action import JsonAction

from giskardpy.executor import Executor
from giskardpy.middleware.ros2.action_server import ActionServerHandler


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
        data["motion_statechart"] = (
            self.executor.motion_statechart.create_structure_copy().to_json()
        )
        data["goal_id"] = self.action_server.goal_id
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
        data["goal_id"] = self.action_server.goal_id
        self.send(data)

    def publish(self) -> None:
        """
        Send feedback regardless of whether anything changed.
        """
        if self.executor.motion_statechart is None:
            return
        data = self.create_states()
        data["goal_id"] = self.action_server.goal_id
        self.send(data)

    def create_states(self) -> Dict[str, Any]:
        """
        Collect the life cycle and observation state of the motion statechart.
        """
        return {
            "life_cycle_state": self.executor.motion_statechart.life_cycle_state.to_json(),
            "observation_state": self.executor.motion_statechart.observation_state.to_json(),
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

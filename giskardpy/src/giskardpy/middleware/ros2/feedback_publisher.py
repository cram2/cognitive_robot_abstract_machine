from __future__ import annotations

from cramph.data_types import StatechartPayloadKey
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum

from json_msgs.action import JsonAction
from typing_extensions import TYPE_CHECKING, Any

from cramph.composites import CompositeNodeChoosingItsChild
from cramph.executor import StatechartExecutor
from giskardpy.middleware.ros2.action_server import ActionServerHandler
from krrood.adapters.json_serializer import to_json

if TYPE_CHECKING:
    from giskardpy.middleware.ros2.world_updates import OutgoingWorldUpdates


class MotionStatechartPayloadKey(StrEnum):
    """
    The keys of the motion statechart feedback sent to the action client on top of the
    :class:`~cramph.data_types.StatechartPayloadKey`.
    """

    WAITING_FOR_CHILD = "waiting_for_child"
    """
    The nodes waiting for the client to choose their child, by index, each with the
    number of children it already holds.
    """

    PUBLISHED_POSITION = "published_position"
    """
    The position Giskard's world published its changes up to while the goal ran, which
    the client's world has to contain before the client reads it; left out if the goal
    published nothing.
    """


@dataclass
class ActionFeedbackPublisher:
    """
    Reports the state of the running motion statechart to the action client.
    """

    executor: StatechartExecutor
    """
    The executor holding the motion statechart that is reported on.
    """

    action_server: ActionServerHandler
    """
    The action server the feedback is sent through.
    """

    outgoing_world_updates: OutgoingWorldUpdates
    """
    Tells which changes of Giskard's world the running goal published.
    """

    child_request_repeat_period: float = 0.1
    """
    Seconds after which the feedback is sent again while a node waits for a child, so
    that a client that could not choose yet is asked again.
    """

    last_history_length: int = field(init=False, default=-1)
    """
    Length of the statechart history at the most recent feedback, used to detect state
    changes.
    """

    sent_node_count: int = field(init=False, default=0)
    """
    How many nodes of the statechart the feedback sent the structure of so far.
    """

    _last_sent_time: float = field(init=False, default=-float("inf"))
    """
    The :func:`time.monotonic` time the most recent feedback was sent at.
    """

    def publish_structure(self) -> None:
        """
        Send the state of the motion statechart together with its structure.

        Serializing the structure is expensive, so it is sent once while the goal is
        compiled instead of from inside a control cycle.
        """
        if self.executor.statechart is None:
            return
        data = self.create_states()
        data[StatechartPayloadKey.STATECHART] = (
            self.executor.statechart.create_structure_copy().to_json()
        )
        data[StatechartPayloadKey.GOAL_ID] = self.action_server.goal_id
        self.last_history_length = len(self.executor.statechart.state_history)
        self.sent_node_count = len(self.executor.statechart.nodes)
        self.send(data)

    def publish_if_changed(self) -> None:
        """
        Send feedback when the state of the motion statechart changed, and every
        :attr:`child_request_repeat_period` while a node waits for a child.
        """
        if self.executor.statechart is None:
            return
        if not self.has_state_changed() and not self._is_child_request_due():
            return
        self.publish()

    def _is_child_request_due(self) -> bool:
        """
        :return: Whether a node waits for a child and :attr:`child_request_repeat_period`
            passed since the most recent feedback.
        """
        if not self._nodes_waiting_for_a_child():
            return False
        return (
            time.monotonic() - self._last_sent_time >= self.child_request_repeat_period
        )

    def _nodes_waiting_for_a_child(self) -> list[CompositeNodeChoosingItsChild]:
        """
        :return: The nodes of the motion statechart waiting for the client to choose
            their child.
        """
        return [
            node
            for node in self.executor.statechart.get_nodes_by_type(
                CompositeNodeChoosingItsChild
            )
            if node.is_waiting_for_a_child
        ]

    def publish(self) -> None:
        """
        Send feedback regardless of whether anything changed, with the structure of the
        nodes that joined the statechart since the structure was sent last.
        """
        statechart = self.executor.statechart
        if statechart is None:
            return
        data = self.create_states()
        data[StatechartPayloadKey.GOAL_ID] = self.action_server.goal_id
        if len(statechart.nodes) > self.sent_node_count:
            data[StatechartPayloadKey.JOINED_NODES] = (
                statechart.create_structure_copy().nodes_from_to_json(
                    first_node_index=self.sent_node_count
                )
            )
            self.sent_node_count = len(statechart.nodes)
        self.send(data)

    def create_states(self) -> dict[str, Any]:
        """
        Collect the life cycle, observation and last observation state of the motion
        statechart, the tick they were taken at, the nodes waiting for a child, and how
        far the changes of the goal were published.
        """
        motion_statechart = self.executor.statechart
        states = {
            StatechartPayloadKey.TICK_COUNT: self.executor.tick_count,
            StatechartPayloadKey.LIFE_CYCLE_STATE: motion_statechart.life_cycle_state.to_json(),
            StatechartPayloadKey.OBSERVATION_STATE: motion_statechart.observation_state.to_json(),
            StatechartPayloadKey.LAST_OBSERVATION_STATE: motion_statechart.last_observation_state.to_json(),
            MotionStatechartPayloadKey.WAITING_FOR_CHILD: {
                str(node.index): len(node.children)
                for node in self._nodes_waiting_for_a_child()
            },
        }
        published_position = self.outgoing_world_updates.published_position_of_goal()
        if published_position is not None:
            states[MotionStatechartPayloadKey.PUBLISHED_POSITION] = to_json(
                published_position
            )
        return states

    def has_state_changed(self) -> bool:
        """
        Whether the statechart history grew since the last feedback.
        """
        history_length = len(self.executor.statechart.state_history)
        has_changed = self.last_history_length != history_length
        if has_changed:
            self.last_history_length = history_length
        return has_changed

    def send(self, data: dict[str, Any]) -> None:
        """
        Publish the given data as action feedback.
        """
        message = JsonAction.Feedback()
        message.feedback = json.dumps(data)
        self.action_server.send_feedback(message)
        self._last_sent_time = time.monotonic()

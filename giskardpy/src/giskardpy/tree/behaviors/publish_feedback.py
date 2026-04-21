import json
from typing import Optional

from json_msgs.action import JsonAction
from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard


class PublishFeedback(GiskardBehavior):

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)
        self.move_action_server = GiskardBlackboard().move_action_server
        self.last_goal_id = -1
        self.last_history_length = -1

    def has_state_changed(self):
        history_length = len(GiskardBlackboard().motion_statechart.history)
        has_changed = self.last_history_length != history_length
        if has_changed:
            self.last_history_length = history_length
        return has_changed

    def has_new_goal(self):
        return self.last_goal_id != self.move_action_server.goal_id

    @record_time
    def update(self):
        data = {}
        if self.has_new_goal():
            self.last_goal_id = self.move_action_server.goal_id
            data["motion_statechart"] = (
                GiskardBlackboard().motion_statechart.create_structure_copy().to_json()
            )
        data["goal_id"] = self.last_goal_id

        data["life_cycle_state"] = (
            GiskardBlackboard().motion_statechart.life_cycle_state.to_json()
        )
        data["observation_state"] = (
            GiskardBlackboard().motion_statechart.observation_state.to_json()
        )

        if self.has_state_changed() or self.has_new_goal():
            msg = JsonAction.Feedback()
            msg.feedback = json.dumps(data)
            self.move_action_server.send_feedback(msg)
        return Status.SUCCESS


class ForcePublishFeedback(GiskardBehavior):

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)
        self.move_action_server = GiskardBlackboard().move_action_server

    @record_time
    def update(self):
        data = {
            "goal_id": self.move_action_server.goal_id,
            "life_cycle_state": GiskardBlackboard().motion_statechart.life_cycle_state.to_json(),
            "observation_state": GiskardBlackboard().motion_statechart.observation_state.to_json(),
        }

        msg = JsonAction.Feedback()
        msg.feedback = json.dumps(data)
        self.move_action_server.send_feedback(msg)
        return Status.SUCCESS

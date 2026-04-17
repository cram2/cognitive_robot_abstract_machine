import json
from typing import Union

from json_msgs.action import JsonAction
from py_trees.common import Status

from giskardpy.motion_statechart.goals.base_traj_follower import BaseTrajFollower
from giskardpy.motion_statechart.monitors.monitors import (
    LocalMinimumReached,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world_description.connections import OmniDrive


class ParseActionGoal(GiskardBehavior):
    @record_time
    def __init__(self, name):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        move_goal: JsonAction.Goal = GiskardBlackboard().move_action_server.goal_msg
        rospy.node.get_logger().info(
            f"Parsing goal #{GiskardBlackboard().move_action_server.goal_id} message."
        )
        tracker = WorldEntityWithIDKwargsTracker.from_world(
            GiskardBlackboard().executor.context.world
        )
        kwargs = tracker.create_kwargs()
        kwargs["world"] = GiskardBlackboard().executor.context.world
        motion_statechart = MotionStatechart.from_json(
            json.loads(move_goal.goal), **kwargs
        )
        GiskardBlackboard().executor.compile(motion_statechart)
        rospy.node.get_logger().info("Done parsing goal message.")
        return Status.SUCCESS


def get_ros_msgs_constant_name_by_value(
    ros_msg_class, value: Union[str, int, float]
) -> str:
    for attr_name in dir(ros_msg_class):
        if not attr_name.startswith("_"):
            attr_value = getattr(ros_msg_class, attr_name)
            if attr_value == value:
                return attr_name
    raise AttributeError(
        f"Message type {ros_msg_class} has no constant that matches {value}."
    )


class SetExecutionMode(GiskardBehavior):
    @record_time
    def __init__(self, name: str = "set execution mode"):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        # rospy.node.get_logger().info(
        #     f"Goal is of type {get_ros_msgs_constant_name_by_value(type(GiskardBlackboard().move_action_server.goal_msg))}"
        # )
        # if GiskardBlackboard().move_action_server.is_goal_msg_type_projection():
        #     GiskardBlackboard().tree.switch_to_projection()
        # elif GiskardBlackboard().move_action_server.is_goal_msg_type_execute():
        #     GiskardBlackboard().tree.switch_to_execution()
        # else:
        #     raise InvalidGoalException(
        #         f"Goal of type {god_map.goal_msg.type} is not supported."
        #     )
        return Status.SUCCESS


class AddBaseTrajFollowerGoal(GiskardBehavior):
    def __init__(self, name: str = "add base traj goal"):
        super().__init__(name)
        joints = GiskardBlackboard().executor.context.world.get_connections_by_type(
            OmniDrive
        )
        assert len(joints) == 1
        self.joint = joints[0]

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        local_min = LocalMinimumReached("local min")
        god_map.motion_statechart_manager.add_monitor(local_min)

        time_monitor = TimeAbove(
            threshold=len(GiskardBlackboard().trajectory)
            * GiskardBlackboard().executor.qp_controller.config.mpc_dt,
            name="timeout",
        )
        god_map.motion_statechart_manager.add_monitor(time_monitor)

        end_motion = EndMotion(name="end motion")
        end_motion.start_condition = f"{local_min.name} and {time_monitor.name}"
        god_map.motion_statechart_manager.add_monitor(end_motion)

        goal = BaseTrajFollower(
            connection=self.joint.name,
            track_only_velocity=True,
            name="BaseTrajFollower",
        )
        goal.end_condition = f"{local_min.name}"
        goal.apply_end_condition_to_nodes(time_monitor.name)
        god_map.motion_statechart_manager.add_goal(goal)
        god_map.motion_statechart_manager.parse_conditions()
        return Status.SUCCESS

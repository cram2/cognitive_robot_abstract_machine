#!/usr/bin/env python
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState


def main():
    rospy.init_node("giskard_client")
    giskard = GiskardWrapper(node_handle=rospy.get_node())
    goal_positions_in_radians = {"shoulder_pan_joint": 0.5, "elbow_joint": 1.0}

    motion_statechart = MotionStatechart()
    motion_statechart.add_node(
        joint_goal := JointPositionList(
            goal_state=JointState.from_str_dict(
                goal_positions_in_radians, giskard.world
            )
        )
    )
    motion_statechart.add_node(EndMotion.when_true(joint_goal))
    giskard.execute(motion_statechart)

    rospy.shutdown()


if __name__ == "__main__":
    main()

from typing import Tuple, Dict

from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
)
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF


class SyncTfFrames(GiskardBehavior):
    joint_map: Dict[Connection6DoF, Tuple[str, str]]

    def __init__(self, name):
        super().__init__(name)
        self.joint_map = {}
        self.tf = TFWrapper(node=rospy.node)

    def sync_6dof_joint_with_tf_frame(
        self, joint: Connection6DoF, tf_parent_frame: str, tf_child_frame: str
    ):
        if joint in self.joint_map:
            raise AttributeError(
                f"Joint '{joint.name}' is already being tracking with a tf frame: "
                f"'{self.joint_map[joint][0]}'<-'{self.joint_map[joint][1]}'"
            )
        if not isinstance(joint, Connection6DoF):
            raise AttributeError(
                f"Can only sync Connection6DoF with tf but '{joint.name}' is of type '{type(joint)}'."
            )
        self.joint_map[joint] = (tf_parent_frame, tf_child_frame)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        for joint, (tf_parent_frame, tf_child_frame) in self.joint_map.items():
            parent_T_child = self.tf.lookup_pose(tf_parent_frame, tf_child_frame)
            joint.origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=parent_T_child.pose.position.x,
                pos_y=parent_T_child.pose.position.y,
                pos_z=parent_T_child.pose.position.z,
                quat_w=parent_T_child.pose.orientation.w,
                quat_x=parent_T_child.pose.orientation.x,
                quat_y=parent_T_child.pose.orientation.y,
                quat_z=parent_T_child.pose.orientation.z,
            )

        return Status.SUCCESS

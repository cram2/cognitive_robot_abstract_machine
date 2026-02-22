from __future__ import division

from dataclasses import field, dataclass
from typing import Optional, Type, Tuple

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Connection
from ..context import MotionStatechartContext
from ..exceptions import NodeInitializationError
from ..graph_node import MotionStatechartNode, NodeArtifacts
from ..tasks.joint_tasks import JointState


@dataclass(eq=False, repr=False)
class SetSeedConfiguration(MotionStatechartNode):
    """
    Overwrite the configuration of the world to allow starting the planning from a different state.
    CAUTION! don't use this to overwrite the robot's state outside standalone mode!
    :param seed_configuration: maps joint name to float
    :param group_name: if joint names are not unique, it will search in this group for matches.
    """

    seed_configuration: JointState = field(kw_only=True)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())

    def on_start(self, context: MotionStatechartContext):
        # TODO does notify state change too often
        for connection, value in self.seed_configuration.items():
            connection.position = value


@dataclass(eq=False, repr=False)
class SetOdometry(MotionStatechartNode):
    """
    Sets the odometry of the robot to the given pose.
    """

    base_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    """The pose of the robot base."""
    odom_connection: Optional[OmniDrive] = field(default=None, kw_only=True)
    """
    The odometry connection to use. 
    If it is None and there is only one drive in the world, it will be used.
    """
    _odom_joints: Tuple[Type[Connection], ...] = field(default=(OmniDrive,), init=False)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        if self.odom_connection is None:
            drive_connections = context.world.get_connections_by_type(self._odom_joints)
            if len(drive_connections) == 0:
                raise NodeInitializationError(
                    node=self, reason="No drive joints in world"
                )
            elif len(drive_connections) == 1:
                self.odom_connection = drive_connections[0]
            else:
                raise NodeInitializationError(
                    node=self,
                    reason="Multiple drive joint found in world, please set 'group_name'",
                )
        return NodeArtifacts(observation=sm.Scalar.const_true())

    def on_start(self, context: MotionStatechartContext):
        parent_T_pose_ref = HomogeneousTransformationMatrix(
            context.world.compute_forward_kinematics_np(
                self.odom_connection.parent, self.base_pose.reference_frame
            )
        )
        parent_T_pose = parent_T_pose_ref @ self.base_pose

        self.odom_connection.origin = parent_T_pose

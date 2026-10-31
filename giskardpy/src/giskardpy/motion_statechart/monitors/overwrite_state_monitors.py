from __future__ import division

from dataclasses import field, dataclass
from typing_extensions import Self

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Connection
from cramph.node import SucceedsOnObservingTrue
from giskardpy.motion_statechart.exceptions import UnexpectedWorldEntityCountError
from giskardpy.motion_statechart.goals.base_pose_goals import MobileBasePoseGoal
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from cramph.node import NodeArtifacts
from giskardpy.motion_statechart.tasks.joint_tasks import JointState


@dataclass(eq=False, repr=False, kw_only=True)
class SetSeedConfiguration(SucceedsOnObservingTrue, MotionStatechartNode):
    """
    Overwrite the configuration of the world to allow starting the planning from a
    different state.

    CAUTION! don't use this to overwrite the robot's state outside standalone mode!
    :param seed_configuration: maps joint name to float
    :param group_name: if joint names are not unique, it will search in this group for
        matches.
    """

    seed_configuration: JointState

    def build_artifacts(self) -> NodeArtifacts:
        """
        Applying the configuration is what this node is for and it cannot fail, so it
        counts as succeeded from the moment it runs.
        """
        return NodeArtifacts(observation=sm.Scalar.const_true())

    def on_start(self) -> None:
        self.seed_configuration.apply_to(self.context.world)


@dataclass(eq=False, repr=False, kw_only=True)
class SetOdometry(SucceedsOnObservingTrue, MotionStatechartNode, MobileBasePoseGoal):
    """
    Sets the odometry of the robot to the given pose.
    """

    base_pose: HomogeneousTransformationMatrix
    """
    The pose of the robot base.
    """

    odom_connection: OmniDrive | None = None
    """
    The odometry connection to use.

    If it is None and there is only one drive in the world, it will be used.
    """

    _odom_joints: tuple[type[Connection], ...] = field(default=(OmniDrive,), init=False)
    """
    The kinds of drive searched for when :attr:`odom_connection` is None.
    """

    @classmethod
    def moving_base_of(
        cls, robot: AbstractRobot, goal_pose: Pose, threshold: float
    ) -> Self:
        """
        Differs from :meth:`MobileBasePoseGoal.moving_base_of` in setting the odometry
        of the drive attaching `robot` to its parent, which puts the base exactly at
        `goal_pose` at once, so `threshold` is ignored.
        """
        return cls(
            base_pose=goal_pose.homogeneous_matrix,
            odom_connection=robot.root.parent_connection,
        )

    def build_artifacts(self) -> NodeArtifacts:
        self.odom_connection = self._current_odom_connection()
        return NodeArtifacts(observation=sm.Scalar.const_true())

    def _current_odom_connection(self) -> OmniDrive:
        """
        The drive that moves the body this node sets the odometry of, as the world holds
        it now.

        A drive given at construction time is re-read from its child body, because a
        world model change between then and now (re-parenting the body onto a carrier,
        say) replaces the connection object while keeping the body.

        :return: The drive to write the odometry into.
        """
        if self.odom_connection is not None:
            return self.odom_connection.child.parent_connection

        drive_connections = self.context.world.get_connections_by_type(
            self._odom_joints
        )
        if len(drive_connections) != 1:
            raise UnexpectedWorldEntityCountError(
                node=self,
                expected_count=1,
                actual_count=len(drive_connections),
                entity_type=self._odom_joints,
            )
        return drive_connections[0]

    def on_start(self) -> None:
        parent_T_pose_ref = HomogeneousTransformationMatrix(
            self.context.world.compute_forward_kinematics_np(
                self.odom_connection.parent, self.base_pose.reference_frame
            )
        )
        parent_T_pose = parent_T_pose_ref @ self.base_pose

        self.odom_connection.origin = parent_T_pose

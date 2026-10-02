from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from typing_extensions import Dict, List, Tuple, Union

from semantic_digital_twin.adapters.ros.latest_message_subscriber import (
    LatestMessageSubscriber,
    MessageType,
)
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.exceptions import (
    AlreadyTrackedByTfFrameError,
    ConnectionCannotBeTrackedByTfFrameError,
)
from semantic_digital_twin.input_synchronization import InputSynchronizer
from semantic_digital_twin.robots.input_source import (
    BasePoseSource,
    JointPositionSource,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    DifferentialDrive,
    OmniDrive,
)

# %% base classes


@dataclass
class TopicInputSynchronizer(
    LatestMessageSubscriber[MessageType], InputSynchronizer, ABC
):
    """
    Applies the latest message of a topic on demand.
    """

    def apply(self) -> bool:
        message = self.take_message()
        if message is None:
            return False
        self.apply_message(message)
        return True

    def take_message(self) -> MessageType | None:
        """
        The message to write in this cycle, or ``None`` when there is nothing to write.
        """
        return self.latest_message

    @abstractmethod
    def apply_message(self, message: MessageType) -> None:
        """
        Write the message into the world state.
        """


# %% joint states


@dataclass
class JointStateInputSynchronizer(TopicInputSynchronizer[JointState], ABC):
    """
    Writes the positions of a joint state message into the world state.
    """

    def apply_message(self, message: JointState) -> None:
        for joint_name, position in zip(message.name, message.position):
            connection: ActiveConnection1DOF = self.world.get_connection_by_name(
                joint_name
            )
            if not self.writes(connection):
                continue
            self.world.state[connection.raw_dof.id].position = position

    def writes(self, connection: ActiveConnection1DOF) -> bool:
        """
        Whether the position this synchronizer received for a connection is its to
        write.

        :param connection: The connection the message reports a position for.
        """
        return True

    @abstractmethod
    def take_message(self) -> JointState | None:
        """
        The message to write in this cycle, or ``None`` when there is nothing to write.

        Each joint state synchronizer decides here whether writing a message consumes
        it.
        """


@dataclass
class PendingJointStateSynchronizer(JointStateInputSynchronizer):
    """
    Writes every joint state message exactly once, leaving nothing pending.

    Reports that it wrote nothing in cycles without a new message, so that the world
    state is not announced for positions the observers already know.
    """

    def take_message(self) -> JointState | None:
        message = self.latest_message
        self.latest_message = None
        return message


@dataclass
class LatestJointStateSynchronizer(JointStateInputSynchronizer):
    """
    Writes the most recent joint state message in every cycle, however old it is.

    Keeps the world state on the last measurement of the robot even when the cycle
    itself moved the state away from it, as a control cycle does when it integrates the
    commanded velocities.
    """

    def take_message(self) -> JointState | None:
        return self.latest_message


@dataclass
class SubscribedJointPositionSource(
    JointPositionSource, JointStateInputSynchronizer, ABC
):
    """
    The joint positions one robot part reports on a ROS 2 topic.

    A robot publishes all of its joints on one topic, so this source writes only the
    connections of the part it is attached to and leaves the rest of the robot to
    whatever reads it.
    """

    connections: List[ActiveConnection1DOF] = field(kw_only=True)
    """
    The connections of the part this source is attached to.
    """

    def writes(self, connection: ActiveConnection1DOF) -> bool:
        return connection in self.connections


@dataclass
class PendingJointPositionSource(
    SubscribedJointPositionSource, PendingJointStateSynchronizer
):
    """
    The joint positions of one robot part, written once per message the robot publishes.
    """

    def rewriting_every_cycle(self) -> LatestJointPositionSource:
        return LatestJointPositionSource(
            world=self.world,
            node=self.node,
            topic_name=self.topic_name,
            connections=self.connections,
        )


@dataclass
class LatestJointPositionSource(
    SubscribedJointPositionSource, LatestJointStateSynchronizer
):
    """
    The joint positions of one robot part, written again in every cycle however old the
    last message is.
    """


# %% base pose


@dataclass
class OdometrySynchronizer(TopicInputSynchronizer[Odometry]):
    """
    Writes the pose of an odometry message into a drive connection.
    """

    connection: Union[OmniDrive, DifferentialDrive] = field(kw_only=True)
    """
    The drive connection whose origin follows the odometry.
    """

    def apply_message(self, message: Odometry) -> None:
        pose = message.pose.pose
        self.connection.origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=pose.position.x,
            pos_y=pose.position.y,
            pos_z=pose.position.z,
            quat_w=pose.orientation.w,
            quat_x=pose.orientation.x,
            quat_y=pose.orientation.y,
            quat_z=pose.orientation.z,
        )


@dataclass
class SubscribedBasePoseSource(BasePoseSource, OdometrySynchronizer):
    """
    The pose a real mobile base reports as odometry on a ROS 2 topic.
    """


@dataclass
class TfFrameSynchronizer(InputSynchronizer, HasROS2Node):
    """
    Writes tf transforms into 6 degree of freedom connections.
    """

    connection_to_frames: Dict[Connection6DoF, Tuple[str, str]] = field(
        init=False, default_factory=dict
    )
    """
    Maps each tracked connection to its tf parent and child frame.
    """

    tf_wrapper: TFWrapper = field(init=False)
    """
    Provides the tf lookups.
    """

    def __post_init__(self):
        self.tf_wrapper = TFWrapper(node=self.node)

    def track(
        self, connection: Connection6DoF, tf_parent_frame: str, tf_child_frame: str
    ) -> None:
        """
        Make the origin of ``connection`` follow the transform between the two frames.

        :raises AlreadyTrackedByTfFrameError: If the connection is already tracked.
        :raises ConnectionCannotBeTrackedByTfFrameError: If the connection has not
            exactly 6 degrees of freedom.
        """
        if connection in self.connection_to_frames:
            raise AlreadyTrackedByTfFrameError(
                connection_name=str(connection.name),
                tf_parent_frame=self.connection_to_frames[connection][0],
                tf_child_frame=self.connection_to_frames[connection][1],
            )
        if not isinstance(connection, Connection6DoF):
            raise ConnectionCannotBeTrackedByTfFrameError(connection=connection)
        self.connection_to_frames[connection] = (tf_parent_frame, tf_child_frame)

    def apply(self) -> bool:
        for connection, (
            tf_parent_frame,
            tf_child_frame,
        ) in self.connection_to_frames.items():
            parent_T_child = self.tf_wrapper.lookup_pose(
                tf_parent_frame, tf_child_frame
            ).pose
            connection.origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=parent_T_child.position.x,
                pos_y=parent_T_child.position.y,
                pos_z=parent_T_child.position.z,
                quat_w=parent_T_child.orientation.w,
                quat_x=parent_T_child.orientation.x,
                quat_y=parent_T_child.orientation.y,
                quat_z=parent_T_child.orientation.z,
                reference_frame=connection.parent,
                child_frame=connection.child,
            )
        return bool(self.connection_to_frames)

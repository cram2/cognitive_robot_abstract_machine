from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Tuple, Type, Union

from nav_msgs.msg import Odometry
from rclpy.subscription import Subscription
from sensor_msgs.msg import JointState
from typing_extensions import TypeVar

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.exceptions import (
    AlreadyTrackedByTfFrameError,
    ConnectionCannotBeTrackedByTfFrameError,
    UnboundMessageTypeError,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    DifferentialDrive,
    OmniDrive,
)

# %% base classes

MessageType = TypeVar("MessageType")


@dataclass
class InputSynchronizer(ABC):
    """
    Writes an external source of truth, e.g. a robot's joint states, into the world
    state.
    """

    world: World
    """
    The world whose state is kept in sync with the external source.
    """

    @abstractmethod
    def apply(self) -> bool:
        """
        Write the most recent input into the world state.

        :return: Whether anything was written.
        """

    def close(self) -> None:
        """
        Release the resources used to receive inputs.
        """


@dataclass
class WorldStateInputs:
    """
    All inputs that one loop of Giskard reads before it computes anything.
    """

    world: World
    """
    The world whose state is written and whose observers are notified.
    """

    synchronizers: List[InputSynchronizer] = field(default_factory=list)
    """
    The inputs, applied in the order they were added.
    """

    def apply_inputs(self) -> bool:
        """
        Write all inputs into the world state, in the order they were added.

        :return: Whether any of them wrote something.
        """
        wrote_something = False
        for synchronizer in self.synchronizers:
            wrote_something |= synchronizer.apply()
        return wrote_something

    def synchronize(self) -> None:
        """
        Write all inputs into the world state and announce the change.

        Nothing is announced when no input wrote, because announcing recomputes the
        forward kinematics and reaches every observer of the world.
        """
        if not self.apply_inputs():
            return
        self.announce_state()

    def synchronize_and_announce(self) -> None:
        """
        Write all inputs into the world state and announce the state even if no input
        wrote.

        Use this where nothing else announces, so that the observers of the world do not
        go stale.
        """
        self.apply_inputs()
        self.announce_state()

    def announce_state(self) -> None:
        """
        Hand the current world state to the observers of the world.

        Nothing is announced while the world model is being modified, because the
        observers would see an inconsistent model.
        """
        if self.world.world_is_being_modified:
            return
        self.world.notify_state_change()


@dataclass
class TopicInputSynchronizer(
    InputSynchronizer, Generic[MessageType], SubClassSafeGeneric, ABC
):
    """
    Buffers the latest message of a topic and applies it on demand.

    Subclasses name the type of their messages by binding the generic parameter, as in
    ``TopicInputSynchronizer[Odometry]``.
    """

    topic_name: str
    """
    Name of the topic the inputs are read from.
    """

    latest_message: MessageType | None = field(init=False, default=None)
    """
    The most recently received message, or ``None`` if nothing was received yet.
    """

    subscription: Subscription = field(init=False)
    """
    The subscription feeding ``latest_message``.
    """

    def __post_init__(self):
        if not self.topic_name.startswith("/"):
            self.topic_name = f"/{self.topic_name}"
        self.subscription = rospy.node.create_subscription(
            self.message_type(), self.topic_name, self.buffer_message, 1
        )
        rospy.node.get_logger().info(f"Subscribed to {self.topic_name}")

    @classmethod
    def message_type(cls) -> Type[MessageType]:
        """
        The type of the messages published on ``topic_name``.

        :raises UnboundMessageTypeError: If the class does not bind the generic
            parameter.
        """
        message_types = cls.get_generic_type_parameters()
        if not message_types or isinstance(message_types[0], TypeVar):
            raise UnboundMessageTypeError(synchronizer_type=cls)
        return message_types[0]

    def buffer_message(self, message: MessageType) -> None:
        """
        Remember the message so that the next :meth:`apply` can use it.
        """
        self.latest_message = message

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

    def close(self) -> None:
        rospy.node.destroy_subscription(self.subscription)


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
            self.world.state[connection.raw_dof.id].position = position

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
class TfFrameSynchronizer(InputSynchronizer):
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
        self.tf_wrapper = TFWrapper(node=rospy.node)

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

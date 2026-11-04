import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from time import sleep
from uuid import UUID

import numpy as np

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_msgs.msg import TFMessage
from typing_extensions import Self

from krrood.symbolic_math.symbolic_math import (
    Matrix,
    VariableParameters,
    CompiledFunction,
)
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.callbacks.callback import (
    StateChangeCallback,
    ModelChangeCallback,
)
from semantic_digital_twin.datastructures.unique_entity_names import UniqueEntityNames
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Connection,
    KinematicStructureEntity,
)

logger = logging.getLogger(__name__)


class TfTopic(StrEnum):
    """
    The topics tf is published on.
    """

    STATIC = "tf_static"
    """
    Transforms that never change, latched for late subscribers.
    """


@dataclass
class TfFrameNames(UniqueEntityNames):
    """
    Names the tf frame every kinematic structure entity is published under, unique
    across the whole tf tree.

    An entity another publisher already broadcasts keeps that publisher's frame name,
    which is where our tree joins theirs.
    """

    prefix: str = field(default="", kw_only=True)
    """
    Put in front of every frame name, so a copy of a world can be published next to
    the world it copies without the two trees claiming the same frames.

    It is prepended as it is, so it carries its own separator: ``"copy/"`` names a
    body ``copy/milk``, while ``"copy"`` would name it ``copymilk``.
    """

    def preferred_name(self, entity: KinematicStructureEntity) -> str:
        return f"{self.prefix}{entity.name}"


@dataclass(eq=False)
class TfPublisherModelCallback(ModelChangeCallback, HasROS2Node):
    """
    Publishes the TF tree of the world.
    """

    ignored_kinematic_structure_entities: set[KinematicStructureEntity] = field(
        default_factory=set
    )
    """
    Kinematic structure entities that should not be published in the tf tree.

    Useful, if the robot is already publishing some tf.
    """

    connections_to_expression: dict[tuple[UUID, UUID], Matrix] = field(
        init=False, default_factory=OrderedDict
    )
    """
    Maps the ids of the parent and child of every published connection whose origin
    depends on the world state to its position and quaternion expression.
    """

    fixed_connections_to_expression: dict[tuple[UUID, UUID], Matrix] = field(
        init=False, default_factory=OrderedDict
    )
    """
    Maps the ids of the parent and child of every published connection whose origin
    does not depend on the world state to its position and quaternion expression.
    """

    tf_message: TFMessage = field(init=False)
    """
    Cache for the tf message that is published on every state change.
    """

    static_tf_message: TFMessage = field(init=False)
    """
    The transforms of :attr:`fixed_connections_to_expression`, published once per model
    change.
    """

    static_tf_pub: Publisher = field(init=False)
    """
    Publisher of :attr:`static_tf_message`, latched for late subscribers.
    """

    compiled_tf: CompiledFunction = field(init=False)
    """
    Compiled function for evaluating the tf expressions.
    """

    frame_names: TfFrameNames = field(default_factory=TfFrameNames)
    """
    The tf frame name of every entity published so far.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.static_tf_pub = self.node.create_publisher(
            TFMessage,
            TfTopic.STATIC,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

    def on_model_change(self, **kwargs):
        self.update_connections_to_expression()
        self.compile_tf_expression()
        self.init_tf_message()
        self.static_tf_pub.publish(self.static_tf_message)

    def stop(self) -> None:
        self.node.destroy_publisher(self.static_tf_pub)
        super().stop()

    def update_connections_to_expression(self):
        """
        Sorts every connection that is not ignored into
        :attr:`connections_to_expression` or :attr:`fixed_connections_to_expression`.
        """
        self.connections_to_expression.clear()
        self.fixed_connections_to_expression.clear()
        for connection in self._world.connections:
            if (
                connection.parent in self.ignored_kinematic_structure_entities
                and connection.child in self.ignored_kinematic_structure_entities
            ):
                continue
            expression = connection.origin_as_position_quaternion()
            if expression.is_constant():
                published = self.fixed_connections_to_expression
            else:
                published = self.connections_to_expression
            published[(connection.parent.id, connection.child.id)] = expression

    def compile_tf_expression(self):
        tf = Matrix.vstack([pose for pose in self.connections_to_expression.values()])
        self.compiled_tf = tf.compile(
            parameters=VariableParameters.from_lists(
                self._world.state.position_float_variables
            )
        )
        if self.compiled_tf.is_result_empty():
            return
        self.compiled_tf.bind_args_to_memory_view(0, self._world.state.positions)

    def init_tf_message(self):
        """
        Builds :attr:`tf_message` with the frames of every connection whose origin
        depends on the world state, and :attr:`static_tf_message` complete with the
        transforms of every other one.

        Frame names are assigned in the order of the world's connections, whichever
        message a connection goes to.
        """
        self.tf_message = TFMessage()
        self.static_tf_message = TFMessage()
        current_time = self.node.get_clock().now().to_msg()
        for connection in self._world.connections:
            ids = (connection.parent.id, connection.child.id)
            if ids in self.connections_to_expression:
                self.tf_message.transforms.append(self._transform_between(connection))
            elif ids in self.fixed_connections_to_expression:
                transform = self._transform_between(connection)
                transform.header.stamp = current_time
                self._write_pose(
                    transform, self.fixed_connections_to_expression[ids].to_np().ravel()
                )
                self.static_tf_message.transforms.append(transform)

    def _transform_between(self, connection: Connection) -> TransformStamped:
        """
        :param connection: A published connection.
        :return: A transform from the frame of its parent to the frame of its child.
        """
        transform = TransformStamped()
        transform.header.frame_id = self.frame_names.assign(connection.parent)
        transform.child_frame_id = self.frame_names.assign(connection.child)
        return transform

    @staticmethod
    def _write_pose(transform: TransformStamped, pose: np.ndarray) -> None:
        """
        :param transform: The transform to write into.
        :param pose: The position followed by the quaternion of the transform.
        """
        transform.transform.translation.x = pose[0]
        transform.transform.translation.y = pose[1]
        transform.transform.translation.z = pose[2]
        transform.transform.rotation.x = pose[3]
        transform.transform.rotation.y = pose[4]
        transform.transform.rotation.z = pose[5]
        transform.transform.rotation.w = pose[6]

    def update_tf_message(self):
        if self.compiled_tf.is_result_empty():
            return
        tf_data = self.compiled_tf.evaluate()
        current_time = self.node.get_clock().now().to_msg()
        for p_T_c, pose in zip(self.tf_message.transforms, tf_data):
            p_T_c.header.stamp = current_time
            self._write_pose(p_T_c, pose)


@dataclass(eq=False)
class TFPublisher(StateChangeCallback, HasROS2Node):
    """
    On state change, publishes the TF tree of the world.

    Puts a frame in every kinematic structure entity that is not in the ignored_bodies
    set.
    """

    ignored_kinematic_structure_entities: set[KinematicStructureEntity] = field(
        default_factory=set
    )
    """
    Kinematic structure entities that should not be published in the tf tree.

    Useful, if the robot is already publishing some tf.
    """

    tf_topic: str = field(default="tf")
    """
    Topic to which tf messages should be published.
    """

    tf_pub: Publisher = field(init=False)
    """
    Publisher for tf messages.
    """

    tf_model_callback: TfPublisherModelCallback = field(init=False)
    """
    Callback for updating the tf message cache on model update.
    """

    throttle_state_updates: int = 1
    """
    Only published every n-th state update.
    """

    frame_names: TfFrameNames = field(default_factory=TfFrameNames, kw_only=True)
    """
    The tf frame name of every entity this publisher publishes.
    """

    root_link_pub: Publisher | None = field(init=False, default=None)
    """
    Publisher of the static transform joining a prefixed tree to the unprefixed one,
    kept for as long as this publisher lives so the transform stays latched.
    """

    def __post_init__(self):
        super().__post_init__()

        self.tf_pub = self.node.create_publisher(TFMessage, self.tf_topic, 10)
        sleep(0.2)
        self.tf_model_callback = TfPublisherModelCallback(
            node=self.node,
            _world=self._world,
            ignored_kinematic_structure_entities=self.ignored_kinematic_structure_entities,
            frame_names=self.frame_names,
        )
        self.tf_model_callback.notify_model_change()
        if self.frame_names.prefix:
            self._publish_root_link()
        self.on_state_change()

    def _publish_root_link(self):
        """
        Publish the prefixed root as coinciding with the unprefixed root, the frame the
        world this one copies is published under, so both trees can be shown in one
        fixed frame.
        """
        self.root_link_pub = self.node.create_publisher(
            TFMessage,
            TfTopic.STATIC,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        root_link = TransformStamped()
        root_link.header.stamp = self.node.get_clock().now().to_msg()
        root_link.header.frame_id = str(self._world.root.name)
        root_link.child_frame_id = self.frame_names.assign(self._world.root)
        root_link.transform.rotation.w = 1.0
        self.root_link_pub.publish(TFMessage(transforms=[root_link]))

    def stop(self):
        """
        Deregister this publisher and the model callback it owns.

        The model callback registers itself on the world, so stopping only the state
        callback would leave it publishing on a node that may already be gone.
        """
        self.tf_model_callback.stop()
        if self.root_link_pub is not None:
            self.node.destroy_publisher(self.root_link_pub)
        super().stop()

    @classmethod
    def create_with_ignore_robot(cls, robot: AbstractRobot, node: Node) -> Self:
        """
        Creates a TF publisher that ignores the robot's kinematic structure.

        Useful, if the robot is already publishing some tf.
        :param robot: The robot for which to create the TF publisher.
        :param node: The ROS2 node used to create the publisher.
        """
        ignored_bodies = set(robot.bodies)
        return cls(
            node=node,
            _world=robot._world,
            ignored_kinematic_structure_entities=ignored_bodies,
        )

    @classmethod
    def create_with_ignore_existing_tf(
        cls,
        world: World,
        node: Node,
        wait_for_existing_tf: int = 5,
    ) -> Self:
        """
        Checks if any kinematic structure entity is already published in tf and ignores
        them.

        :param world: The world for which to create the TF publisher.
        :param node: The ROS2 node used to create the publisher.
        :param wait_for_existing_tf: The time to wait for existing tf to be published.
        """
        tf_wrapper = TFWrapper(node=node)
        sleep(wait_for_existing_tf)
        all_frames = set(tf_wrapper.get_tf_frames())
        tf_wrapper.stop()
        ignored_bodies = set(
            kse
            for kse in world.kinematic_structure_entities
            if str(kse.name) in all_frames
        )
        return cls(
            node=node,
            _world=world,
            ignored_kinematic_structure_entities=ignored_bodies,
        )

    def on_state_change(self, **kwargs):
        if self._world.state.version % self.throttle_state_updates != 0:
            return
        self.tf_model_callback.update_tf_message()
        self.tf_pub.publish(self.tf_model_callback.tf_message)

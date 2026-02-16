import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from time import sleep
from typing import Optional
from uuid import UUID

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.publisher import Publisher
from tf2_msgs.msg import TFMessage
from typing_extensions import Self

from krrood.symbolic_math.symbolic_math import (
    Matrix,
    VariableParameters,
    CompiledFunction,
)
from .tfwrapper import TFWrapper
from ...callbacks.callback import StateChangeCallback, ModelChangeCallback
from ...robots.abstract_robot import AbstractRobot
from ...world import World
from ...world_description.world_entity import KinematicStructureEntity

logger = logging.getLogger(__name__)


@dataclass
class TfPublisherModelCallback(ModelChangeCallback):
    """
    Publishes the TF tree of the world.
    """

    node: Node
    """
    ros2 node used to publish tf messages
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
    Maps kinematic structure entity ids which are directly connected to the corresponding position and quaternion expressions.
    If either parent or child is in the ignored_kinematic_structure_entities set, the connection is not included in this dictionary.
    """
    tf_message: TFMessage = field(init=False)
    """Cache for the tf message that is published."""
    compiled_tf: CompiledFunction = field(init=False)
    """Compiled function for evaluating the tf expressions."""

    def _notify(self, **kwargs):
        self.update_connections_to_expression()
        self.compile_tf_expression()
        self.init_tf_message()

    def update_connections_to_expression(self):
        self.connections_to_expression.clear()
        for connection in self.world.connections:
            if (
                connection.parent in self.ignored_kinematic_structure_entities
                and connection.child in self.ignored_kinematic_structure_entities
            ):
                continue
            self.connections_to_expression[
                (connection.parent.id, connection.child.id)
            ] = connection.origin_as_position_quaternion()

    def compile_tf_expression(self):
        tf = Matrix.vstack([pose for pose in self.connections_to_expression.values()])
        params = [v.variables.position for v in self.world.degrees_of_freedom]
        self.compiled_tf = tf.compile(parameters=VariableParameters.from_lists(params))
        if self.compiled_tf.is_result_empty():
            return
        self.compiled_tf.bind_args_to_memory_view(0, self.world.state.positions)

    def init_tf_message(self):
        self.tf_message = TFMessage()
        self.tf_message.transforms = [
            TransformStamped() for _ in range(len(self.connections_to_expression))
        ]
        for i, (parent_link_id, child_link_id) in enumerate(
            self.connections_to_expression
        ):
            parent_link = self.world.get_kinematic_structure_entity_by_id(
                parent_link_id
            )
            child_link = self.world.get_kinematic_structure_entity_by_id(child_link_id)

            self.tf_message.transforms[i].header.frame_id = str(parent_link.name)
            self.tf_message.transforms[i].child_frame_id = str(child_link.name)

    def update_tf_message(self):
        if self.compiled_tf.is_result_empty():
            return
        tf_data = self.compiled_tf.evaluate()
        current_time = self.node.get_clock().now().to_msg()
        for i, (p_T_c, pose) in enumerate(zip(self.tf_message.transforms, tf_data)):
            p_T_c.header.stamp = current_time
            p_T_c.transform.translation.x = pose[0]
            p_T_c.transform.translation.y = pose[1]
            p_T_c.transform.translation.z = pose[2]
            p_T_c.transform.rotation.x = pose[3]
            p_T_c.transform.rotation.y = pose[4]
            p_T_c.transform.rotation.z = pose[5]
            p_T_c.transform.rotation.w = pose[6]


@dataclass
class TFPublisher(StateChangeCallback):
    """
    On state change, publishes the TF tree of the world.
    Puts a frame in every kinematic structure entity that is not in the ignored_bodies set.
    """

    node: Node
    """ros2 node used to publish tf messages"""
    world: World
    """World for which to publish tf messages."""
    ignored_kinematic_structure_entities: set[KinematicStructureEntity] = field(
        default_factory=set
    )
    """
    Kinematic structure entities that should not be published in the tf tree.
    Useful, if the robot is already publishing some tf.
    """
    tf_topic: str = field(default="tf")
    """Topic to which tf messages should be published."""
    tf_pub: Publisher = field(init=False)
    """Publisher for tf messages."""

    tf_model_cb: TfPublisherModelCallback = field(init=False)
    """Callback for updating the tf message cache on model update."""

    throttle_state_updates: int = 1
    """
    Only published every n-th state update.
    """

    def __post_init__(self):
        super().__post_init__()
        self.tf_pub = self.node.create_publisher(TFMessage, self.tf_topic, 10)
        sleep(0.2)
        self.tf_model_cb = TfPublisherModelCallback(
            node=self.node,
            world=self.world,
            ignored_kinematic_structure_entities=self.ignored_kinematic_structure_entities,
        )
        self.tf_model_cb.notify()
        self._notify()

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
            world=robot._world,
            ignored_kinematic_structure_entities=ignored_bodies,
        )

    @classmethod
    def create_with_ignore_existing_tf(cls, world: World, node: Node) -> Self:
        """
        Checks if any kinematic structure entity is already published in tf and ignores them.
        :param world: The world for which to create the TF publisher.
        :param node: The ROS2 node used to create the publisher.
        """
        tf_wrapper = TFWrapper(node=node)
        for i in range(20):
            all_frames = set(tf_wrapper.get_tf_frames())
            if len(all_frames) > 0:
                break
            sleep(0.1)
        else:
            all_frames = set()
            logging.info("Could not find any tf frames, publishing all tf")
        ignored_bodies = set(
            kse
            for kse in world.kinematic_structure_entities
            if str(kse.name) in all_frames
        )
        return cls(
            node=node,
            world=world,
            ignored_kinematic_structure_entities=ignored_bodies,
        )

    def _notify(self, **kwargs):
        if self.world.state.version % self.throttle_state_updates != 0:
            return
        self.tf_model_cb.update_tf_message()
        self.tf_pub.publish(self.tf_model_cb.tf_message)

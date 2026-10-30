from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.node import MsgType, Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from typing_extensions import Generic

import krrood.symbolic_math.symbolic_math as sm
from krrood.ormatic.utils import classproperty
from cramph.node import EndedByOwner
from cramph.context import ContextExtension
from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from cramph.node import NodeArtifacts
from giskardpy.motion_statechart.ros_context import RosContextExtension


@dataclass(eq=False, repr=False, kw_only=True)
class TopicNode(EndedByOwner, MotionStatechartNode, Generic[MsgType]):
    """
    Superclass for nodes that use ROS topics.
    """

    topic_name: str
    """
    Name of the ROS topic to subscribe to.
    """

    msg_type: type[MsgType]
    """
    Type of the ROS message.
    """

    qos_profile: QoSProfile = field(default_factory=lambda: QoSProfile(depth=10))
    """
    QoS profile to use when subscribing to the topic.
    """

    ros2_node: Node = field(init=False)

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (RosContextExtension,)

    def set_up(self) -> None:
        super().set_up()
        ros_context_extension = self.context.require_extension(RosContextExtension)
        self.ros2_node = ros_context_extension.ros_node


@dataclass(eq=False, repr=False, kw_only=True)
class TopicSubscriberNode(TopicNode[MsgType]):
    """
    Superclass for all nodes that subscribe to a ROS topic.

    This node will automatically create a subscriber on set up and cache the last
    message in `current_msg` on_tick.
    """

    _subscriber: Subscription = field(init=False)
    """
    Internal ROS subscription object.
    """

    __last_msg: MsgType | None = field(init=False, default=None)
    """
    The callback updates this variable.

    Don't use it directly, use `current_msg` instead.
    """

    current_msg: MsgType | None = field(init=False, default=None)
    """
    __last_msg is copied to this variable on every tick while this node is RUNNING.
    """

    def set_up(self) -> None:
        super().set_up()
        self._subscriber = self.ros2_node.create_subscription(
            msg_type=self.msg_type,
            topic=self.topic_name,
            callback=self.callback,
            qos_profile=self.qos_profile,
        )

    def callback(self, msg: MsgType):
        self.__last_msg = msg

    def has_msg(self) -> bool:
        return self.current_msg is not None

    def clear_msg(self):
        self.__last_msg = None

    def on_tick(self) -> ObservationStateValues | None:
        """
        .. warning:: If you override this method, make sure to call `super().on_tick(context)`.
        """
        self.current_msg = self.__last_msg

    def on_reset(self) -> None:
        self.clear_msg()


@dataclass(eq=False, repr=False, kw_only=True)
class TopicPublisherNode(TopicNode[MsgType]):
    """
    Superclass for all nodes that publish to a ROS topic.

    This node will automatically create a publisher on set up.
    """

    _publisher: Publisher = field(init=False)
    """
    Internal ROS publisher object.
    """

    def set_up(self) -> None:
        super().set_up()
        self._publisher = self.ros2_node.create_publisher(
            msg_type=self.msg_type,
            topic=self.topic_name,
            qos_profile=self.qos_profile,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class WaitForMessage(TopicSubscriberNode[MsgType]):
    """
    This node will turn to True once a message was received on its topic.
    """

    def on_tick(self) -> ObservationStateValues | None:
        super().on_tick()
        if self.has_msg():
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False, kw_only=True)
class PublishOnStart(TopicPublisherNode[MsgType]):
    """
    This node will publish its message when on_start is called.

    This is not repeated on every tick, but will be repeated after a reset, if the node
    is started again.
    """

    msg: MsgType
    """
    Message to publish.
    """

    msg_type: type[MsgType] = field(init=False)
    """
    Init=False, because we can figure out the type from the msg parameter.
    """

    def __post_init__(self):
        super().__post_init__()
        self.msg_type = type(self.msg)

    def build(self) -> NodeArtifacts:
        node_artifacts = super().build()
        node_artifacts.observation = sm.Scalar.const_true()
        return node_artifacts

    def on_start(self) -> None:
        self._publisher.publish(self.msg)

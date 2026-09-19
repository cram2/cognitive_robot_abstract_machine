from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from rclpy.node import Node
from rclpy.subscription import Subscription
from typing_extensions import Generic, Type, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.exceptions import UnboundMessageTypeError

MessageType = TypeVar("MessageType")


@dataclass
class LatestMessageSubscriber(Generic[MessageType], SubClassSafeGeneric, ABC):
    """
    Subscribes to a topic and keeps the most recently received message.

    Subclasses name the type of their messages by binding the generic parameter, as in
    ``LatestMessageSubscriber[Odometry]``.
    """

    node: Node = field(kw_only=True)
    """
    The node the messages are received on.
    """

    topic_name: str = field(kw_only=True)
    """
    Name of the topic the messages are read from.
    """

    latest_message: MessageType | None = field(init=False, default=None)
    """
    The most recently received message, or ``None`` if nothing was received yet.
    """

    subscription: Subscription = field(init=False)
    """
    The subscription feeding :attr:`latest_message`.
    """

    def __post_init__(self):
        if not self.topic_name.startswith("/"):
            self.topic_name = f"/{self.topic_name}"
        self.subscription = self.node.create_subscription(
            self.message_type(), self.topic_name, self.buffer_message, 1
        )
        self.node.get_logger().info(f"Subscribed to {self.topic_name}")

    @classmethod
    def message_type(cls) -> Type[MessageType]:
        """
        The type of the messages published on :attr:`topic_name`.

        :raises UnboundMessageTypeError: If the class does not bind the generic
            parameter.
        """
        message_types = cls.get_generic_type_parameters()
        if not message_types or isinstance(message_types[0], TypeVar):
            raise UnboundMessageTypeError(subscriber_type=cls)
        return message_types[0]

    def buffer_message(self, message: MessageType) -> None:
        """
        Keep the message as the most recently received one.
        """
        self.latest_message = message

    def close(self) -> None:
        """
        Stop receiving messages.
        """
        self.node.destroy_subscription(self.subscription)

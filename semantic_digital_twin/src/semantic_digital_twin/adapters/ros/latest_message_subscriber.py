from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from rclpy.subscription import Subscription
from typing_extensions import Any, Callable, Generic, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.exceptions import UnboundMessageTypeError

MessageType = TypeVar("MessageType")

# %% handing one subscription's messages to every reader of a topic


@dataclass(eq=False)
class TopicMessageDistributor:
    """
    The callback of a subscription that several readers of a topic share.
    """

    receivers: list[Callable[[Any], None]] = field(default_factory=list)
    """
    What each message of the subscription is handed to.
    """

    def __call__(self, message: Any) -> None:
        """
        Hand the message to every receiver.

        :param message: The message the subscription received.
        """
        for receiver in tuple(self.receivers):
            receiver(message)


# %% keeping the latest message of a topic


@dataclass
class LatestMessageSubscriber(
    Generic[MessageType], SubClassSafeGeneric, HasROS2Node, ABC
):
    """
    Subscribes to a topic and keeps the most recently received message.

    Subscribers of the same topic and message type on one node share a subscription, but
    each keeps its own message.

    Subclasses name the type of their messages by binding the generic parameter, as in
    ``LatestMessageSubscriber[Odometry]``.
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
    The subscription feeding :attr:`latest_message`, shared with the other subscribers
    of this topic on :attr:`node`.
    """

    def __post_init__(self) -> None:
        if not self.topic_name.startswith("/"):
            self.topic_name = f"/{self.topic_name}"
        self.subscription = self._join_or_create_subscription()

    def _join_or_create_subscription(self) -> Subscription:
        """
        :return: The subscription of this topic and message type that :attr:`node`
            already shares, with this subscriber added to it, or else a new one.
        """
        for subscription in self.node.subscriptions:
            if (
                subscription.topic_name == self.topic_name
                and subscription.msg_type is self.message_type()
                and isinstance(subscription.callback, TopicMessageDistributor)
            ):
                subscription.callback.receivers.append(self.buffer_message)
                return subscription
        self.node.get_logger().info(f"Subscribed to {self.topic_name}")
        return self.node.create_subscription(
            self.message_type(),
            self.topic_name,
            TopicMessageDistributor(receivers=[self.buffer_message]),
            1,
        )

    @classmethod
    def message_type(cls) -> type[MessageType]:
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
        Stop receiving messages, destroying the subscription once no subscriber is left
        on it.
        """
        receivers = self.subscription.callback.receivers
        if self.buffer_message in receivers:
            receivers.remove(self.buffer_message)
        if not receivers:
            self.node.destroy_subscription(self.subscription)

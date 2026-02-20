import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
import threading
from typing import ClassVar, Optional, Set, Type, List, Dict
from uuid import UUID

import numpy as np
import rclpy  # type: ignore
import std_msgs.msg
from krrood.ormatic.dao import to_dao
from krrood.adapters.json_serializer import from_json, to_json
from rclpy.node import Node as RosNode
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sqlalchemy import select
from sqlalchemy.orm import Session

from .messages import (
    MetaData,
    WorldStateUpdate,
    Message,
    ModificationBlock,
    LoadModel,
    Acknowledgment,
)
from ..world_entity_kwargs_tracker import WorldEntityWithIDKwargsTracker
from ...callbacks.callback import Callback, StateChangeCallback, ModelChangeCallback
from ...exceptions import MissingPublishChangesKWARG
from ...orm.ormatic_interface import *
from ...world import World


@dataclass
class Synchronizer(ABC):
    """
    Abstract synchronizer to manage world synchronizations between processes running semantic digital twin.

    It manages publishers and subscribers, ensuring proper cleanup after use.
    The communication is JSON string based.

    .. warning::

        When ``synchronous=True`` is used on a :class:`SynchronizerOnCallback`, publication
        blocks until **all** current subscribers acknowledge receipt or a 5-second timeout
        elapses. If a subscriber process crashes or exits without unsubscribing, the publisher
        will wait for the full timeout on every synchronous publish because the dead process
        never acknowledges.

        To mitigate this, always clean up synchronizers when shutting down:

        .. code-block:: python

            import atexit
            atexit.register(synchronizer.close)

        This gives some assurance that the ROS subscriber is destroyed on exit, so other publishers
        will no longer expect an acknowledgment from the terminated process.
    """

    node: RosNode
    """
    The rclpy node used to create the publishers and subscribers.
    """

    world: World
    """
    The world to synchronize.
    """

    topic_name: Optional[str] = None
    """
    The topic name of the publisher and subscriber.
    """

    acknowledge_topic_name: Optional[str] = "/acknowledge"
    """
    The name of the acknowledgment topic. Synchronous publication of world state waits until all subscribers have acknowledged on this topic.
    """

    publisher: Optional[Publisher] = field(init=False, default=None)
    """
    The publisher used to publish the world state.
    """

    subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the world state.
    """

    acknowledge_publisher: Optional[Publisher] = field(init=False, default=None)
    """
    The publisher used to send acknowledgment messages on the acknowledge topic.
    """

    acknowledge_subscriber: Optional[Subscription] = field(init=False, default=None)
    """
    The subscriber that receives acknowledgment messages from other nodes.
    """

    message_type: ClassVar[Optional[Type[Message]]] = None
    """The type of the message that is sent and received."""

    _current_publication_event_id: Optional[UUID] = None
    """The UUID of the most recently published message awaiting acknowledgment."""

    _expected_acknowledgment_count: int = 0
    """Number of remote subscribers that must acknowledge the current event before synchronous publication unblocks."""

    _received_acknowledgments: Set[MetaData] = field(default_factory=set)
    """Metadata of subscribers that have acknowledged the current event so far."""

    _acknowledge_condition_variable: threading.Condition = field(
        default_factory=threading.Condition
    )
    """
    Condition variable used to block synchronous publication until all expected acknowledgments have been received.
    """

    def __post_init__(self):
        self.publisher = self.node.create_publisher(
            std_msgs.msg.String, topic=self.topic_name, qos_profile=10
        )
        self.subscriber = self.node.create_subscription(
            std_msgs.msg.String,
            topic=self.topic_name,
            callback=self.subscription_callback,
            qos_profile=10,
        )
        self.acknowledge_publisher = self.node.create_publisher(
            std_msgs.msg.String, topic=self.acknowledge_topic_name, qos_profile=10
        )
        self.acknowledge_subscriber = self.node.create_subscription(
            std_msgs.msg.String,
            topic=self.acknowledge_topic_name,
            callback=self.acknowledge_callback,
            qos_profile=10,
        )

    @cached_property
    def meta_data(self) -> MetaData:
        """
        The metadata of the synchronizer which can be used to compare origins of messages.
        """
        return MetaData(
            world_id=self.world._id,
            node_name=self.node.get_name(),
            process_id=os.getpid(),
        )

    def subscription_callback(self, msg: std_msgs.msg.String):
        """
        Wrap the origin subscription callback by self-skipping and disabling the next world callback.

        :param msg: The incoming ROS string message containing a serialized synchronization message.
        """
        tracker = WorldEntityWithIDKwargsTracker.from_world(self.world)

        msg = from_json(json.loads(msg.data), **tracker.create_kwargs())

        if msg.meta_data == self.meta_data:
            return

        acknowledgment = Acknowledgment(
            publication_event_id=msg.publication_event_id,
            node_meta_data=self.meta_data,
        )
        self.acknowledge_publisher.publish(
            std_msgs.msg.String(data=json.dumps(to_json(acknowledgment)))
        )
        self._subscription_callback(msg)

    def acknowledge_callback(self, msg: std_msgs.msg.String):
        """
        Called when subscribers of the sync topic acknowledge receipt of synchronization notifications.

        :param msg: The incoming ROS string message containing a serialized acknowledgment.
        """
        acknowledgment = from_json(json.loads(msg.data))

        with self._acknowledge_condition_variable:
            if (
                self._expected_acknowledgment_count == 0
                or self._current_publication_event_id is None
            ):
                # Not waiting for any acknowledgments at the moment
                return

            if (
                acknowledgment.publication_event_id
                != self._current_publication_event_id
            ):
                # This acknowledgment is not about the event we want to have acknowledged
                return

            self._received_acknowledgments.add(acknowledgment.node_meta_data)

            if (
                len(self._received_acknowledgments)
                >= self._expected_acknowledgment_count
            ):
                self._acknowledge_condition_variable.notify_all()

    def _snapshot_subscribers(self) -> int:
        """
        Count the remote subscribers to the synchronization topic.

        The publishing node's own subscription is excluded because self-originated
        messages are already filtered out in :meth:`subscription_callback`.

        :return: Number of remote subscriptions on this synchronizer's topic.
        """
        infos = self.node.get_subscriptions_info_by_topic(self.topic_name)
        own_name = self.node.get_name()
        own_count = sum(1 for info in infos if info.node_name == own_name)
        return len(infos) - own_count

    @abstractmethod
    def _subscription_callback(self, msg: message_type):
        """
        Callback function called when receiving new messages from other publishers.
        """
        raise NotImplementedError

    def publish(self, msg: Message, synchronous: bool = False):
        """
        Publish a message to the synchronization topic.

        :param msg: The message to publish.
        :param synchronous: If True, block until all subscribers acknowledge receipt.
        """
        self._current_publication_event_id = msg.publication_event_id

        if synchronous:
            with self._acknowledge_condition_variable:
                self._expected_acknowledgment_count = self._snapshot_subscribers()
                self._received_acknowledgments = set()

                self.publisher.publish(
                    std_msgs.msg.String(data=json.dumps(to_json(msg)))
                )

                success = self._acknowledge_condition_variable.wait_for(
                    lambda: len(self._received_acknowledgments)
                    >= self._expected_acknowledgment_count,
                    timeout=5,
                )
                if not success:
                    self.node.get_logger().warning(
                        "Message was not acknowledged, timeout"
                    )

                self._current_publication_event_id = None
                self._expected_acknowledgment_count = 0
                self._received_acknowledgments = set()
        else:
            self.publisher.publish(std_msgs.msg.String(data=json.dumps(to_json(msg))))

    def close(self):
        """
        Clean up publishers and subscribers.
        """

        # Destroy subscribers
        if self.subscriber is not None:
            self.node.destroy_subscription(self.subscriber)
            self.subscriber = None

        # Destroy publishers
        if self.publisher is not None:
            self.node.destroy_publisher(self.publisher)
            self.publisher = None


@dataclass
class SynchronizerOnCallback(Synchronizer, Callback, ABC):
    """
    Synchronizer that does something on callbacks by the world.
    Additionally, ensures that the callback is cleaned up on close.
    """

    synchronous: bool = False
    """
    If True, world_callback will block until all subscribers acknowledge receipt of the published message.
    """

    missed_messages: List = field(default_factory=list, init=False, repr=False)
    """
    The messages that the callback did not trigger due to being paused.
    """

    def _notify(self, **kwargs):
        """
        Wrapper method around world_callback that checks if this time the callback should be triggered.
        """
        publish_changes = kwargs.get("publish_changes", None)
        if publish_changes is None:
            raise MissingPublishChangesKWARG(kwargs)

        if not publish_changes:
            return

        self.world_callback(
            publish_changes=publish_changes, synchronous=self.synchronous
        )

    def _subscription_callback(self, msg):
        if self._is_paused:
            self.missed_messages.append(msg)
        else:
            self.apply_message(msg)

    @abstractmethod
    def apply_message(self, msg):
        """
        Apply the received message to the world.
        """
        raise NotImplementedError

    @abstractmethod
    def world_callback(self, publish_changes: bool = True, synchronous: bool = False):
        """
        Called when the world notifies and update that is not caused by this synchronizer.
        """
        raise NotImplementedError

    def apply_missed_messages(self):
        """
        Applies the missed messages to the world.
        """
        if not self.missed_messages:
            return
        with self.world.modify_world(publish_changes=False):
            for msg in self.missed_messages:
                self.apply_message(msg)

        self.missed_messages = []


@dataclass
class StateSynchronizer(StateChangeCallback, SynchronizerOnCallback):
    """
    Synchronizes the state (values of free variables) of the semantic digital twin with the associated ROS topic.
    """

    message_type: ClassVar[Optional[Type[Message]]] = WorldStateUpdate

    topic_name: str = "/semantic_digital_twin/world_state"

    def __post_init__(self):
        super().__post_init__()
        SynchronizerOnCallback.__post_init__(self)

    def apply_message(self, msg: WorldStateUpdate):
        """
        Update the world state with the provided message.

        :param msg: The message containing the new state information.
        """
        # Parse incoming states: WorldState has 'states' only
        indices = [self.world.state._index[_id] for _id in msg.ids]

        if indices:
            self.world.state.data[0, indices] = np.asarray(msg.states, dtype=float)
            self.update_previous_world_state()
            self.world.notify_state_change(publish_changes=False)

    def world_callback(self, publish_changes: bool = True, synchronous: bool = False):
        """
        Publish the current world state to the ROS topic.
        """
        if not publish_changes:
            return

        changes = self.compute_state_changes()

        if not changes:
            return

        self.update_previous_world_state()

        msg = WorldStateUpdate(
            ids=list(changes.keys()),
            states=list(changes.values()),
            meta_data=self.meta_data,
        )
        self.publish(msg, synchronous=synchronous)

    def compute_state_changes(self) -> Dict[UUID, float]:
        """
        Compute and return only the position changes since the last published snapshot.

        Returns a mapping of DOF name to current position for entries whose position
        differs from the previous snapshot, using a vectorized tolerance-based diff.
        """
        ids = self.world.state.keys()  # List[PrefixedName] in column order
        curr = self.world.state.positions  # np.ndarray shape (N,)
        prev = self.previous_world_state_data  # np.ndarray shape (N,)

        # If the number of DOFs changed (model update), send everything once
        # so the other side can resynchronize, then the snapshot will be updated afterward.
        if prev.shape != curr.shape:
            return {n: float(v) for n, v in zip(ids, curr)}

        # Vectorized comparison: O(N) with minimal Python overhead
        changed_mask = ~np.isclose(curr, prev, rtol=1e-8, atol=1e-12, equal_nan=True)
        if not np.any(changed_mask):
            return {}

        idx = np.nonzero(changed_mask)[0]
        return {ids[i]: float(curr[i]) for i in idx}


@dataclass
class ModelSynchronizer(
    ModelChangeCallback,
    SynchronizerOnCallback,
):
    """
    Synchronizes the model (addition/removal of bodies/DOFs/connections) with the associated ROS topic.
    """

    message_type: ClassVar[Type[Message]] = ModificationBlock
    topic_name: str = "/semantic_digital_twin/world_model"

    def __post_init__(self):
        super().__post_init__()
        SynchronizerOnCallback.__post_init__(self)

    def apply_message(self, msg: ModificationBlock):
        running_callbacks = [
            callback
            for callback in self.world.state.state_change_callbacks
            if not callback._is_paused
        ]
        for callback in running_callbacks:
            callback.pause()

        with self.world.modify_world(publish_changes=False):
            msg.modifications.apply(self.world)
        for callback in running_callbacks:
            callback.resume()

    def world_callback(self, publish_changes: bool = True, synchronous: bool = False):

        if not publish_changes:
            return

        msg = ModificationBlock(
            meta_data=self.meta_data,
            modifications=self.world.get_world_model_manager().model_modification_blocks[
                -1
            ],
        )
        self.publish(msg, synchronous=synchronous)


@dataclass
class ModelReloadSynchronizer(Synchronizer):
    """
    Synchronizes the model reloading process across different systems using ROS messaging.
    The database must be the same across the different processes, otherwise the synchronizer will fail.

    Use this when you did changes to the model that cannot be communicated via the ModelSynchronizer and hence need
    to force all processes to load your world model. Note that this may take a couple of seconds.
    """

    message_type: ClassVar[Type[Message]] = LoadModel

    session: Session = None
    """
    The session used to perform persistence interaction. 
    """

    topic_name: str = "/semantic_digital_twin/reload_model"

    def __post_init__(self):
        super().__post_init__()

    def publish_reload_model(self):
        """
        Save the current world model to the database and publish the primary key to the ROS topic such that other
        processes can subscribe to the model changes and update their worlds.
        """
        dao: WorldMappingDAO = to_dao(self.world)
        self.session.add(dao)
        self.session.commit()
        message = LoadModel(primary_key=dao.database_id, meta_data=self.meta_data)
        self.publish(message)

    def _subscription_callback(self, msg: LoadModel):
        """
        Update the world with the new model by fetching it from the database.

        :param msg: The message containing the primary key of the model to be fetched.
        """
        query = select(WorldMappingDAO).where(
            WorldMappingDAO.database_id == msg.primary_key
        )
        new_world = self.session.scalars(query).one().from_dao()
        self._replace_world(new_world)
        self.world._notify_model_change(publish_changes=False)

    def _replace_world(self, new_world: World):
        """
        Replaces the current world with a new one, updating all relevant attributes.
        This method modifies the existing world state, kinematic structure, degrees
        of freedom, and semantic annotation based on the `new_world` provided.

        If you encounter any issues with references to dead objects, it is most likely due to this method not doing
        everything needed.

        :param new_world: The new world instance to replace the current world.
        """
        self.world.clear()
        self.world.merge_world(new_world)

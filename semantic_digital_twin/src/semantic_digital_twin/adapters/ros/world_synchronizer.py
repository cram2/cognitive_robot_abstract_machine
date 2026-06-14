import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Optional, Set, Type, List, Dict
from uuid import UUID

import numpy as np
import rclpy  # type: ignore
import std_msgs.msg
from krrood.adapters.json_serializer import from_json, to_json
from krrood.ormatic.data_access_objects.helper import to_dao
from rclpy.node import Node as RosNode
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    WorldStateUpdate,
    Message,
    ModificationBlock,
    LoadModel,
    Acknowledgment,
    WorldUpdate,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.callbacks.callback import (
    StateChangeCallback,
    ModelChangeCallback,
)
from semantic_digital_twin.exceptions import (
    MissingPublishChangesKWARG,
    ApplyMissedMessagesWhileWorldIsBeingModifiedError,
    StateUpdateContainsUnknownDegreesOfFreedomError,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityWithClassBasedID,
)
from sqlalchemy import select
from sqlalchemy.orm import Session


@dataclass
class Synchronizer(WorldEntityWithClassBasedID):
    """
    Abstract synchronizer to manage world synchronizations between processes running semantic digital twin.

    It manages publishers and subscribers, ensuring proper cleanup after use.
    The communication is JSON string based.

    .. warning::

        When ``synchronous=True``, publication blocks until **all** current subscribers acknowledge receipt or
        a 5-second timeout elapses. If a subscriber process crashes or exits without unsubscribing, the publisher
        will wait for the full timeout on every synchronous publish because the dead process
        never acknowledges.

        To mitigate this, always clean up synchronizers when shutting down:

        .. code-block:: python

            import atexit
            atexit.register(synchronizer.close)

        This gives some assurance that the ROS subscriber is destroyed on exit, so other publishers
        will no longer expect an acknowledgment from the terminated process.
    """

    node: RosNode = field(kw_only=True)
    """
    The rclpy node used to create the publishers and subscribers.
    """

    topic_name: Optional[str] = None
    """
    The topic name of the publisher and subscriber.
    """

    synchronous: bool = False
    """
    If ``True``, publish blocks until all subscribers acknowledge receipt.
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

    wait_for_synchronization_timeout: float = field(default=30.0)
    """Timeout in seconds for waiting for synchronization."""

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
        self.subscriber = self.node.create_subscription(
            std_msgs.msg.String,
            topic=self.topic_name,
            callback=self.subscription_callback,
            qos_profile=10,
        )
        self.publisher = self.node.create_publisher(
            std_msgs.msg.String, topic=self.topic_name, qos_profile=10
        )
        self.acknowledge_subscriber = self.node.create_subscription(
            std_msgs.msg.String,
            topic=self.acknowledge_topic_name,
            callback=self.acknowledge_callback,
            qos_profile=10,
        )
        self.acknowledge_publisher = self.node.create_publisher(
            std_msgs.msg.String, topic=self.acknowledge_topic_name, qos_profile=10
        )

    @cached_property
    def meta_data(self) -> MetaData:
        """
        The metadata of the synchronizer which can be used to compare origins of messages.
        """
        return MetaData(
            world_id=self._world._id,
            node_name=self.node.get_name(),
            process_id=os.getpid(),
        )

    def subscription_callback(self, msg: std_msgs.msg.String):
        """
        Wrap the origin subscription callback by self-skipping and disabling the next world callback.

        :param msg: The incoming ROS string message containing a serialized synchronization message.
        """
        tracker = WorldEntityWithIDKwargsTracker.from_world(self._world)

        msg = from_json(json.loads(msg.data), **tracker.create_kwargs())

        if msg.meta_data == self.meta_data:
            return

        self._subscription_callback(msg)

    def acknowledge_message(self, message: message_type):
        if self.acknowledge_publisher is None:
            return
        acknowledgment = Acknowledgment(
            publication_event_id=message.publication_event_id,
            node_meta_data=self.meta_data,
        )
        self.acknowledge_publisher.publish(
            std_msgs.msg.String(data=json.dumps(to_json(acknowledgment)))
        )

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
                return

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

    def publish(self, msg: Message):
        """
        Publish a message to the synchronization topic.

        :param msg: The message to publish.
        """
        self._current_publication_event_id = msg.publication_event_id

        if self.synchronous:
            with self._acknowledge_condition_variable:
                self._expected_acknowledgment_count = self._snapshot_subscribers()
                self._received_acknowledgments = set()
                self.publisher.publish(
                    std_msgs.msg.String(data=json.dumps(to_json(msg)))
                )

                success = self._acknowledge_condition_variable.wait_for(
                    lambda: len(self._received_acknowledgments)
                    >= self._expected_acknowledgment_count,
                    timeout=self.wait_for_synchronization_timeout,
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
        if self.subscriber is not None:
            self.node.destroy_subscription(self.subscriber)
            self.subscriber = None

        if self.acknowledge_subscriber is not None:
            self.node.destroy_subscription(self.acknowledge_subscriber)
            self.acknowledge_subscriber = None

        if self.publisher is not None:
            self.node.destroy_publisher(self.publisher)
            self.publisher = None

        if self.acknowledge_publisher is not None:
            self.node.destroy_publisher(self.acknowledge_publisher)
            self.acknowledge_publisher = None


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
        from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO

        dao = to_dao(self._world)
        self.session.add(dao)
        self.session.commit()
        message = LoadModel(primary_key=dao.database_id, meta_data=self.meta_data)
        self.publish(message)

    def _subscription_callback(self, msg: LoadModel):
        """
        Update the world with the new model by fetching it from the database.

        :param msg: The message containing the primary key of the model to be fetched.
        """
        from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO

        query = select(WorldMappingDAO).where(
            WorldMappingDAO.database_id == msg.primary_key
        )
        new_world = self.session.scalars(query).one().from_dao()
        self._replace_world(new_world)
        self._world._notify_model_change(publish_changes=False)

    def _replace_world(self, new_world: World):
        """
        Replaces the current world with a new one, updating all relevant attributes.
        This method modifies the existing world state, kinematic structure, degrees
        of freedom, and semantic annotation based on the `new_world` provided.

        If you encounter any issues with references to dead objects, it is most likely due to this method not doing
        everything needed.

        :param new_world: The new world instance to replace the current world.
        """
        self._world.clear()
        self._world.merge_world(new_world)


@dataclass(eq=False)
class WorldSynchronizer(Synchronizer, ModelChangeCallback, StateChangeCallback):
    """
    Single ``/world_sync`` topic synchronizer for ordered model + state delivery.

    Publishing both model and state updates on the **same** ROS topic provides DDS FIFO
    ordering guarantees — a model update published before a state update will always be
    received first, eliminating the cross-topic race that causes ``KeyError`` when state
    messages arrive before the model update that introduced the referenced DOF UUIDs.

    The ``synchronize_model`` and ``synchronize_state`` flags control whether outgoing
    changes are published.  Incoming messages from other nodes are always received and
    applied regardless of these flags.
    """

    message_type: ClassVar[Optional[Type[Message]]] = WorldUpdate

    topic_name: str = "/semantic_digital_twin/world_sync"

    synchronize_model: bool = True
    """
    If ``True``, model changes on this world are published to the synchronization topic.
    If ``False``, this synchronizer acts as a receive-only participant for model changes.
    """

    synchronize_state: bool = True
    """
    If ``True``, state changes on this world are published to the synchronization topic.
    If ``False``, this synchronizer acts as a receive-only participant for state changes.
    """

    missed_messages: List[WorldUpdate] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Buffer for messages received while the synchronizer is paused. 
    These messages can be applied later by calling ``apply_missed_messages()``.
    """

    def __post_init__(self):
        Synchronizer.__post_init__(self)
        if self.synchronize_model:
            self._world.get_world_model_manager().model_change_callbacks.append(self)
        if self.synchronize_state:
            self._world.state.state_change_callbacks.append(self)
        self.update_previous_world_state()

    def on_model_change(self, **kwargs):
        publish_changes = kwargs.get("publish_changes")
        if publish_changes is None:
            raise MissingPublishChangesKWARG(kwargs)
        if not publish_changes:
            return

        model_block = ModificationBlock(
            meta_data=self.meta_data,
            modifications=self._world.get_world_model_manager().model_modification_blocks[
                -1
            ],
        )
        update = WorldUpdate(meta_data=self.meta_data, modification_block=model_block)
        self.publish(update)

    def on_state_change(self, **kwargs):
        publish_changes = kwargs.get("publish_changes")
        if publish_changes is None:
            raise MissingPublishChangesKWARG(kwargs)
        if not publish_changes:
            return

        changes = self.compute_state_changes()
        if not changes:
            return

        state_message = WorldStateUpdate(
            meta_data=self.meta_data,
            ids=list(changes.keys()),
            states=list(changes.values()),
        )
        update = WorldUpdate(meta_data=self.meta_data, state_update=state_message)
        self.publish(update)
        self.update_previous_world_state()

    def compute_state_changes(self) -> Dict[UUID, float]:
        """Return only DOF positions that changed since the last snapshot."""
        degree_of_freedom_identifiers = self._world.state.keys()
        current_positions = self._world.state.positions
        previous_positions = self.previous_world_state_data

        if previous_positions.shape != current_positions.shape:
            return {
                identifier: float(value)
                for identifier, value in zip(
                    degree_of_freedom_identifiers, current_positions
                )
            }

        changed_mask = ~np.isclose(
            current_positions, previous_positions, rtol=1e-8, atol=1e-12, equal_nan=True
        )
        if not np.any(changed_mask):
            return {}

        changed_indices = np.nonzero(changed_mask)[0]
        return {
            degree_of_freedom_identifiers[index]: float(current_positions[index])
            for index in changed_indices
        }

    def _subscription_callback(self, message: WorldUpdate):
        if self._is_paused:
            self.missed_messages.append(message)
        else:
            self.apply_message(message)
            self.acknowledge_message(message)

    def apply_message(self, message: WorldUpdate):
        """Apply model first, then state — preserves the ordering invariant."""
        if message.modification_block is not None:
            self._apply_model(message.modification_block)
        if message.state_update is not None:
            self._apply_state(message.state_update)

    def _apply_model(self, modification_block_message: ModificationBlock):
        with self._world.modify_world(publish_changes=False):
            modification_block_message.modifications.apply(self._world)

    def _apply_state(self, state_update_message: WorldStateUpdate):
        identifier_index_state_triples = [
            (identifier, self._world.state._index.get(identifier), state_value)
            for identifier, state_value in zip(
                state_update_message.ids, state_update_message.states
            )
        ]
        unknown_identifiers = [
            identifier
            for identifier, index, _ in identifier_index_state_triples
            if index is None
        ]
        if unknown_identifiers:
            raise StateUpdateContainsUnknownDegreesOfFreedomError(
                unknown_identifiers=unknown_identifiers
            )
        indices = [index for _, index, _ in identifier_index_state_triples]
        state_values = [
            state_value for _, _, state_value in identifier_index_state_triples
        ]
        with self._world._world_lock:
            self._world.state._data[0, indices] = np.asarray(state_values, dtype=float)
            self.update_previous_world_state()
        self._world.notify_state_change(publish_changes=False)

    def apply_missed_messages(self):
        """Apply buffered messages accumulated while the synchronizer was paused.

        Each message is applied independently so that model-change notifications
        fire between messages, which is required for state messages that follow
        model messages (``compiled_all_fks`` must exist before
        ``notify_state_change`` is called).

        :raises ApplyMissedMessagesWhileWorldIsBeingModifiedError: If called while a
            ``modify_world`` context is active on this synchronizer's world.
        """
        if self._world.world_is_being_modified:
            raise ApplyMissedMessagesWhileWorldIsBeingModifiedError()
        if not self.missed_messages:
            return
        pending_messages = self.missed_messages
        self.missed_messages = []
        for message in pending_messages:
            self.apply_message(message)
        for message in pending_messages:
            self.acknowledge_message(message)

    def resume(self):
        """Resume publishing and subscribing. Missed messages are NOT applied automatically."""
        super().resume()

    def stop(self):
        if self.synchronize_model:
            try:
                self._world.get_world_model_manager().model_change_callbacks.remove(
                    self
                )
            except ValueError:
                pass
        if self.synchronize_state:
            try:
                self._world.state.state_change_callbacks.remove(self)
            except ValueError:
                pass
        super().stop()

    def close(self):
        self.stop()
        super().close()

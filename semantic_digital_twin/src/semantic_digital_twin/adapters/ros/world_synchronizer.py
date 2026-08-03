from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Optional, Type, List, Dict
from uuid import UUID

import numpy as np
import rclpy  # type: ignore
import std_msgs.msg
from rclpy.node import Node as RosNode
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.adapters.json_serializer import from_json, to_json
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    WorldStateUpdate,
    Message,
    ModificationBlock,
    LoadModel,
    StateWatermark,
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
    WorldHasMultipleSynchronizersError,
    WorldHasNoSynchronizerError,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityWithClassBasedID,
)


class PublicationProgress(ABC):
    """
    Reports how far its own changes were published to the other processes.

    Whoever hands out a :class:`StateWatermark` of its own stream tells others what they
    have to catch up with before they may read the world it publishes.
    """

    @property
    @abstractmethod
    def published_sequence_number(self) -> int:
        """
        Position of the message that was published last.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def latest_published_watermark(self) -> StateWatermark:
        """
        The position of the last published message, together with its publisher.
        """
        raise NotImplementedError


@dataclass
class Synchronizer(WorldEntityWithClassBasedID, PublicationProgress):
    """
    Abstract synchronizer to manage world synchronizations between processes running
    semantic digital twin.

    It manages publishers and subscribers, ensuring proper cleanup after use. The
    communication is JSON string based.

    Every published message carries its position in this synchronizer's stream, and
    every applied message is remembered per publisher, so that two processes can tell
    each other what to catch up with without comparing world models they cannot compare.
    """

    node: RosNode = field(kw_only=True)
    """
    The rclpy node used to create the publishers and subscribers.
    """

    topic_name: Optional[str] = None
    """
    The topic name of the publisher and subscriber.
    """

    publisher: Optional[Publisher] = field(init=False, default=None)
    """
    The publisher used to publish the world state.
    """

    subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the world state.
    """

    message_type: ClassVar[Optional[Type[Message]]] = None
    """
    The type of the message that is sent and received.
    """

    _published_sequence_number: int = field(default=0, init=False, repr=False)
    """
    Position of the message this synchronizer published last.
    """

    _applied_sequence_numbers: Dict[MetaData, int] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The highest position applied so far, per publisher.
    """

    _sequence_number_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """
    Guards the published and applied positions.

    Messages are published and applied on different threads, and both positions are read
    by whoever waits for one of them.
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

    @cached_property
    def meta_data(self) -> MetaData:
        """
        The metadata of the synchronizer which can be used to compare origins of
        messages.
        """
        return MetaData(
            world_id=self._world._id,
            node_name=self.node.get_name(),
            process_id=os.getpid(),
        )

    def subscription_callback(self, message: std_msgs.msg.String):
        """
        Wrap the origin subscription callback by self-skipping and disabling the next
        world callback. Holds the world lock while deserializing to ensure no changes
        happen while building the tracker and running from_json.

        :param message: The incoming ROS string message containing a serialized
            synchronization message.
        """
        with self._world._world_lock:
            tracker = WorldEntityWithIDKwargsTracker.from_world(self._world)
            deserialized_message = from_json(
                json.loads(message.data), **tracker.create_kwargs()
            )

            if deserialized_message.meta_data == self.meta_data:
                return

            self._subscription_callback(deserialized_message)

    @property
    def published_sequence_number(self) -> int:
        with self._sequence_number_lock:
            return self._published_sequence_number

    @property
    def latest_published_watermark(self) -> StateWatermark:
        return StateWatermark(
            origin=self.meta_data, sequence_number=self.published_sequence_number
        )

    def has_applied(self, watermark: StateWatermark) -> bool:
        """
        Whether everything up to ``watermark`` was applied to this world.

        A publisher this world never heard from is treated as being at the start of its
        stream, so its first message is still awaited.
        """
        with self._sequence_number_lock:
            applied = self._applied_sequence_numbers.get(watermark.origin, 0)
        return applied >= watermark.sequence_number

    def record_applied(self, message: Message):
        """
        Remember how far this world caught up with the publisher of ``message``.

        Called when a message is applied rather than when it is received, so that a
        buffered message does not count as caught up with.
        """
        with self._sequence_number_lock:
            origin = message.meta_data
            self._applied_sequence_numbers[origin] = max(
                self._applied_sequence_numbers.get(origin, 0), message.sequence_number
            )

    @abstractmethod
    def _subscription_callback(self, msg: message_type):
        """
        Callback function called when receiving new messages from other publishers.
        """
        raise NotImplementedError

    def publish(self, msg: Message):
        """
        Publish a message to the synchronization topic, stamped with its position in
        this synchronizer's stream.

        Waits for its turn in the stream of publications of the world, so that a message
        cannot overtake the changes that were published before the one it describes.

        :param msg: The message to publish.
        """
        with self._world.get_world_model_manager().publishing_in_order():
            with self._sequence_number_lock:
                self._published_sequence_number += 1
                msg.sequence_number = self._published_sequence_number
            self.publisher.publish(std_msgs.msg.String(data=json.dumps(to_json(msg))))

    def close(self):
        """
        Clean up publishers and subscribers.
        """
        if self.subscriber is not None:
            self.node.destroy_subscription(self.subscriber)
            self.subscriber = None

        if self.publisher is not None:
            self.node.destroy_publisher(self.publisher)
            self.publisher = None


@dataclass
class ModelReloadSynchronizer(Synchronizer):
    """
    Synchronizes the model reloading process across different systems using ROS
    messaging. The database must be the same across the different processes, otherwise
    the synchronizer will fail.

    Use this when you did changes to the model that cannot be communicated via the
    ModelSynchronizer and hence need to force all processes to load your world model.
    Note that this may take a couple of seconds.
    """

    message_type: ClassVar[Type[Message]] = LoadModel

    session: Session = None
    """
    The session used to perform persistence interaction.
    """

    topic_name: str = "/semantic_digital_twin/reload_model"

    defer_incoming_reloads: bool = False
    """
    If ``True``, an incoming reload is remembered instead of applied on the receiving
    thread.

    Use this when another thread owns the world and has to decide itself when it may be
    replaced, for example a controller that must not have the world changed under a
    running motion.
    """

    pending_reload: Optional[LoadModel] = field(default=None, init=False, repr=False)
    """
    The reload that was received but not applied yet.

    Only the most recent one is kept, because a reload replaces the whole world model
    and therefore makes every earlier request obsolete.
    """

    def __post_init__(self):
        super().__post_init__()

    @property
    def has_pending_reload(self) -> bool:
        """
        Whether a reload is waiting to be applied.
        """
        return self.pending_reload is not None

    def publish_reload_model(self):
        """
        Save the current world model to the database and publish the primary key to the
        ROS topic such that other processes can subscribe to the model changes and
        update their worlds.
        """
        from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO  # type: ignore

        dao = to_dao(self._world)
        self.session.add(dao)
        self.session.commit()
        message = LoadModel(primary_key=dao.database_id, meta_data=self.meta_data)
        self.publish(message)

    def _subscription_callback(self, msg: LoadModel):
        if self.defer_incoming_reloads:
            self.pending_reload = msg
            return
        self.apply_reload(msg)

    def apply_pending_reload(self):
        """
        Apply the reload that was deferred, if there is one.
        """
        if self.pending_reload is None:
            return
        message = self.pending_reload
        self.pending_reload = None
        self.apply_reload(message)

    def apply_reload(self, msg: LoadModel):
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
        self.record_applied(msg)

    def _replace_world(self, new_world: World):
        """
        Replaces the current world with a new one, updating all relevant attributes.
        This method modifies the existing world state, kinematic structure, degrees of
        freedom, and semantic annotation based on the `new_world` provided.

        If you encounter any issues with references to dead objects, it is most likely
        due to this method not doing everything needed.

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
    """

    message_type: ClassVar[Optional[Type[Message]]] = WorldUpdate

    topic_name: str = "/semantic_digital_twin/world_sync"

    defer_incoming_updates: bool = False
    """
    If ``True``, incoming messages are buffered instead of applied on the subscription
    thread.

    Outgoing publishing is unaffected, unlike with ``pause()``. Use this when another
    thread owns the world and has to decide itself when an update may be applied, for
    example a controller that must not have the world changed under a running motion.
    """

    missed_messages: List[WorldUpdate] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Buffer for messages that were received but not applied yet.

    These messages can be applied later by calling ``apply_missed_messages()`` or, up to
    the next model modification, ``apply_missed_state_updates()``.
    """

    _missed_message_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """
    Guards ``missed_messages``.

    The subscription thread appends while the owning thread drains. Deliberately not
    ``_world_lock``: receiving a message must never wait for the world.
    """

    @classmethod
    def for_world(cls, world: World) -> WorldSynchronizer:
        """
        The synchronizer that publishes the changes of ``world``.

        :raises WorldHasNoSynchronizerError: If the world publishes its changes nowhere.
        :raises WorldHasMultipleSynchronizersError: If several synchronizers publish the
            changes of the world, leaving it undecided which stream to refer to.
        """
        synchronizers = [
            callback
            for callback in world.get_world_model_manager().model_change_callbacks
            if isinstance(callback, cls)
        ]
        if not synchronizers:
            raise WorldHasNoSynchronizerError(world=world)
        if len(synchronizers) > 1:
            raise WorldHasMultipleSynchronizersError(
                world=world, synchronizer_count=len(synchronizers)
            )
        return synchronizers[0]

    def __post_init__(self):
        # Called explicitly instead of via super(): Synchronizer does not chain to its own
        # base, so only naming both branches of the MRO runs the ros setup and the callback
        # registration.
        Synchronizer.__post_init__(self)
        ModelChangeCallback.__post_init__(self)

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
        self._publish_or_defer(update)

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
        self._publish_or_defer(update)
        self.update_previous_world_state()

    def _publish_or_defer(self, update: WorldUpdate) -> None:
        """
        Publishes ``update`` now, or defers it until the world lock is released.

        A modification that is still running has not produced its model block yet, so
        publishing a state change from inside it would send degrees of freedom that the
        receivers cannot know yet. Deferring to the world's ``pending_publications``,
        which are flushed after the lock is released, keeps a state update behind the
        model update that introduces what it refers to. Outside a modification (usually
        just during state changes) no lock is held, so we publish directly.
        """
        if self._world.world_is_being_modified:
            self._world.get_world_model_manager().pending_publications.append(
                lambda: self.publish(update)
            )
        else:
            self.publish(update)

    def compute_state_changes(self) -> Dict[UUID, float]:
        """
        Return only DOF positions that changed since the last snapshot.
        """
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
        if self._is_paused or self.defer_incoming_updates:
            with self._missed_message_lock:
                self.missed_messages.append(message)
            return
        self.apply_message(message)

    @property
    def has_buffered_model_modification(self) -> bool:
        """
        Whether any buffered message carries a model modification.

        A model modification invalidates anything compiled against the current world
        structure, so an owner of the world uses this to decide whether it can keep
        going.
        """
        with self._missed_message_lock:
            return any(
                message.modification_block is not None
                for message in self.missed_messages
            )

    def apply_message(self, message: WorldUpdate):
        """
        Apply model first, then state — preserves the ordering invariant.

        Both are applied under a single hold of ``_world_lock`` so a combined update is
        atomic: no other thread can observe the new model structure without its accompanying state.
        """
        with self._world._world_lock:
            if message.modification_block is not None:
                self._apply_model(message.modification_block)
            if message.state_update is not None:
                self._apply_state(message.state_update)
        self.record_applied(message)

    def _apply_model(self, modification_block_message: ModificationBlock):
        """
        Applies the model and recompiles the world structure before applying the new
        state.
        """
        with self._world.modify_world(publish_changes=False):
            modification_block_message.modifications.apply(self._world)

    def _apply_state(self, state_update_message: WorldStateUpdate):
        """
        Applies the state, and raises a StateUpdateContainsUnknownDegreesOfFreedomError
        if we receive unknown degree of freedom.
        """
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
        """
        Apply every buffered message.

        :raises ApplyMissedMessagesWhileWorldIsBeingModifiedError: If called while a
            ``modify_world`` context is active on this synchronizer's world.
        """
        self._apply_leading_messages(len(self.missed_messages))

    def apply_missed_state_updates(self):
        """
        Apply the buffered messages up to the next model modification.

        Everything from that modification onwards stays buffered, so the order in which
        the messages were published survives. Use this to keep consuming state while a
        model modification cannot be applied yet.

        :raises ApplyMissedMessagesWhileWorldIsBeingModifiedError: If called while a
            ``modify_world`` context is active on this synchronizer's world.
        """
        self._apply_leading_messages(self._count_leading_state_only_messages())

    def _count_leading_state_only_messages(self) -> int:
        """
        Number of buffered messages before the first one carrying a model modification.
        """
        with self._missed_message_lock:
            for position, message in enumerate(self.missed_messages):
                if message.modification_block is not None:
                    return position
            return len(self.missed_messages)

    def _apply_leading_messages(self, count: int):
        """
        Apply the first ``count`` buffered messages.

        Each message is applied independently so that model-change notifications fire
        between messages, which is required for state messages that follow model
        messages (``compiled_all_fks`` must exist before ``notify_state_change`` is
        called).

        :raises ApplyMissedMessagesWhileWorldIsBeingModifiedError: If called while a
            ``modify_world`` context is active on this synchronizer's world.
        """
        if self._world.world_is_being_modified:
            raise ApplyMissedMessagesWhileWorldIsBeingModifiedError()
        with self._missed_message_lock:
            pending_messages = self.missed_messages[:count]
            if not pending_messages:
                return
            # Mutate in place instead of rebinding: the subscription thread appends to
            # the same list and its message must survive the drain.
            del self.missed_messages[: len(pending_messages)]
        # Hold the world lock across the whole batch so the buffered messages apply atomically: a
        # concurrent modify_world on another thread serializes behind it instead of interleaving.
        with self._world._world_lock:
            for message in pending_messages:
                self.apply_message(message)

    def resume(self):
        """
        Resume publishing and subscribing.

        Missed messages are NOT applied automatically.
        """
        super().resume()

    def close(self):
        self.stop()
        super().close()

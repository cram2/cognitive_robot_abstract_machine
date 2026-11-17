"""
Synchronizers that keep the worlds of different processes in step over ROS 2 topics.
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cached_property
from typing_extensions import Callable, ClassVar
from uuid import UUID

from numbers import Real

import numpy as np
import rclpy  # type: ignore
import std_msgs.msg
from rclpy.node import Node as RosNode
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.adapters.json_serializer import to_json
from krrood.exceptions import DataclassException
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.synchronization.exceptions import DivergedFromPublisherError
from krrood.synchronization.messages import Message, MessageOrigin
from krrood.synchronization.stream_participants import (
    LostMessages,
    StreamPublisher,
    StreamReceiver,
)
from semantic_digital_twin.adapters.ros.messages import (
    WorldStateUpdate,
    ModificationBlock,
    LoadModel,
    ReceivedMessage,
    WorldUpdate,
)
from semantic_digital_twin.callbacks.callback import (
    StateChangeCallback,
    ModelChangeCallback,
)
from semantic_digital_twin.exceptions import (
    StateUpdateContainsNonNumericValuesError,
    SynchronizerNotConnectedError,
    MissingPublishChangesKWARG,
    ApplyMissedMessagesWhileWorldIsBeingModifiedError,
    StateUpdateContainsUnknownDegreesOfFreedomError,
    WorldHasMultipleSynchronizersError,
    WorldHasNoSynchronizerError,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_modification import (
    AddDegreeOfFreedomModification,
    RemoveDegreeOfFreedomModification,
)
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityWithClassBasedID,
)


@dataclass
class Synchronizer(WorldEntityWithClassBasedID, StreamPublisher, StreamReceiver):
    """
    Abstract synchronizer to manage world synchronizations between processes running
    semantic digital twin.

    It manages publishers and subscribers, ensuring proper cleanup after use. The
    communication is JSON string based.

    Every published message carries its position in this synchronizer's stream, and
    every applied message is remembered per publisher, so that two processes can tell
    each other what to catch up with without comparing world models they cannot compare.
    A message that does not follow the one received from its publisher before reveals
    that messages in between were lost, which is logged. A message this world cannot
    apply is logged too, and its publisher no longer counts as caught up with until
    :meth:`forget_divergence_from` is called for it.
    """

    node: RosNode = field(kw_only=True)
    """
    The rclpy node used to create the publishers and subscribers.
    """

    topic_name: str | None = None
    """
    The topic name of the publisher and subscriber.
    """

    connection_timeout: timedelta = timedelta(seconds=5)
    """
    How long to wait for the topic of this synchronizer to become usable.
    """

    discovery_settle_time: timedelta = timedelta(seconds=0.2)
    """
    How long no subscriber may appear before the topic counts as connected.
    """

    discovery_poll_interval: timedelta = timedelta(seconds=0.02)
    """
    How long to wait between two looks at the subscribers of the topic while connecting.
    """

    message_queue_depth: int = 1000
    """
    How many messages the publisher and the subscriber keep for a receiver that has not
    taken them yet; once more are waiting, the oldest are lost.
    """

    publisher: Publisher | None = field(init=False, default=None)
    """
    The publisher used to publish the world state.
    """

    subscriber: Subscription | None = field(default=None, init=False)
    """
    The subscriber to the world state.
    """

    message_type: ClassVar[type[Message] | None] = None
    """
    The type of the message that is sent and received.
    """

    def __post_init__(self):
        self.subscriber = self.node.create_subscription(
            std_msgs.msg.String,
            topic=self.topic_name,
            callback=self.subscription_callback,
            qos_profile=self.message_queue_depth,
        )
        self.publisher = self.node.create_publisher(
            std_msgs.msg.String,
            topic=self.topic_name,
            qos_profile=self.message_queue_depth,
        )
        self.wait_until_connected()

    def wait_until_connected(self) -> None:
        """
        Block until what this synchronizer publishes reaches the processes that already
        listen on its topic.

        A publisher drops what it sends to subscribers it has not discovered yet, so a
        world that is modified right after its synchronizer was created would keep those
        changes to itself. Ros names no moment at which every process is known, so this
        waits for the subscribers it can see to be matched and for no further one to
        turn up.

        :raises SynchronizerNotConnectedError: If not even the subscriber of this
            synchronizer itself was matched in time.
        """
        deadline = time.monotonic() + self.connection_timeout.total_seconds()
        subscriber_count = self.publisher.get_subscription_count()
        settled_since = time.monotonic()
        while time.monotonic() < deadline:
            current_count = self.publisher.get_subscription_count()
            if current_count != subscriber_count:
                subscriber_count = current_count
                settled_since = time.monotonic()
            elif (
                current_count > 0
                and time.monotonic() - settled_since
                >= self.discovery_settle_time.total_seconds()
            ):
                return
            time.sleep(self.discovery_poll_interval.total_seconds())
        raise SynchronizerNotConnectedError(
            topic_name=self.topic_name, timeout=self.connection_timeout
        )

    @cached_property
    def origin(self) -> MessageOrigin:
        return MessageOrigin(
            model_id=self._world._id,
            node_name=self.node.get_name(),
            process_id=os.getpid(),
        )

    def subscription_callback(self, message: std_msgs.msg.String) -> None:
        """
        Hand a message of another synchronizer to :meth:`_subscription_callback`; the
        echo of this synchronizer's own messages is dropped unread.

        :param message: The incoming ROS string message containing a serialized
            synchronization message.
        """
        received = ReceivedMessage.from_json(content=json.loads(message.data))
        if received.position.origin == self.origin:
            return
        lost = self.record_received(position=received.position)
        if lost is not None:
            self._warn_about_lost_messages(lost=lost)
        self._handle_reporting_divergence(
            received=received, handle=self._subscription_callback
        )

    def _handle_reporting_divergence(
        self,
        *,
        received: ReceivedMessage,
        handle: Callable[[ReceivedMessage], None],
        error_throttle_period: float = 1.0,
    ) -> None:
        """
        Handle a received message; if it cannot be applied, log that, and stop counting
        its publisher as caught up with, instead of ending the spinning of the node or
        the batch the message is applied in.

        .. warning:: A model modification that fails part-way stays applied as far as it
            got, so the world may hold part of the failed message.

        :param received: The message to handle.
        :param handle: What to do with the message.
        :param error_throttle_period: The fewest seconds between two logged errors.
        """
        try:
            handle(received)
        except DataclassException as error:
            divergence = DivergedFromPublisherError(
                position=received.position, raised_from=error
            )
            self.record_divergence(divergence=divergence)
            self.node.get_logger().error(
                f"{divergence.error_message()} Cause: {error}",
                throttle_duration_sec=error_throttle_period,
            )

    def _warn_about_lost_messages(
        self, *, lost: LostMessages, warning_throttle_period: float = 1.0
    ) -> None:
        """
        Warn that messages of another synchronizer were lost.

        Only what changed is published, so the world may differ from the publisher's in
        what the lost messages carried.

        :param lost: The messages that never arrived.
        :param warning_throttle_period: The fewest seconds between two warnings.
        """
        self.node.get_logger().warning(
            f"Updates #{lost.first_sequence_number} to #{lost.last_sequence_number} of "
            f"'{lost.publisher.node_name}' were lost, so this world may differ from the "
            f"one of the publisher.",
            throttle_duration_sec=warning_throttle_period,
        )

    @abstractmethod
    def _subscription_callback(self, message: ReceivedMessage) -> None:
        """
        Callback function called when receiving new messages from other publishers.

        :param message: The message, not yet built against the world.
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
            self.assign_next_sequence_number(message=msg)
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

    message_type: ClassVar[type[Message]] = LoadModel

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

    pending_reload: LoadModel | None = field(default=None, init=False, repr=False)
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
        message = LoadModel(primary_key=dao.database_id, origin=self.origin)
        self.publish(message)

    def _subscription_callback(self, message: ReceivedMessage) -> None:
        reload = message.to_message(world=self._world)
        if self.defer_incoming_reloads:
            self.pending_reload = reload
            return
        self.apply_reload(reload)

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
        self._world._replace_with(new_world)
        self.record_applied(position=msg.position)
        for synchronizer in [
            self,
            *WorldSynchronizer.all_callbacks_of_this_type_from_world(self._world),
        ]:
            synchronizer.forget_divergences_from_same_model(publisher=msg.origin)


@dataclass(eq=False)
class WorldSynchronizer(Synchronizer, ModelChangeCallback, StateChangeCallback):
    """
    Single ``/world_sync`` topic synchronizer for ordered model + state delivery.

    Publishing both model and state updates on the **same** ROS topic provides DDS FIFO
    ordering guarantees — a model update published before a state update will always be
    received first, eliminating the cross-topic race that causes ``KeyError`` when state
    messages arrive before the model update that introduced the referenced DOF UUIDs.
    """

    message_type: ClassVar[type[Message] | None] = WorldUpdate

    topic_name: str = "/semantic_digital_twin/world_sync"

    defer_incoming_updates: bool = False
    """
    If ``True``, incoming messages are buffered instead of applied on the subscription
    thread.

    Outgoing publishing is unaffected, unlike with ``pause()``. Use this when another
    thread owns the world and has to decide itself when an update may be applied, for
    example a controller that must not have the world changed under a running motion.
    """

    missed_messages: list[ReceivedMessage] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Buffer for messages that were received but not applied yet, each built against the
    world only when it is applied, so it may refer to what an earlier one introduces.

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
    def of_world(cls, world: World) -> WorldSynchronizer:
        """
        The synchronizer that publishes the changes of ``world``.

        :raises WorldHasNoSynchronizerError: If the world publishes its changes nowhere.
        :raises WorldHasMultipleSynchronizersError: If several synchronizers publish the
            changes of the world, leaving it undecided which stream to refer to.
        """
        synchronizers = cls.all_callbacks_of_this_type_from_world(world)
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

    def on_model_change(self, publish_changes: bool | None = None, **kwargs):
        """
        Send the modification block the world just recorded to the other synchronizers.

        Implements
        :meth:`~semantic_digital_twin.callbacks.callback.ModelChangeCallback.on_model_change`.

        :param publish_changes: Whether this modification may leave the world.
        :raises MissingPublishChangesKWARG: If the notification did not say whether to
            publish.
        """
        if publish_changes is None:
            raise MissingPublishChangesKWARG(kwargs)
        if not publish_changes:
            return

        model_block = ModificationBlock(
            origin=self.origin,
            modifications=self._world.get_world_model_manager().model_modification_blocks[
                -1
            ],
        )
        update = WorldUpdate(origin=self.origin, modification_block=model_block)
        self._publish_or_defer(update)

    def on_state_change(
        self,
        publish_changes: bool | None = None,
        force_republish: bool = False,
        **kwargs,
    ):
        """
        Send the positions of the degrees of freedom to the other synchronizers.

        Implements
        :meth:`~semantic_digital_twin.callbacks.callback.StateChangeCallback.on_state_change`.

        :param publish_changes: Whether this state change may leave the world.
        :param force_republish: Whether to send every degree of freedom instead of only
            the ones that moved since the last message, so a receiver that has just
            applied a model modification is given the whole state.
        :raises MissingPublishChangesKWARG: If the notification did not say whether to
            publish.
        """
        if publish_changes is None:
            raise MissingPublishChangesKWARG(kwargs)
        if not publish_changes:
            return

        changes = (
            self._world.state.to_uuid_position_dict()
            if force_republish
            else self.compute_state_changes()
        )
        if not changes:
            return

        state_message = WorldStateUpdate(
            origin=self.origin,
            ids=list(changes.keys()),
            states=list(changes.values()),
        )
        update = WorldUpdate(origin=self.origin, state_update=state_message)
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

    def compute_state_changes(self) -> dict[UUID, float]:
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

    def _subscription_callback(self, message: ReceivedMessage) -> None:
        if self._is_paused or self.defer_incoming_updates:
            with self._missed_message_lock:
                self.missed_messages.append(message)
            return
        self.apply_message(message.to_message(world=self._world))

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
                message.carries_model_modification for message in self.missed_messages
            )

    def apply_message(self, message: WorldUpdate):
        """
        Apply model first, then state — preserves the ordering invariant.

        Both are applied under a single hold of ``_world_lock`` so a combined update is
        atomic: no other thread can observe the new model structure without its
        accompanying state. The state is checked before the model is applied, so a
        state that cannot be applied leaves the model untouched too.
        """
        with self._world._world_lock:
            if message.state_update is not None:
                self._check_state_update(
                    state_update=message.state_update,
                    modification_block=message.modification_block,
                )
            if message.modification_block is not None:
                self._apply_model(message.modification_block)
            if message.state_update is not None:
                self._apply_state(message.state_update)
        self.record_applied(position=message.position)

    def _apply_model(self, modification_block_message: ModificationBlock):
        """
        Applies the model and recompiles the world structure before applying the new
        state.
        """
        with self._world.modify_world(publish_changes=False):
            modification_block_message.modifications.apply(self._world)

    def _check_state_update(
        self,
        *,
        state_update: WorldStateUpdate,
        modification_block: ModificationBlock | None = None,
    ) -> None:
        """
        :param state_update: A received state.
        :param modification_block: The model modification received with it, whose
            degrees of freedom the state may already refer to.
        :raises StateUpdateContainsUnknownDegreesOfFreedomError: If the state refers to
            a degree of freedom the world will not hold.
        :raises StateUpdateContainsNonNumericValuesError: If a value is not a number.
        """
        known_identifiers = set(self._world.state.keys())
        if modification_block is not None:
            for modification in modification_block.modifications:
                match modification:
                    case AddDegreeOfFreedomModification():
                        known_identifiers.add(modification.degree_of_freedom.id)
                    case RemoveDegreeOfFreedomModification():
                        known_identifiers.discard(modification.degree_of_freedom_id)
        unknown_identifiers = [
            identifier
            for identifier in state_update.ids
            if identifier not in known_identifiers
        ]
        if unknown_identifiers:
            raise StateUpdateContainsUnknownDegreesOfFreedomError(
                unknown_identifiers=unknown_identifiers
            )
        non_numeric_states = [
            state for state in state_update.states if not isinstance(state, Real)
        ]
        if non_numeric_states:
            raise StateUpdateContainsNonNumericValuesError(
                non_numeric_states=non_numeric_states
            )

    def _apply_state(self, state_update_message: WorldStateUpdate):
        """
        Applies the state.

        :raises StateUpdateContainsUnknownDegreesOfFreedomError: If the state refers to
            a degree of freedom the world does not hold.
        :raises StateUpdateContainsNonNumericValuesError: If a value is not a number.
        """
        self._check_state_update(state_update=state_update_message)
        indices = [
            self._world.state._index[identifier]
            for identifier in state_update_message.ids
        ]
        with self._world._world_lock:
            self._world.state._data[0, indices] = np.asarray(
                state_update_message.states, dtype=float
            )
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
                if message.carries_model_modification:
                    return position
            return len(self.missed_messages)

    def _apply_leading_messages(self, count: int):
        """
        Apply the first ``count`` buffered messages.

        Each message is applied independently so that model-change notifications fire
        between messages, which is required for state messages that follow model
        messages (``compiled_all_fks`` must exist before ``notify_state_change`` is
        called).

        A message that cannot be applied is reported as in :meth:`subscription_callback`
        and does not keep the messages after it from being applied.

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
                self._handle_reporting_divergence(
                    received=message,
                    handle=lambda received: self.apply_message(
                        received.to_message(world=self._world)
                    ),
                )

    def resume(self):
        """
        Resume publishing and subscribing.

        Missed messages are NOT applied automatically.
        """
        super().resume()

    def close(self):
        self.stop()
        super().close()

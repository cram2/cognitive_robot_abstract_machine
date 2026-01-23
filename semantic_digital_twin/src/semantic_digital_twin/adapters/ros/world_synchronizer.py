import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Optional, Type, List, Dict
from uuid import UUID

import numpy as np
import rclpy  # type: ignore
import std_msgs.msg
from krrood.ormatic.dao import to_dao
from krrood.adapters.json_serializer import SubclassJSONSerializer
from rclpy.node import Node as RosNode
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sqlalchemy import select
from sqlalchemy.orm import Session

from .messages import MetaData, WorldStateUpdate, Message, ModificationBlock, LoadModel
from ..world_entity_kwargs_tracker import WorldEntityWithIDKwargsTracker
from ...callbacks.callback import Callback, StateChangeCallback, ModelChangeCallback
from ...orm.ormatic_interface import *
from ...world import World


@dataclass
class Synchronizer(ABC):
    """
    Abstract Synchronizer class to manage world synchronizations between processes running semantic digital twin.
    It manages publishers and subscribers, ensuring proper cleanup after use.
    The communication is JSON string based.
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

    publisher: Optional[Publisher] = field(init=False, default=None)
    """
    The publisher used to publish the world state.
    """

    subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the world state.
    """

    message_type: ClassVar[Optional[Type[SubclassJSONSerializer]]] = None
    """
    The type of the message that is sent and received.
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

    @cached_property
    def meta_data(self) -> MetaData:
        """
        The metadata of the synchronizer which can be used to compare origins of messages.
        """
        return MetaData(
            node_name=self.node.get_name(), process_id=os.getpid(), object_id=id(self)
        )

    def subscription_callback(self, msg: std_msgs.msg.String):
        """
        Wrap the origin subscription callback by self-skipping and disabling the next world callback.
        """
        tracker = WorldEntityWithIDKwargsTracker.from_world(self.world)
        msg = self.message_type.from_json(
            json.loads(msg.data), **tracker.create_kwargs()
        )
        if msg.meta_data == self.meta_data:
            return
        self._skip_next_world_callback = True
        self._subscription_callback(msg)

    @abstractmethod
    def _subscription_callback(self, msg: message_type):
        """
        Callback function called when receiving new messages from other publishers.
        """
        raise NotImplementedError

    def publish(self, msg: Message):
        self.publisher.publish(std_msgs.msg.String(data=json.dumps(msg.to_json())))

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

    _skip_next_world_callback: bool = False
    """
    Flag to indicate if the next world callback should be skipped.
    
    An incoming message from some other world might trigger a change in this world that produces a notify callback that 
    will try to send a message. 
    If the callback is triggered by a message, this synchronizer should not republish the change.
    """

    missed_messages: List = field(default_factory=list, init=False, repr=False)
    """
    The messages that the callback did not trigger due to being paused.
    """

    def _notify(self):
        """
        Wrapper method around world_callback that checks if this time the callback should be triggered.
        """
        if self._skip_next_world_callback:
            self._skip_next_world_callback = False
        else:
            self.world_callback()

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
    def world_callback(self):
        """
        Called when the world notifies and update that is not caused by this synchronizer.
        """
        raise NotImplementedError

    def apply_missed_messages(self):
        """
        Applies the missed messages to the world.
        """
        self._skip_next_world_callback = True
        with self.world.modify_world():
            for msg in self.missed_messages:
                self.apply_message(msg)

        self.missed_messages = []


@dataclass
class StateSynchronizer(StateChangeCallback, SynchronizerOnCallback):
    """
    Synchronizes the state (values of free variables) of the semantic digital twin with the associated ROS topic.
    """

    message_type: ClassVar[Optional[Type[SubclassJSONSerializer]]] = WorldStateUpdate

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
            self.world.notify_state_change()

    def world_callback(self):
        """
        Publish the current world state to the ROS topic.
        """
        changes = self.compute_state_changes()

        if not changes:
            return

        msg = WorldStateUpdate(
            ids=list(changes.keys()),
            states=list(changes.values()),
            meta_data=self.meta_data,
        )
        self.update_previous_world_state()
        self.publish(msg)

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
        # so the other side can resync, then the snapshot will be updated afterward.
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

    message_type: ClassVar[Type[SubclassJSONSerializer]] = ModificationBlock
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
        msg.modifications.apply(self.world)
        for callback in running_callbacks:
            callback.resume()

    def world_callback(self):
        msg = ModificationBlock(
            meta_data=self.meta_data,
            modifications=self.world.get_world_model_manager().model_modification_blocks[
                -1
            ],
        )
        self.publish(msg)


@dataclass
class ModelReloadSynchronizer(Synchronizer):
    """
    Synchronizes the model reloading process across different systems using ROS messaging.
    The database must be the same across the different processes, otherwise the synchronizer will fail.

    Use this when you did changes to the model that cannot be communicated via the ModelSynchronizer and hence need
    to force all processes to load your world model. Note that this may take a couple of seconds.
    """

    message_type: ClassVar[Type[SubclassJSONSerializer]] = LoadModel

    session: Session = None
    """
    The session used to perform persistence interaction. 
    """

    topic_name: str = "/semantic_digital_twin/reload_model"

    def __post_init__(self):
        super().__post_init__()
        assert self.session is not None

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
        self.world._notify_model_change()

    def _replace_world(self, new_world: World):
        """
        Replaces the current world with a new one, updating all relevant attributes.
        This method modifies the existing world state, kinematic structure, degrees
        of freedom, and semantic annotation based on the `new_world` provided.

        If you encounter any issues with references to dead objects, it is most likely due to this method not doing
        everything needed.

        :param new_world: The new world instance to replace the current world.
        """
        with self.world.modify_world():
            self.world.clear()
            self.world.merge_world(new_world)

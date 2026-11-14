"""
Tests that a synchronizer keeps receiving after an update it cannot apply and tells the
owner of the world, ignores its own echo without reading it, and buffers updates that
build on each other.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
import std_msgs.msg
from typing_extensions import Any

from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    ModificationBlock,
    StreamPosition,
    WorldStateUpdate,
    WorldUpdate,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.exceptions import (
    StateUpdateContainsUnknownDegreesOfFreedomError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    AddConnectionModification,
    WorldModelModificationBlock,
)
from .test_world_synchronizer import (
    create_dummy_world,
    unique_topic,
    wait_for_condition,
)


# %% helpers


@pytest.fixture
def stranger() -> MetaData:
    """
    A publisher no receiver shares a model with.
    """
    return MetaData(node_name="stranger", process_id=0)


def state_update(
    *, publisher: MetaData, ids: list[UUID], states: list[Any]
) -> std_msgs.msg.String:
    """
    :param publisher: The publisher of the update.
    :param ids: The degrees of freedom the update is about.
    :param states: Their positions.
    :return: The serialized state update, as the first message of its publisher.
    """
    update = WorldUpdate(
        meta_data=publisher,
        state_update=WorldStateUpdate(meta_data=publisher, ids=ids, states=states),
        sequence_number=1,
    )
    return std_msgs.msg.String(data=json.dumps(to_json(update)))


def state_update_of_unknown_degree_of_freedom(
    publisher: MetaData,
) -> std_msgs.msg.String:
    """
    :param publisher: The publisher of the update.
    :return: A state update about a degree of freedom no receiver has.
    """
    return state_update(publisher=publisher, ids=[uuid4()], states=[1.0])


# %% updates a receiver cannot apply


def test_update_a_receiver_cannot_apply_does_not_raise(rclpy_node, stranger):
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=create_dummy_world(),
        topic_name=unique_topic("cannot_apply"),
    )

    synchronizer.subscription_callback(
        state_update_of_unknown_degree_of_freedom(stranger)
    )

    synchronizer.close()


def test_receiver_keeps_applying_after_an_update_it_cannot_apply(rclpy_node, stranger):
    topic = unique_topic("keeps_applying")
    publishing_world = create_dummy_world()
    receiving_world = create_dummy_world()
    publisher = WorldSynchronizer(
        node=rclpy_node, _world=publishing_world, topic_name=topic
    )
    receiver = WorldSynchronizer(
        node=rclpy_node, _world=receiving_world, topic_name=topic
    )

    stranger_publisher = rclpy_node.create_publisher(std_msgs.msg.String, topic, 10)
    stranger_publisher.publish(state_update_of_unknown_degree_of_freedom(stranger))
    publishing_world.state._data[0, 0] = 4.2
    publishing_world.notify_state_change()

    assert wait_for_condition(lambda: receiving_world.state._data[0, 0] == 4.2)
    publisher.close()
    receiver.close()


def test_receiver_does_not_report_caught_up_with_a_publisher_it_diverged_from(
    rclpy_node,
):
    topic = unique_topic("diverged")
    publishing_world = create_dummy_world()
    receiving_world = create_dummy_world()
    publisher = WorldSynchronizer(
        node=rclpy_node, _world=publishing_world, topic_name=topic
    )
    receiver = WorldSynchronizer(
        node=rclpy_node, _world=receiving_world, topic_name=topic
    )
    unknown_to_receiver = Body(name=PrefixedName("unknown"))
    with publishing_world.modify_world(publish_changes=False):
        publishing_world.add_connection(
            FixedConnection(parent=publishing_world.root, child=unknown_to_receiver)
        )
    with publishing_world.modify_world():
        publishing_world.add_connection(
            FixedConnection(
                parent=unknown_to_receiver, child=Body(name=PrefixedName("child"))
            )
        )
    publishing_world.state._data[0, 0] = 4.2
    publishing_world.notify_state_change()

    assert wait_for_condition(lambda: receiving_world.state._data[0, 0] == 4.2)
    assert not receiver.has_applied(publisher.latest_published_position)
    publisher.close()
    receiver.close()


def test_receiver_caught_up_again_once_divergence_is_forgotten(rclpy_node, stranger):
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=create_dummy_world(),
        topic_name=unique_topic("forgotten"),
    )
    synchronizer.subscription_callback(
        state_update_of_unknown_degree_of_freedom(stranger)
    )

    synchronizer.forget_divergence_from(publisher=stranger)

    assert synchronizer.has_applied(StreamPosition(origin=stranger, sequence_number=0))
    synchronizer.close()


# %% telling the owner of the world


def test_owner_is_told_why_an_update_could_not_be_applied(rclpy_node, stranger):
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=create_dummy_world(),
        topic_name=unique_topic("owner_told"),
    )

    synchronizer.subscription_callback(
        state_update_of_unknown_degree_of_freedom(stranger)
    )

    divergence = synchronizer.divergence_from(publisher=stranger)
    assert isinstance(
        divergence.raised_from, StateUpdateContainsUnknownDegreesOfFreedomError
    )
    synchronizer.close()


def test_asking_for_a_divergence_does_not_consume_it(rclpy_node, stranger):
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=create_dummy_world(),
        topic_name=unique_topic("not_consumed"),
    )
    synchronizer.subscription_callback(
        state_update_of_unknown_degree_of_freedom(stranger)
    )
    first_answer = synchronizer.divergence_from(publisher=stranger)

    assert synchronizer.divergence_from(publisher=stranger) is first_answer
    synchronizer.close()


def test_update_whose_state_is_no_number_is_reported(rclpy_node, stranger):
    world = create_dummy_world()
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=unique_topic("no_number")
    )

    synchronizer.subscription_callback(
        state_update(
            publisher=stranger,
            ids=[world.degrees_of_freedom[0].id],
            states=["not a number"],
        )
    )

    divergence = synchronizer.divergence_from(publisher=stranger)
    assert isinstance(divergence.raised_from, ValueError)
    synchronizer.close()


# %% buffered updates that cannot be applied


def test_buffered_update_that_cannot_be_applied_keeps_the_rest_of_the_batch(
    rclpy_node, stranger
):
    world = create_dummy_world()
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=world,
        topic_name=unique_topic("batch"),
        defer_incoming_updates=True,
    )
    other_publisher = MetaData(node_name="other", process_id=1)
    synchronizer.subscription_callback(
        state_update_of_unknown_degree_of_freedom(stranger)
    )
    synchronizer.subscription_callback(
        state_update(
            publisher=other_publisher,
            ids=[world.degrees_of_freedom[0].id],
            states=[4.2],
        )
    )

    synchronizer.apply_missed_messages()

    assert world.state._data[0, 0] == 4.2
    synchronizer.close()


def test_buffered_update_that_cannot_be_applied_is_reported(rclpy_node, stranger):
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=create_dummy_world(),
        topic_name=unique_topic("batch_reported"),
        defer_incoming_updates=True,
    )
    synchronizer.subscription_callback(
        state_update_of_unknown_degree_of_freedom(stranger)
    )

    synchronizer.apply_missed_messages()

    divergence = synchronizer.divergence_from(publisher=stranger)
    assert isinstance(
        divergence.raised_from, StateUpdateContainsUnknownDegreesOfFreedomError
    )
    synchronizer.close()


# %% own echo


def test_own_echo_is_ignored_without_reading_it(rclpy_node):
    world = create_dummy_world()
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=unique_topic("own_echo")
    )
    gone = FixedConnection(
        parent=Body(name=PrefixedName("gone_parent")),
        child=Body(name=PrefixedName("gone_child")),
    )
    echo = WorldUpdate(
        meta_data=synchronizer.meta_data,
        modification_block=ModificationBlock(
            meta_data=synchronizer.meta_data,
            modifications=WorldModelModificationBlock(
                modifications=[AddConnectionModification(gone)]
            ),
        ),
        sequence_number=1,
    )

    synchronizer.subscription_callback(
        std_msgs.msg.String(data=json.dumps(to_json(echo)))
    )

    synchronizer.close()


# %% buffered updates building on each other


def test_deferred_updates_building_on_each_other_are_applied(rclpy_node):
    topic = unique_topic("deferred_chain")
    publishing_world = World()
    receiving_world = World()
    with publishing_world.modify_world():
        publishing_world.add_kinematic_structure_entity(Body(name=PrefixedName("root")))
    with receiving_world.modify_world():
        receiving_world.add_kinematic_structure_entity(
            Body(name=PrefixedName("root"), id=publishing_world.root.id)
        )
    publisher = WorldSynchronizer(
        node=rclpy_node, _world=publishing_world, topic_name=topic
    )
    receiver = WorldSynchronizer(
        node=rclpy_node,
        _world=receiving_world,
        topic_name=topic,
        defer_incoming_updates=True,
    )
    tray = Body(name=PrefixedName("tray"))
    with publishing_world.modify_world():
        publishing_world.add_connection(
            FixedConnection(parent=publishing_world.root, child=tray)
        )
    with publishing_world.modify_world():
        publishing_world.add_connection(
            FixedConnection(parent=tray, child=Body(name=PrefixedName("cup")))
        )
    wait_for_condition(lambda: receiver.has_buffered_model_modification)
    wait_for_condition(lambda: len(receiver.missed_messages) >= 4)

    receiver.apply_missed_messages()

    assert sorted(
        entity.name.name for entity in receiving_world.kinematic_structure_entities
    ) == ["cup", "root", "tray"]
    publisher.close()
    receiver.close()

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
from krrood_synchronization.messages import MessageOrigin, Sender, StreamPosition
from semantic_digital_twin.adapters.ros.messages import (
    ModificationBlock,
    ReceivedMessage,
    WorldStateUpdate,
    WorldUpdate,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelReloadSynchronizer,
    WorldSynchronizer,
)
from semantic_digital_twin.exceptions import (
    StateUpdateContainsNonNumericValuesError,
    StateUpdateContainsUnknownDegreesOfFreedomError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    AddConnectionModification,
    AddKinematicStructureEntityModification,
    WorldModelModificationBlock,
)
from .test_world_synchronizer import (
    create_dummy_world,
    unique_topic,
    wait_for_condition,
)


# %% helpers


@pytest.fixture
def stranger() -> MessageOrigin:
    """
    A publisher no receiver shares a model with.
    """
    return MessageOrigin(sender=Sender(node_name="stranger", process_id=0))


def state_update(
    *, publisher: MessageOrigin, ids: list[UUID], states: list[Any]
) -> std_msgs.msg.String:
    """
    :param publisher: The publisher of the update.
    :param ids: The degrees of freedom the update is about.
    :param states: Their positions.
    :return: The serialized state update, as the first message of its publisher.
    """
    update = WorldUpdate(
        position=StreamPosition(origin=publisher, sequence_number=1),
        state_update=WorldStateUpdate(
            position=StreamPosition(origin=publisher), ids=ids, states=states
        ),
    )
    return std_msgs.msg.String(data=json.dumps(to_json(update)))


def state_update_of_unknown_degree_of_freedom(
    publisher: MessageOrigin,
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
    """
    An update that builds on a change the receiver never got cannot be applied, so the
    receiver no longer counts as caught up with that publisher, even after later
    updates.
    """
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
    assert not receiver.has_applied(position=publisher.latest_published_position)
    publisher.close()
    receiver.close()


def test_receiver_caught_up_again_once_divergence_is_forgotten(rclpy_node, stranger):
    """
    Forgetting the divergence from a publisher counts it as caught up with again.
    """
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=create_dummy_world(),
        topic_name=unique_topic("forgotten"),
    )
    synchronizer.subscription_callback(
        state_update_of_unknown_degree_of_freedom(stranger)
    )

    synchronizer.forget_divergence_from(publisher=stranger)

    assert synchronizer.has_applied(
        position=StreamPosition(origin=stranger, sequence_number=0)
    )
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
    assert isinstance(divergence.raised_from, StateUpdateContainsNonNumericValuesError)
    synchronizer.close()


def test_programming_error_while_applying_is_not_swallowed(
    rclpy_node, stranger, monkeypatch
):
    """
    An error that is not a :class:`~krrood.exceptions.DataclassException` is a
    programming error, so it is raised instead of logged.
    """
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=create_dummy_world(),
        topic_name=unique_topic("programming_error"),
    )

    def fail_like_a_bug(message: ReceivedMessage) -> None:
        """
        Fail as a programming error would, not as an update the world cannot apply.
        """
        raise KeyError(message.position.sequence_number)

    monkeypatch.setattr(synchronizer, "_subscription_callback", fail_like_a_bug)

    with pytest.raises(KeyError):
        synchronizer.subscription_callback(
            state_update_of_unknown_degree_of_freedom(stranger)
        )
    synchronizer.close()


# %% updates applied entirely or not at all


def test_combined_update_with_unknown_state_leaves_the_model_untouched(
    rclpy_node, stranger
):
    """
    A combined update whose state cannot be applied leaves out its model modification
    too.
    """
    world = create_dummy_world()
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=unique_topic("all_or_nothing")
    )
    combined = WorldUpdate(
        position=StreamPosition(origin=stranger, sequence_number=1),
        modification_block=ModificationBlock(
            position=StreamPosition(origin=stranger),
            modifications=WorldModelModificationBlock(
                modifications=[
                    AddKinematicStructureEntityModification(
                        Body(name=PrefixedName("extra"))
                    )
                ]
            ),
        ),
        state_update=WorldStateUpdate(
            position=StreamPosition(origin=stranger), ids=[uuid4()], states=[1.0]
        ),
    )

    synchronizer.subscription_callback(
        std_msgs.msg.String(data=json.dumps(to_json(combined)))
    )

    assert "extra" not in [
        entity.name.name for entity in world.kinematic_structure_entities
    ]
    synchronizer.close()


# %% recovering by reloading the world


def test_reload_clears_the_divergence_from_the_reloaded_world(
    rclpy_node, in_memory_session_maker
):
    """
    Reloading the world of a publisher counts every synchronizer of that world as caught
    up with again.
    """
    world_topic = unique_topic("reload_world")
    reload_topic = unique_topic("reload_model")
    publishing_world = create_dummy_world()
    receiving_world = create_dummy_world()
    publisher = WorldSynchronizer(
        node=rclpy_node, _world=publishing_world, topic_name=world_topic
    )
    receiver = WorldSynchronizer(
        node=rclpy_node, _world=receiving_world, topic_name=world_topic
    )
    publishing_reloader = ModelReloadSynchronizer(
        node=rclpy_node,
        _world=publishing_world,
        session=in_memory_session_maker(),
        topic_name=reload_topic,
    )
    receiving_reloader = ModelReloadSynchronizer(
        node=rclpy_node,
        _world=receiving_world,
        session=in_memory_session_maker(),
        topic_name=reload_topic,
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
    assert wait_for_condition(
        lambda: receiver.divergence_from(publisher=publisher.origin) is not None
    )

    publishing_reloader.publish_reload_model()

    assert wait_for_condition(
        lambda: receiver.divergence_from(publisher=publisher.origin) is None
    )
    for synchronizer in (publisher, receiver, publishing_reloader, receiving_reloader):
        synchronizer.close()


# %% buffered updates that cannot be applied


def test_buffered_update_that_cannot_be_applied_keeps_the_rest_of_the_batch(
    rclpy_node, stranger
):
    """
    A buffered update that cannot be applied does not keep the updates after it from
    being applied.
    """
    world = create_dummy_world()
    synchronizer = WorldSynchronizer(
        node=rclpy_node,
        _world=world,
        topic_name=unique_topic("batch"),
        defer_incoming_updates=True,
    )
    other_publisher = MessageOrigin(sender=Sender(node_name="other", process_id=1))
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
    """
    A synchronizer drops its own echo before building it against the world, which need
    not hold what the echo refers to.
    """
    world = create_dummy_world()
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=unique_topic("own_echo")
    )
    gone = FixedConnection(
        parent=Body(name=PrefixedName("gone_parent")),
        child=Body(name=PrefixedName("gone_child")),
    )
    echo = WorldUpdate(
        position=StreamPosition(origin=synchronizer.origin, sequence_number=1),
        modification_block=ModificationBlock(
            position=StreamPosition(origin=synchronizer.origin),
            modifications=WorldModelModificationBlock(
                modifications=[AddConnectionModification(gone)]
            ),
        ),
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

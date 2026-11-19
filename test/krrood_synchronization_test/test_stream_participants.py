"""
Tests that a publisher numbers its messages and that a receiver knows, per publisher,
how far it received and applied their streams.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from krrood.adapters.json_serializer import from_json, to_json
from krrood_synchronization.exceptions import DivergedFromPublisherError
from krrood_synchronization.messages import MessageOrigin, Sender, StreamPosition
from krrood_synchronization.stream_participants import LostMessages

from .dataset.counter_peers import CounterPeer, CounterReading

# %% fixtures


@pytest.fixture
def sender() -> CounterPeer:
    """
    A process that publishes counter readings.
    """
    return CounterPeer(
        own_origin=MessageOrigin(sender=Sender(node_name="sender", process_id=1))
    )


@pytest.fixture
def receiver() -> CounterPeer:
    """
    A process that receives the readings of others.
    """
    return CounterPeer(
        own_origin=MessageOrigin(sender=Sender(node_name="receiver", process_id=2))
    )


def stream_position(*, publisher: CounterPeer, sequence_number: int) -> StreamPosition:
    """
    :return: The position ``sequence_number`` in the stream of ``publisher``.
    """
    return StreamPosition(origin=publisher.origin, sequence_number=sequence_number)


# %% origins and messages


def test_origin_survives_json_round_trip():
    """
    An origin sent along with a message is the same publisher on the receiving side.
    """
    origin = MessageOrigin(sender=Sender(node_name="sender", process_id=1))

    assert from_json(to_json(origin)) == origin


def test_message_survives_json_round_trip(sender):
    """
    A message keeps its origin, its position and its content on the way through json.
    """
    reading = sender.read_counter(reading=7)

    assert from_json(to_json(reading)) == reading


def test_origins_of_one_model_in_one_process_publish_the_same_copy():
    """
    Two publishers of one process that carry the changes of the same model publish the
    same copy of it.
    """
    model_id = uuid4()
    reloads = MessageOrigin(
        sender=Sender(node_name="reloads", process_id=1), model_id=model_id
    )
    updates = MessageOrigin(
        sender=Sender(node_name="updates", process_id=1), model_id=model_id
    )

    assert reloads.publishes_same_model_copy_as(other=updates)


def test_origins_of_one_model_in_two_processes_publish_different_copies():
    """
    Two processes never publish the same copy of a model, even under the same model id.
    """
    model_id = uuid4()
    first = MessageOrigin(
        sender=Sender(node_name="updates", process_id=1), model_id=model_id
    )
    second = MessageOrigin(
        sender=Sender(node_name="updates", process_id=2), model_id=model_id
    )

    assert not first.publishes_same_model_copy_as(other=second)


# %% publishing


def test_nothing_is_published_at_first(sender):
    """
    A publisher that sent nothing yet is at the start of its stream.
    """
    assert sender.latest_published_position == stream_position(
        publisher=sender, sequence_number=0
    )


def test_messages_are_numbered_in_the_order_they_are_sent(sender):
    """
    Every message gets the position after the one numbered before it, starting at one.
    """
    readings = [sender.read_counter(reading=reading) for reading in range(3)]

    assert [reading.position.sequence_number for reading in readings] == [1, 2, 3]


def test_numbered_message_stays_in_the_stream_of_its_publisher(sender):
    """
    Numbering a message moves it along the stream it was built in, not into another.
    """
    reading = sender.read_counter(reading=1)

    assert reading.position.origin == sender.origin


def test_latest_published_position_is_the_last_numbered_message(sender):
    """
    What others have to catch up with is the message that was numbered last.
    """
    sender.read_counter(reading=1)
    last = sender.read_counter(reading=2)

    assert sender.latest_published_position == last.position


# %% receiving and applying


def test_first_message_of_a_publisher_reveals_no_loss(sender, receiver):
    """
    A receiver that hears from a publisher for the first time may have joined late, so
    whatever came before is not counted as lost.
    """
    sender.read_counter(reading=1)
    later = sender.read_counter(reading=2)

    assert receiver.record_received(position=later.position) is None


def test_consecutive_messages_reveal_no_loss(sender, receiver):
    """
    A message that follows the one received before reveals no loss.
    """
    first = sender.read_counter(reading=1)
    second = sender.read_counter(reading=2)
    receiver.record_received(position=first.position)

    assert receiver.record_received(position=second.position) is None


def test_gap_in_the_stream_reveals_the_lost_messages(sender, receiver):
    """
    A message that skips positions reveals exactly the messages in between as lost.
    """
    first = sender.read_counter(reading=1)
    second = sender.read_counter(reading=2)
    third = sender.read_counter(reading=3)
    fourth = sender.read_counter(reading=4)
    receiver.record_received(position=first.position)

    assert receiver.record_received(position=fourth.position) == LostMessages(
        publisher=sender.origin,
        first_sequence_number=second.position.sequence_number,
        last_sequence_number=third.position.sequence_number,
    )


def test_start_of_an_unknown_stream_counts_as_applied(sender, receiver):
    """
    Nothing has to be applied to be at the start of a stream never heard from.
    """
    assert receiver.has_applied(
        position=stream_position(publisher=sender, sequence_number=0)
    )


def test_message_of_an_unknown_stream_is_not_applied(sender, receiver):
    """
    The first message of a publisher never heard from is still awaited.
    """
    reading = sender.read_counter(reading=1)

    assert not receiver.has_applied(position=reading.position)


def test_received_message_is_not_applied_before_it_is_recorded_so(sender, receiver):
    """
    A message that was received but not applied yet, for example because it was
    buffered, does not count as caught up with.
    """
    reading = sender.read_counter(reading=1)
    receiver.record_received(position=reading.position)

    assert not receiver.has_applied(position=reading.position)


def test_applied_message_covers_the_messages_before_it(sender, receiver):
    """
    Having applied a position means having applied everything before it.
    """
    first = sender.read_counter(reading=1)
    second = sender.read_counter(reading=2)
    receiver.record_applied(position=second.position)

    assert receiver.has_applied(position=first.position)


def test_applying_an_older_message_does_not_move_back(sender, receiver):
    """
    Applying a message older than one applied before keeps the later position.
    """
    first = sender.read_counter(reading=1)
    second = sender.read_counter(reading=2)
    receiver.record_applied(position=second.position)
    receiver.record_applied(position=first.position)

    assert receiver.has_applied(position=second.position)


def test_streams_of_different_publishers_are_kept_apart(sender, receiver):
    """
    Catching up with one publisher says nothing about another.
    """
    other = CounterPeer(
        own_origin=MessageOrigin(sender=Sender(node_name="other", process_id=3))
    )
    receiver.record_applied(position=sender.read_counter(reading=1).position)

    assert not receiver.has_applied(position=other.read_counter(reading=1).position)


# %% divergences


def test_divergence_stops_counting_the_publisher_as_caught_up(sender, receiver):
    """
    A publisher a message of which could not be applied is not caught up with, not even
    at the start of its stream.
    """
    reading = sender.read_counter(reading=1)
    receiver.record_applied(position=reading.position)
    receiver.record_divergence(
        divergence=DivergedFromPublisherError(position=reading.position)
    )

    assert not receiver.has_applied(
        position=stream_position(publisher=sender, sequence_number=0)
    )


def test_first_divergence_from_a_publisher_is_kept(sender, receiver):
    """
    The first message that could not be applied is the one the copy may differ from
    since, so later failures do not replace it.
    """
    first = DivergedFromPublisherError(position=sender.read_counter(reading=1).position)
    second = DivergedFromPublisherError(
        position=sender.read_counter(reading=2).position
    )
    receiver.record_divergence(divergence=first)
    receiver.record_divergence(divergence=second)

    assert receiver.divergence_from(publisher=sender.origin) is first


def test_forgotten_divergence_counts_the_publisher_as_caught_up(sender, receiver):
    """
    Once a divergence is forgotten, the positions applied before count again.
    """
    reading = sender.read_counter(reading=1)
    receiver.record_applied(position=reading.position)
    receiver.record_divergence(
        divergence=DivergedFromPublisherError(position=reading.position)
    )

    receiver.forget_divergence_from(publisher=sender.origin)

    assert receiver.has_applied(position=reading.position)


def test_replaced_model_forgets_divergences_from_every_publisher_of_it(receiver):
    """
    Replacing the copy by the one of a publisher makes it equal to every publisher that
    shares that model, such as its reloads and its updates.
    """
    model_id = uuid4()
    reloads = MessageOrigin(
        sender=Sender(node_name="reloads", process_id=1), model_id=model_id
    )
    updates = MessageOrigin(
        sender=Sender(node_name="updates", process_id=1), model_id=model_id
    )
    for publisher in [reloads, updates]:
        receiver.record_divergence(
            divergence=DivergedFromPublisherError(
                position=StreamPosition(origin=publisher, sequence_number=1)
            )
        )

    receiver.forget_divergences_from_same_model(publisher=reloads)

    assert [
        receiver.divergence_from(publisher=publisher)
        for publisher in [reloads, updates]
    ] == [None, None]


def test_replaced_model_keeps_divergences_from_other_models(sender, receiver):
    """
    Replacing the copy by the one of a publisher says nothing about other models.
    """
    divergence = DivergedFromPublisherError(
        position=sender.read_counter(reading=1).position
    )
    receiver.record_divergence(divergence=divergence)

    receiver.forget_divergences_from_same_model(
        publisher=MessageOrigin(sender=Sender(node_name="reloads", process_id=1))
    )

    assert receiver.divergence_from(publisher=sender.origin) is divergence

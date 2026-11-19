"""
The messages of a stream, who sends them and where they sit in it.

A stream carries the changes of a model that several processes hold copies of, so that
every copy can follow the one its publisher changes.
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from typing_extensions import Any, Self

from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    SubclassJSONSerializer,
)
from krrood.patterns.caching import memoize

# %% sender, origin and position


@dataclass(kw_only=True, frozen=True)
class Sender:
    """
    A node of a process that sends messages.
    """

    node_name: str
    """
    The name of the node that sends the messages.
    """

    process_id: int
    """
    The id of the process the node runs in.
    """


@dataclass(kw_only=True, frozen=True)
class MessageOrigin(SubclassJSONSerializer):
    """
    The publisher of a stream of messages: a sender together with the copy of the model
    whose changes it publishes.

    Two messages come from the same stream exactly if their origins are equal.
    """

    sender: Sender
    """
    The node that publishes the stream.
    """

    model_id: UUID = field(default_factory=uuid.uuid4)
    """
    The id of the model whose changes the stream carries.

    The copies of a model in different processes may share this id. Several publishers of
    one process may carry the changes of the same copy.
    """

    def publishes_same_model_copy_as(self, *, other: MessageOrigin) -> bool:
        """
        :param other: Another publisher.
        :return: Whether both publish the changes of the same copy of a model, which is
            the copy of that model in the process they both run in.
        """
        return (
            self.sender.process_id == other.sender.process_id
            and self.model_id == other.model_id
        )

    @memoize
    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        :meth:`SubclassJSONSerializer.to_json`, computed once, since every message of
        the stream carries the same origin.
        """
        return DataclassJSONSerializer.to_json(instance=self, **kwargs)

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        The inverse of :meth:`to_json`.
        """
        return DataclassJSONSerializer.from_json(data=data, clazz=cls, **kwargs)


@dataclass(kw_only=True, frozen=True)
class StreamPosition:
    """
    A position in the stream of messages that one publisher sent.

    Positions of different publishers are not comparable, which is why the publisher is
    part of the position.
    """

    origin: MessageOrigin
    """
    The publisher whose stream this position belongs to.
    """

    sequence_number: int = 0
    """
    The position in the stream of that publisher.

    Counts up by one per publication, starting at one. Zero is the start of the stream,
    before the first message.
    """


# %% messages


@dataclass(kw_only=True)
class Message(ABC):
    """
    A message of a stream.
    """

    position: StreamPosition
    """
    Where this message sits in the stream of its publisher.

    A message that was never published sits at the start of the stream.
    """


class MessageField(StrEnum):
    """
    The parts of a serialized message that are read without building it.
    """

    POSITION = "position"
    """
    Where the message sits in the stream of its publisher.
    """

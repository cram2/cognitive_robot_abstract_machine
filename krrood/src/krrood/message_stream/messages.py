"""
The messages of a stream, where they come from and where they sit in it.
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

# %% origin and position


@dataclass(kw_only=True, frozen=True)
class MessageOrigin:
    """
    The publisher of a stream of messages.

    Two messages come from the same stream exactly if their origins are equal.
    """

    node_name: str
    """
    The name of the node that published the message.
    """

    process_id: int
    """
    The id of the process that published the message.
    """

    model_id: UUID = field(default_factory=uuid.uuid4)
    """
    The id of the copy of the model whose changes the stream carries.

    Several publishers of one process may carry the changes of the same copy.
    """

    def shares_model_with(self, *, other: MessageOrigin) -> bool:
        """
        :param other: Another publisher.
        :return: Whether both publish the changes of the same copy of a model, from the
            same process.
        """
        return self.process_id == other.process_id and self.model_id == other.model_id


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

    sequence_number: int
    """
    The position in the stream of that publisher.
    """


# %% messages


@dataclass(kw_only=True)
class Message(ABC):
    """
    A message of a stream.
    """

    origin: MessageOrigin
    """
    The publisher of the message.
    """

    sequence_number: int = 0
    """
    Position of this message in the stream of its publisher.

    Counts up by one per publication, starting at one. A message that was never
    published has the position zero.
    """

    @property
    def position(self) -> StreamPosition:
        """
        Where this message sits in the stream of its publisher.
        """
        return StreamPosition(origin=self.origin, sequence_number=self.sequence_number)


class MessageField(StrEnum):
    """
    The parts of a serialized message that are read without building it.
    """

    ORIGIN = "origin"
    """
    The publisher of the message.
    """

    SEQUENCE_NUMBER = "sequence_number"
    """
    The position of the message in the stream of its publisher.
    """

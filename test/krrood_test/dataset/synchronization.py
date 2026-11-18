"""
A process that publishes a stream of counter readings and receives the streams of
others.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.synchronization.messages import Message, MessageOrigin, StreamPosition
from krrood.synchronization.stream_participants import StreamPublisher, StreamReceiver

# %% messages


@dataclass(kw_only=True)
class CounterReading(Message):
    """
    A message that carries one reading of a counter.
    """

    reading: int
    """
    The value the counter showed.
    """


# %% participants


@dataclass(kw_only=True)
class CounterPeer(StreamPublisher, StreamReceiver):
    """
    A process that publishes its counter readings and receives those of others.
    """

    own_origin: MessageOrigin
    """
    The origin every reading of this process carries.
    """

    @property
    def origin(self) -> MessageOrigin:
        return self.own_origin

    def read_counter(self, *, reading: int) -> CounterReading:
        """
        :param reading: The value the counter shows.
        :return: The reading, numbered as the next message of this process.
        """
        message = CounterReading(
            position=StreamPosition(origin=self.origin), reading=reading
        )
        self.assign_next_sequence_number(message=message)
        return message

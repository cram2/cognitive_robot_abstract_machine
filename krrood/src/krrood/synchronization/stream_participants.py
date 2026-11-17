"""
What a process that publishes or receives a stream of messages keeps track of: how far
it published, and how far it caught up with every other publisher.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from krrood.synchronization.exceptions import DivergedFromPublisherError
from krrood.synchronization.messages import Message, MessageOrigin, StreamPosition

# %% publishing


class PublicationProgress(ABC):
    """
    Reports how far its own changes were published to the other processes.

    Whoever hands out a :class:`StreamPosition` of its own stream tells others what they
    have to catch up with before they may read the model it publishes.
    """

    @property
    @abstractmethod
    def published_sequence_number(self) -> int:
        """
        Position of the message that was published last, zero before the first one.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def latest_published_position(self) -> StreamPosition:
        """
        The position of the last published message, together with its publisher.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class StreamPublisher(PublicationProgress, ABC):
    """
    Publishes a stream of messages and numbers each message by its position in it.

    Thread-safe: messages may be numbered while others read how far it published.
    """

    _published_sequence_number: int = field(default=0, init=False, repr=False)
    """
    Position of the message that was numbered last.
    """

    _publication_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """
    Guards :attr:`_published_sequence_number`.
    """

    @property
    @abstractmethod
    def origin(self) -> MessageOrigin:
        """
        The publisher every message of this stream comes from.
        """
        raise NotImplementedError

    @property
    def published_sequence_number(self) -> int:
        with self._publication_lock:
            return self._published_sequence_number

    @property
    def latest_published_position(self) -> StreamPosition:
        return StreamPosition(
            origin=self.origin, sequence_number=self.published_sequence_number
        )

    def assign_next_sequence_number(self, *, message: Message) -> None:
        """
        Give ``message`` the position after the message numbered last.

        Call it right before ``message`` is sent, in the order the messages are sent.
        """
        with self._publication_lock:
            self._published_sequence_number += 1
            message.sequence_number = self._published_sequence_number


# %% receiving


@dataclass(kw_only=True)
class LostMessages:
    """
    Messages of a publisher that never arrived, found because a later one did.
    """

    publisher: MessageOrigin
    """
    The publisher whose messages were lost.
    """

    first_sequence_number: int
    """
    The position of the first lost message.
    """

    last_sequence_number: int
    """
    The position of the last lost message.
    """


@dataclass(kw_only=True)
class StreamReceiver:
    """
    Receives the streams of other publishers and remembers, per publisher, how far it
    received them, how far it applied them and which message it could not apply.

    Thread-safe: messages may be received and applied on different threads while others
    ask how far it caught up.
    """

    _received_sequence_numbers: dict[MessageOrigin, int] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The position of the message received last, per publisher.
    """

    _applied_sequence_numbers: dict[MessageOrigin, int] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The highest position applied so far, per publisher.
    """

    _divergences: dict[MessageOrigin, DivergedFromPublisherError] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The first message that could not be applied, per publisher this copy may differ from
    since.
    """

    _reception_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """
    Guards the received and applied positions and the divergences.
    """

    def record_received(self, *, position: StreamPosition) -> LostMessages | None:
        """
        Remember that the message at ``position`` arrived.

        :param position: The position of a message just received.
        :return: The messages of its publisher between the one received before and this
            one, or ``None`` if none are missing.
        """
        publisher = position.origin
        with self._reception_lock:
            last_received = self._received_sequence_numbers.get(publisher)
            self._received_sequence_numbers[publisher] = position.sequence_number
        if last_received is None or position.sequence_number <= last_received + 1:
            return None
        return LostMessages(
            publisher=publisher,
            first_sequence_number=last_received + 1,
            last_sequence_number=position.sequence_number - 1,
        )

    def record_applied(self, *, position: StreamPosition) -> None:
        """
        Remember that everything up to ``position`` was applied.

        Call it when a message is applied rather than when it is received, so that a
        buffered message does not count as caught up with.
        """
        with self._reception_lock:
            self._applied_sequence_numbers[position.origin] = max(
                self._applied_sequence_numbers.get(position.origin, 0),
                position.sequence_number,
            )

    def has_applied(self, *, position: StreamPosition) -> bool:
        """
        Whether everything up to ``position`` was applied.

        A publisher never heard from counts as being at the start of its stream. A
        publisher a message of which could not be applied is not caught up with, see
        :meth:`divergence_from`.
        """
        with self._reception_lock:
            if position.origin in self._divergences:
                return False
            applied = self._applied_sequence_numbers.get(position.origin, 0)
        return applied >= position.sequence_number

    def record_divergence(self, *, divergence: DivergedFromPublisherError) -> None:
        """
        Remember that a message could not be applied, unless an earlier message of the
        same publisher could not be applied either.
        """
        with self._reception_lock:
            self._divergences.setdefault(divergence.position.origin, divergence)

    def divergence_from(
        self, *, publisher: MessageOrigin
    ) -> DivergedFromPublisherError | None:
        """
        :param publisher: Another publisher.
        :return: The first message of ``publisher`` that could not be applied since this
            copy last caught up with it in full, or ``None`` if every one was applied.
        """
        with self._reception_lock:
            return self._divergences.get(publisher)

    def forget_divergence_from(self, *, publisher: MessageOrigin) -> None:
        """
        Count ``publisher`` as caught up with again, once this copy was made equal to
        the publisher's, for example by fetching it again.
        """
        with self._reception_lock:
            self._divergences.pop(publisher, None)

    def forget_divergences_from_same_model(self, *, publisher: MessageOrigin) -> None:
        """
        Count every publisher that shares its copy of the model with ``publisher`` as
        caught up with again, once this copy was replaced by that copy.
        """
        with self._reception_lock:
            caught_up = [
                diverged
                for diverged in self._divergences
                if diverged.shares_model_with(other=publisher)
            ]
            for diverged in caught_up:
                del self._divergences[diverged]

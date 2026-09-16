"""
A feed of the events a watched run detects, kept for whoever reads them later or as
they come, such as the live event dashboard.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Dict, List, Optional, Protocol, runtime_checkable

from segmind.datastructures.events import DetectionEvent, EventWithTrackedObjects


class FeedField(StrEnum):
    """
    The fields an event is shown by.
    """

    TRACKED_OBJECT = "tracked_object"
    """
    The name of the object the event is about.
    """

    EVENT_TYPE = "event_type"
    """
    The kind of event.
    """

    WITH_OBJECT = "with_object"
    """
    The name of the object it happened with, if any.
    """

    TIMESTAMP = "timestamp"
    """
    When it happened, in ISO 8601.
    """


@dataclass(frozen=True)
class EventRow:
    """
    An event reduced to the fields it is shown by.
    """

    tracked_object: Optional[str]
    """
    See :attr:`FeedField.TRACKED_OBJECT`; None for an event about no object.
    """

    event_type: str
    """
    See :attr:`FeedField.EVENT_TYPE`.
    """

    with_object: Optional[str]
    """
    See :attr:`FeedField.WITH_OBJECT`; None for an event with no other object.
    """

    timestamp: str
    """
    See :attr:`FeedField.TIMESTAMP`.
    """

    @classmethod
    def of(cls, event: DetectionEvent) -> EventRow:
        """
        :param event: The event to show.
        """
        if not isinstance(event, EventWithTrackedObjects):
            return cls(
                tracked_object=None,
                event_type=type(event).__name__,
                with_object=None,
                timestamp=event.timestamp.isoformat(),
            )
        return cls(
            tracked_object=str(event.tracked_object.name),
            event_type=type(event).__name__,
            with_object=(
                None if event.with_object is None else str(event.with_object.name)
            ),
            timestamp=event.timestamp.isoformat(),
        )

    def to_json(self) -> Dict[FeedField, Optional[str]]:
        """
        :return: The row, keyed by the field each value is.
        """
        return {
            FeedField.TRACKED_OBJECT: self.tracked_object,
            FeedField.EVENT_TYPE: self.event_type,
            FeedField.WITH_OBJECT: self.with_object,
            FeedField.TIMESTAMP: self.timestamp,
        }


@runtime_checkable
class ReceivesDetectedEvents(Protocol):
    """
    Anything told about events as a run detects them.
    """

    def receive(self, events: List[DetectionEvent]) -> None:
        """
        Take newly detected events.

        :param events: The events, oldest first.
        """


@dataclass
class EventFeed:
    """
    Keeps every event it is told about, for any number of readers on any thread.

    Listens to a :class:`~segmind.live_segmenter.LiveSegmenter` directly: it is one of
    the events it is watched by.
    """

    _events: List[DetectionEvent] = field(init=False, default_factory=list, repr=False)
    """
    Every event received, oldest first.
    """

    _arrived: threading.Condition = field(
        init=False, default_factory=threading.Condition, repr=False
    )
    """
    Guards :attr:`_events` and wakes readers waiting for more.
    """

    def receive(self, events: List[DetectionEvent]) -> None:
        """
        Keep newly detected events and wake whoever waits for them.

        :param events: The events, oldest first.
        """
        with self._arrived:
            self._events.extend(events)
            self._arrived.notify_all()

    def snapshot(self) -> List[DetectionEvent]:
        """
        :return: Every event received so far, oldest first.
        """
        with self._arrived:
            return list(self._events)

    def subscribe(self) -> Subscription:
        """
        :return: A subscription starting at the first event this feed ever received.
        """
        return Subscription(feed=self)

    def events_after(self, count: int, timeout: float) -> List[DetectionEvent]:
        """
        The events received after the first ``count``, waiting up to ``timeout`` seconds
        for one if there are none yet.

        :param count: How many events the reader has already taken.
        :param timeout: The longest to wait, in seconds.
        :return: The new events, or none if none arrived in time.
        """
        with self._arrived:
            self._arrived.wait_for(lambda: len(self._events) > count, timeout=timeout)
            return self._events[count:]


@dataclass
class Subscription:
    """
    A reader's place in an :class:`EventFeed`: it takes every event exactly once, in the
    order the feed received them, starting with the ones received before it existed.
    """

    feed: EventFeed
    """
    The feed read.
    """

    _taken: int = field(init=False, default=0, repr=False)
    """
    How many of the feed's events this subscription has taken.
    """

    def take(self, timeout: float = 0.0) -> List[DetectionEvent]:
        """
        The events not taken yet, waiting up to ``timeout`` seconds for one.

        :param timeout: The longest to wait, in seconds.
        :return: The new events, or none if none arrived in time.
        """
        events = self.feed.events_after(self._taken, timeout)
        self._taken += len(events)
        return events

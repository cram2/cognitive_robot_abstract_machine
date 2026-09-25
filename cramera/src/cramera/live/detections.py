"""
A running demo's segmind detections, offered to the viewer's queries.

Reading a detector's events is what needs segmind, so it happens here rather than in the
knowledge layer, which the recorded viewer imports and which must stay servable without
segmind's package -- and without the ROS overlay it needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from segmind.datastructures.events import DetectionEvent, EventWithTrackedObjects
from segmind.event_logger import EventLogger
from typing_extensions import List, Optional, Tuple, Type

from cramera.knowledge.detected_events import EVENT_VARIABLE, DetectedEventRecord
from cramera.knowledge.presets import DetectedEventQuestions, Preset
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope


def record_of(event: DetectionEvent) -> DetectedEventRecord:
    """
    One detected event as a record.

    :param event: The event a detector produced.
    """
    tracked, second = _involved_names(event)
    return DetectedEventRecord(
        name=" ".join(part for part in [tracked, type(event).__name__] if part),
        event_type=type(event).__name__,
        timestamp=event.timestamp,
        tracked_object=tracked,
        with_object=second,
    )


def records_of(events: List[DetectionEvent]) -> List[DetectedEventRecord]:
    """
    Every detected event as a record, in the order they were detected.

    :param events: The events a detector produced.
    """
    return [record_of(event) for event in events]


def detectable_event_types() -> Tuple[str, ...]:
    """
    The type of every event a record can be written for, in alphabetical order.

    A type that only serves as the base of others is left out: no detector produces
    one, so no question can ask for it. Only the types this process has imported are
    found, which for a running demo is every type its own detectors can produce.
    """
    return tuple(sorted(event_type.__name__ for event_type in _leaf_event_types()))


def _leaf_event_types() -> List[Type[DetectionEvent]]:
    """
    Every event type a detector can produce: the leaves of segmind's event tree.
    """

    def leaves(base: Type[DetectionEvent]) -> List[Type[DetectionEvent]]:
        """
        Collect the concrete event types below one event class.

        :param base: The event class whose descendants are inspected.
        :return: Event classes with no subclasses.
        """
        found: List[Type[DetectionEvent]] = []
        for subclass in base.__subclasses__():
            found.extend(leaves(subclass) if subclass.__subclasses__() else [subclass])
        return found

    return leaves(DetectionEvent)


def _involved_names(event: DetectionEvent) -> Tuple[Optional[str], Optional[str]]:
    """
    The names of the bodies an event happened to, the primary one first.

    :param event: The event a detector produced.
    """
    if not isinstance(event, EventWithTrackedObjects):
        return None, None
    second = event.with_object
    return (
        event.tracked_object.name.name,
        second.name.name if second is not None else None,
    )


@dataclass
class DetectedEvents:
    """
    A running demo's detections, as a body of knowledge questions can be asked of.

    Any demo that ticks segmind detectors offers this alongside its own knowledge, so
    what a question about the detections may name does not have to be written twice.
    """

    logger: EventLogger
    """
    The event logger the demo's detectors write their events to.
    """

    def knowledge(self) -> QueryableKnowledge:
        """
        What a question about the detections may range over.

        Read fresh on every call, so an answer names every moment detected up to now.
        """
        return QueryableKnowledge(
            scope=QueryScope.DETECTED_EVENTS,
            domains=[
                QueryDomain(
                    EVENT_VARIABLE,
                    DetectedEventRecord,
                    self.records(),
                )
            ],
        )

    def records(self) -> List[DetectedEventRecord]:
        """
        Everything detected so far, oldest first.
        """
        with self.logger.timeline_lock:
            return records_of(list(self.logger.timeline))

    def presets(self) -> List[Preset]:
        """
        The ready-made questions the panel offers as buttons.
        """
        return self._questions().listed()

    def unlisted_presets(self) -> List[Preset]:
        """
        The questions recognized when one is asked, but not shown as buttons.
        """
        return self._questions().unlisted()

    def _questions(self) -> DetectedEventQuestions:
        """
        The questions this demo's detectors give rise to, one per type they can produce.
        """
        return DetectedEventQuestions(detectable_event_types())

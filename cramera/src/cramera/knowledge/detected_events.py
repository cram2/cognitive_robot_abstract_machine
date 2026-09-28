"""
What a run's detectors saw, as something a query can range over.

A detector produces events about bodies of the running world; a question is asked about
them in the same language as everything else, so each event is flattened to the names
and the moment a question actually asks for.

Only that flattened form lives here, and nothing of the detectors themselves: this
module is read by the recorded viewer, which is meant to serve without the detectors'
own package -- and without the ROS overlay that one needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from typing_extensions import Any, Dict, List, Optional

from cramera.recording_fields import SceneField, TrajectoryField
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope

EVENT_VARIABLE = "event"
"""
Name a question binds one detected event to.
"""

EVENT_CLASS_SUFFIX = "Event"
"""
The word every detected event's class name ends in, which is what one of them is called
when a question asks for it out loud.
"""


class EventField(StrEnum):
    """
    Key one detected event is written to a bundle under.
    """

    NAME = "name"
    EVENT_TYPE = "eventType"
    TIMESTAMP = "timestamp"
    TRACKED_OBJECT = "trackedObject"
    WITH_OBJECT = "withObject"


@dataclass(frozen=True)
class DetectedEventRecord:
    """
    One event segmind detected, flattened to what a query asks about it.
    """

    name: str
    """
    The event's label, e.g. ``"milk PickUpEvent"``.
    """

    event_type: str
    """
    The detected event's own type.
    """

    timestamp: datetime
    """
    When the event was detected.
    """

    tracked_object: Optional[str] = None
    """
    Name of the primary body the event was detected on, or None for an event about no
    body in particular.
    """

    with_object: Optional[str] = None
    """
    Name of the second body the event relates the first one to, if any.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The record as a recorded bundle carries it, with the moment as an ISO instant.
        """
        return {
            EventField.NAME.value: self.name,
            EventField.EVENT_TYPE.value: self.event_type,
            EventField.TIMESTAMP.value: self.timestamp.isoformat(),
            EventField.TRACKED_OBJECT.value: self.tracked_object,
            EventField.WITH_OBJECT.value: self.with_object,
        }

    @classmethod
    def of_payload(cls, payload: Dict[str, Any]) -> DetectedEventRecord:
        """
        One record as a bundle carries it.

        :param payload: The event as :meth:`to_payload` wrote it.
        """
        return cls(
            name=payload[EventField.NAME.value],
            event_type=payload[EventField.EVENT_TYPE.value],
            timestamp=datetime.fromisoformat(payload[EventField.TIMESTAMP.value]),
            tracked_object=payload.get(EventField.TRACKED_OBJECT.value),
            with_object=payload.get(EventField.WITH_OBJECT.value),
        )

    @classmethod
    def of_scene(cls, scene: Dict[str, Any]) -> List[DetectedEventRecord]:
        """
        Every detection a recorded scene carries, or none for a scene recorded without
        detectors.

        :param scene: The scene bundle's ``scene.json`` content.
        """
        return [
            cls.of_payload(payload)
            for payload in scene.get(SceneField.DETECTED_EVENTS.value) or []
        ]

"""
Watching a world with SegMind while something else, such as a plan, changes it, and
handing what was detected over as records of names.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from giskardpy.motion_statechart.context import MotionStatechartContext
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json
from typing_extensions import Any, Dict, List, Optional, Self, Tuple, Type

from segmind.datastructures.events import EventWithTrackedObjects
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector,
    MotionDetector,
    RotationDetector,
    StopRotationDetector,
    StopTranslationDetector,
    TranslationDetector,
)
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.detectors.grasp_detector_nodes import (
    GraspDetector,
    LossOfGraspDetector,
)
from segmind.detectors.spatial_relation_detector_nodes import (
    ContainmentDetector,
    LossOfContainmentDetector,
    LossOfSupportDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger
from segmind.statecharts.segmind_statechart import SegmindStatechart
from segmind.utils import PropagatingThread
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

OBJECT_DETECTOR_TYPES: Tuple[Type[AbstractDetector], ...] = (
    ContactDetector,
    LossOfContactDetector,
    GraspDetector,
    LossOfGraspDetector,
    SupportDetector,
    LossOfSupportDetector,
    ContainmentDetector,
    LossOfContainmentDetector,
    TranslationDetector,
    StopTranslationDetector,
    RotationDetector,
    StopRotationDetector,
)
"""
The detectors that each watch a single body.
"""

EVENT_COMBINING_DETECTOR_TYPES: Tuple[Type[AbstractDetector], ...] = (
    PickUpDetector,
    PlacingDetector,
)
"""
The detectors that combine the events already detected about any body.
"""


class SegmindEnvironmentVariable(StrEnum):
    """
    Environment variables a watched run reads.
    """

    EVENTS_FILE = "SEGMIND_EVENTS_FILE"
    """
    When set, the file a watched run writes the events SegMind detected to.
    """


# %% records of detected events


class EventRecordField(StrEnum):
    """
    The keys an event record is written under.
    """

    EVENT_TYPE = "event_type"
    TRACKED_OBJECT = "tracked_object"
    WITH_OBJECT = "with_object"


@dataclass(frozen=True)
class EventRecord(SubclassJSONSerializer):
    """
    A detected event, named by its type and the names of its bodies, so it can be read
    in a process that holds no world.
    """

    event_type: str
    """
    The name of the event's class.
    """

    tracked_object: str
    """
    The name of the body the event is about.
    """

    with_object: Optional[str] = None
    """
    The name of the other body involved, if any.
    """

    @classmethod
    def of(cls, event: EventWithTrackedObjects) -> Self:
        """
        :return: The record of ``event``.
        """
        return cls(
            event_type=type(event).__name__,
            tracked_object=event.tracked_object.name.name,
            with_object=(
                None if event.with_object is None else event.with_object.name.name
            ),
        )

    @classmethod
    def read_all(cls, path: Path) -> List[Self]:
        """
        :return: The records a watched run wrote to ``path``.
        """
        return [from_json(item) for item in json.loads(path.read_text())]

    def to_json(self, **kwargs) -> Dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            EventRecordField.EVENT_TYPE: self.event_type,
            EventRecordField.TRACKED_OBJECT: self.tracked_object,
            EventRecordField.WITH_OBJECT: self.with_object,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            event_type=data[EventRecordField.EVENT_TYPE],
            tracked_object=data[EventRecordField.TRACKED_OBJECT],
            with_object=data[EventRecordField.WITH_OBJECT],
        )


# %% watching on a thread of its own


@dataclass(eq=False)
class LiveSegmenter(PropagatingThread):
    """
    Ticks SegMind detectors against a world on a thread of its own, from when it is
    started until it is stopped.

    A tick holds the world's lock, so it never reads the world while another thread is in
    the middle of changing it. After each tick the world is left to other threads for at
    least as long as the tick held it, so a plan changing the world keeps going while it
    is watched.

    ..note:: The detectors read the world as plain numbers and call no CasADi, which is
       what lets them tick beside a plan that does.
    """

    world: World = field(kw_only=True)
    """
    The world the detectors watch.
    """

    detectors: List[AbstractDetector] = field(kw_only=True)
    """
    The detectors ticked.
    """

    pause_between_ticks: float = field(kw_only=True, default=0.01)
    """
    The least time, in seconds, the world is left to other threads after a tick.
    """

    executor: EpisodeSegmenterExecutor = field(init=False)
    """
    Compiles and ticks the detectors.
    """

    def __post_init__(self):
        super().__post_init__()
        self.daemon = True
        self.executor = EpisodeSegmenterExecutor(
            context=MotionStatechartContext(world=self.world)
        )
        self.executor.compile(SegmindStatechart().build_statechart(self.detectors))

    @classmethod
    def watching(cls, world: World, bodies: List[Body]) -> Self:
        """
        A segmenter detecting what happens to ``bodies``: each is watched by every
        detector of :data:`OBJECT_DETECTOR_TYPES`, and their events are combined by the
        detectors of :data:`EVENT_COMBINING_DETECTOR_TYPES`.
        """
        detectors = [
            detector_type(tracked_object=body)
            for body in bodies
            for detector_type in OBJECT_DETECTOR_TYPES
        ]
        # A watched run has an agent in it, so what it did to an object is the better
        # reading of a pick-up or a placing than the object's own motion alone.
        detectors.extend(
            detector_type(use_grasp_logic=True)
            for detector_type in EVENT_COMBINING_DETECTOR_TYPES
        )
        return cls(world=world, detectors=detectors)

    @property
    def event_logger(self) -> EventLogger:
        """
        The logger holding every event the detectors detected.
        """
        return self.executor.context.require_extension(SegmindContext).logger

    @property
    def ticks_to_see_the_world_at_rest(self) -> int:
        """
        How many ticks of a still world the detectors need to conclude what it ended in:
        the longest motion window among them, and at least one.
        """
        return max(
            [
                detector.window_size
                for detector in self.detectors
                if isinstance(detector, MotionDetector)
            ],
            default=1,
        )

    def tick(self) -> None:
        """
        Tick the detectors once, while no other thread changes the world.
        """
        with self.world._world_lock:
            self.executor.tick()

    def write_event_records(self, path: Path) -> None:
        """
        Write the record of every event detected so far to ``path``.
        """
        records = [EventRecord.of(event) for event in self.event_logger.get_events()]
        path.write_text(json.dumps([record.to_json() for record in records]))

    def write_event_records_where_requested(self) -> None:
        """
        Write the record of every event detected so far to the file named by
        :attr:`SegmindEnvironmentVariable.EVENTS_FILE`, if that variable is set.
        """
        requested = os.environ.get(SegmindEnvironmentVariable.EVENTS_FILE)
        if requested is None:
            return
        self.write_event_records(Path(requested))

    def _run(self) -> None:
        while not self.kill_event.is_set():
            started = time.monotonic()
            self.tick()
            held_for = time.monotonic() - started
            self.kill_event.wait(max(self.pause_between_ticks, held_for))
        for _ in range(self.ticks_to_see_the_world_at_rest):
            self.tick()

    def _join(self, timeout=None) -> None:
        self.join(timeout)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.stop()

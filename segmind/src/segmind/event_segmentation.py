"""
SegMind itself: what it watches and what it is asked to detect while a run goes on.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List, Optional, Self, Sequence, Type

from segmind.detector_selection import DetectorSelection
from segmind.detectors.atomic_event_detectors_nodes import MotionDetector
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger
from segmind.exceptions import NoSemanticAnnotationToWatch
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Segmind:
    """
    SegMind segmenting a run into events while it goes on.

    Entering it starts ticking the detectors on a thread of their own, and leaving it
    stops them and reports what they detected, so a run states what it wants watched and
    what it wants detected, and keeps the rest to its plan.

    A tick holds the world's lock, so it never reads the world while another thread is
    in the middle of changing it. Ticks begin :attr:`tick_period` apart, so the share
    of the run spent watching falls as ticks get cheaper, and every tick is followed by
    a pause, so a plan changing the world keeps going while it is watched.
    """

    world: World
    """
    The world the run takes place in.
    """

    detectors: List[AbstractDetector]
    """
    The detectors ticked.
    """

    tick_period: float = field(kw_only=True, default=0.5)
    """
    The time, in seconds, from the start of one tick to the start of the next.

    A tick that takes longer is followed by :attr:`pause_between_ticks` instead.
    """

    pause_between_ticks: float = field(kw_only=True, default=0.01)
    """
    The least time, in seconds, the world is left to other threads after a tick.
    """

    executor: EpisodeSegmenterExecutor = field(init=False)
    """
    Compiles and ticks the detectors.
    """

    stopping: threading.Event = field(init=False, default_factory=threading.Event)
    """
    Set when the detectors are to stop being ticked.
    """

    thread: Optional[threading.Thread] = field(init=False, default=None)
    """
    The thread the detectors are ticked on, once started.
    """

    def __post_init__(self) -> None:
        self.executor = EpisodeSegmenterExecutor(
            context=MotionStatechartContext(world=self.world)
        )
        self.executor.compile(SegmindStatechart().build_statechart(self.detectors))

    @classmethod
    def create_for_bodies(
        cls,
        world: World,
        bodies: Sequence[Body],
        detector_types: Sequence[Type[AbstractDetector]] = (),
    ) -> Self:
        """
        SegMind detecting what happens to ``bodies``.

        :param world: The world the run takes place in.
        :param bodies: The bodies the run handles, which the detectors watch.
        :param detector_types: The kinds of detector asked for. Every kind they are read
            from is brought along (see
            :class:`~segmind.detector_selection.DetectorSelection`), so asking for what is
            to be detected is enough. Without any, every kind SegMind defines.
        """
        selection = (
            DetectorSelection.of(*detector_types)
            if detector_types
            else DetectorSelection.of_every_kind()
        )
        detectors: List[AbstractDetector] = [
            detector
            for detector_type in selection.detector_types
            for detector in detector_type.create_for_run(
                bodies, selection.detector_types
            )
        ]
        return cls(world=world, detectors=detectors)

    @classmethod
    def create_for_semantic_annotation_types(
        cls,
        world: World,
        semantic_annotation_types: Sequence[Type[SemanticAnnotation]],
        detector_types: Sequence[Type[AbstractDetector]] = (),
    ) -> Self:
        """
        SegMind watching the bodies of every semantic annotation of the types a run
        names.

        :param world: The world the run takes place in.
        :param semantic_annotation_types: The types whose annotations in ``world`` have
            their bodies watched.
        :param detector_types: The kinds of detector asked for.
        :raises NoSemanticAnnotationToWatch: When ``world`` holds no annotation of a
            type asked for.
        """
        bodies = []
        for semantic_annotation_type in semantic_annotation_types:
            annotations = world.get_semantic_annotations_by_type(
                semantic_annotation_type
            )
            if not annotations:
                raise NoSemanticAnnotationToWatch(semantic_annotation_type)
            bodies.extend(
                body for annotation in annotations for body in annotation.bodies
            )
        return cls.create_for_bodies(world, list(dict.fromkeys(bodies)), detector_types)

    @property
    def watched_bodies(self) -> List[Body]:
        """
        The bodies the detectors watch, each named once, in the order they are ticked.
        """
        return list(
            dict.fromkeys(
                detector.tracked_object
                for detector in self.detectors
                if detector.tracked_object is not None
            )
        )

    @property
    def detector_names(self) -> List[str]:
        """
        The kinds of detector ticked, each named once, in the order they are ticked.
        """
        return list(
            dict.fromkeys(type(detector).__name__ for detector in self.detectors)
        )

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

    @property
    def is_watching(self) -> bool:
        """
        Whether the detectors are being ticked on their thread.
        """
        return self.thread is not None and self.thread.is_alive()

    def tick(self) -> None:
        """
        Tick the detectors once, while no other thread changes the world.
        """
        with self.world._world_lock:
            self.executor.tick()

    def start(self) -> None:
        """
        Start ticking the detectors on a thread of their own.
        """
        self.stopping.clear()
        self.thread = threading.Thread(target=self._tick_until_stopped, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """
        Stop ticking the detectors, once they have seen the world at rest.
        """
        self.stopping.set()
        self.thread.join()

    def _tick_until_stopped(self) -> None:
        """
        Tick the detectors until told to stop, then as long as they need to see the
        world at rest.
        """
        while not self.stopping.is_set():
            started = time.monotonic()
            self.tick()
            held_for = time.monotonic() - started
            self.stopping.wait(
                max(self.tick_period - held_for, self.pause_between_ticks)
            )
        for _ in range(self.ticks_to_see_the_world_at_rest):
            self.tick()

    def report_detected_events(self) -> None:
        """
        Put every event detected on the console, which is where a run is read.

        SegMind reports what it detected at debug level, which nothing shows by default,
        and a run that already has a handler of its own would otherwise show each event
        through both.
        """
        detected_events = logging.getLogger(EventLogger.__module__)
        detected_events.setLevel(logging.DEBUG)
        if not detected_events.handlers:
            detected_events.addHandler(logging.StreamHandler())
        propagated = detected_events.propagate
        detected_events.propagate = False
        self.event_logger.print_events()
        detected_events.propagate = propagated

    def __enter__(self) -> Self:
        logger.info("SegMind detectors: %s", ", ".join(self.detector_names))
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.stop()
        self.report_detected_events()

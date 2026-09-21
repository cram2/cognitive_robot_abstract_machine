"""
Watching a world with SegMind while something else, such as a plan, changes it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List, Self, Sequence, Tuple, Type

from segmind.detector_selection import DetectorSelection
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
from segmind.detectors.agent_event_detector_nodes import (
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
    def watching(
        cls,
        world: World,
        bodies: List[Body],
        detectors: Sequence[Type[AbstractDetector]] = (),
    ) -> Self:
        """
        A segmenter detecting what happens to ``bodies``.

        :param detectors: The kinds of detector asked for. Every kind they are read from
            is brought along (see :class:`~segmind.detector_selection.DetectorSelection`),
            so asking for what is to be detected is enough. Without any, every kind of
            :data:`OBJECT_DETECTOR_TYPES` and :data:`EVENT_COMBINING_DETECTOR_TYPES`.
        """
        asked_for = detectors or (
            *OBJECT_DETECTOR_TYPES,
            *EVENT_COMBINING_DETECTOR_TYPES,
        )
        chosen: List[AbstractDetector] = []
        for detector_type in DetectorSelection.of(*asked_for).detector_types:
            if detector_type.watches_a_body():
                chosen.extend(detector_type(tracked_object=body) for body in bodies)
            else:
                chosen.append(detector_type())
        return cls(world=world, detectors=chosen)

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

"""
A detector that detects nothing and records its ticks, for tests about when and where
detectors are ticked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List

from segmind.datastructures.events import DetectionEvent
from segmind.detectors.base import AbstractDetector, SegmindContext
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class DetectorCountingItsTicks(AbstractDetector):
    """
    Detects nothing, and records how often and on which thread it was ticked.
    """

    ticks: int = field(default=0, init=False)
    """
    How often this detector was ticked.
    """

    tick_threads: List[int] = field(default_factory=list, init=False)
    """
    The identifier of the thread each tick ran on, in order.
    """

    ticked: threading.Event = field(default_factory=threading.Event, init=False)
    """
    Set once this detector has been ticked.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        self.ticks += 1
        self.tick_threads.append(threading.get_ident())
        self.ticked.set()
        return []

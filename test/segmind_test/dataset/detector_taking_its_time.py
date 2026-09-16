"""
A detector that detects nothing and takes a while to do it, for tests about what a slow
tick costs whatever else uses the world.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List

from segmind.datastructures.events import DetectionEvent
from segmind.detectors.base import AbstractDetector, SegmindContext
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class DetectorTakingItsTime(AbstractDetector):
    """
    Detects nothing, and spends a fixed time on every tick.
    """

    seconds_per_tick: float = 0.02
    """
    How long each tick takes.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        time.sleep(self.seconds_per_tick)
        return []

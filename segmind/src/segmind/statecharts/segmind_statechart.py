from __future__ import annotations

from dataclasses import dataclass, field

from cramph.context import StatechartContext
from cramph.statechart import Statechart
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    TranslationDetector,
)
from segmind.detectors.base import AbstractDetector
from segmind.detectors.coarse_event_detector_nodes import (
    PlacingDetector,
    PickUpDetector,
)
from segmind.detectors.spatial_relation_detector_nodes import (
    SupportDetector,
    ContainmentDetector,
    InsertionDetector,
)


@dataclass
class DetectorStatechartBuilder:
    """
    Builds a statechart that runs detectors against a shared
    :class:`~segmind.detectors.base.SegmindContext`.
    """

    detectors: list[AbstractDetector] = field(
        default_factory=lambda: [
            ContactDetector(),
            SupportDetector(),
            ContainmentDetector(),
            TranslationDetector(),
            PlacingDetector(),
            InsertionDetector(),
            PickUpDetector(),
        ]
    )
    """
    The detectors the statechart runs; by default every detector except the rotation
    detectors.
    """

    def build(self, context: StatechartContext) -> Statechart:
        """
        :param context: The context of the executor that runs the statechart.
        :return: A statechart holding :attr:`detectors` as its nodes.
        """
        statechart = Statechart(context=context)
        statechart.add_nodes(self.detectors)
        return statechart

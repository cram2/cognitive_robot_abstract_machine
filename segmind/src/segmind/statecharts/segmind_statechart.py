from dataclasses import dataclass, field
from typing import List

from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from segmind.detectors.atomic_event_detectors_nodes import ContactDetector, LossOfContactDetector, TranslationDetector, \
    StopTranslationDetector, RotationDetector, StopRotationDetector
from segmind.detectors.base import DetectorStateChart, AbstractDetector
from segmind.detectors.coarse_event_detector_nodes import PlacingDetector, PickUpDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector, LossOfSupportDetector, \
    ContainmentDetector, InsertionDetector, LossOfContainmentDetector


@dataclass
class SegmindStatechart(MotionStatechart):
    """
    Represents the statechart for Segmind, encapsulating its construction and management.

    This class is used to build a statechart for Segmind by establishing various detectors
    that act as nodes within the statechart. Each detector is instantiated with a unique
    name and a shared context. These detectors are then added as nodes to the statechart.
    """


    def build_statechart(self, detectors:List[AbstractDetector]=None) -> DetectorStateChart:
        """
        Build a statechart with various detector nodes.

        This method constructs a statechart used to manage different states and transitions
        within a detection system. Each detector node corresponds to a specific event or
        state in the system, such as contact detection, loss of contact, support, and
        containment detection. Once initialized, the statechart is populated with these
        nodes for future state management.

        :return: A statechart instance with detector nodes.
        """

        sc = DetectorStateChart()
        default_detectors = [
            ContactDetector(),
            LossOfContactDetector(),
            SupportDetector(),
            LossOfSupportDetector(),
            ContainmentDetector(),
            TranslationDetector(),
            StopTranslationDetector(),
            PlacingDetector(),
            InsertionDetector(),
            PickUpDetector(),
            LossOfContainmentDetector(),
        ]

        detectors = detectors if detectors else default_detectors

        sc.add_nodes(detectors)


        return sc
from dataclasses import dataclass, field


from segmind.detectors.atomic_event_detectors_nodes import ContactDetector, LossOfContactDetector, TranslationDetector, \
    StopTranslationDetector
from segmind.detectors.base import SegmindContext, DetectorStateChart
from segmind.detectors.coarse_event_detector_nodes import PlacingDetector, PickUpDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector, LossOfSupportDetector, \
    ContainmentDetector, InsertionDetector, LossOfContainmentDetector


@dataclass
class SegmindStatechart:
    """
    Represents the statechart for Segmind, encapsulating its construction and management.

    This class is used to build a statechart for Segmind by establishing various detectors
    that act as nodes within the statechart. Each detector is instantiated with a unique
    name and a shared context. These detectors are then added as nodes to the statechart.
    A `SegmindContext` instance is required to initialize and use the statechart effectively.

    """

    context: SegmindContext = field(init=False)
    """
    The shared context for the statechart, providing access to world information,
    relation history, and logging utilities.
    """

    def build_statechart(self, context: SegmindContext) -> DetectorStateChart:
        """
        Build a statechart with various detector nodes.

        This method constructs a statechart used to manage different states and transitions
        within a detection system. Each detector node corresponds to a specific event or
        state in the system, such as contact detection, loss of contact, support, and
        containment detection. Once initialized, the statechart is populated with these
        nodes for future state management.

        :param context: The shared context for the statechart.
        :return: A statechart instance with detector nodes.
        """

        sc = DetectorStateChart()

        self.context = context

        detectors = [
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

        sc.add_nodes(detectors)

        return sc
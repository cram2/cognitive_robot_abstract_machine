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

    def build_statechart(self, context: SegmindContext):
        """
        Build a statechart with various detector nodes.

        This method constructs a statechart used to manage different states and transitions
        within a detection system. Each detector node corresponds to a specific event or
        state in the system, such as contact detection, loss of contact, support, and
        containment detection. Once initialized, the statechart is populated with these
        nodes for future state management.

        Parameters:
        context (SegmindContext): The context object that contains the necessary information
                                  and resources required by all detectors.

        Returns:
        DetectorStateChart: A statechart instance populated with initialized detection nodes.
        """

        sc = DetectorStateChart()

        self.context = context

        contact_detector = ContactDetector(
            name="contact_detector", context=self.context
        )
        loss_of_contact_detector = LossOfContactDetector(
            name="loss_of_contact_detector",
            context=self.context,
        )
        support_detector = SupportDetector(
            name="support_detector",
            context=self.context,
        )
        loss_of_support_detector = LossOfSupportDetector(
            name="los_detector",
            context=self.context,
        )
        containment_detector = ContainmentDetector(
            name="containment_detector",
            context=self.context,
        )
        translation_detector = TranslationDetector(
            name="translation_detector", context=self.context
        )

        stop_translation_detector = StopTranslationDetector(
            name="stop_translation_detector", context=self.context
        )

        placing_detector = PlacingDetector(
            name="placing_detector", context=self.context
        )

        insertion_detector = InsertionDetector(
            name="insertion_detector", context=self.context
        )

        pickup_detector = PickUpDetector(name="pickup_detector", context=self.context)

        loss_of_containment_detector = LossOfContainmentDetector(name="loss_of_containment_detector", context=self.context)
        sc.add_nodes(
            [
                contact_detector,
                loss_of_contact_detector,
                support_detector,
                loss_of_support_detector,
                translation_detector,
                containment_detector,
                stop_translation_detector,
                placing_detector,
                insertion_detector,
                pickup_detector,
                loss_of_containment_detector,
            ]
        )



        return sc
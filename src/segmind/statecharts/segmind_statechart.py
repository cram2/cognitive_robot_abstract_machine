from dataclasses import dataclass, field

from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from segmind.detectors.atomic_event_detectors import DetectorStateChart
from segmind.detectors.atomic_event_detectors_nodes import ContactDetector, LossOfContactDetector, TranslationDetector, \
    StopTranslationDetector
from segmind.detectors.base import SegmindContext
from segmind.detectors.coarse_event_detector_nodes import PlacingDetector, PickUpDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector, LossOfSupportDetector, \
    ContainmentDetector, InsertionDetector


@dataclass
class SegmindStatechart:

    context: SegmindContext = field(init=False)

    def build_statechart(self, context: SegmindContext):
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
            ]
        )

        support_detector.start_condition = contact_detector.observation_variable
        loss_of_support_detector.start_condition = (
            loss_of_contact_detector.observation_variable
        )
        containment_detector.start_condition = support_detector.observation_variable
        placing_detector.start_condition = support_detector.observation_variable
        insertion_detector.start_condition = containment_detector.observation_variable
        pickup_detector.start_condition = loss_of_support_detector.observation_variable

        return sc
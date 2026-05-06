from ripple_down_rules.datastructures.case import Case
from segmind.detectors.coarse_event_detectors import PlacingDetector
from typing_extensions import Dict, Optional, Union
from segmind.datastructures.events import PlacingEvent, StopTranslationEvent, TranslationEvent, SupportEvent
from types import NoneType


def conditions_73076948442184610140058021308995926047(case) -> bool:
    def conditions_for_placing_detector_get_interaction_event(self_: PlacingDetector, output_: Union[NoneType, PlacingEvent]) -> bool:
        """Get conditions on whether it's possible to conclude a value for PlacingDetector_get_interaction_event.output_  of type PlacingEvent."""
        return isinstance(self_.starter_event, SupportEvent)
    return conditions_for_placing_detector_get_interaction_event(**case)


def conclusion_73076948442184610140058021308995926047(case) -> Optional[PlacingEvent]:
    def placing_detector_get_interaction_event(self_: PlacingDetector, output_: Union[NoneType, PlacingEvent]) -> Union[NoneType, PlacingEvent]:
        """Get possible value(s) for PlacingDetector_get_interaction_event.output_  of type PlacingEvent."""
        # latest_translation_event = self_.starter_event.object_tracker.get_first_event_of_type_before_event(TranslationEvent, self_.starter_event)
        return PlacingEvent(tracked_object=self_.tracked_object,
                            timestamp=self_.starter_event.timestamp - self_.wait_time.total_seconds(),
                            end_timestamp=self_.starter_event.timestamp)
    return placing_detector_get_interaction_event(**case)



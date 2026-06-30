from typing_extensions import Dict, Optional, Union
from ripple_down_rules.datastructures.case import Case
from segmind.datastructures.events import LossOfInterferenceEvent, PickUpEvent, TranslationEvent, LossOfSupportEvent
from segmind.detectors.coarse_event_detectors import GeneralPickUpDetector
from types import NoneType


def conditions_204728260923142752317583736648507389376(case) -> bool:
    def conditions_for_general_pick_up_detector_get_interaction_event(self_: GeneralPickUpDetector, output_: Union[NoneType, PickUpEvent]) -> bool:
        """Get conditions on whether it's possible to conclude a value for GeneralPickUpDetector_get_interaction_event.output_  of type PickUpEvent."""
        return isinstance(self_.starter_event, LossOfSupportEvent)
    return conditions_for_general_pick_up_detector_get_interaction_event(**case)


def conclusion_204728260923142752317583736648507389376(case) -> Optional[PickUpEvent]:
    def general_pick_up_detector_get_interaction_event(self_: GeneralPickUpDetector, output_: Union[NoneType, PickUpEvent]) -> Union[NoneType, PickUpEvent]:
        """Get possible value(s) for GeneralPickUpDetector_get_interaction_event.output_  of type PickUpEvent."""
        nearest_translation_event = self_.object_tracker.get_nearest_event_of_type_to_event(self_.starter_event, TranslationEvent)
        end_timestamp = max(nearest_translation_event.timestamp, self_.starter_event.timestamp) if nearest_translation_event is not None else self_.starter_event.timestamp
        timestamp = min(nearest_translation_event.timestamp, self_.starter_event.timestamp) if nearest_translation_event is not None else self_.starter_event.timestamp - self_.wait_time.total_seconds()
        return PickUpEvent(tracked_object=self_.tracked_object, timestamp=timestamp, end_timestamp=end_timestamp)
    return general_pick_up_detector_get_interaction_event(**case)



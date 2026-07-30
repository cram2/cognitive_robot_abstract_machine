from ripple_down_rules.datastructures.case import Case
from typing_extensions import Dict, Optional, Type, Union
from segmind.datastructures.events import EventWithOneTrackedObject, SupportEvent
from segmind.detectors.coarse_event_detectors import PlacingDetector
from types import NoneType


def conditions_124701724059586364685260806439568356831(case) -> bool:
    def conditions_for_placing_detector_start_condition_checker(cls_: Type[PlacingDetector], event: EventWithOneTrackedObject, output_: bool) -> bool:
        """Get conditions on whether it's possible to conclude a value for PlacingDetector_start_condition_checker.output_  of type ."""
        return isinstance(event, SupportEvent)
    return conditions_for_placing_detector_start_condition_checker(**case)


def conclusion_124701724059586364685260806439568356831(case) -> bool:
    def placing_detector_start_condition_checker(cls_: Type[PlacingDetector], event: EventWithOneTrackedObject, output_: bool) -> bool:
        """Get possible value(s) for PlacingDetector_start_condition_checker.output_  of type ."""
        return True
    return placing_detector_start_condition_checker(**case)



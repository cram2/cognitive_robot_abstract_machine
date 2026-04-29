from types import NoneType
from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Optional
from .placing_detector_get_object_to_track_from_starter_event_output__scrdr_defs import *


attribute_name = 'output_'
conclusion_type = (NoneType, Object,)
mutually_exclusive = True


def classify(case: Dict, **kwargs) -> Optional[Object]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_157980724165316319734190707637955950660(case):
        return conclusion_157980724165316319734190707637955950660(case)
    else:
        return None

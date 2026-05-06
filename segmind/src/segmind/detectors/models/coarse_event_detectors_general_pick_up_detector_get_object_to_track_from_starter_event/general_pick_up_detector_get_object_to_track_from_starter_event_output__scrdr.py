from types import NoneType
from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Optional
from .general_pick_up_detector_get_object_to_track_from_starter_event_output__scrdr_defs import *


attribute_name = 'output_'
conclusion_type = (NoneType, Object,)
mutually_exclusive = True


def classify(case: Dict, **kwargs) -> Optional[Object]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_132752200528268219213179872006260451656(case):
        return conclusion_132752200528268219213179872006260451656(case)
    else:
        return None

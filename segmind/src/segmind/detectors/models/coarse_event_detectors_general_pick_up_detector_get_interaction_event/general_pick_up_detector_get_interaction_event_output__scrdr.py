from types import NoneType
from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Optional
from .general_pick_up_detector_get_interaction_event_output__scrdr_defs import *


attribute_name = 'output_'
conclusion_type = (NoneType, PickUpEvent,)
mutually_exclusive = True


def classify(case: Dict, **kwargs) -> Optional[PickUpEvent]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_204728260923142752317583736648507389376(case):
        return conclusion_204728260923142752317583736648507389376(case)
    else:
        return None

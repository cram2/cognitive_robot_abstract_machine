from types import NoneType
from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Optional
from .placing_detector_get_interaction_event_output__scrdr_defs import *


attribute_name = 'output_'
conclusion_type = (NoneType, PlacingEvent,)
mutually_exclusive = True


def classify(case: Dict, **kwargs) -> Optional[PlacingEvent]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_73076948442184610140058021308995926047(case):
        return conclusion_73076948442184610140058021308995926047(case)
    else:
        return None

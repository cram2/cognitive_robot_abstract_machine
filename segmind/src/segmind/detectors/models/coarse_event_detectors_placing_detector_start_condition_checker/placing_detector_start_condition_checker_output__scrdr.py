from typing_extensions import Optional
from ripple_down_rules.datastructures.case import Case, create_case
from types import NoneType
from .placing_detector_start_condition_checker_output__scrdr_defs import *


attribute_name = 'output_'
conclusion_type = (bool,)
mutually_exclusive = True


def classify(case: Dict, **kwargs) -> Optional[bool]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_124701724059586364685260806439568356831(case):
        return conclusion_124701724059586364685260806439568356831(case)
    else:
        return None

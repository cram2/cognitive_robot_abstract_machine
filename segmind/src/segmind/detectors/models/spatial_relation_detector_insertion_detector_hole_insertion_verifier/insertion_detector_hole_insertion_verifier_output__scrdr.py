from typing_extensions import Optional
from ripple_down_rules.datastructures.case import Case, create_case
from types import NoneType
from .insertion_detector_hole_insertion_verifier_output__scrdr_defs import *


attribute_name = 'output_'
conclusion_type = (bool,)
mutually_exclusive = True


def classify(case: Dict, **kwargs) -> Optional[bool]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_38355037295796650033371896063976531277(case):

        if conditions_21738774625860220488991060484462427733(case):

            if conditions_313966059252436144481394373657043070884(case):
                return conclusion_313966059252436144481394373657043070884(case)
            return conclusion_21738774625860220488991060484462427733(case)
        return conclusion_38355037295796650033371896063976531277(case)
    else:
        return None

from types import NoneType
from typing import Dict

from ripple_down_rules.datastructures.case import create_case, Case
from typing_extensions import Optional
from .general_pick_up_detector_start_condition_checker_output__scrdr_defs import *


attribute_name = 'output_'
conclusion_type = (bool,)
mutually_exclusive = True


def classify(case: Dict, **kwargs) -> Optional[bool]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_313968436519149281932112885344716145224(case):

        if conditions_185374571278911955422500089733244334209(case):
            return conclusion_185374571278911955422500089733244334209(case)

        elif conditions_259334668447470890319338447268411809573(case):
            return conclusion_259334668447470890319338447268411809573(case)

        elif conditions_1084517150588802996811823834978763787(case):
            return conclusion_1084517150588802996811823834978763787(case)

        elif conditions_257426441376249650265480473188288026956(case):
            return conclusion_257426441376249650265480473188288026956(case)
        return conclusion_313968436519149281932112885344716145224(case)
    else:
        return None

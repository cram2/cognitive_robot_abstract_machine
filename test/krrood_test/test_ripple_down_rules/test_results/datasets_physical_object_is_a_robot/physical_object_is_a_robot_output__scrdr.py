from typing_extensions import Optional
from krrood.ripple_down_rules.datastructures.case import Case, create_case
from types import NoneType
from .physical_object_is_a_robot_output__scrdr_defs import *

attribute_name = "output_"
conclusion_type = (bool,)
mutually_exclusive = True
name = "output_"
case_type = Dict
case_name = "PhysicalObject_is_a_robot"


def classify(case: Dict, **kwargs) -> Optional[bool]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)

    if conditions_226969243620390858682731042391766665817(case):
        return conclusion_226969243620390858682731042391766665817(case)
    else:
        return None

from typing_extensions import Optional, Set
from krrood.ripple_down_rules.datastructures.case import Case, create_case
from krrood.ripple_down_rules.utils import make_set
from krrood.ripple_down_rules.helpers import get_an_updated_case_copy
from .physical_object_select_objects_that_are_parts_of_robot_output__mcrdr_defs import *

attribute_name = "output_"
conclusion_type = (PhysicalObject,)
mutually_exclusive = False
name = "output_"
case_type = Dict
case_name = "PhysicalObject_select_objects_that_are_parts_of_robot"


def classify(case: Dict, **kwargs) -> Set[PhysicalObject]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_164855806603893754507167918997373216146(case):
        conclusions.update(
            make_set(conclusion_164855806603893754507167918997373216146(case))
        )
    return conclusions

from krrood.ripple_down_rules.datastructures.case import Case, create_case
from krrood.ripple_down_rules.helpers import general_rdr_classify
from typing_extensions import Any, Dict
from . import physical_object_is_a_robot_output__scrdr as output__classifier

name = "output_"
case_type = Dict
case_name = "PhysicalObject_is_a_robot"
classifiers_dict = dict()
classifiers_dict["output_"] = output__classifier


def classify(case: Dict, **kwargs) -> Dict[str, Any]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    return general_rdr_classify(classifiers_dict, case, **kwargs)

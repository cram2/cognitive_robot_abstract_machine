from semantic_digital_twin.world import World
from typing_extensions import Any, Dict
from krrood.ripple_down_rules.datastructures.case import Case, create_case
from krrood.ripple_down_rules.helpers import general_rdr_classify
from semantic_digital_twin.reasoning.world_rdr import world_semantic_annotations_mcrdr as semantic_annotations_classifier

name = "world"
case_type = World
case_name = "World"
classifiers_dict = dict()
classifiers_dict["semantic_annotations"] = semantic_annotations_classifier


def classify(case: World, **kwargs) -> Dict[str, Any]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    return general_rdr_classify(classifiers_dict, case, **kwargs)

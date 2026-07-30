from typing_extensions import Any, Dict
from ripple_down_rules.helpers import general_rdr_classify
from ripple_down_rules.datastructures.case import Case, create_case
from . import general_pick_up_detector_get_object_to_track_from_starter_event_output__scrdr as output__classifier

classifiers_dict = dict()
classifiers_dict['output_'] = output__classifier


def classify(case: Dict, **kwargs) -> Dict[str, Any]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    return general_rdr_classify(classifiers_dict, case, **kwargs)

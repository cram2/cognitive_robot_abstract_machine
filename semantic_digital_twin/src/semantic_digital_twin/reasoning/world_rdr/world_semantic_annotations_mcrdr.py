from ripple_down_rules.utils import make_set
from typing_extensions import Optional, Set
from ripple_down_rules.datastructures.case import Case, create_case
from .world_semantic_annotations_mcrdr_defs import *


attribute_name = "semantic_annotations"
conclusion_type = (
    Drawer,
    Handle,
    Door,
    Fridge,
    Wardrobe,
)
mutually_exclusive = False
name = "semantic_annotations"
case_type = World
case_name = "World"


def classify(
    case: World, **kwargs
) -> Set[Union[Drawer, Handle, Door, Fridge, Wardrobe]]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_90574698325129464513441443063592862114(case):
        conclusions.update(
            make_set(conclusion_90574698325129464513441443063592862114(case))
        )

    if conditions_331345798360792447350644865254855982739(case):
        conclusions.update(
            make_set(conclusion_331345798360792447350644865254855982739(case))
        )

    if conditions_35528769484583703815352905256802298589(case):
        conclusions.update(
            make_set(conclusion_35528769484583703815352905256802298589(case))
        )

    if conditions_59112619694893607910753808758642808601(case):
        conclusions.update(
            make_set(conclusion_59112619694893607910753808758642808601(case))
        )

    if conditions_10840634078579061471470540436169882059(case):
        conclusions.update(
            make_set(conclusion_10840634078579061471470540436169882059(case))
        )
    return conclusions

from krrood.ripple_down_rules.utils import make_set
from typing_extensions import Set, Union
from krrood.ripple_down_rules.datastructures.case import Case, create_case
from semantic_digital_twin.reasoning.world_rdr.world_semantic_annotations_mcrdr_defs import *

attribute_name = "semantic_annotations"
conclusion_type = (
    Handle,
    Drawer,
    Door,
    Wardrobe,
    Dishwasher,
    Fridge,
    Cabinet,
    Oven,
    Sink,
    Cooktop,
    CounterTop,
    Sofa,
    Wall,
    ShelfLayer,
    CoffeeMachine,
    CoffeeTable,
    SideTable,
    Table,
)
mutually_exclusive = False
name = "semantic_annotations"
case_type = World
case_name = "World"


def classify(case: World, **kwargs) -> Set[
    Union[
        Handle,
        Drawer,
        Door,
        Wardrobe,
        Dishwasher,
        Fridge,
        Cabinet,
        Oven,
        Sink,
        Cooktop,
        CounterTop,
        Sofa,
        Wall,
        ShelfLayer,
        CoffeeMachine,
        CoffeeTable,
        SideTable,
        Table,
    ]
]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_111115858654502671279494400838612635168(case):
        conclusions.update(
            make_set(conclusion_111115858654502671279494400838612635168(case))
        )

    if conditions_248676565933633239858802311504377126882(case):
        conclusions.update(
            make_set(conclusion_248676565933633239858802311504377126882(case))
        )

    if conditions_317079062995625045064213599855869838805(case):
        conclusions.update(
            make_set(conclusion_317079062995625045064213599855869838805(case))
        )

    if conditions_326953014145210418398679055851359282662(case):
        conclusions.update(
            make_set(conclusion_326953014145210418398679055851359282662(case))
        )

    if conditions_218985522231414995632246132988515574217(case):
        conclusions.update(
            make_set(conclusion_218985522231414995632246132988515574217(case))
        )

    if conditions_26541319268144667585806659965812960774(case):
        conclusions.update(
            make_set(conclusion_26541319268144667585806659965812960774(case))
        )

    if conditions_158937834096912333482822714868490776940(case):
        conclusions.update(
            make_set(conclusion_158937834096912333482822714868490776940(case))
        )

    if conditions_316106723856289315485855687643584759560(case):
        conclusions.update(
            make_set(conclusion_316106723856289315485855687643584759560(case))
        )

    if conditions_273868058932563223747162125459578705794(case):
        conclusions.update(
            make_set(conclusion_273868058932563223747162125459578705794(case))
        )

    if conditions_48470732943666389031275241446759040158(case):
        conclusions.update(
            make_set(conclusion_48470732943666389031275241446759040158(case))
        )

    if conditions_30511584820652557323555645212699735534(case):
        conclusions.update(
            make_set(conclusion_30511584820652557323555645212699735534(case))
        )

    if conditions_53140626725687000784569602523600018998(case):
        conclusions.update(
            make_set(conclusion_53140626725687000784569602523600018998(case))
        )

    if conditions_91766461504596272560719260519280649820(case):
        conclusions.update(
            make_set(conclusion_91766461504596272560719260519280649820(case))
        )

    if conditions_64426787522073749385484386687820455248(case):
        conclusions.update(
            make_set(conclusion_64426787522073749385484386687820455248(case))
        )

    if conditions_127273513110689344453536255694523763442(case):
        conclusions.update(
            make_set(conclusion_127273513110689344453536255694523763442(case))
        )

    if conditions_66825188434526833455942009106856873396(case):
        conclusions.update(
            make_set(conclusion_66825188434526833455942009106856873396(case))
        )

    if conditions_129346935101384221720160031395011916810(case):
        conclusions.update(
            make_set(conclusion_129346935101384221720160031395011916810(case))
        )

    if conditions_269752152718035292505919835246449120303(case):
        conclusions.update(
            make_set(conclusion_269752152718035292505919835246449120303(case))
        )

    if conditions_206298218301184229027724450338144229416(case):
        conclusions.update(
            make_set(conclusion_206298218301184229027724450338144229416(case))
        )

    if conditions_289159079186754023252507394066467767351(case):
        conclusions.update(
            make_set(conclusion_289159079186754023252507394066467767351(case))
        )

    return conclusions

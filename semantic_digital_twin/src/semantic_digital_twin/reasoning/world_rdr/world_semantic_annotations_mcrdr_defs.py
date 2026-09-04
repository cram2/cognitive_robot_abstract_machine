"""
What the world semantic-annotation classifier calls.

Each ``conditions_``/``conclusion_`` pair names the precondition of one rule and the rule
that runs under it. The rules themselves are in
:mod:`semantic_digital_twin.reasoning.world_rdr.rules`.
"""

from semantic_digital_twin.reasoning.world_rdr.rules import (
    cabinets,
    coffee_machines,
    coffee_tables,
    cooktops,
    counter_tops,
    dishwashers,
    doors_with_a_handle,
    doors_without_a_handle,
    drawers_with_a_handle,
    drawers_without_a_handle,
    fridges,
    handles,
    ovens,
    shelf_layers,
    side_tables,
    sinks,
    sofas,
    tables,
    walls,
    wardrobes,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    CoffeeMachine,
    CoffeeTable,
    Cooktop,
    CounterTop,
    Dishwasher,
    Door,
    Drawer,
    Fridge,
    Handle,
    Oven,
    ShelfLayer,
    SideTable,
    Sink,
    Sofa,
    Table,
    Wall,
    Wardrobe,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)
from typing_extensions import List


def conditions_111115858654502671279494400838612635168(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_111115858654502671279494400838612635168(case) -> List[Handle]:
    def get_handles(case: World) -> List[Handle]:
        """
        Get possible value(s) for World.semantic_annotations of type Handle.
        """
        return handles(case)

    return get_handles(case)


def conditions_248676565933633239858802311504377126882(case) -> bool:
    def world_holds_sliders(case: World) -> bool:
        """
        Whether anything in the world slides, which is what a drawer needs.
        """
        return any(isinstance(c, PrismaticConnection) for c in case.connections)

    return world_holds_sliders(case)


def conclusion_248676565933633239858802311504377126882(case) -> List[Drawer]:
    def get_drawers_without_a_handle(case: World) -> List[Drawer]:
        """
        Get possible value(s) for World.semantic_annotations of type Drawer.
        """
        return drawers_without_a_handle(case)

    return get_drawers_without_a_handle(case)


def conditions_317079062995625045064213599855869838805(case) -> bool:
    def world_holds_hinges(case: World) -> bool:
        """
        Whether anything in the world swings, which is what a door needs.
        """
        return any(isinstance(c, RevoluteConnection) for c in case.connections)

    return world_holds_hinges(case)


def conclusion_317079062995625045064213599855869838805(case) -> List[Door]:
    def get_doors_without_a_handle(case: World) -> List[Door]:
        """
        Get possible value(s) for World.semantic_annotations of type Door.
        """
        return doors_without_a_handle(case)

    return get_doors_without_a_handle(case)


def conditions_326953014145210418398679055851359282662(case) -> bool:
    def world_holds_handles(case: World) -> bool:
        """
        Whether any handle is already known, since it takes one to say what it opens.
        """
        return any(isinstance(a, Handle) for a in case.semantic_annotations)

    return world_holds_handles(case)


def conclusion_326953014145210418398679055851359282662(case) -> List[Drawer]:
    def get_drawers_with_a_handle(case: World) -> List[Drawer]:
        """
        Get possible value(s) for World.semantic_annotations of type Drawer.
        """
        return drawers_with_a_handle(case)

    return get_drawers_with_a_handle(case)


def conditions_218985522231414995632246132988515574217(case) -> bool:
    def world_holds_handles(case: World) -> bool:
        """
        Whether any handle is already known, since it takes one to say what it opens.
        """
        return any(isinstance(a, Handle) for a in case.semantic_annotations)

    return world_holds_handles(case)


def conclusion_218985522231414995632246132988515574217(case) -> List[Door]:
    def get_doors_with_a_handle(case: World) -> List[Door]:
        """
        Get possible value(s) for World.semantic_annotations of type Door.
        """
        return doors_with_a_handle(case)

    return get_doors_with_a_handle(case)


def conditions_26541319268144667585806659965812960774(case) -> bool:
    def world_holds_openable_parts(case: World) -> bool:
        """
        Whether any door or drawer is already known, since a container is recognised by
        what opens out of it.
        """
        return any(isinstance(a, (Door, Drawer)) for a in case.semantic_annotations)

    return world_holds_openable_parts(case)


def conclusion_26541319268144667585806659965812960774(case) -> List[Wardrobe]:
    def get_wardrobes(case: World) -> List[Wardrobe]:
        """
        Get possible value(s) for World.semantic_annotations of type Wardrobe.
        """
        return wardrobes(case)

    return get_wardrobes(case)


def conditions_158937834096912333482822714868490776940(case) -> bool:
    def world_holds_openable_parts(case: World) -> bool:
        """
        Whether any door or drawer is already known, since a container is recognised by
        what opens out of it.
        """
        return any(isinstance(a, (Door, Drawer)) for a in case.semantic_annotations)

    return world_holds_openable_parts(case)


def conclusion_158937834096912333482822714868490776940(case) -> List[Dishwasher]:
    def get_dishwashers(case: World) -> List[Dishwasher]:
        """
        Get possible value(s) for World.semantic_annotations of type Dishwasher.
        """
        return dishwashers(case)

    return get_dishwashers(case)


def conditions_316106723856289315485855687643584759560(case) -> bool:
    def world_holds_openable_parts(case: World) -> bool:
        """
        Whether any door or drawer is already known, since a container is recognised by
        what opens out of it.
        """
        return any(isinstance(a, (Door, Drawer)) for a in case.semantic_annotations)

    return world_holds_openable_parts(case)


def conclusion_316106723856289315485855687643584759560(case) -> List[Fridge]:
    def get_fridges(case: World) -> List[Fridge]:
        """
        Get possible value(s) for World.semantic_annotations of type Fridge.
        """
        return fridges(case)

    return get_fridges(case)


def conditions_273868058932563223747162125459578705794(case) -> bool:
    def world_holds_openable_parts(case: World) -> bool:
        """
        Whether any door or drawer is already known, since a container is recognised by
        what opens out of it.
        """
        return any(isinstance(a, (Door, Drawer)) for a in case.semantic_annotations)

    return world_holds_openable_parts(case)


def conclusion_273868058932563223747162125459578705794(case) -> List[Cabinet]:
    def get_cabinets(case: World) -> List[Cabinet]:
        """
        Get possible value(s) for World.semantic_annotations of type Cabinet.
        """
        return cabinets(case)

    return get_cabinets(case)


def conditions_48470732943666389031275241446759040158(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_48470732943666389031275241446759040158(case) -> List[Oven]:
    def get_ovens(case: World) -> List[Oven]:
        """
        Get possible value(s) for World.semantic_annotations of type Oven.
        """
        return ovens(case)

    return get_ovens(case)


def conditions_30511584820652557323555645212699735534(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_30511584820652557323555645212699735534(case) -> List[Sink]:
    def get_sinks(case: World) -> List[Sink]:
        """
        Get possible value(s) for World.semantic_annotations of type Sink.
        """
        return sinks(case)

    return get_sinks(case)


def conditions_53140626725687000784569602523600018998(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_53140626725687000784569602523600018998(case) -> List[Cooktop]:
    def get_cooktops(case: World) -> List[Cooktop]:
        """
        Get possible value(s) for World.semantic_annotations of type Cooktop.
        """
        return cooktops(case)

    return get_cooktops(case)


def conditions_91766461504596272560719260519280649820(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_91766461504596272560719260519280649820(case) -> List[CounterTop]:
    def get_counter_tops(case: World) -> List[CounterTop]:
        """
        Get possible value(s) for World.semantic_annotations of type CounterTop.
        """
        return counter_tops(case)

    return get_counter_tops(case)


def conditions_64426787522073749385484386687820455248(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_64426787522073749385484386687820455248(case) -> List[Sofa]:
    def get_sofas(case: World) -> List[Sofa]:
        """
        Get possible value(s) for World.semantic_annotations of type Sofa.
        """
        return sofas(case)

    return get_sofas(case)


def conditions_127273513110689344453536255694523763442(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_127273513110689344453536255694523763442(case) -> List[Wall]:
    def get_walls(case: World) -> List[Wall]:
        """
        Get possible value(s) for World.semantic_annotations of type Wall.
        """
        return walls(case)

    return get_walls(case)


def conditions_66825188434526833455942009106856873396(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_66825188434526833455942009106856873396(case) -> List[ShelfLayer]:
    def get_shelf_layers(case: World) -> List[ShelfLayer]:
        """
        Get possible value(s) for World.semantic_annotations of type ShelfLayer.
        """
        return shelf_layers(case)

    return get_shelf_layers(case)


def conditions_129346935101384221720160031395011916810(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_129346935101384221720160031395011916810(case) -> List[CoffeeMachine]:
    def get_coffee_machines(case: World) -> List[CoffeeMachine]:
        """
        Get possible value(s) for World.semantic_annotations of type CoffeeMachine.
        """
        return coffee_machines(case)

    return get_coffee_machines(case)


def conditions_269752152718035292505919835246449120303(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_269752152718035292505919835246449120303(case) -> List[CoffeeTable]:
    def get_coffee_tables(case: World) -> List[CoffeeTable]:
        """
        Get possible value(s) for World.semantic_annotations of type CoffeeTable.
        """
        return coffee_tables(case)

    return get_coffee_tables(case)


def conditions_206298218301184229027724450338144229416(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_206298218301184229027724450338144229416(case) -> List[SideTable]:
    def get_side_tables(case: World) -> List[SideTable]:
        """
        Get possible value(s) for World.semantic_annotations of type SideTable.
        """
        return side_tables(case)

    return get_side_tables(case)


def conditions_289159079186754023252507394066467767351(case) -> bool:
    def world_holds_bodies(case: World) -> bool:
        """
        Whether the world holds anything that could be named.
        """
        return len(case.kinematic_structure_entities) > 0

    return world_holds_bodies(case)


def conclusion_289159079186754023252507394066467767351(case) -> List[Table]:
    def get_tables(case: World) -> List[Table]:
        """
        Get possible value(s) for World.semantic_annotations of type Table.
        """
        return tables(case)

    return get_tables(case)

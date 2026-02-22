from typing_extensions import List

from krrood.entity_query_language.factories import (
    contains,
    entity,
    inference,
    variable,
    match_variable,
)
from ...semantic_annotations.semantic_annotations import (
    Wardrobe,
    Door,
    Drawer,
    Fridge,
    Handle,
)
from ...world import World
from ...world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from ...world_description.world_entity import Body


def conditions_90574698325129464513441443063592862114(case) -> bool:
    def has_bodies_named_handle(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Handle."""
        return True

    return has_bodies_named_handle(case)


def conclusion_90574698325129464513441443063592862114(case) -> List[Handle]:
    def get_handles(case: World) -> List[Handle]:
        """Get possible value(s) for World.semantic_annotations of types list/set of Handle"""
        kse = variable(Body, case.kinematic_structure_entities)
        return (
            entity(inference(Handle)(root=kse))
            .where(contains(kse.name.name.lower(), "handle"))
            .tolist()
        )

    return get_handles(case)


def conditions_331345798360792447350644865254855982739(case) -> bool:
    def has_handles_and_HasCaseAsMainBodys(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Drawer."""
        return True

    return has_handles_and_HasCaseAsMainBodys(case)


def conclusion_331345798360792447350644865254855982739(case) -> List[Drawer]:
    def get_drawers(case: World) -> List[Drawer]:
        """Get possible value(s) for World.semantic_annotations of types list/set of Drawer"""
        handle = variable(Handle, case.semantic_annotations)
        prismatic_connection = variable(PrismaticConnection, case.connections)
        fixed_connection = match_variable(FixedConnection, case.connections)(
            parent=prismatic_connection.child, child=handle.root
        )
        return inference(Drawer)(
            root=fixed_connection.parent, handle=fixed_connection.child
        ).tolist()

    return get_drawers(case)


def conditions_35528769484583703815352905256802298589(case) -> bool:
    def has_drawers(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Wardrobe."""
        return True

    return has_drawers(case)


def conclusion_35528769484583703815352905256802298589(case) -> List[Wardrobe]:
    def get_wardrobes(case: World) -> List[Wardrobe]:
        """Get possible value(s) for World.semantic_annotations of types list/set of Wardrobe"""
        drawer = variable(Drawer, case.semantic_annotations)
        prismatic_connection = variable(PrismaticConnection, case.connections)
        return (
            entity(
                inference(Wardrobe)(
                    root=prismatic_connection.parent,
                    drawers=drawer,
                )
            )
            .where(prismatic_connection.child == drawer.root)
            .grouped_by(prismatic_connection.parent)
            .tolist()
        )

    return get_wardrobes(case)


def conditions_59112619694893607910753808758642808601(case) -> bool:
    def has_handles_and_revolute_connections(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Door."""
        return True

    return has_handles_and_revolute_connections(case)


def conclusion_59112619694893607910753808758642808601(case) -> List[Door]:
    def get_doors(case: World) -> List[Door]:
        """Get possible value(s) for World.semantic_annotations  of type Door."""
        handle = variable(Handle, case.semantic_annotations)
        revolute_connection = variable(RevoluteConnection, case.connections)
        fixed_connection = match_variable(FixedConnection, case.connections)(
            parent=revolute_connection.child, child=handle.root
        )
        return inference(Door)(
            root=fixed_connection.parent, handle=fixed_connection.child
        ).tolist()

    return get_doors(case)


def conditions_10840634078579061471470540436169882059(case) -> bool:
    def has_doors_with_fridge_in_their_name(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Fridge."""
        return True

    return has_doors_with_fridge_in_their_name(case)


def conclusion_10840634078579061471470540436169882059(case) -> List[Fridge]:
    def get_fridges(case: World) -> List[Fridge]:
        """Get possible value(s) for World.semantic_annotations of type Fridge."""
        door = variable(Door, case.semantic_annotations)
        revolute_connection = variable(RevoluteConnection, case.connections)
        return (
            entity(inference(Fridge)(root=revolute_connection.parent, doors=door))
            .where(
                revolute_connection.child == door.root,
                contains(door.root.name.name.lower(), "fridge"),
            )
            .grouped_by(revolute_connection.parent)
            .tolist()
        )

    return get_fridges(case)

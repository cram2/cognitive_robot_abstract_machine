from typing_extensions import List, Union

from ...semantic_annotations.semantic_annotations import (
    Wardrobe,
    Door,
    Drawer,
    Fridge,
    Handle,
)
from ...world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from ...world import World


def conditions_90574698325129464513441443063592862114(case) -> bool:
    def has_bodies_named_handle(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Handle."""
        return any(
            "handle" in b.name.name.lower() for b in case.kinematic_structure_entities
        )

    return has_bodies_named_handle(case)


def conclusion_90574698325129464513441443063592862114(case) -> List[Handle]:
    def get_handles(case: World) -> Union[set, list, Handle]:
        """Get possible value(s) for World.semantic_annotations of types list/set of Handle"""
        return [
            Handle(root=b)
            for b in case.kinematic_structure_entities
            if "handle" in b.name.name.lower()
        ]

    return get_handles(case)


def conditions_331345798360792447350644865254855982739(case) -> bool:
    def has_handles_and_HasCaseAsMainBodys(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Drawer."""
        return any(v for v in case.semantic_annotations if type(v) is Handle)

    return has_handles_and_HasCaseAsMainBodys(case)


def conclusion_331345798360792447350644865254855982739(case) -> List[Drawer]:
    def get_drawers(case: World) -> Union[set, list, Drawer]:
        """Get possible value(s) for World.semantic_annotations of types list/set of Drawer"""
        handles = [v for v in case.semantic_annotations if type(v) is Handle]
        fixed_connections = [
            c
            for c in case.connections
            if isinstance(c, FixedConnection) and c.child in [h.root for h in handles]
        ]
        prismatic_connections = [
            c for c in case.connections if isinstance(c, PrismaticConnection)
        ]
        drawer_handle_connections = [
            fc
            for fc in fixed_connections
            if fc.parent in [pc.child for pc in prismatic_connections]
        ]
        drawers = [
            Drawer(root=fc.parent, handle=[h for h in handles if h.root == fc.child][0])
            for fc in drawer_handle_connections
        ]
        return drawers

    return get_drawers(case)


def conditions_35528769484583703815352905256802298589(case) -> bool:
    def has_drawers(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Wardrobe."""
        return any(v for v in case.semantic_annotations if type(v) is Drawer)

    return has_drawers(case)


def conclusion_35528769484583703815352905256802298589(case) -> List[Wardrobe]:
    def get_wardrobes(case: World) -> Union[set, Wardrobe, list]:
        """Get possible value(s) for World.semantic_annotations of types list/set of Wardrobe"""
        drawers = [v for v in case.semantic_annotations if isinstance(v, Drawer)]
        prismatic_connections = [
            c
            for c in case.connections
            if isinstance(c, PrismaticConnection)
            and c.child in [drawer.root for drawer in drawers]
        ]
        wardrobe_HasCaseAsMainBody_bodies = [pc.parent for pc in prismatic_connections]
        wardrobes = []
        for ccb in wardrobe_HasCaseAsMainBody_bodies:
            if ccb in [wardrobe.root for wardrobe in wardrobes]:
                continue
            cc_prismatic_connections = [
                pc for pc in prismatic_connections if pc.parent is ccb
            ]
            wardrobe_drawer_HasCaseAsMainBody_bodies = [
                pc.child for pc in cc_prismatic_connections
            ]
            wardrobe_drawers = [
                d for d in drawers if d.root in wardrobe_drawer_HasCaseAsMainBody_bodies
            ]
            wardrobes.append(Wardrobe(root=ccb, drawers=wardrobe_drawers))

        return wardrobes

    return get_wardrobes(case)


def conditions_59112619694893607910753808758642808601(case) -> bool:
    def has_handles_and_revolute_connections(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Door."""
        return any(
            v for v in case.semantic_annotations if isinstance(v, Handle)
        ) and any(c for c in case.connections if isinstance(c, RevoluteConnection))

    return has_handles_and_revolute_connections(case)


def conclusion_59112619694893607910753808758642808601(case) -> List[Door]:
    def get_doors(case: World) -> List[Door]:
        """Get possible value(s) for World.semantic_annotations  of type Door."""
        handles = [v for v in case.semantic_annotations if isinstance(v, Handle)]
        handle_bodies = [h.root for h in handles]
        connections_with_handles = [
            c
            for c in case.connections
            if isinstance(c, FixedConnection) and c.child in handle_bodies
        ]

        revolute_connections = [
            c for c in case.connections if isinstance(c, RevoluteConnection)
        ]
        bodies_connected_to_handles = [
            c.parent if c.child in handle_bodies else c.child
            for c in connections_with_handles
        ]
        bodies_that_have_revolute_joints = [
            b
            for b in bodies_connected_to_handles
            for c in revolute_connections
            if b == c.child
        ]
        body_handle_connections = [
            c
            for c in connections_with_handles
            if c.parent in bodies_that_have_revolute_joints
        ]
        doors = [
            Door(root=c.parent, handle=[h for h in handles if h.root == c.child][0])
            for c in body_handle_connections
        ]
        return doors

    return get_doors(case)


def conditions_10840634078579061471470540436169882059(case) -> bool:
    def has_doors_with_fridge_in_their_name(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.semantic_annotations  of type Fridge."""
        return any(
            v
            for v in case.semantic_annotations
            if isinstance(v, Door) and "fridge" in v.root.name.name.lower()
        )

    return has_doors_with_fridge_in_their_name(case)


def conclusion_10840634078579061471470540436169882059(case) -> List[Fridge]:
    def get_fridges(case: World) -> List[Fridge]:
        """Get possible value(s) for World.semantic_annotations of type Fridge."""
        # Get fridge-related doors
        fridge_doors = [
            v
            for v in case.semantic_annotations
            if isinstance(v, Door) and "fridge" in v.root.name.name.lower()
        ]
        # Precompute bodies of the fridge doors
        fridge_doors_bodies = [d.root for d in fridge_doors]
        # Filter relevant revolute connections
        fridge_door_connections = [
            c
            for c in case.connections
            if isinstance(c, RevoluteConnection)
            and c.child in fridge_doors_bodies
            and "fridge" in c.parent.name.name.lower()
        ]
        return [
            Fridge(
                root=c.parent,
                doors=[fridge_doors[fridge_doors_bodies.index(c.child)]],
            )
            for c in fridge_door_connections
        ]

    return get_fridges(case)

import re

from ...datastructures.prefixed_name import PrefixedName
from ...spatial_types.spatial_types import HomogeneousTransformationMatrix
from ...semantic_annotations.factories import (
    HandleFactory,
    ContainerFactory,
    Direction,
    DrawerFactory,
    DoorFactory,
    DresserFactory,
    SemanticPositionDescription,
    HorizontalSemanticDirection,
    VerticalSemanticDirection,
)
from ...world_description.geometry import Scale
from ...world_description.world_entity import Body


def drawer_factory_from_body(drawer: Body) -> DrawerFactory:
    """
    Create a DrawerFactory from a drawer body.
    This function assumes that the drawer body has a bounding box that can be used to determine its
    scale and that a handle can be created with a standard size.
    """
    handle_factory = HandleFactory(
        name=PrefixedName(drawer.name.name + "_handle", drawer.name.prefix),
        scale=Scale(0.05, 0.1, 0.02),
    )
    container_factory = ContainerFactory(
        name=PrefixedName(drawer.name.name + "_container", drawer.name.prefix),
        scale=drawer.collision.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=drawer._world.root)
        )
        .bounding_boxes[0]
        .scale,
        direction=Direction.Z,
    )
    drawer_T_handle = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=container_factory.scale.x / 2
    )
    drawer_factory = DrawerFactory(
        name=drawer.name,
        handle_factory=handle_factory,
        parent_T_handle=drawer_T_handle,
        container_factory=container_factory,
    )
    return drawer_factory


def door_factory_from_body(door: Body) -> DoorFactory:
    """
    Create a DoorFactory from a door body.
    This function assumes that the door body has a bounding box that can be used to determine its
    scale and that a handle can be created with a standard size.
    """
    handle_factory = HandleFactory(
        name=PrefixedName(door.name.name + "_handle", door.name.prefix),
        scale=Scale(0.05, 0.1, 0.02),
    )

    door_factory = DoorFactory(
        name=door.name,
        scale=door.collision.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=door._world.root)
        )
        .bounding_boxes[0]
        .scale,
        handle_factory=handle_factory,
        semantic_position=SemanticPositionDescription(
            horizontal_direction_chain=[
                HorizontalSemanticDirection.RIGHT,
                HorizontalSemanticDirection.FULLY_CENTER,
            ],
            vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
        ),
    )
    return door_factory


def dresser_factory_from_body(dresser: Body) -> DresserFactory:
    """
    Replace a dresser body with a DresserFactory.
    This function identifies drawers and doors in the dresser based on naming conventions
    and creates corresponding factories for them.
    It assumes that drawer bodies have names containing '_drawer_' and door bodies have names
    containing '_door_'.
    """
    drawer_pattern = re.compile(r"^.*_drawer_.*$")
    door_pattern = re.compile(r"^.*_door_.*$")
    drawer_factories = []
    drawer_transforms = []
    door_factories = []
    door_transforms = []
    for child in dresser._world.compute_child_kinematic_structure_entities(dresser):
        child: Body
        if bool(drawer_pattern.fullmatch(child.name.name)):
            drawer_transforms.append(child.parent_connection.origin_expression)
            drawer_factory = drawer_factory_from_body(child)
            drawer_factories.append(drawer_factory)
        elif bool(door_pattern.fullmatch(child.name.name)):
            door_transforms.append(child.parent_connection.origin_expression)
            door_factory = door_factory_from_body(child)
            door_factories.append(door_factory)

    dresser_container_factory = ContainerFactory(
        name=PrefixedName(dresser.name.name + "_container", dresser.name.prefix),
        scale=dresser.collision.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=dresser._world.root)
        )
        .bounding_boxes[0]
        .scale,
        direction=Direction.X,
    )
    dresser_factory = DresserFactory(
        name=dresser.name,
        container_factory=dresser_container_factory,
        drawers_factories=drawer_factories,
        parent_T_drawers=drawer_transforms,
        door_factories=door_factories,
        door_transforms=door_transforms,
    )

    return dresser_factory

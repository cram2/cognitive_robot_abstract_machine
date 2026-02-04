import re

from ...datastructures.prefixed_name import PrefixedName
from ...datastructures.variables import SpatialVariables
from ...semantic_annotations.position_descriptions import (
    SemanticPositionDescription,
    HorizontalSemanticDirection,
    VerticalSemanticDirection,
)
from ...semantic_annotations.semantic_annotations import (
    Handle,
    Dresser,
    Drawer,
    Door,
    Hinge,
)
from ...spatial_types.spatial_types import HomogeneousTransformationMatrix, Vector3
from ...world import World
from ...world_description.geometry import Scale
from ...world_description.world_entity import Body


def drawer_from_body_in_world(drawer_body: Body, world: World) -> Drawer:
    """
    Create a DrawerFactory from a drawer body.
    This function assumes that the drawer body has a bounding box that can be used to determine its
    scale and that a handle can be created with a standard size.
    """
    drawer_scale = drawer_body.collision.scale

    with world.modify_world():
        drawer = Drawer.create_with_new_body_in_world(
            name=drawer_body.name,
            scale=drawer_scale,
            world=world,
        )
        world_T_drawer = drawer_body.global_pose
        drawer_T_handle = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=drawer_scale.x / 2
        )
        world_T_handle = world_T_drawer @ drawer_T_handle
        handle = Handle.create_with_new_body_in_world(
            name=PrefixedName(
                drawer_body.name.name + "_handle", drawer_body.name.prefix
            ),
            scale=Scale(0.05, 0.1, 0.02),
            world=world,
            world_root_T_self=world_T_handle,
        )
        drawer.add_handle(handle)

    return drawer


def door_from_body_in_world(door_body: Body, world: World) -> Door:
    """
    Create a DoorFactory from a door body.
    This function assumes that the door body has a bounding box that can be used to determine its
    scale and that a handle can be created with a standard size.
    """

    semantic_handle_position = SemanticPositionDescription(
        horizontal_direction_chain=[
            HorizontalSemanticDirection.RIGHT,
            HorizontalSemanticDirection.FULLY_CENTER,
        ],
        vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
    )
    scale = door_body.collision.scale
    sampled_2d_point = semantic_handle_position.sample_point_from_event(
        scale.to_simple_event().as_composite_set().marginal(SpatialVariables.yz)
    )
    door_T_handle = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=scale.x / 2, y=sampled_2d_point[0], z=sampled_2d_point[1]
    )
    world_T_door = door_body.global_pose
    world_T_handle = world_T_door @ door_T_handle

    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name=door_body.name,
            scale=door_body.collision.scale,
            world=world,
        )

        handle = Handle.create_with_new_body_in_world(
            name=PrefixedName(door_body.name.name + "_handle", door_body.name.prefix),
            scale=Scale(0.05, 0.1, 0.02),
            world=world,
            world_root_T_self=world_T_handle,
        )
        door.add_handle(handle)
    with world.modify_world():
        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName(door_body.name.name + "_hinge", door_body.name.prefix),
            world=world,
            world_root_T_self=world_T_hinge,
            active_axis=Vector3.Z(),
        )
        door.add_hinge(hinge)

    return door


def dresser_from_body_in_world(dresser: Body, world: World) -> Dresser:
    """
    Replace a dresser body with a DresserFactory.
    This function identifies drawers and doors in the dresser based on naming conventions
    and creates corresponding factories for them.
    It assumes that drawer bodies have names containing '_drawer_' and door bodies have names
    containing '_door_'.
    """
    drawer_pattern = re.compile(r"^.*_drawer_.*$")
    door_pattern = re.compile(r"^.*_door_.*$")
    with world.modify_world():
        dresser = Dresser.create_with_new_body_in_world(
            name=dresser.name,
            scale=dresser.collision.scale,
            world=world,
        )
        for child in dresser._world.compute_child_kinematic_structure_entities(
            dresser.root
        ):
            child: Body
            if bool(drawer_pattern.fullmatch(child.name.name)):
                drawer = drawer_from_body_in_world(child, world)
                dresser.add_drawer(drawer)
            elif bool(door_pattern.fullmatch(child.name.name)):
                door = door_from_body_in_world(child, world)
                dresser.add_door(door)

    return dresser

"""
General methods to access the current World.
Reasoning about alternate world states is done in the corresponding Annotators.
"""

import sys

from semantic_digital_twin.adapters.world_entity_kwargs_tracker import WorldEntityWithIDKwargsTracker
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World, Body
from semantic_digital_twin.world_description.connections import Connection6DoF

# Module-level singleton-like variables
this = sys.modules[__name__]
this.world = None

this.world_entity_tracker = None


def get_world_entity_tracker() -> WorldEntityWithIDKwargsTracker:
    return this.world_entity_tracker


def init_world_with_entity_tracker() -> WorldEntityWithIDKwargsTracker:
    world = World()
    this.world_entity_tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    set_world(world)
    return this.world_entity_tracker


def init_world_entity_tracker_from_world(world: World) -> WorldEntityWithIDKwargsTracker:
    this.world_entity_tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    return this.world_entity_tracker


def world_instance() -> World:
    """
    A singleton-like World instance.

    :return: The world state for the currently running perception pipeline.
    This is NOT necessarily the belief state World based on the previous analysis results.
    :rtype: World
    """
    if this.world is None:
        this.clear_world()

        # Setup of this world is currently the responsibility of the other nodes, loaded URDF
        # and/or camera interface.

    return this.world


def set_world(world: World) -> None:
    this.world = world


def clear_world() -> None:
    this.world = World()


def world_has_body_by_name(world: World, body_name: str) -> int:
    bodies = world.get_bodies_by_name(name=body_name)
    return len(bodies) > 0


# def add_dummy_frame_if_non_existent(frame_name: str) -> None:
#     if not world_has_body_by_name(world=world_instance(), body_name=frame_name):
#         with world_instance().modify_world():
#             world_instance().add_body(
#                 semantic_digital_twin.world.Body(name=PrefixedName(name=frame_name)))


def setup_world_for_camera_frame(world_frame: str, camera_frame: str) -> None:
    world_exists = world_has_body_by_name(world=world_instance(), body_name=world_frame)
    camera_exists = world_has_body_by_name(world=world_instance(), body_name=camera_frame)

    if world_exists and camera_exists:
        return

    if not world_exists and not camera_exists:
        with world_instance().modify_world():
            world_body = Body(name=PrefixedName(name=world_frame))
            camera_body = Body(name=PrefixedName(name=camera_frame))
            world_c_camera = Connection6DoF.create_with_dofs(parent=world_body, child=camera_body,
                                                             world=world_instance())
            world_instance().add_connection(world_c_camera)

        return

    raise AssertionError(f"This method can currently only be called when neither the world or camera frame exist. "
                         f"Existence of camera frame: {camera_exists}, world frame: {world_exists}.")

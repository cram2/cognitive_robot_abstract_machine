"""General methods to access the current World. Reasoning about alternate world states is done in the corresponding
Annotators.
"""

import sys
from threading import Lock
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World, Body
from semantic_digital_twin.world_description.connections import Connection6DoF

# Module-level singleton-like variables
this = sys.modules[__name__]

this.world = None
"""RoboKudo's central world state."""

this.world_entity_tracker = None
"""RoboKudo's central entity tracker."""

_rk_world_lock = Lock()
"""Lock for safe creation of the central SemDT World and entity tracker."""


def get_world_entity_tracker() -> WorldEntityWithIDKwargsTracker:
    """Get the entity tracker instance of for the current world.

    :return: The current entity tracker instance.
    """
    return this.world_entity_tracker


def init_world_with_entity_tracker() -> WorldEntityWithIDKwargsTracker:
    """Initialize the world and entity tracker and return the entity tracker.

    :return: The newly created entity tracker instance.
    """
    with _rk_world_lock:
        world = World()
        this.world_entity_tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        unsafe_set_world(world)
    return this.world_entity_tracker


def init_world_entity_tracker_from_world(
    world: World,
) -> WorldEntityWithIDKwargsTracker:
    """Initialize the entity tracker from the given world and return the entity tracker.

    :return: The newly created entity tracker instance.
    """
    with _rk_world_lock:
        this.world_entity_tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    return this.world_entity_tracker


def world_instance() -> World:
    """Return the world state for the currently running perception pipeline.

    .. warning::
    This is NOT necessarily the belief state World based on the previous analysis results.

    :return: A singleton-like World instance.
    """
    if this.world is None:
        with _rk_world_lock:
            # Check again with lock
            if this.world is None:
                this.unsafe_clear_world()

        # Setup of this world is currently the responsibility of the other nodes, loaded URDF
        # and/or camera interface.
    return this.world


def set_world(world: World) -> None:
    """Clear the world state safely by overwriting it with the given instance.

    :param world: The new world state.
    """
    with _rk_world_lock:
        this.unsafe_set_world(world=world)


def unsafe_set_world(world: World) -> None:
    """Unsafely set the world state without acquiring the lock.

    .. warning::
        Always acquire the lock manually before calling this method. Take a look at `this.set_world()` or
        `this.init_world_with_entity_tracker()` for example.

    :param world: The new world state to set.
    """
    this.world = world


def clear_world() -> None:
    """Clear the world state by instantiating a new World."""
    with _rk_world_lock:
        this.unsafe_clear_world()


def unsafe_clear_world() -> None:
    """Unsafely clear the world state without acquiring the lock.

    .. warning::
        Always acquire the lock manually before calling this method. Take a look at `this.clear_world()` or
        `this.world_instance()` for examples.
    """
    this.world = World()


def world_has_body_by_name(world: World, body_name: str) -> bool:
    """Check whether a body with a given name exists in the given world.

    :param world: The world to check in.
    :param body_name: The body name to search for.
    :return: True if the body exists, False otherwise.
    """
    bodies = world.get_bodies_by_name(name=body_name)
    return len(bodies) > 0


# def add_dummy_frame_if_non_existent(frame_name: str) -> None:
#     if not world_has_body_by_name(world=world_instance(), body_name=frame_name):
#         with world_instance().modify_world():
#             world_instance().add_body(
#                 semantic_digital_twin.world.Body(name=PrefixedName(name=frame_name)))


def setup_world_for_camera_frame(world_frame: str, camera_frame: str) -> None:
    """Set up the world and camera frames if they do not exist yet.

    :param world_frame: The name of the world frame.
    :param camera_frame: The name of the camera frame.
    """
    world_exists = world_has_body_by_name(world=world_instance(), body_name=world_frame)
    camera_exists = world_has_body_by_name(
        world=world_instance(), body_name=camera_frame
    )

    if world_exists and camera_exists:
        return

    if not world_exists and not camera_exists:
        with world_instance().modify_world():
            world_body = Body(name=PrefixedName(name=world_frame))
            camera_body = Body(name=PrefixedName(name=camera_frame))
            world_c_camera = Connection6DoF.create_with_dofs(
                parent=world_body, child=camera_body, world=world_instance()
            )
            world_instance().add_connection(world_c_camera)

        return

    raise AssertionError(
        f"This method can currently only be called when neither the world or camera frame exist. "
        f"Existence of camera frame: {camera_exists}, world frame: {world_exists}."
    )

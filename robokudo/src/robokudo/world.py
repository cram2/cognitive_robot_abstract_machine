"""
General methods to access the current World.

Reasoning about alternate world states is done in the corresponding Annotators.
"""

import numpy as np

from robokudo.cas import CAS
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import get_camera_to_world_transform_matrix
from robokudo.utils.transform import (
    get_transform_matrix_from_q,
    get_quaternion_from_transform_matrix,
    get_translation_from_transform_matrix,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
import trimesh.creation
from semantic_digital_twin.world_description.geometry import Mesh, Scale, Box
import sys
from uuid import UUID
from threading import Lock
from typing_extensions import Dict

from robokudo.types.belief_state import ObjectBeliefState
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World, Body
from semantic_digital_twin.world_description.connections import Connection6DoF

# Module-level singleton-like variables
this = sys.modules[__name__]

this.world = World()
"""
RoboKudo's central world state.
"""

this.world_entity_tracker = WorldEntityWithIDKwargsTracker.from_world(this.world)
"""
RoboKudo's central entity tracker.
"""

_rk_world_lock = Lock()
"""
Lock for safe creation of the central SemDT World and entity tracker.
"""

_tracked_objects: Dict[UUID, ObjectBeliefState] = {}


def get_world_entity_tracker() -> WorldEntityWithIDKwargsTracker:
    """
    Get the entity tracker instance of for the current world.

    :return: The current entity tracker instance.
    """
    return this.world_entity_tracker


def init_world_with_entity_tracker() -> WorldEntityWithIDKwargsTracker:
    """
    Initialize the world and entity tracker and return the entity tracker.

    :return: The newly created entity tracker instance.
    """
    with _rk_world_lock:
        unsafe_clear_world()
        this.world_entity_tracker = WorldEntityWithIDKwargsTracker.from_world(
            world_instance()
        )
    return this.world_entity_tracker


def init_world_entity_tracker_from_world(
    world: World,
) -> WorldEntityWithIDKwargsTracker:
    """
    Initialize the entity tracker from the given world and return the entity tracker.

    :return: The newly created entity tracker instance.
    """
    with _rk_world_lock:
        this.world_entity_tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    return this.world_entity_tracker


def world_instance() -> World:
    """
    Return the world state for the currently running perception pipeline.

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
    """
    Clear the world state safely by overwriting it with the given instance.

    :param world: The new world state.
    """
    with _rk_world_lock:
        this.unsafe_set_world(world=world)


def unsafe_set_world(world: World) -> None:
    """
    Unsafely set the world state without acquiring the lock.

    .. warning::
        Always acquire the lock manually before calling this method. Take a look at `this.set_world()` or
        `this.init_world_with_entity_tracker()` for example.

    :param world: The new world state to set.
    """
    this.world = world


def clear_world() -> None:
    """
    Clear the world state by instantiating a new World.
    """
    with _rk_world_lock:
        this.unsafe_clear_world()


def unsafe_clear_world() -> None:
    """
    Unsafely clear the world state without acquiring the lock.

    .. warning::
        Always acquire the lock manually before calling this method. Take a look at `this.clear_world()` or
        `this.world_instance()` for examples.
    """
    with this.world.modify_world():
        this.world.clear()

    model_manager = this.world.get_world_model_manager()
    model_manager.model_modification_blocks.clear()
    model_manager.current_model_modification_block.modifications.clear()


def world_has_body_by_name(world: World, body_name: str) -> bool:
    """
    Check whether a body with a given name exists in the given world.

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


def setup_world_for_camera_frame(
    world_frame: str,
    camera_frame: str,
) -> None:
    """
    Set up the world and camera frames if they do not exist yet.

    :param world_frame: The name of the world frame.
    :param camera_frame: The name of the camera frame.
    """
    rk_world = world_instance()
    world_bodies = rk_world.get_bodies_by_name(name=world_frame)
    camera_bodies = rk_world.get_bodies_by_name(name=camera_frame)

    world_body = world_bodies[0] if len(world_bodies) > 0 else None
    camera_body = camera_bodies[0] if len(camera_bodies) > 0 else None

    with rk_world.modify_world():
        if world_body is None:
            world_body = Body(name=PrefixedName(name=world_frame))
            rk_world.add_body(world_body)

            world_body.visual.append(
                Mesh.from_trimesh(
                    mesh=trimesh.creation.axis(),
                    origin=HomogeneousTransformationMatrix(reference_frame=world_body),
                )
            )

        if camera_body is None:
            camera_body = Body(name=PrefixedName(name=camera_frame))
            rk_world.add_body(camera_body)

            camera_body.visual.append(
                Mesh.from_trimesh(
                    mesh=trimesh.creation.axis(),
                    origin=HomogeneousTransformationMatrix(reference_frame=camera_body),
                )
            )

        connection_name = PrefixedName(name=f"{camera_frame}_T_{world_frame}")
        if len(rk_world.get_connections_by_name(connection_name)) == 0:
            camera_T_world = Connection6DoF.create_with_dofs(
                parent=world_body,
                child=camera_body,
                world=rk_world,
                name=connection_name,
            )
            rk_world.add_connection(camera_T_world)


def update_connection_transform(
    to_name: PrefixedName,
    from_name: PrefixedName,
    transform: HomogeneousTransformationMatrix,
) -> None:
    """
    Update the connection connecting from_name to to_name with the given transformation.

    :param to_name: The connection target name.
    :param from_name: The connection source name.
    :param transform: The transformation matrix.
    """
    world = world_instance()

    connection = world.get_connection_by_name(
        PrefixedName(name=f"{from_name.name}_T_{to_name.name}")
    )

    with world.modify_world():
        connection.origin = transform


def get_object_belief_states() -> Dict[UUID, ObjectBeliefState]:
    """
    Get all object belief states as a map of UUID to belief state.

    :return: A map of all UUIDs to their object belief states.
    """
    return _tracked_objects


def _get_world_body_from_cas(cas: CAS, rk_world: World) -> Body | None:
    """
    Return the world-frame body referenced by the CAS, if available.
    """
    world_frame = cas.world_frame
    if world_frame is None:
        return None
    return rk_world.get_body_by_name(PrefixedName(name=world_frame))


def _create_world_origin_and_scale_from_latest_bbox(
    object_belief: ObjectBeliefState,
    cas: CAS,
    world_body: Body,
) -> tuple[HomogeneousTransformationMatrix, Scale] | None:
    """
    Convert the latest camera-frame bbox into a world-frame SemDT origin and scale.
    """
    bb = object_belief.latest_bbox_3d
    if bb is None:
        return None

    pose_mat = get_transform_matrix_from_q(bb.pose.rotation, bb.pose.translation)

    camera_to_world_transform = get_camera_to_world_transform_matrix(cas)
    pose_in_world_mat = np.matmul(camera_to_world_transform, pose_mat)

    rotation = list(get_quaternion_from_transform_matrix(pose_in_world_mat))
    translation = list(get_translation_from_transform_matrix(pose_in_world_mat))

    object_belief_body = object_belief.body
    origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=translation[0],
        pos_y=translation[1],
        pos_z=translation[2],
        quat_x=rotation[0],
        quat_y=rotation[1],
        quat_z=rotation[2],
        quat_w=rotation[3],
        reference_frame=world_body,
        child_frame=object_belief_body,
    )

    scale = Scale(x=bb.x_length, y=bb.y_length, z=bb.z_length)
    return origin, scale


def _update_belief_body_from_latest_bbox(
    object_belief: ObjectBeliefState,
    cas: CAS,
    rk_world: World,
    world_body: Body,
    world_T_object_belief: Connection6DoF,
    replace_existing_visuals: bool,
) -> None:
    """
    Update the SemDT body origin and box geometry from the latest bbox.
    """
    origin_and_scale = _create_world_origin_and_scale_from_latest_bbox(
        object_belief=object_belief,
        cas=cas,
        world_body=world_body,
    )
    if origin_and_scale is None:
        return

    origin, scale = origin_and_scale
    object_belief_body = object_belief.body

    with rk_world.modify_world():
        if replace_existing_visuals:
            object_belief_body.visual.shapes.clear()
        object_belief_body.visual.append(Box(scale=scale))
        world_T_object_belief.origin = origin


def add_object_hypothesis_as_belief_state(
    object_hypothesis: ObjectHypothesis, cas: CAS
) -> ObjectBeliefState:
    """
    Create a new object belief from the given hypothesis and add it to the world.

    .. note::
        This currently assumes that all object hypotheses are rooted in the camera space.

    :param object_hypothesis: The object hypothesis to create a belief from.
    :param cas: The CAS to use for transform lookups.
    :return: The new object belief.
    """
    rk_world = world_instance()

    object_belief = ObjectBeliefState.create_with_new_body().add_hypothesis(
        object_hypothesis
    )
    object_belief_body = object_belief.body
    _tracked_objects[object_belief.uuid] = object_belief

    world_body = _get_world_body_from_cas(cas, rk_world)
    if world_body is None:
        return object_belief

    with rk_world.modify_world():
        world_T_object_belief = Connection6DoF.create_with_dofs(
            world=rk_world,
            parent=world_body,
            child=object_belief_body,
        )
        rk_world.add_connection(world_T_object_belief)

    _update_belief_body_from_latest_bbox(
        object_belief=object_belief,
        cas=cas,
        rk_world=rk_world,
        world_body=world_body,
        world_T_object_belief=world_T_object_belief,
        replace_existing_visuals=False,
    )
    return object_belief


def update_belief_state_with_object_hypothesis(
    object_belief: ObjectBeliefState, object_hypothesis: ObjectHypothesis, cas: CAS
) -> None:
    """
    Update the given object belief state with the object hypothesis.

    .. note::
        This currently assumes that all object hypotheses are rooted in the camera space.

    :param object_belief: Object belief state to update.
    :param object_hypothesis: Object hypothesis state to update the belief state with.
    :param cas: The CAS to use for transform lookups.
    """
    rk_world = world_instance()

    object_belief.add_hypothesis(object_hypothesis)
    object_belief_body = object_belief.body

    world_body = _get_world_body_from_cas(cas, rk_world)
    if world_body is None:
        return

    _update_belief_body_from_latest_bbox(
        object_belief=object_belief,
        cas=cas,
        rk_world=rk_world,
        world_body=world_body,
        world_T_object_belief=object_belief_body.get_first_parent_connection_of_type(
            Connection6DoF
        ),
        replace_existing_visuals=True,
    )


def get_object_belief_state(uuid: UUID) -> ObjectBeliefState:
    """
    Get an object belief state by UUID.

    :param uuid: The UUID of the object.
    :return: The object belief state corresponding to the given UUID.
    """
    return _tracked_objects[uuid]

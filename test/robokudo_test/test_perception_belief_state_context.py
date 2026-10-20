from __future__ import annotations

import numpy as np
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS
from robokudo.perception_belief_state_context import PerceptionBeliefStateContext

from ._object_hypotheses import make_object_hypothesis
from ._perception_world import (
    CAMERA_FRAME,
    WORLD_FRAME,
    camera_pose,
)

# %% blackboard


class TestContextOnBlackboard:
    def test_creates_and_stores_a_context_when_none_is_stored(self) -> None:
        context = PerceptionBeliefStateContext.from_blackboard()

        assert PerceptionBeliefStateContext.from_blackboard() is context
        assert not context.is_synchronized

    def test_returns_the_stored_context(self) -> None:
        context = PerceptionBeliefStateContext()
        context.store_on_blackboard()

        assert PerceptionBeliefStateContext.from_blackboard() is context


# %% world replacement


class TestWorldReplacement:
    def test_replaced_world_drops_the_objects_perceived_in_the_previous_one(
        self,
    ) -> None:
        context = PerceptionBeliefStateContext()
        cas = CAS()
        cas.annotations.append(make_object_hypothesis())
        context.perceived_objects.apply(cas)
        new_world = World()

        context.replace_world(new_world)

        assert context.world is new_world
        assert context.perceived_objects.world is new_world
        assert context.perceived_objects.beliefs == {}


# %% camera placement


class TestCameraPlacement:
    def test_adds_missing_frames_and_moves_the_camera(self) -> None:
        context = PerceptionBeliefStateContext()
        world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=2.0, z=3.0
        )

        bound_world_T_camera = context.place_camera(
            WORLD_FRAME, CAMERA_FRAME, world_T_camera
        )

        world_body = context.world.get_body_by_name(WORLD_FRAME)
        camera_body = context.world.get_body_by_name(CAMERA_FRAME)
        assert context.world.root is world_body
        assert bound_world_T_camera.reference_frame is world_body
        assert bound_world_T_camera.child_frame is camera_body
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_moves_a_placed_camera_again(self) -> None:
        context = PerceptionBeliefStateContext()
        context.place_camera(WORLD_FRAME, CAMERA_FRAME, camera_pose(x=1.0))
        world_T_camera = camera_pose(x=2.0)

        context.place_camera(WORLD_FRAME, CAMERA_FRAME, world_T_camera)

        camera_body = context.world.get_body_by_name(CAMERA_FRAME)
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_leaves_a_camera_located_by_another_kinematic_structure_in_place(
        self,
    ) -> None:
        world = World()
        world_body = Body(name=PrefixedName(WORLD_FRAME))
        head_body = Body(name=PrefixedName("head"))
        camera_body = Body(name=PrefixedName(CAMERA_FRAME))
        with world.modify_world():
            world.add_body(world_body)
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=world_body, child=head_body, world=world
                )
            )
            world.add_connection(FixedConnection(parent=head_body, child=camera_body))
        head_body.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                z=1.5, reference_frame=world_body
            )
        )
        camera_pose_before = camera_body.global_transform.to_np()
        context = PerceptionBeliefStateContext(world=world)
        world_T_camera = camera_pose(x=4.0)

        bound_world_T_camera = context.place_camera(
            WORLD_FRAME, CAMERA_FRAME, world_T_camera
        )

        assert np.allclose(camera_body.global_transform.to_np(), camera_pose_before)
        assert bound_world_T_camera.child_frame is camera_body
        assert np.allclose(bound_world_T_camera.to_np(), world_T_camera.to_np())

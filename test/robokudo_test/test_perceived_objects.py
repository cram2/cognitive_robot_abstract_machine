from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from robokudo.cas import CAS
from robokudo.perceived_objects import PerceivedObjects
from robokudo.types.annotation import StampedPoseAnnotation
from robokudo.types.scene import ObjectHypothesis

from ._object_hypotheses import make_object_hypothesis
from ._perception_world import WORLD_FRAME, camera_pose, cas_seen_from


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def perceived_objects(world: World) -> PerceivedObjects:
    return PerceivedObjects(world=world)


def cas_holding(cas: CAS, *hypotheses: ObjectHypothesis) -> CAS:
    """
    ``cas`` after a pass that found ``hypotheses``.
    """
    cas.annotations.extend(hypotheses)
    return cas


def world_poses_of(hypothesis: ObjectHypothesis) -> list[StampedPoseAnnotation]:
    return [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, StampedPoseAnnotation)
        and annotation.source == PerceivedObjects.__name__
    ]


def scale_of_latest_box(perceived_objects: PerceivedObjects, body_id) -> Scale:
    bounding_box = perceived_objects.beliefs[body_id].latest_bbox_3d
    return Scale(bounding_box.x_length, bounding_box.y_length, bounding_box.z_length)


# %% new objects


class TestNewObjects:
    def test_new_object_is_a_box_at_the_world_pose_of_its_hypothesis(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        hypothesis = make_object_hypothesis(translation=(0.0, 0.5, 2.0))
        cas = cas_holding(cas_seen_from(world, camera_pose(x=1.0)), hypothesis)

        [body_id] = perceived_objects.apply(cas)

        body = world.get_kinematic_structure_entity_by_id(body_id)
        bounding_box = perceived_objects.beliefs[body_id].latest_bbox_3d
        assert perceived_objects.beliefs[body_id].latest_hypothesis is hypothesis
        assert np.allclose(
            body.global_transform.position.to_np()[:3],
            np.array([1.0, 0.0, 0.0]) + np.array(bounding_box.pose.translation),
        )
        assert [shape.scale for shape in body.visual.shapes] == [
            scale_of_latest_box(perceived_objects, body_id)
        ]

    def test_reports_the_bodies_of_the_objects_in_the_order_they_were_found(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        hypotheses = [
            make_object_hypothesis(roi=(5, 5, 10, 10)),
            make_object_hypothesis(roi=(25, 25, 10, 10), translation=(1.0, 0.0, 1.0)),
        ]
        cas = cas_holding(cas_seen_from(world, camera_pose()), *hypotheses)

        body_ids = perceived_objects.apply(cas)

        assert [
            perceived_objects.beliefs[body_id].latest_hypothesis for body_id in body_ids
        ] == hypotheses

    def test_object_without_a_camera_pose_is_kept_outside_the_world(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [body_id] = perceived_objects.apply(
            cas_holding(CAS(), make_object_hypothesis())
        )

        assert body_id in perceived_objects.beliefs
        assert body_id not in [body.id for body in world.bodies]

    def test_pass_without_objects_reports_none(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        assert perceived_objects.apply(cas_seen_from(world, camera_pose())) == []


# %% objects seen again


class TestObjectsSeenAgain:
    def test_object_seen_again_moves_its_body_and_takes_the_new_shape(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [first_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), make_object_hypothesis())
        )
        hypothesis = make_object_hypothesis(
            translation=(0.03, 0.0, 1.0), extent=(0.12, 0.22, 0.32)
        )

        [second_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), hypothesis)
        )

        body = world.get_kinematic_structure_entity_by_id(second_id)
        assert second_id == first_id
        assert perceived_objects.beliefs[second_id].latest_hypothesis is hypothesis
        assert np.allclose(
            body.global_transform.position.to_np()[:3],
            perceived_objects.beliefs[second_id].latest_bbox_3d.pose.translation,
        )
        assert [shape.scale for shape in body.visual.shapes] == [
            scale_of_latest_box(perceived_objects, second_id)
        ]

    def test_object_seen_from_a_moved_camera_keeps_its_body(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [first_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(translation=(0.0, 0.0, 1.0)),
            )
        )
        # Same world pose as before, seen from a camera one meter further along x.
        same_object = make_object_hypothesis(
            roi=(10, 10, 12, 12), translation=(-1.0, 0.0, 1.0)
        )
        other_object = make_object_hypothesis(
            roi=(80, 80, 12, 12), translation=(2.0, 0.0, 1.0)
        )

        body_ids = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose(x=1.0)), same_object, other_object
            )
        )

        assert body_ids[0] == first_id
        assert body_ids[1] != first_id
        assert set(perceived_objects.beliefs) == {first_id, body_ids[1]}

    def test_object_too_unlike_any_known_one_gets_a_body_of_its_own(
        self, world: World
    ) -> None:
        perceived_objects = PerceivedObjects(world=world, match_threshold=0.99)
        first_hypothesis = make_object_hypothesis()
        [first_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), first_hypothesis)
        )

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(translation=(5.0, 0.0, 1.0)),
            )
        )

        assert second_id != first_id
        assert perceived_objects.beliefs[first_id].latest_hypothesis is first_hypothesis


# %% world poses


class TestWorldPoses:
    def test_each_object_is_annotated_with_its_pose_in_the_world_frame(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        hypothesis = make_object_hypothesis()

        perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), hypothesis)
        )

        assert [pose.frame for pose in world_poses_of(hypothesis)] == [WORLD_FRAME]

    def test_object_without_a_camera_pose_gets_no_world_pose(
        self, perceived_objects: PerceivedObjects
    ) -> None:
        hypothesis = make_object_hypothesis()

        perceived_objects.apply(cas_holding(CAS(), hypothesis))

        assert world_poses_of(hypothesis) == []

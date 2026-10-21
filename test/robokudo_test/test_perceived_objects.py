from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS
from robokudo.perceived_objects import PerceivedObjects
from robokudo.types.annotation import BoundingBox3DAnnotation, StampedPoseAnnotation
from robokudo.types.scene import ObjectHypothesis

from ._object_hypotheses import make_object_hypothesis
from ._perception_world import FrameName, camera_pose, cas_seen_from


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


def add_perceivable_object(
    world: World,
    *,
    position: tuple[float, float, float],
    yaw: float = 0.0,
    class_label: str | None = None,
    fixed: bool = False,
) -> Body:
    """
    Add a perceivable object of the size :func:`make_object_hypothesis` reports by
    default to ``world``, below its world frame.

    :param world: The world, whose world frame a camera was placed against.
    :param position: Position of the object in the world frame.
    :param yaw: Rotation of the object about the z axis of the world frame.
    :param class_label: The class label of the object's annotation.
    :param fixed: Whether the object is attached with a fixed connection instead of a
        connection with six degrees of freedom.
    :return: The body of the object.
    """
    world_body = world.get_body_by_name(PrefixedName(name=FrameName.WORLD))
    world_T_object = HomogeneousTransformationMatrix.from_xyz_rpy(
        *position, yaw=yaw, reference_frame=world_body
    )
    body = Body(
        name=PrefixedName("perceivable_object"),
        visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.2, 0.3))]),
    )
    with world.modify_world():
        if fixed:
            world.add_connection(
                FixedConnection.create_with_dofs(
                    world=world,
                    parent=world_body,
                    child=body,
                    parent_T_connection_expression=world_T_object,
                )
            )
        else:
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world, parent=world_body, child=body
                )
            )
        world.add_semantic_annotations([Milk(root=body, class_label=class_label)])
    if not fixed:
        body.parent_connection.origin = world_T_object
    return body


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

    def test_new_object_has_the_size_of_its_bounding_box(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [body_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), make_object_hypothesis())
        )

        size = world.get_kinematic_structure_entity_by_id(body_id).visual.scale
        expected = scale_of_latest_box(perceived_objects, body_id)
        assert np.allclose(
            [size.x, size.y, size.z], [expected.x, expected.y, expected.z]
        )

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

    def test_object_without_a_camera_pose_gets_no_body(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        body_ids = perceived_objects.apply(cas_holding(CAS(), make_object_hypothesis()))

        assert body_ids == [None]
        assert perceived_objects.beliefs == {}

    def test_object_without_a_bounding_box_gets_no_body(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        bodies_before = list(world.bodies)
        hypothesis = make_object_hypothesis()
        hypothesis.annotations = [
            annotation
            for annotation in hypothesis.annotations
            if not isinstance(annotation, BoundingBox3DAnnotation)
        ]

        body_ids = perceived_objects.apply(cas_holding(cas, hypothesis))

        assert body_ids == [None]
        assert perceived_objects.beliefs == {}
        assert list(world.bodies) == bodies_before

    def test_object_seen_by_a_camera_placed_in_another_world_gets_no_body(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        body_ids = perceived_objects.apply(
            cas_holding(cas_seen_from(World(), camera_pose()), make_object_hypothesis())
        )

        assert body_ids == [None]
        assert world.bodies == []

    def test_object_seen_against_a_prefixed_world_frame_becomes_a_body_below_it(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        world_body = Body(name=PrefixedName(FrameName.WORLD, prefix="robot"))
        with world.modify_world():
            world.add_body(world_body)
        cas = cas_seen_from(world, camera_pose())

        [body_id] = perceived_objects.apply(cas_holding(cas, make_object_hypothesis()))

        body = world.get_kinematic_structure_entity_by_id(body_id)
        assert body.parent_connection.parent is world_body

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

    def test_object_another_process_attached_rigidly_is_reported_unchanged(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [body_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), make_object_hypothesis())
        )
        body = world.get_kinematic_structure_entity_by_id(body_id)
        parent = body.parent_connection.parent
        parent_T_body = body.parent_connection.origin
        with world.modify_world():
            world.remove_connection(body.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=parent,
                    child=body,
                    parent_T_connection_expression=parent_T_body,
                )
            )
        pose_before = body.global_transform.to_np()
        scales_before = [shape.scale for shape in body.visual.shapes]

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(
                    translation=(0.03, 0.0, 1.0), extent=(0.12, 0.22, 0.32)
                ),
            )
        )

        assert second_id == body_id
        assert np.allclose(body.global_transform.to_np(), pose_before)
        assert [shape.scale for shape in body.visual.shapes] == scales_before


# %% matching


class TestMatching:
    def test_object_beyond_the_match_distance_gets_a_body_of_its_own(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        first_hypothesis = make_object_hypothesis(translation=(0.0, 0.0, 1.0))
        [first_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), first_hypothesis)
        )
        beyond = 2 * perceived_objects.matcher.maximum_match_distance

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(translation=(beyond, 0.0, 1.0)),
            )
        )

        assert second_id != first_id
        assert perceived_objects.beliefs[first_id].latest_hypothesis is first_hypothesis

    def test_object_of_another_class_gets_a_body_of_its_own(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [first_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label="cup"),
            )
        )

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label="bowl"),
            )
        )

        assert second_id != first_id

    def test_object_classified_in_only_one_pass_keeps_its_body(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [first_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), make_object_hypothesis())
        )

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label="cup"),
            )
        )

        assert second_id == first_id

    def test_object_of_a_very_different_size_gets_a_body_of_its_own(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        extent = (0.1, 0.2, 0.3)
        [first_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(extent=extent),
            )
        )
        factor = 2 * perceived_objects.matcher.maximum_match_size_ratio

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(
                    extent=tuple(factor * length for length in extent)
                ),
            )
        )

        assert second_id != first_id

    def test_objects_are_matched_so_that_as_many_as_possible_keep_their_body(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        distance = perceived_objects.matcher.maximum_match_distance
        left_id, right_id = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(translation=(0.0, 0.0, 1.0)),
                make_object_hypothesis(translation=(1.75 * distance, 0.0, 1.0)),
            )
        )
        # Closer to the left object, but also within reach of the right one, which
        # nothing else can take.
        between = make_object_hypothesis(translation=(0.85 * distance, 0.0, 1.0))
        # Within reach of the left object only.
        left = make_object_hypothesis(translation=(-0.1 * distance, 0.0, 1.0))

        body_ids = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), between, left)
        )

        assert body_ids == [right_id, left_id]


# %% objects of the world


class TestPerceivableObjectsOfTheWorld:
    def test_object_at_a_perceivable_body_is_taken_as_that_body(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0))

        [body_id] = perceived_objects.apply(
            cas_holding(cas, make_object_hypothesis(translation=(0.05, 0.0, 1.0)))
        )

        assert body_id == body.id
        assert perceived_objects.beliefs == {}

    def test_perceivable_body_moves_to_where_it_was_seen_and_keeps_its_orientation(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0), yaw=np.pi / 2)
        rotation_before = body.global_transform.to_np()[:3, :3]
        hypothesis = make_object_hypothesis(translation=(0.05, 0.0, 1.0))

        perceived_objects.apply(cas_holding(cas, hypothesis))

        assert np.allclose(
            body.global_transform.position.to_np()[:3],
            hypothesis.latest_annotation_of_type(
                BoundingBox3DAnnotation
            ).pose.translation,
        )
        assert np.allclose(body.global_transform.to_np()[:3, :3], rotation_before)

    def test_perceivable_body_keeps_its_shape(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0))
        scales_before = [shape.scale for shape in body.visual.shapes]

        perceived_objects.apply(
            cas_holding(cas, make_object_hypothesis(extent=(0.12, 0.22, 0.32)))
        )

        assert [shape.scale for shape in body.visual.shapes] == scales_before

    def test_perceivable_body_on_a_fixed_connection_is_reported_without_moving(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0), fixed=True)
        pose_before = body.global_transform.to_np()

        [body_id] = perceived_objects.apply(
            cas_holding(cas, make_object_hypothesis(translation=(0.05, 0.0, 1.0)))
        )

        assert body_id == body.id
        assert np.allclose(body.global_transform.to_np(), pose_before)

    def test_object_of_another_class_than_a_perceivable_body_gets_a_body_of_its_own(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(
            world, position=(0.0, 0.0, 1.0), class_label="milk"
        )

        [body_id] = perceived_objects.apply(
            cas_holding(cas, make_object_hypothesis(class_label="cup"))
        )

        assert body_id != body.id


# %% world poses


class TestWorldPoses:
    def test_each_object_is_annotated_with_its_pose_in_the_world_frame(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        hypothesis = make_object_hypothesis()

        perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), hypothesis)
        )

        assert [pose.frame for pose in world_poses_of(hypothesis)] == [FrameName.WORLD]

    def test_object_without_a_camera_pose_gets_no_world_pose(
        self, perceived_objects: PerceivedObjects
    ) -> None:
        hypothesis = make_object_hypothesis()

        perceived_objects.apply(cas_holding(CAS(), hypothesis))

        assert world_poses_of(hypothesis) == []

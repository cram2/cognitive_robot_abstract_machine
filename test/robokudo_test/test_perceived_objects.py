from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cup,
    Milk,
    PerceivedObject,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.geometry import Color as SemDTColor
from semantic_digital_twin.world_description.geometry import (
    Cylinder as SemDTCylinder,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color
from robokudo.cas import CAS
from robokudo.perceived_objects import PerceivedObjects
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Cylinder,
    SemanticColor,
    StampedPoseAnnotation,
)
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


def annotations_of(world: World, body_id: UUID) -> list[IsPerceivable]:
    """
    :return: The perceivable annotations of ``world`` rooted at the body ``body_id``.
    """
    return [
        annotation
        for annotation in world.get_semantic_annotations_by_type(IsPerceivable)
        if annotation.root.id == body_id
    ]


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
    def test_unclassified_object_becomes_a_perceived_object_at_its_world_pose(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        translation = (0.0, 0.5, 2.0)
        cas = cas_holding(
            cas_seen_from(world, camera_pose(x=1.0)),
            make_object_hypothesis(translation=translation),
        )

        [body_id] = perceived_objects.apply(cas)

        body = world.get_kinematic_structure_entity_by_id(body_id)
        [annotation] = annotations_of(world, body_id)
        assert type(annotation) is PerceivedObject
        assert annotation.class_label is None
        assert np.allclose(
            body.global_transform.position.to_np()[:3],
            np.array([1.0, 0.0, 0.0]) + np.array(translation),
        )

    def test_new_object_is_a_box_of_the_size_of_its_bounding_box(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        extent = (0.1, 0.2, 0.3)

        [body_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(extent=extent),
            )
        )

        [shape] = world.get_kinematic_structure_entity_by_id(body_id).visual.shapes
        assert type(shape) is Box
        assert shape.scale == Scale(*extent)

    def test_new_object_collides_with_the_shape_it_was_seen_with(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        extent = (0.1, 0.2, 0.3)

        [body_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(extent=extent),
            )
        )

        [shape] = world.get_kinematic_structure_entity_by_id(body_id).collision.shapes
        assert type(shape) is Box
        assert shape.scale == Scale(*extent)

    def test_confidently_classified_object_becomes_its_semantic_annotation(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [body_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label="cup"),
            )
        )

        [annotation] = annotations_of(world, body_id)
        assert type(annotation) is Cup

    def test_semantic_annotation_keeps_the_label_the_object_was_classified_with(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        class_label = "teacup"

        [body_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label=class_label),
            )
        )

        [annotation] = annotations_of(world, body_id)
        assert type(annotation) is Cup
        assert annotation.class_label == class_label

    def test_object_whose_label_names_no_semantic_annotation_keeps_the_label(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        class_label = "fork"

        [body_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label=class_label),
            )
        )

        [annotation] = annotations_of(world, body_id)
        assert type(annotation) is PerceivedObject
        assert annotation.class_label == class_label

    def test_object_classified_without_confidence_becomes_a_perceived_object(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        class_label = "cup"
        hypothesis = make_object_hypothesis(
            class_label=class_label,
            class_confidence=perceived_objects.minimum_class_confidence / 2,
        )

        [body_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), hypothesis)
        )

        [annotation] = annotations_of(world, body_id)
        assert type(annotation) is PerceivedObject
        assert annotation.class_label == class_label

    def test_new_object_takes_the_shape_fitted_to_it(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        bounding_box_center = (0.0, 0.0, 1.0)
        cylinder_center = (0.0, 0.01, 1.02)
        cylinder = Cylinder(
            geometry=SemDTCylinder(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(*cylinder_center),
                width=0.08,
                height=0.12,
            )
        )
        hypothesis = make_object_hypothesis(translation=bounding_box_center)
        hypothesis.annotations.append(cylinder)

        [body_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), hypothesis)
        )

        [shape] = world.get_kinematic_structure_entity_by_id(body_id).visual.shapes
        assert type(shape) is SemDTCylinder
        assert (shape.width, shape.height) == (
            cylinder.geometry.width,
            cylinder.geometry.height,
        )
        assert np.allclose(
            shape.origin.position.to_np()[:3],
            np.array(cylinder_center) - np.array(bounding_box_center),
        )

    def test_new_object_takes_its_dominant_color(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        hypothesis = make_object_hypothesis()
        hypothesis.annotations.extend(
            [
                SemanticColor(color=Color.RED.name.lower(), ratio=0.3),
                SemanticColor(color=Color.BLUE.name.lower(), ratio=0.7),
            ]
        )

        [body_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), hypothesis)
        )

        [shape] = world.get_kinematic_structure_entity_by_id(body_id).visual.shapes
        blue_in_bgr = ClusterColorAnnotator.color_name_to_bgr_values[Color.BLUE]
        assert shape.color == SemDTColor.from_rgb(
            [value / 255 for value in reversed(blue_in_bgr)]
        )

    def test_reports_the_bodies_of_the_objects_in_the_order_they_were_found(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        class_labels = ["fork", "knife"]
        hypotheses = [
            make_object_hypothesis(roi=(5, 5, 10, 10), class_label=class_labels[0]),
            make_object_hypothesis(
                roi=(25, 25, 10, 10),
                translation=(1.0, 0.0, 1.0),
                class_label=class_labels[1],
            ),
        ]
        cas = cas_holding(cas_seen_from(world, camera_pose()), *hypotheses)

        body_ids = perceived_objects.apply(cas)

        assert [
            annotation.class_label
            for body_id in body_ids
            for annotation in annotations_of(world, body_id)
        ] == class_labels

    def test_object_without_a_camera_pose_gets_no_body(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        body_ids = perceived_objects.apply(cas_holding(CAS(), make_object_hypothesis()))

        assert body_ids == [None]
        assert world.bodies == []

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
    def test_object_seen_again_moves_its_body_and_keeps_its_orientation_and_shape(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [first_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), make_object_hypothesis())
        )
        body = world.get_kinematic_structure_entity_by_id(first_id)
        rotation_before = body.global_transform.to_np()[:3, :3]
        scales_before = [shape.scale for shape in body.visual.shapes]
        translation = (0.03, 0.0, 1.0)

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(
                    translation=translation, extent=(0.12, 0.22, 0.32)
                ),
            )
        )

        assert second_id == first_id
        assert np.allclose(body.global_transform.position.to_np()[:3], translation)
        assert np.allclose(body.global_transform.to_np()[:3, :3], rotation_before)
        assert [shape.scale for shape in body.visual.shapes] == scales_before

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
        assert len(world.get_semantic_annotations_by_type(PerceivedObject)) == 2

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

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(translation=(0.03, 0.0, 1.0)),
            )
        )

        assert second_id == body_id
        assert np.allclose(body.global_transform.to_np(), pose_before)


# %% matching


class TestMatching:
    def test_object_beyond_the_match_distance_gets_a_body_of_its_own(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [first_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(translation=(0.0, 0.0, 1.0)),
            )
        )
        beyond = 2 * perceived_objects.matcher.maximum_match_distance

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(translation=(beyond, 0.0, 1.0)),
            )
        )

        assert second_id != first_id

    def test_object_seen_as_another_class_at_the_same_place_keeps_its_body(
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

        assert second_id == first_id

    def test_object_is_taken_as_the_nearby_object_of_its_own_class(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        cup_id, bowl_id = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(
                    translation=(-0.05, 0.0, 1.0), class_label="cup"
                ),
                make_object_hypothesis(
                    translation=(0.05, 0.0, 1.0), class_label="bowl"
                ),
            )
        )
        # Closer to the cup, but classified as a bowl.
        bowl = make_object_hypothesis(translation=(-0.01, 0.0, 1.0), class_label="bowl")

        [body_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), bowl)
        )

        assert body_id == bowl_id

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


# %% classification of objects seen again


class TestClassificationOfObjectsSeenAgain:
    def test_perceived_object_becomes_its_semantic_annotation_once_classified(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        class_label = "cup"
        [first_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), make_object_hypothesis())
        )

        [second_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label=class_label),
            )
        )

        [annotation] = annotations_of(world, second_id)
        assert second_id == first_id
        assert type(annotation) is Cup
        assert annotation.class_label == class_label

    def test_perceived_object_classified_without_confidence_stays_as_it_is(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [body_id] = perceived_objects.apply(
            cas_holding(cas_seen_from(world, camera_pose()), make_object_hypothesis())
        )

        perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(
                    class_label="cup",
                    class_confidence=perceived_objects.minimum_class_confidence / 2,
                ),
            )
        )

        [annotation] = annotations_of(world, body_id)
        assert type(annotation) is PerceivedObject
        assert annotation.class_label is None

    def test_semantic_annotation_keeps_its_class_when_seen_as_another(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        [body_id] = perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label="cup"),
            )
        )

        perceived_objects.apply(
            cas_holding(
                cas_seen_from(world, camera_pose()),
                make_object_hypothesis(class_label="bowl"),
            )
        )

        [annotation] = annotations_of(world, body_id)
        assert type(annotation) is Cup


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
        assert world.get_semantic_annotations_by_type(PerceivedObject) == []

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

    def test_perceivable_body_seen_as_another_class_keeps_its_annotation(
        self, world: World, perceived_objects: PerceivedObjects
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0))

        [body_id] = perceived_objects.apply(
            cas_holding(cas, make_object_hypothesis(class_label="cup"))
        )

        [annotation] = annotations_of(world, body_id)
        assert body_id == body.id
        assert type(annotation) is Milk


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

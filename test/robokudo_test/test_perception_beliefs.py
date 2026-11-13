"""
Tests of how the objects a pipeline pass found update the world perception perceives
into.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Thread

import logging

import numpy as np
import open3d as o3d
import pytest
from rclpy.node import Node
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cup,
    Human,
    Milk,
    PerceivedObject,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description import geometry
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color
from robokudo.cas import CAS
from robokudo.defs import PACKAGE_NAME
from robokudo.exceptions import CameraPoseOutsideWorldError
from robokudo.perception_beliefs import ObjectMatcher, PerceptionBelief
from robokudo.process import RoboKudo
from robokudo.types.annotation import BoundingBox3DAnnotation, Cylinder, SemanticColor
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis

from ._object_hypotheses import make_object_hypothesis
from ._perception_process import (
    EngineRunningABehaviour,
    put_pass_into_world,
    robokudo_of_its_own,
)
from ._perception_world import FrameName, camera_pose, cas_seen_from

# %% fixtures and helpers


@pytest.fixture
def robokudo(node: Node) -> Iterator[RoboKudo]:
    """
    A RoboKudo perceiving into a world of its own.
    """
    robokudo = robokudo_of_its_own(node, EngineRunningABehaviour())
    yield robokudo
    robokudo.close()


@pytest.fixture
def world(robokudo: RoboKudo) -> World:
    """
    The world :func:`robokudo` perceives into.
    """
    return robokudo.world


def perceivable_objects(world: World) -> list[IsPerceivable]:
    """
    :return: The perceivable annotations of ``world``.
    """
    return world.get_semantic_annotations_by_type(IsPerceivable)


def position_of(body: Body) -> np.ndarray:
    """
    :return: The position of ``body`` relative to the root of its world.
    """
    return body.global_transform.position.to_np()[:3]


def box_corners(
    center: tuple[float, float, float], scale: tuple[float, float, float]
) -> np.ndarray:
    """
    :return: The eight corners of the box of ``scale`` around ``center``, aligned with
        the axes of the frame ``center`` is given in.
    """
    half_lengths = np.array(scale) / 2
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float
    )
    return np.array(center) + signs * half_lengths


def hypothesis_without_a_bounding_box(points: np.ndarray | None) -> ObjectHypothesis:
    """
    :param points: The points of the object relative to the camera, or None to give it
        none.
    :return: A hypothesis with a pose but no 3D bounding box.
    """
    hypothesis = make_object_hypothesis()
    hypothesis.annotations = [
        annotation
        for annotation in hypothesis.annotations
        if not isinstance(annotation, BoundingBox3DAnnotation)
    ]
    if points is not None:
        hypothesis.points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return hypothesis


def warning_levels(caplog: pytest.LogCaptureFixture) -> list[int]:
    """
    :return: The levels of the warnings and errors RoboKudo logged during the test.
    """
    return [
        record.levelno
        for record in caplog.records
        if record.name.split(".")[0] == PACKAGE_NAME
        and record.levelno >= logging.WARNING
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
        visual=ShapeCollection(
            shapes=[geometry.Box(scale=geometry.Scale(0.1, 0.2, 0.3))]
        ),
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


@dataclass(kw_only=True)
class MatcherFacingAConcurrentChange(ObjectMatcher):
    """
    Matches while another thread tries to add a body to :attr:`changed_world`, as a
    process sharing the world does when its changes arrive.
    """

    changed_world: World
    """
    The world the other thread changes.
    """

    patience: timedelta = timedelta(milliseconds=500)
    """
    How long matching waits for the other thread's change.
    """

    change: Thread | None = field(init=False, default=None)
    """
    The thread changing :attr:`changed_world`; None until matching started.
    """

    change_was_made_while_matching: bool | None = field(init=False, default=None)
    """
    Whether the other thread changed :attr:`changed_world` before matching went on; None
    until matching started.
    """

    def match(
        self, *, beliefs: list[PerceptionBelief], world: World
    ) -> dict[PerceptionBelief, IsPerceivable]:
        self.change = Thread(target=self._add_a_body, daemon=True)
        self.change.start()
        self.change.join(timeout=self.patience.total_seconds())
        self.change_was_made_while_matching = not self.change.is_alive()
        return super().match(beliefs=beliefs, world=world)

    def _add_a_body(self) -> None:
        """
        Add a body below the root of :attr:`changed_world`.
        """
        with self.changed_world.modify_world():
            self.changed_world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=self.changed_world,
                    parent=self.changed_world.root,
                    child=Body(name=PrefixedName("added_by_another_process")),
                )
            )


# %% new objects


class TestNewObjects:
    """
    Beliefs no object of the world is about become objects of their own.
    """

    def test_unclassified_object_becomes_a_perceived_object_at_its_world_pose(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        translation = (0.0, 0.5, 2.0)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose(x=1.0)),
            make_object_hypothesis(translation=translation),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is PerceivedObject
        assert annotation.class_label is None
        assert np.allclose(
            position_of(annotation.root),
            np.array([1.0, 0.0, 0.0]) + np.array(translation),
        )

    def test_new_object_is_a_box_of_the_size_of_its_bounding_box(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        scale = (0.1, 0.2, 0.3)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(scale=scale),
        )

        [annotation] = perceivable_objects(world)
        [shape] = annotation.root.visual.shapes
        assert type(shape) is geometry.Box
        assert shape.scale == geometry.Scale(*scale)

    def test_new_object_collides_with_the_shape_it_was_seen_with(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        scale = (0.1, 0.2, 0.3)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(scale=scale),
        )

        [annotation] = perceivable_objects(world)
        [shape] = annotation.root.collision.shapes
        assert type(shape) is geometry.Box
        assert shape.scale == geometry.Scale(*scale)

    def test_confidently_classified_object_becomes_its_semantic_annotation(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label="cup"),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is Cup

    def test_semantic_annotation_keeps_the_label_the_object_was_classified_with(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        class_label = "teacup"

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label=class_label),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is Cup
        assert annotation.class_label == class_label

    def test_object_whose_label_names_no_semantic_annotation_keeps_the_label(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        class_label = "fork"

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label=class_label),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is PerceivedObject
        assert annotation.class_label == class_label

    def test_object_classified_without_confidence_becomes_a_perceived_object(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        class_label = "cup"

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label=class_label, class_confidence=0.0),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is PerceivedObject
        assert annotation.class_label == class_label

    def test_new_object_takes_the_shape_fitted_to_it_at_its_pose(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cylinder_center = (0.0, 0.01, 1.02)
        cylinder = Cylinder(
            geometry=geometry.Cylinder(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(*cylinder_center),
                width=0.08,
                height=0.12,
            )
        )
        hypothesis = make_object_hypothesis(translation=(0.0, 0.0, 1.0))
        hypothesis.annotations.append(cylinder)

        put_pass_into_world(robokudo, cas_seen_from(world, camera_pose()), hypothesis)

        [annotation] = perceivable_objects(world)
        [shape] = annotation.root.visual.shapes
        assert type(shape) is geometry.Cylinder
        assert (shape.width, shape.height) == (
            cylinder.geometry.width,
            cylinder.geometry.height,
        )
        assert np.allclose(position_of(annotation.root), cylinder_center)
        assert np.allclose(shape.origin.to_np(), np.eye(4))

    def test_object_with_only_a_fitted_shape_is_placed_at_it(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cylinder_center = (0.0, 0.01, 1.02)
        hypothesis = hypothesis_without_a_bounding_box(None)
        hypothesis.annotations.append(
            Cylinder(
                geometry=geometry.Cylinder(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        *cylinder_center
                    ),
                    width=0.08,
                    height=0.12,
                )
            )
        )

        put_pass_into_world(robokudo, cas_seen_from(world, camera_pose()), hypothesis)

        [annotation] = perceivable_objects(world)
        assert np.allclose(position_of(annotation.root), cylinder_center)

    def test_new_object_takes_its_dominant_color(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        hypothesis = make_object_hypothesis()
        hypothesis.annotations.extend(
            [
                SemanticColor(color=Color.RED.name.lower(), ratio=0.3),
                SemanticColor(color=Color.BLUE.name.lower(), ratio=0.7),
            ]
        )

        put_pass_into_world(robokudo, cas_seen_from(world, camera_pose()), hypothesis)

        [annotation] = perceivable_objects(world)
        [shape] = annotation.root.visual.shapes
        blue, green, red = ClusterColorAnnotator.color_name_to_bgr_values[Color.BLUE]
        assert shape.color == geometry.Color.from_rgba(
            red=red / 255, green=green / 255, blue=blue / 255
        )

    def test_each_object_of_a_pass_becomes_an_object_of_its_own(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        class_labels = ["fork", "knife"]

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(roi=(5, 5, 10, 10), class_label=class_labels[0]),
            make_object_hypothesis(
                roi=(25, 25, 10, 10),
                translation=(1.0, 0.0, 1.0),
                class_label=class_labels[1],
            ),
        )

        assert sorted(
            annotation.class_label for annotation in perceivable_objects(world)
        ) == sorted(class_labels)

    def test_pass_without_a_camera_pose_leaves_the_world_as_it_is_and_warns(
        self, *, robokudo: RoboKudo, world: World, caplog: pytest.LogCaptureFixture
    ) -> None:
        put_pass_into_world(robokudo, CAS(), make_object_hypothesis())

        assert world.bodies == []
        assert warning_levels(caplog) == [logging.WARNING]

    def test_pass_seen_by_a_camera_placed_in_another_world_raises(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cas = cas_seen_from(World(), camera_pose())

        with pytest.raises(CameraPoseOutsideWorldError):
            put_pass_into_world(robokudo, cas, make_object_hypothesis())

    def test_object_seen_against_a_prefixed_world_frame_becomes_a_body_below_it(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        world_body = Body(name=PrefixedName(FrameName.WORLD, prefix="robot"))
        with world.modify_world():
            world.add_body(world_body)
        cas = cas_seen_from(world, camera_pose(), world_frame=str(world_body.name))

        put_pass_into_world(robokudo, cas, make_object_hypothesis())

        [annotation] = perceivable_objects(world)
        assert annotation.root.parent_connection.parent is world_body

    def test_pass_without_objects_leaves_the_world_as_it_is(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        bodies_before = list(world.bodies)

        put_pass_into_world(robokudo, cas)

        assert list(world.bodies) == bodies_before

    def test_putting_a_pass_into_the_world_leaves_its_hypotheses_as_they_are(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        hypothesis = make_object_hypothesis()
        annotations_before = list(hypothesis.annotations)

        put_pass_into_world(robokudo, cas_seen_from(world, camera_pose()), hypothesis)

        assert hypothesis.annotations == annotations_before


# %% objects without a 3D bounding box


class TestObjectsWithoutABoundingBox:
    """
    Objects whose hypothesis has no 3D bounding box, located by their points instead.
    """

    def test_object_without_a_bounding_box_becomes_a_box_around_its_points(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        center = (0.0, 0.0, 1.0)
        scale = (0.1, 0.2, 0.3)
        hypothesis = hypothesis_without_a_bounding_box(box_corners(center, scale))

        put_pass_into_world(robokudo, cas_seen_from(world, camera_pose()), hypothesis)

        [annotation] = perceivable_objects(world)
        [shape] = annotation.root.visual.shapes
        assert np.allclose(position_of(annotation.root), center)
        assert np.allclose(
            sorted((shape.scale.x, shape.scale.y, shape.scale.z)), sorted(scale)
        )

    def test_bounding_box_is_used_over_the_points(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        translation = (0.0, 0.0, 1.0)
        scale = (0.1, 0.2, 0.3)
        hypothesis = make_object_hypothesis(translation=translation, scale=scale)
        hypothesis.points = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(box_corners((0.5, 0.0, 1.0), (0.4, 0.4, 0.4)))
        )

        put_pass_into_world(robokudo, cas_seen_from(world, camera_pose()), hypothesis)

        [annotation] = perceivable_objects(world)
        [shape] = annotation.root.visual.shapes
        assert np.allclose(position_of(annotation.root), translation)
        assert shape.scale == geometry.Scale(*scale)

    def test_object_without_a_bounding_box_or_points_gets_no_body_and_a_warning(
        self, *, robokudo: RoboKudo, world: World, caplog: pytest.LogCaptureFixture
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        bodies_before = list(world.bodies)

        put_pass_into_world(robokudo, cas, hypothesis_without_a_bounding_box(None))

        assert list(world.bodies) == bodies_before
        assert warning_levels(caplog) == [logging.WARNING]

    def test_object_with_too_few_points_for_a_box_gets_no_body_and_a_warning(
        self, *, robokudo: RoboKudo, world: World, caplog: pytest.LogCaptureFixture
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        bodies_before = list(world.bodies)
        points = box_corners((0.0, 0.0, 1.0), (0.1, 0.2, 0.3))[
            : PerceptionBelief.minimum_box_point_count - 1
        ]

        put_pass_into_world(robokudo, cas, hypothesis_without_a_bounding_box(points))

        assert list(world.bodies) == bodies_before
        assert warning_levels(caplog) == [logging.WARNING]

    def test_object_with_a_bounding_box_logs_no_warning(
        self, *, robokudo: RoboKudo, world: World, caplog: pytest.LogCaptureFixture
    ) -> None:
        put_pass_into_world(
            robokudo, cas_seen_from(world, camera_pose()), make_object_hypothesis()
        )

        assert warning_levels(caplog) == []


# %% humans


class TestHumans:
    """
    Hypotheses about humans, which become :class:`Human` annotations.
    """

    def test_human_becomes_a_human_with_its_class_label(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        class_label = "person"

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(
                class_label=class_label, hypothesis_type=HumanHypothesis
            ),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is Human
        assert annotation.class_label == class_label

    def test_human_seen_again_keeps_its_body(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(hypothesis_type=HumanHypothesis),
        )
        [human] = perceivable_objects(world)
        translation = (0.05, 0.0, 1.0)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(
                translation=translation, hypothesis_type=HumanHypothesis
            ),
        )

        assert perceivable_objects(world) == [human]
        assert np.allclose(position_of(human.root), translation)

    def test_perceived_object_seen_as_a_human_becomes_a_human(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo, cas_seen_from(world, camera_pose()), make_object_hypothesis()
        )
        [perceived_object] = perceivable_objects(world)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(hypothesis_type=HumanHypothesis),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is Human
        assert annotation.root is perceived_object.root


# %% objects seen again


class TestObjectsSeenAgain:
    """
    Beliefs about an object perception put into the world before.
    """

    def test_object_seen_again_moves_its_body_and_keeps_its_orientation_and_shape(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo, cas_seen_from(world, camera_pose()), make_object_hypothesis()
        )
        [known_object] = perceivable_objects(world)
        body = known_object.root
        rotation_before = body.global_transform.to_np()[:3, :3]
        scales_before = [shape.scale for shape in body.visual.shapes]
        translation = (0.03, 0.0, 1.0)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=translation, scale=(0.12, 0.22, 0.32)),
        )

        assert [annotation.root for annotation in perceivable_objects(world)] == [body]
        assert np.allclose(position_of(body), translation)
        assert np.allclose(body.global_transform.to_np()[:3, :3], rotation_before)
        assert [shape.scale for shape in body.visual.shapes] == scales_before

    def test_object_seen_from_a_moved_camera_keeps_its_body(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        translation = (0.0, 0.0, 1.0)
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=translation),
        )
        [known_object] = perceivable_objects(world)
        # Same world pose as before, seen from a camera one meter further along x.
        same_object = make_object_hypothesis(
            roi=(10, 10, 12, 12), translation=(-1.0, 0.0, 1.0)
        )
        other_object = make_object_hypothesis(
            roi=(80, 80, 12, 12), translation=(2.0, 0.0, 1.0)
        )

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose(x=1.0)),
            same_object,
            other_object,
        )

        bodies = [annotation.root for annotation in perceivable_objects(world)]
        assert len(bodies) == 2
        assert known_object.root in bodies
        assert np.allclose(position_of(known_object.root), translation)

    def test_object_another_process_attached_rigidly_stays_where_it_is(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo, cas_seen_from(world, camera_pose()), make_object_hypothesis()
        )
        [known_object] = perceivable_objects(world)
        body = known_object.root
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

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=(0.03, 0.0, 1.0)),
        )

        assert [annotation.root for annotation in perceivable_objects(world)] == [body]
        assert np.allclose(body.global_transform.to_np(), pose_before)


# %% matching


class TestMatching:
    """
    Which object of the world a belief is taken to be about.
    """

    def test_object_beyond_the_match_distance_gets_a_body_of_its_own(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=(0.0, 0.0, 1.0)),
        )
        beyond = 2 * ObjectMatcher().maximum_match_distance

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=(beyond, 0.0, 1.0)),
        )

        assert len(perceivable_objects(world)) == 2

    def test_object_seen_as_another_class_at_the_same_place_keeps_its_body(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label="cup"),
        )
        [known_object] = perceivable_objects(world)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label="bowl"),
        )

        assert [annotation.root for annotation in perceivable_objects(world)] == [
            known_object.root
        ]

    def test_object_is_taken_as_the_nearby_object_of_its_own_class(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cup_translation = (-0.05, 0.0, 1.0)
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=cup_translation, class_label="cup"),
            make_object_hypothesis(translation=(0.05, 0.0, 1.0), class_label="bowl"),
        )
        bodies = {
            annotation.class_label: annotation.root
            for annotation in perceivable_objects(world)
        }
        # Closer to the cup, but classified as a bowl.
        bowl_translation = (-0.01, 0.0, 1.0)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=bowl_translation, class_label="bowl"),
        )

        assert len(perceivable_objects(world)) == 2
        assert np.allclose(position_of(bodies["bowl"]), bowl_translation)
        assert np.allclose(position_of(bodies["cup"]), cup_translation)

    def test_object_of_a_very_different_size_gets_a_body_of_its_own(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        scale = (0.1, 0.2, 0.3)
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(scale=scale),
        )
        factor = 2 * ObjectMatcher().maximum_match_size_ratio

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(scale=tuple(factor * length for length in scale)),
        )

        assert len(perceivable_objects(world)) == 2

    def test_objects_are_matched_so_that_as_many_as_possible_keep_their_body(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        distance = ObjectMatcher().maximum_match_distance
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=(0.0, 0.0, 1.0)),
            make_object_hypothesis(translation=(1.75 * distance, 0.0, 1.0)),
        )
        left_body, right_body = sorted(
            (annotation.root for annotation in perceivable_objects(world)),
            key=lambda body: position_of(body)[0],
        )
        # Closer to the left object, but also within reach of the right one, which
        # nothing else can take.
        between_translation = (0.85 * distance, 0.0, 1.0)
        # Within reach of the left object only.
        left_translation = (-0.1 * distance, 0.0, 1.0)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(translation=between_translation),
            make_object_hypothesis(translation=left_translation),
        )

        assert len(perceivable_objects(world)) == 2
        assert np.allclose(position_of(right_body), between_translation)
        assert np.allclose(position_of(left_body), left_translation)


# %% classification of objects seen again


class TestClassificationOfObjectsSeenAgain:
    """
    How the semantic annotation of an object seen again follows its classification.
    """

    def test_perceived_object_becomes_its_semantic_annotation_once_classified(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        class_label = "cup"
        put_pass_into_world(
            robokudo, cas_seen_from(world, camera_pose()), make_object_hypothesis()
        )
        [known_object] = perceivable_objects(world)

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label=class_label),
        )

        [annotation] = perceivable_objects(world)
        assert annotation.root is known_object.root
        assert type(annotation) is Cup
        assert annotation.class_label == class_label

    def test_perceived_object_classified_without_confidence_stays_as_it_is(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo, cas_seen_from(world, camera_pose()), make_object_hypothesis()
        )

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label="cup", class_confidence=0.0),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is PerceivedObject
        assert annotation.class_label is None

    def test_semantic_annotation_keeps_its_class_when_seen_as_another(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label="cup"),
        )

        put_pass_into_world(
            robokudo,
            cas_seen_from(world, camera_pose()),
            make_object_hypothesis(class_label="bowl"),
        )

        [annotation] = perceivable_objects(world)
        assert type(annotation) is Cup


# %% objects of the world


class TestPerceivableObjectsOfTheWorld:
    """
    Beliefs about perceivable objects another process put into the world.
    """

    def test_object_at_a_perceivable_body_is_taken_as_that_body(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0))

        put_pass_into_world(
            robokudo, cas, make_object_hypothesis(translation=(0.05, 0.0, 1.0))
        )

        assert [annotation.root for annotation in perceivable_objects(world)] == [body]

    def test_perceivable_body_moves_to_where_it_was_seen_and_keeps_its_orientation(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0), yaw=np.pi / 2)
        rotation_before = body.global_transform.to_np()[:3, :3]
        hypothesis = make_object_hypothesis(translation=(0.05, 0.0, 1.0))

        put_pass_into_world(robokudo, cas, hypothesis)

        assert np.allclose(
            position_of(body),
            hypothesis.latest_annotation_of_type(
                BoundingBox3DAnnotation
            ).pose.translation,
        )
        assert np.allclose(body.global_transform.to_np()[:3, :3], rotation_before)

    def test_perceivable_body_keeps_its_shape(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0))
        scales_before = [shape.scale for shape in body.visual.shapes]

        put_pass_into_world(
            robokudo, cas, make_object_hypothesis(scale=(0.12, 0.22, 0.32))
        )

        assert [shape.scale for shape in body.visual.shapes] == scales_before

    def test_perceivable_body_on_a_fixed_connection_stays_where_it_is(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0), fixed=True)
        pose_before = body.global_transform.to_np()

        put_pass_into_world(
            robokudo, cas, make_object_hypothesis(translation=(0.05, 0.0, 1.0))
        )

        assert [annotation.root for annotation in perceivable_objects(world)] == [body]
        assert np.allclose(body.global_transform.to_np(), pose_before)

    def test_perceivable_body_seen_as_another_class_keeps_its_annotation(
        self, *, robokudo: RoboKudo, world: World
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        body = add_perceivable_object(world, position=(0.0, 0.0, 1.0))

        put_pass_into_world(robokudo, cas, make_object_hypothesis(class_label="cup"))

        [annotation] = perceivable_objects(world)
        assert annotation.root is body
        assert type(annotation) is Milk


# %% concurrent changes


class TestConcurrentChanges:
    """
    Changes of other threads while a pass is put into the world.
    """

    def test_no_other_thread_changes_the_world_while_a_pass_is_put_into_it(
        self, *, robokudo: RoboKudo, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cas = cas_seen_from(world, camera_pose())
        matcher = MatcherFacingAConcurrentChange(changed_world=world)
        monkeypatch.setattr("robokudo.process.ObjectMatcher", lambda: matcher)

        put_pass_into_world(robokudo, cas, make_object_hypothesis())

        matcher.change.join()
        assert matcher.change_was_made_while_matching is False

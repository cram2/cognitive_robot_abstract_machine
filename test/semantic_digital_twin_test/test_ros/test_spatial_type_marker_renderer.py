"""
Tests rendering evaluated spatial types as RViz markers.
"""

import numpy as np
import pytest
from visualization_msgs.msg import Marker

from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from semantic_digital_twin.adapters.ros.visualization.exceptions import (
    CannotRenderSpatialTypeError,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeMarkerRenderer,
    SpatialTypeVisualization,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
    Point3,
    Pose2D,
    Quaternion,
    Vector3,
)
from krrood_spatial_math.spatial_types import RotationMatrix, SpatialType
from semantic_digital_twin.world_description.world_entity import Body

ROOT_FRAME_NAME = "map"


def rendered_value(spatial_type: SpatialType) -> np.ndarray:
    """
    The evaluated rendered expression of a spatial type.
    """
    renderer = SpatialTypeMarkerRenderer.renderer_for(spatial_type)
    return renderer.rendered_expression(spatial_type).evaluate()


def render(spatial_type: SpatialType, **kwargs) -> list[Marker]:
    """
    Render a spatial type from its evaluated rendered expression.
    """
    request = SpatialTypeVisualization(spatial_type=spatial_type, **kwargs)
    return SpatialTypeMarkerRenderer.render(
        request, rendered_value(spatial_type), ROOT_FRAME_NAME
    )


# %% markers per spatial type


def test_point_renders_single_sphere():
    frame = Body(name=PrefixedName("frame"))
    point = Point3(1, 2, 3, reference_frame=frame)

    markers = render(point)

    assert len(markers) == 1
    marker = markers[0]
    assert marker.type == Marker.SPHERE
    assert marker.header.frame_id == str(frame.name)
    assert (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z) == (
        1,
        2,
        3,
    )
    assert marker.frame_locked is True


def test_point_without_reference_frame_uses_root():
    markers = render(Point3(0, 0, 0))

    assert markers[0].header.frame_id == ROOT_FRAME_NAME


def test_vector_renders_arrow_anchored_at_visualisation_frame():
    reference = Body(name=PrefixedName("reference"))
    visualisation = Body(name=PrefixedName("visualisation"))
    vector = Vector3(
        x=1, y=0, z=0, reference_frame=reference, visualisation_frame=visualisation
    )

    markers = render(vector)

    assert len(markers) == 1
    marker = markers[0]
    assert marker.type == Marker.ARROW
    assert marker.header.frame_id == str(visualisation.name)
    assert (marker.points[0].x, marker.points[0].y, marker.points[0].z) == (0, 0, 0)
    assert (marker.points[1].x, marker.points[1].y, marker.points[1].z) == (1, 0, 0)


def test_homogeneous_matrix_renders_axis_triad():
    frame = Body(name=PrefixedName("frame"))
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(1, 1, 1, reference_frame=frame)

    markers = render(pose)

    assert len(markers) == 3
    assert all(marker.type == Marker.ARROW for marker in markers)
    assert all(marker.header.frame_id == str(frame.name) for marker in markers)
    arrow_length = SpatialTypeVisualization(spatial_type=pose).arrow_length
    assert markers[0].points[1].x == arrow_length
    assert markers[1].points[1].y == arrow_length
    assert markers[2].points[1].z == arrow_length


def test_pose_like_adds_text_marker_when_labeled():
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0)

    markers = render(pose, label="goal")

    assert len(markers) == 4
    assert markers[3].type == Marker.TEXT_VIEW_FACING
    assert markers[3].text == "goal"


def test_rotation_matrix_renders_triad_at_origin():
    markers = render(RotationMatrix())

    assert len(markers) == 3
    for marker in markers:
        assert (
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
        ) == (
            0,
            0,
            0,
        )


def test_quaternion_renders_triad_at_origin():
    markers = render(Quaternion())

    assert len(markers) == 3
    assert all(marker.type == Marker.ARROW for marker in markers)


def test_pose_2d_renders_triad_at_its_position():
    """
    A 2D pose is drawn on the x-y plane of its reference frame.
    """
    markers = render(Pose2D(Point2(1, 2)))

    assert len(markers) == 3
    position = markers[0].pose.position
    assert (position.x, position.y, position.z) == (1, 2, 0)


def test_unsupported_spatial_type_raises():
    with pytest.raises(CannotRenderSpatialTypeError):
        render(Scalar(1.0))


def test_namespace_and_id_offset_propagate():
    markers = render(Point3(0, 0, 0), namespace="debug/point", marker_id_offset=10)

    assert markers[0].ns == "debug/point"
    assert markers[0].id == 10


# %% rendered expressions


def test_point_renders_its_position():
    """
    A point is drawn from its three coordinates, without the homogeneous entry.
    """
    assert rendered_value(Point3(1, 2, 3)).tolist() == [1, 2, 3]


def test_vector_renders_its_components():
    """
    A vector is drawn from its three components, without the homogeneous entry.
    """
    assert rendered_value(Vector3(4, 5, 6)).tolist() == [4, 5, 6]


def test_pose_like_renders_position_and_quaternion():
    """
    A transformation is drawn from its position followed by its quaternion.
    """
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3)

    assert rendered_value(pose).tolist() == [1, 2, 3, 0, 0, 0, 1]


def test_orientation_without_position_renders_at_origin():
    """
    A quaternion has no position, so it is drawn at the origin.
    """
    quaternion = Quaternion(x=0, y=0, z=1, w=0)

    assert rendered_value(quaternion).tolist() == [0, 0, 0, 0, 0, 1, 0]


# %% rendering from a given value


def test_markers_are_drawn_from_the_given_value():
    """
    Rendering takes the given value and never evaluates the spatial type.
    """
    unresolvable = FloatVariable(name="unresolvable")
    point = Point3(x=unresolvable)
    request = SpatialTypeVisualization(spatial_type=point)

    markers = SpatialTypeMarkerRenderer.render(
        request, np.array([7.0, 8.0, 9.0]), ROOT_FRAME_NAME
    )

    position = markers[0].pose.position
    assert (position.x, position.y, position.z) == (7, 8, 9)

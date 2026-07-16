import numpy as np
import pytest
from visualization_msgs.msg import Marker

from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.adapters.ros.visualization.exceptions import (
    CannotRenderSpatialTypeError,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeMarkerRenderer,
    SpatialTypeVisualization,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    RotationMatrix,
    SpatialType,
)
from semantic_digital_twin.world_description.world_entity import Body

ROOT_FRAME_NAME = "map"


def render(spatial_type: SpatialType, **kwargs) -> list[Marker]:
    request = SpatialTypeVisualization(spatial_type=spatial_type, **kwargs)
    return SpatialTypeMarkerRenderer.render(request, ROOT_FRAME_NAME)


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


def test_unsupported_spatial_type_raises():
    with pytest.raises(CannotRenderSpatialTypeError):
        render(Scalar(1.0))


def test_namespace_and_id_offset_propagate():
    markers = render(Point3(0, 0, 0), namespace="debug/point", marker_id_offset=10)

    assert markers[0].ns == "debug/point"
    assert markers[0].id == 10

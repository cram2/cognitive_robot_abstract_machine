from __future__ import annotations

import json

import numpy as np
import pytest

from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.exceptions import (
    MissingCartesianFrame3DResolver,
    ReferenceFrameMismatchError,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    OrthogonalProjection,
    Point3,
    Pose,
    Pose2D,
    RotationMatrix,
)

from .dataset.frames import KnownFrames
from .dataset.mounted_frames import MountedFrame

# %% frames fixed on another frame


@pytest.fixture
def shelf_frame(table_frame) -> MountedFrame:
    return MountedFrame(
        "shelf",
        mount_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1, z=2, reference_frame=table_frame
        ),
    )


# %% combining values expressed in frames


def test_a_transformation_from_parts_in_different_frames_raises(
    table_frame, camera_frame
):
    position = Point3(1, 2, 3, reference_frame=table_frame)
    rotation = RotationMatrix.from_rpy(yaw=0.5, reference_frame=camera_frame)

    with pytest.raises(ReferenceFrameMismatchError):
        HomogeneousTransformationMatrix.from_point_rotation_matrix(position, rotation)


def test_a_transformed_point_is_expressed_in_the_transformations_frame(
    table_frame, camera_frame
):
    table_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1, reference_frame=table_frame, child_frame=camera_frame
    )

    point_in_table = table_T_camera @ Point3(0, 2, 3, reference_frame=camera_frame)

    assert point_in_table.reference_frame is table_frame
    assert np.allclose(point_in_table.to_np()[:3], [1, 2, 3])


def test_a_point_changed_there_and_back_keeps_its_coordinates(
    table_frame, camera_frame
):
    table_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1, y=2, yaw=0.5, reference_frame=table_frame, child_frame=camera_frame
    )
    point_in_camera = Point3(0, 2, 3, reference_frame=camera_frame)

    point_copy = table_T_camera.inverse() @ (table_T_camera @ point_in_camera)

    assert point_copy.reference_frame is camera_frame
    assert np.allclose(point_copy.to_np(), point_in_camera.to_np())


def test_a_transformation_applies_to_a_point_without_a_frame(table_frame, camera_frame):
    table_T_camera = HomogeneousTransformationMatrix(
        reference_frame=table_frame, child_frame=camera_frame
    )

    point_in_table = table_T_camera @ Point3(0, 2, 3)

    assert point_in_table.reference_frame is table_frame


# %% bringing a value onto the plane of another frame


def test_a_point_projected_onto_another_frames_plane_is_expressed_in_that_frame(
    table_frame, shelf_frame
):
    projection = OrthogonalProjection(plane_frame=shelf_frame)

    point_on_shelf = projection.project(Point3(1, 2, 3, reference_frame=table_frame))

    assert point_on_shelf.reference_frame is shelf_frame
    assert np.allclose(
        np.concatenate([point_on_shelf.x.to_np(), point_on_shelf.y.to_np()]), [0, 2]
    )


def test_a_pose_flattened_onto_another_frames_plane_is_expressed_in_that_frame(
    table_frame, shelf_frame
):
    pose_in_table = Pose.from_xyz_rpy(
        x=3, y=2, z=5, yaw=0.5, reference_frame=table_frame
    )

    pose_on_shelf = Pose2D.from_pose(pose_in_table, plane_frame=shelf_frame)

    assert pose_on_shelf.reference_frame is shelf_frame
    assert np.allclose(
        np.concatenate(
            [
                pose_on_shelf.x.to_np(),
                pose_on_shelf.y.to_np(),
                pose_on_shelf.yaw.to_np(),
            ]
        ),
        [2, 2, 0.5],
    )


# %% serializing the frames of a value


def test_a_deserialized_point_refers_to_the_readers_frame(table_frame):
    point = Point3(1, 2, 3, reference_frame=table_frame)
    data = json.loads(json.dumps(to_json(point)))
    known_frames = KnownFrames()
    known_frames.add(table_frame.id, table_frame)

    point_copy = from_json(data, **known_frames.create_kwargs())

    assert point_copy.reference_frame is table_frame
    assert np.allclose(point_copy.to_np(), point.to_np())


def test_a_transformation_keeps_both_of_its_frames(table_frame, camera_frame):
    table_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1, reference_frame=table_frame, child_frame=camera_frame
    )
    known_frames = KnownFrames()
    known_frames.add(table_frame.id, table_frame)
    known_frames.add(camera_frame.id, camera_frame)

    copy = from_json(to_json(table_T_camera), **known_frames.create_kwargs())

    assert copy.reference_frame is table_frame
    assert copy.child_frame is camera_frame


def test_a_point_referring_to_a_frame_cannot_be_read_without_a_resolver(
    table_frame,
):
    data = to_json(Point3(1, 2, 3, reference_frame=table_frame))

    with pytest.raises(MissingCartesianFrame3DResolver):
        from_json(data)


def test_a_point_without_a_frame_is_read_without_a_resolver():
    point = Point3(1, 2, 3)

    point_copy = from_json(to_json(point))

    assert point_copy.reference_frame is None
    assert np.allclose(point_copy.to_np(), point.to_np())

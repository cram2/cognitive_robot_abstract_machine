"""
Camera pose and pixel orientation across rendering conventions.
"""

from __future__ import annotations

# %% Imports

from unittest.mock import Mock

import numpy as np
import pytest

from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.camera_rendering import RayIntersections
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.testing import ray_test_world
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


# %% Optical camera fixture
@pytest.fixture
def optical_camera(
    ray_test_world: tuple[World, Body, Body, Body, Body],
) -> Camera:
    """
    Provide a camera with a rectangular projection and a nonidentity optical pose.
    """
    world, body, *_ = ray_test_world
    with world.modify_world():
        body.parent_connection.origin = HomogeneousTransformationMatrix(
            data=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.2, y=0.1, z=0.3, roll=0.4, pitch=0.2, yaw=np.pi / 2
            ),
            reference_frame=body.parent_connection.parent,
            child_frame=body,
        )
    return Camera(
        name=PrefixedName(name="optical_camera"),
        root=body,
        camera_model=PinholeCameraModel.from_field_of_view(
            resolution=CameraResolution(width=6, height=4), field_of_view=FieldOfView()
        ),
    )


# %% Optical pose composition


def test_viewing_direction_uses_optical_root(optical_camera: Camera) -> None:
    """
    Positive optical z is the camera's root-relative viewing direction.
    """
    np.testing.assert_array_equal(
        optical_camera.viewing_direction.to_np(), Vector3.Z().to_np()
    )
    assert optical_camera.viewing_direction.reference_frame is optical_camera.root


@pytest.mark.parametrize("optical_roll", [0.0, np.pi / 2])
@pytest.mark.parametrize("root_pitch", [0.0, np.pi / 2])
def test_render_preserves_optical_orientation(
    optical_camera: Camera,
    monkeypatch: pytest.MonkeyPatch,
    optical_roll: float,
    root_pitch: float,
) -> None:
    """
    Rendering preserves the full optical orientation, including roll.
    """
    root = optical_camera.root
    with root._world.modify_world():
        root.parent_connection.origin = HomogeneousTransformationMatrix(
            data=HomogeneousTransformationMatrix.from_xyz_rpy(
                roll=optical_roll, pitch=root_pitch
            ),
            reference_frame=root.parent_connection.parent,
            child_frame=root,
        )
    ray_tracer = optical_camera.root._world.ray_tracer
    query = Mock(
        return_value=RayIntersections(
            hit_points=np.empty((0, 3)), ray_indices=np.empty(0, dtype=int), bodies=[]
        )
    )
    monkeypatch.setattr(ray_tracer, "ray_test", query)
    ray_tracer.render(optical_camera)
    np.testing.assert_allclose(
        ray_tracer.scene.camera_transform,
        optical_camera.root.global_transform.to_np() @ np.diag([1.0, -1.0, -1.0, 1.0]),
        atol=1e-12,
    )
    query.assert_called_once()


def test_render_places_top_left_optical_ray_at_top_left_pixel(
    optical_camera: Camera, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A hit on the optical top-left ray occupies the top-left image pixel.
    """
    ray_tracer = optical_camera.root._world.ray_tracer
    model = optical_camera.camera_model
    optical_V_top_left = np.array(
        [
            (0.5 - float(model.principal_point.x)) / model.focal_length_x,
            (0.5 - float(model.principal_point.y)) / model.focal_length_y,
            1.0,
        ]
    )
    optical_V_top_left /= np.linalg.norm(optical_V_top_left)
    world_T_optical = optical_camera.root.global_transform.to_np()
    world_V_top_left = world_T_optical[:3, :3] @ optical_V_top_left

    def intersect_top_left(
        origin_points: np.ndarray,
        target_points: np.ndarray,
        multiple_hits: bool,
        min_distance: float,
        max_distance: float,
    ) -> RayIntersections:
        """
        Return a hit only on the ray matching the optical top-left pixel center.
        """
        directions = target_points - origin_points
        distances = np.linalg.norm(directions - world_V_top_left, axis=1)
        ray_index = int(np.argmin(distances))
        np.testing.assert_allclose(directions[ray_index], world_V_top_left, atol=1e-12)
        return RayIntersections(
            hit_points=origin_points[[ray_index]] + directions[[ray_index]],
            ray_indices=np.array([ray_index]),
            bodies=[optical_camera.root],
        )

    monkeypatch.setattr(ray_tracer, "ray_test", intersect_top_left)
    frame = ray_tracer.render(optical_camera)
    expected = np.full(
        (model.resolution.width, model.resolution.height), -1, dtype=np.int32
    )
    expected[0, 0] = optical_camera.root.index
    np.testing.assert_array_equal(frame.segmentation, expected)


# %% Camera-relative image axes


@pytest.mark.parametrize(
    "root_pose",
    [
        HomogeneousTransformationMatrix.from_xyz_rpy(roll=np.pi / 2),
        HomogeneousTransformationMatrix.from_xyz_rpy(pitch=np.pi / 2),
        HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi / 2),
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.8, y=1.4, z=0.9, roll=0.4, pitch=0.7, yaw=0.3
        ),
    ],
)
def test_forward_view_preserves_optical_root_orientation_under_robot_motion(
    optical_camera: Camera,
    root_pose: HomogeneousTransformationMatrix,
) -> None:
    """
    Camera motion changes the world pose without changing image axes relative to its
    root.
    """
    root = optical_camera.root
    world = root._world
    with world.modify_world():
        root.parent_connection.origin = HomogeneousTransformationMatrix(
            data=root_pose,
            reference_frame=root.parent_connection.parent,
            child_frame=root,
        )
    np.testing.assert_allclose(
        optical_camera.world_T_forward_view.to_np(),
        root.global_transform.to_np() @ Camera.optical_R_forward_view.to_np(),
        atol=1e-12,
    )


# %% Full pinhole calibration


def test_render_preserves_off_center_projection(
    optical_camera: Camera, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Every rendered ray projects to its pixel center under the published calibration.
    """
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=8, height=6),
        focal_length_x=10.0,
        focal_length_y=12.0,
        principal_point=Point2(x=2.0, y=1.0),
    )
    optical_camera.camera_model = model
    tracer = optical_camera.root._world.ray_tracer
    query = Mock(
        return_value=RayIntersections(
            hit_points=np.empty((0, 3)), ray_indices=np.empty(0, dtype=int), bodies=[]
        )
    )
    monkeypatch.setattr(tracer, "ray_test", query)
    frame = tracer.render(optical_camera)
    origins, targets = query.call_args.args
    optical_V_rays = (targets - origins) @ optical_camera.root.global_transform.to_np()[
        :3, :3
    ]
    projected = optical_V_rays @ frame.camera_model.intrinsic_matrix.T
    projected = projected[:, :2] / projected[:, 2:]
    expected = (
        np.stack(
            np.meshgrid(
                np.arange(model.resolution.width),
                np.arange(model.resolution.height),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2)
        + 0.5
    )
    np.testing.assert_allclose(projected, expected, atol=1e-12)

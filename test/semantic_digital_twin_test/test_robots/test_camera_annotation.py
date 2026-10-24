from __future__ import annotations

import numpy as np
import pytest
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.robot_predicates import (
    get_visible_bodies,
    occluding_bodies,
)
from semantic_digital_twin.robots.pr2 import PR2KinectV1
from semantic_digital_twin.robots.garmi import GarmiCamera
from semantic_digital_twin.robots.stretch import (
    StretchCameraColor,
    StretchCameraDepth,
    StretchCameraInfra1,
    StretchCameraInfra2,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobotPart,
    Camera,
    RobotCamera,
    Sensor,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

# %% Camera annotations


def test_camera_can_be_added_without_a_robot():
    world = World()
    root = Body(name=PrefixedName(name="world"))
    with world.modify_world():
        world.add_body(root)
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=640, height=480),
        field_of_view=FieldOfView(
            horizontal_angle=np.radians(90.0),
            vertical_angle=np.radians(60.0),
        ),
    )
    camera = Camera(
        name=PrefixedName(name="camera"),
        root=root,
        camera_model=model,
    )

    with world.modify_world():
        world.add_semantic_annotation(camera)

    assert world.get_semantic_annotations_by_type(Camera) == [camera]
    assert not isinstance(camera, AbstractRobotPart)
    assert camera.resolution == model.resolution
    assert camera.field_of_view == model.field_of_view


def test_robot_camera_preserves_robot_part_and_sensor_types():
    assert issubclass(PR2KinectV1, RobotCamera)
    assert issubclass(PR2KinectV1, Camera)
    assert issubclass(PR2KinectV1, Sensor)
    assert issubclass(PR2KinectV1, AbstractRobotPart)


@pytest.mark.parametrize("camera_type", [PR2KinectV1, StretchCameraColor, GarmiCamera])
def test_robot_camera_accepts_projection_model_and_optical_root(
    camera_type: type[RobotCamera], table_world: World
) -> None:
    """
    Robot cameras retain their supplied projection model and optical root.
    """
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=320, height=240),
        field_of_view=FieldOfView(),
    )
    camera = camera_type(
        name=PrefixedName(name="camera"),
        root=table_world.root,
        camera_model=model,
    )

    np.testing.assert_array_equal(camera.viewing_direction.to_np(), Vector3.Z().to_np())
    assert camera.viewing_direction.reference_frame is table_world.root
    assert camera.camera_model is model


# %% Camera projection resolution


@pytest.mark.parametrize("check_occlusion", [False, True])
def test_camera_predicates_use_projection_model_resolution(
    check_occlusion: bool, table_world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Visibility and occlusion render at the camera model's resolution.
    """
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=320, height=240),
        field_of_view=FieldOfView(),
    )
    camera = Camera(
        name=PrefixedName(name="camera"),
        root=table_world.root,
        camera_model=model,
    )
    with table_world.modify_world():
        table_world.add_semantic_annotation(camera)

    rendered_resolutions: list[CameraResolution] = []

    def record_resolution(
        ray_tracer: RayTracer,
        camera_pose: HomogeneousTransformationMatrix,
        resolution: CameraResolution,
        min_distance: float,
        field_of_view: FieldOfView,
    ) -> np.ndarray:
        """
        Record each projection resolution without tracing the scene.
        """
        rendered_resolutions.append(resolution)
        return np.full((resolution.height, resolution.width), -1, dtype=int)

    monkeypatch.setattr(RayTracer, "create_segmentation_mask", record_resolution)

    if check_occlusion:
        occluding_bodies(camera, table_world.root)
    else:
        get_visible_bodies(camera)

    assert rendered_resolutions == [model.resolution] * (2 if check_occlusion else 1)


# %% Default robot cameras


def test_default_pr2_kinect_uses_rgb_optical_frame(
    pr2_world_state_reset: World,
) -> None:
    """
    The default Kinect observes from its RGB sensor rather than the stereo camera.
    """
    camera = pr2_world_state_reset.get_semantic_annotations_by_type(PR2KinectV1)[0]
    rgb_optical_frame = pr2_world_state_reset.get_body_by_name(
        "head_mount_kinect_rgb_optical_frame"
    )

    assert camera.root is rgb_optical_frame


@pytest.mark.parametrize(
    "camera_type",
    [StretchCameraColor, StretchCameraDepth, StretchCameraInfra1, StretchCameraInfra2],
)
def test_default_stretch_camera_uses_optical_positive_z(
    camera_type: type[RobotCamera], table_world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Default Stretch camera construction uses positive z in its optical frame.
    """
    monkeypatch.setattr(
        table_world, "get_body_in_branch_by_name", lambda root, name: table_world.root
    )

    camera = camera_type.setup_default_configuration_in_world_below_robot_root(
        table_world.root
    )

    assert np.array_equal(
        camera.viewing_direction, Vector3.Z(reference_frame=table_world.root)
    )
    assert camera.viewing_direction.reference_frame is table_world.root

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, fields

import pytest
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.camera_model import (
    FieldOfViewCameraModel,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.robots.robot_parts import Camera, RobotCamera
from semantic_digital_twin.robots.armar7 import AzureKinectRGB
from semantic_digital_twin.robots.daisy import DAiSyCamera
from semantic_digital_twin.robots.garmi import GarmiCamera
from semantic_digital_twin.robots.hsrb import (
    HSRBHandCamera,
    HSRBHeadCenterCamera,
    HSRBHeadLeftCamera,
    HSRBHeadRightCamera,
    HSRBHeadRGBDCamera,
)
from semantic_digital_twin.robots.icub3 import ICub3Camera
from semantic_digital_twin.robots.justin import JustinCamera
from semantic_digital_twin.robots.mmp_dresden import MMPDresdenCamera
from semantic_digital_twin.robots.pr2 import PR2KinectV1
from semantic_digital_twin.robots.stretch import (
    StretchCameraColor,
    StretchCameraDepth,
    StretchCameraInfra1,
    StretchCameraInfra2,
)
from semantic_digital_twin.robots.tiago import TiagoCamera, TiagoMujocoCamera
from semantic_digital_twin.robots.tracy import TracyCamera
from semantic_digital_twin.robots.unitree_g1 import D435

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% Projection model bindings
@dataclass(eq=False)
class PinholeProjection(Camera[PinholeCameraModel]):
    """
    A standalone camera with a pinhole projection contract.
    """


@dataclass(eq=False)
class AngularProjection(Camera[FieldOfViewCameraModel]):
    """
    A standalone camera with an angular projection contract.
    """


@dataclass(eq=False)
class RobotPinholeProjection(RobotCamera[PinholeCameraModel]):
    """
    A robot camera with a pinhole projection contract.
    """


# %% Field narrowing
@pytest.mark.parametrize(
    ("camera_type", "model_type"),
    [
        (PinholeProjection, PinholeCameraModel),
        (AngularProjection, FieldOfViewCameraModel),
        (RobotPinholeProjection, PinholeCameraModel),
    ],
)
def test_camera_subclass_narrows_projection_field(
    camera_type: type[Camera],
    model_type: type[PinholeCameraModel | FieldOfViewCameraModel],
) -> None:
    """
    A bound projection model becomes the camera's dataclass field type.
    """
    model_field = next(
        camera_field
        for camera_field in fields(camera_type)
        if camera_field.name == "camera_model"
    )

    assert model_field.type is model_type


def test_camera_subclass_preserves_specific_projection_attributes(
    table_world: World,
) -> None:
    """
    A pinhole camera exposes the supplied model's focal length.
    """
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(), field_of_view=FieldOfView()
    )
    camera = PinholeProjection(root=table_world.root, camera_model=model)

    assert camera.camera_model.focal_length_x == model.focal_length_x


# %% Robot projection contracts
@pytest.mark.parametrize(
    "camera_type",
    [
        AzureKinectRGB,
        DAiSyCamera,
        GarmiCamera,
        HSRBHandCamera,
        HSRBHeadCenterCamera,
        HSRBHeadLeftCamera,
        HSRBHeadRightCamera,
        HSRBHeadRGBDCamera,
        ICub3Camera,
        JustinCamera,
        MMPDresdenCamera,
        PR2KinectV1,
        StretchCameraColor,
        StretchCameraDepth,
        StretchCameraInfra1,
        StretchCameraInfra2,
        TiagoCamera,
        TiagoMujocoCamera,
        TracyCamera,
        D435,
    ],
)
def test_robot_projection_field_matches_default_model(
    camera_type: type[RobotCamera[FieldOfViewCameraModel]],
    table_world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Each robot camera narrows its projection field to its default model's type.
    """
    monkeypatch.setattr(
        table_world, "get_body_in_branch_by_name", lambda root, name: table_world.root
    )
    camera = camera_type.setup_default_configuration_in_world_below_robot_root(
        table_world.root
    )
    model_field = next(
        camera_field
        for camera_field in fields(camera_type)
        if camera_field.name == "camera_model"
    )

    assert model_field.type is type(camera.camera_model)

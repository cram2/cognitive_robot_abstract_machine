from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy
import pytest
import trimesh

pytest.importorskip(
    "robokudo_foundation_pose_tensorrt",
    reason="needs the GPU build of the ROS package robokudo_foundation_pose_tensorrt",
)

from robokudo.annotators.foundation_pose.pose_estimator import (  # noqa: E402
    ObjectObservation,
)
from robokudo.annotators.foundation_pose.tensorrt_pose_estimator import (  # noqa: E402
    TensorRTFoundationPose,
)


# %% Scene


@dataclass
class PlateOnPlane:
    """
    A thin square plate lying on the plane of the warm-up observation, covering its
    mask.
    """

    mesh_file: Path
    """
    File the plate mesh is written to.
    """

    thickness: float = 0.01
    """
    Thickness of the plate, in meters.
    """

    observation: ObjectObservation = field(
        default_factory=lambda: TensorRTFoundationPose.warm_up_observation
    )
    """
    The plane with the plate's mask.
    """

    def __post_init__(self) -> None:
        """
        Write the plate mesh, sized to cover the mask on the plane.
        """
        region = self.observation.mask_region
        size = (
            region.width
            * self.plane_depth
            / self.observation.frame.camera_model.focal_length_x
        )
        trimesh.creation.box(extents=(size, size, self.thickness)).export(
            self.mesh_file
        )

    @property
    def plane_depth(self) -> float:
        """
        Distance of the plane from the camera, in meters.
        """
        region = self.observation.mask_region
        return float(self.observation.frame.depth_image[region.pos.y, region.pos.x])


# %% Fixtures


@pytest.fixture(scope="module")
def estimator() -> TensorRTFoundationPose:
    """
    The estimator with the default engines, shared by all tests of the module.
    """
    return TensorRTFoundationPose()


@pytest.fixture
def plate(estimator: TensorRTFoundationPose, tmp_path: Path) -> PlateOnPlane:
    """
    A plate whose mesh the estimator has loaded and selected.
    """
    plate = PlateOnPlane(mesh_file=tmp_path / "plate.obj")
    estimator.load_meshes([plate.mesh_file])
    estimator.select_mesh(plate.mesh_file)
    return plate


# %% Pose estimation on the GPU


def test_plate_is_posed_on_the_plane_at_its_mask(
    estimator: TensorRTFoundationPose, plate: PlateOnPlane
):
    pose = estimator.estimate_pose(plate.observation)

    numpy.testing.assert_allclose(
        pose[:3, 3], [0.0, 0.0, plate.plane_depth + plate.thickness / 2], atol=0.01
    )


def test_tracking_an_unmoved_plate_keeps_its_position(
    estimator: TensorRTFoundationPose, plate: PlateOnPlane
):
    pose = estimator.estimate_pose(plate.observation)

    tracked_pose = estimator.track_pose(plate.observation, pose)

    numpy.testing.assert_allclose(tracked_pose[:3, 3], pose[:3, 3], atol=0.005)

"""
Point cloud calibration access through the CAS.
"""

from __future__ import annotations

# %% Imports
import pytest

from robokudo.cas import CAS, CASViews
from robokudo.exceptions import PointCloudCameraModelMissing
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from .test_pointcloud_crop import camera_model


# %% Required point cloud calibration
@pytest.mark.parametrize("views", [{}, {CASViews.POINTCLOUD_CAMERA_MODEL: None}])
def test_missing_pointcloud_camera_model_raises(views: dict[CASViews, None]) -> None:
    """
    Missing calibration raises the specific analysis precondition error.
    """
    cas = CAS(views=views)
    with pytest.raises(PointCloudCameraModelMissing):
        _ = cas.pointcloud_camera_model


def test_pointcloud_camera_model_preserves_identity(
    camera_model: PinholeCameraModel,
) -> None:
    """
    Calibration access returns the exact model stored by the producer.
    """
    cas = CAS()
    cas.pointcloud_camera_model = camera_model
    assert cas.pointcloud_camera_model is camera_model

"""
Tests for point cloud cluster extractor preconditions.
"""

import open3d
import pytest

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import (
    NaivePointCloudClusterExtractor,
    PointCloudClusterExtractor,
)
from robokudo.types.cv import ColorToDepthRatio
from robokudo.cas import CAS, CASViews
from robokudo.exceptions import PointCloudCameraModelMissing

# %% Required point cloud camera model


@pytest.mark.parametrize(
    "extractor_type",
    [PointCloudClusterExtractor, NaivePointCloudClusterExtractor],
)
def test_cluster_extractor_requires_pointcloud_camera_model(
    extractor_type: type[ThreadedAnnotator], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Both clustering paths report a missing point cloud projection model.
    """
    cas = CAS()
    cas.set_ref(CASViews.CLOUD, open3d.geometry.PointCloud())
    cas.set_ref(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
    extractor = extractor_type()
    monkeypatch.setattr(extractor, "get_cas", lambda: cas)

    with pytest.raises(PointCloudCameraModelMissing):
        extractor_type.compute.__wrapped__(extractor)

"""
Checks for contour selection and depth projection during image clustering.
"""

from __future__ import annotations

# %% Imports
import cv2
import numpy as np
import open3d
import pytest

from robokudo.annotators.image_cluster_extractor import ImageClusterExtractor
from robokudo.cas import CAS
from robokudo.exceptions import ImageContourMissing


# %% Extraction configuration
@pytest.fixture
def cluster_extractor() -> ImageClusterExtractor:
    """
    Provide an extractor with independent threshold and filtering parameters.
    """
    descriptor = ImageClusterExtractor.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.outlier_removal = False
    descriptor.parameters.erosion_iterations = 0
    return ImageClusterExtractor(descriptor=descriptor)


# %% Contour selection
def test_contour_selection_rejects_an_empty_mask(
    cluster_extractor: ImageClusterExtractor,
) -> None:
    """
    A mask without foreground pixels reports a missing contour.
    """
    with pytest.raises(ImageContourMissing):
        cluster_extractor._select_contours(np.zeros((10, 10), dtype=np.uint8))


def test_contour_selection_keeps_the_largest_outer_contour(
    cluster_extractor: ImageClusterExtractor,
) -> None:
    """
    Inner contours and smaller objects do not displace the largest object.
    """
    mask = np.zeros((40, 60), dtype=np.uint8)
    cv2.rectangle(mask, (2, 2), (22, 22), 255, cv2.FILLED)
    cv2.rectangle(mask, (5, 5), (18, 18), 0, cv2.FILLED)
    cv2.rectangle(mask, (30, 2), (40, 12), 255, cv2.FILLED)
    cluster_extractor.descriptor.parameters.contour_min_size = 50
    cluster_extractor.descriptor.parameters.num_of_objects = 1

    contours = cluster_extractor._select_contours(mask)

    assert [cv2.boundingRect(contour) for contour in contours] == [(2, 2, 21, 21)]


# %% Depth projection
def test_point_cloud_excludes_masked_depth_pixels(
    cluster_extractor: ImageClusterExtractor,
) -> None:
    """
    Only foreground depth pixels contribute projected points.
    """
    cluster_extractor.depth = np.full((2, 2), 1000, dtype=np.uint16)
    cluster_extractor.camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(
        2, 2, 1.0, 1.0, 0.0, 0.0
    )
    color_image = np.zeros((2, 2, 3), dtype=np.uint8)
    contour_mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    point_cloud = cluster_extractor._create_point_cloud(color_image, contour_mask)

    np.testing.assert_allclose(np.asarray(point_cloud.points), [[0, 0, 1], [1, 1, 1]])


# %% Annotation threshold
@pytest.mark.parametrize("minimum_points, annotation_count", [(64, 1), (65, 0)])
def test_cluster_annotation_requires_minimum_points(
    cluster_extractor: ImageClusterExtractor,
    monkeypatch: pytest.MonkeyPatch,
    minimum_points: int,
    annotation_count: int,
) -> None:
    """
    Small clouds remain visible while only qualifying clouds are annotated.
    """
    cas = CAS()
    monkeypatch.setattr(cluster_extractor, "get_cas", lambda: cas)
    cluster_extractor.depth = np.full((12, 12), 1000, dtype=np.uint16)
    cluster_extractor.camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(
        12, 12, 1.0, 1.0, 0.0, 0.0
    )
    cluster_extractor.descriptor.parameters.min_points_threshold = minimum_points
    contour = np.array([[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]], dtype=np.int32)
    color_image = np.zeros((12, 12, 3), dtype=np.uint8)

    clusters = cluster_extractor._extract_clusters(color_image, [contour])

    assert len(clusters.point_clouds) == 1
    assert len(clusters.object_hypotheses) == annotation_count
    assert cas.annotations == clusters.object_hypotheses

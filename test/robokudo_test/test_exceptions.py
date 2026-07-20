import numpy as np
import pytest

from robokudo.cas import CASViews
from robokudo.exceptions import (
    CASCheckConfigurationError,
    CASCheckFailed,
    CameraDataMissing,
    ColorToDepthRatioMissing,
    CVBridgeImageConversionError,
    CVBridgeImageShapeError,
    CVBridgeUnsupportedEncoding,
    CVBridgeUnsupportedTargetEncoding,
    CVBridgeROSImageShapeError,
    CVBridgeROSImageStepError,
    CVBridgeROSImagePayloadError,
    CVBridgeUnsupportedImageData,
    EmptyPointCloud,
    ImageContourMissing,
    PlaneModelMissing,
    PointCloudThresholdError,
    PointCloudThresholdRelation,
    PointCloudTooSmallForClustering,
    StoredCameraTransformFrameMetadataMissing,
    UnknownMode,
    WorldDescriptorBootstrapError,
    WorldDescriptorLoadError,
)
from robokudo.utils.cv_helper import (
    get_scale_coordinates,
    get_scaled_color_image_for_depth_image,
)


class _FakeCAS:
    def get(self, key):
        if key == CASViews.COLOR2DEPTH_RATIO:
            return None
        raise KeyError(key)


class TestRoboKudoExceptions:
    def test_color_to_depth_ratio_missing_message(self):
        exception = ColorToDepthRatioMissing(operation="scale coordinates")

        assert "COLOR2DEPTH_RATIO is missing from the CAS" in str(exception)
        assert "Can't scale coordinates" in str(exception)
        assert "Suggestion:" in str(exception)

    def test_unknown_mode_message(self):
        exception = UnknownMode(mode="not-a-mode", context="TestAnnotator")

        assert "TestAnnotator received an unknown mode" in str(exception)
        assert "not-a-mode" in str(exception)

    def test_camera_data_missing_message(self):
        exception = CameraDataMissing(
            data_name="Depth data",
            context="compressed depth conversion",
        )

        assert "Depth data is required for compressed depth conversion" in str(
            exception
        )
        assert "check the camera subscriptions and synchronization setup" in str(
            exception
        )

    def test_cv_bridge_image_conversion_error_message(self):
        exception = CVBridgeImageConversionError(
            source_encoding="bgr8",
            target_encoding="32FC1",
            reason="source image is not single-channel",
        )

        assert "Cannot convert ROS image encoding 'bgr8' to '32FC1'" in str(exception)
        assert "source image is not single-channel" in str(exception)

    def test_cv_bridge_image_shape_error_message(self):
        exception = CVBridgeImageShapeError(shape=(1, 2, 3, 4), dimensions=4)

        assert "Expected 2D or 3D image array" in str(exception)
        assert "(1, 2, 3, 4)" in str(exception)

    def test_cv_bridge_unsupported_encoding_message(self):
        exception = CVBridgeUnsupportedEncoding(encoding="not_an_encoding")

        assert "Unsupported ROS image encoding 'not_an_encoding'" in str(exception)

    def test_cv_bridge_unsupported_target_encoding_message(self):
        exception = CVBridgeUnsupportedTargetEncoding(target_encoding="rgb8")

        assert "Unsupported desired encoding 'rgb8'" in str(exception)

    def test_cv_bridge_ros_image_shape_error_message(self):
        exception = CVBridgeROSImageShapeError(height=0, width=640)

        assert "Invalid ROS image shape height=0, width=640" in str(exception)

    def test_cv_bridge_ros_image_step_error_message(self):
        exception = CVBridgeROSImageStepError(row_bytes=2, pixel_row_bytes=3)

        assert "ROS image step (2) is smaller than pixel row bytes (3)" in str(
            exception
        )

    def test_cv_bridge_ros_image_payload_error_message(self):
        exception = CVBridgeROSImagePayloadError(
            actual_bytes=3,
            required_bytes=4,
        )

        assert "ROS image payload too small: got 3 bytes" in str(exception)
        assert "expected at least 4" in str(exception)

    def test_cv_bridge_unsupported_image_data_message(self):
        exception = CVBridgeUnsupportedImageData(dtype="bool", channel_count=1)

        assert "Unsupported dtype for ROS image conversion" in str(exception)
        assert "dtype=bool" in str(exception)
        assert "channels=1" in str(exception)

    def test_stored_camera_transform_frame_metadata_missing_message(self):
        exception = StoredCameraTransformFrameMetadataMissing()

        assert "CAMERA_TO_WORLD_TRANSFORM is missing frame-name metadata" in str(
            exception
        )
        assert "Recreate the recording with the current storage format" in str(
            exception
        )

    def test_cas_check_configuration_error_message(self):
        exception = CASCheckConfigurationError(component_name="CASCheckFunc")

        assert "CASCheckFunc needs a check function" in str(exception)

    def test_cas_check_failed_message(self):
        exception = CASCheckFailed(reason="object hypothesis missing")

        assert "CAS check failed" in str(exception)
        assert "object hypothesis missing" in str(exception)

    def test_point_cloud_threshold_error_message(self):
        exception = PointCloudThresholdError(
            point_count=12,
            threshold=100,
            relation=PointCloudThresholdRelation.BELOW,
        )

        assert "Scene point cloud size (12) is below" in str(exception)
        assert "100" in str(exception)

    def test_plane_model_missing_message(self):
        exception = PlaneModelMissing(context="point cloud clustering")

        assert "point cloud clustering requires a plane model" in str(exception)

    def test_point_cloud_too_small_for_clustering_message(self):
        exception = PointCloudTooSmallForClustering(
            point_count=4,
            minimum_point_count=20,
            context="above-plane point cloud clustering",
        )

        assert "requires at least 20 points" in str(exception)
        assert "only 4 points" in str(exception)

    def test_empty_point_cloud_message(self):
        exception = EmptyPointCloud(context="point cloud clustering")

        assert "point cloud clustering requires a non-empty point cloud" in str(
            exception
        )

    def test_image_contour_missing_message(self):
        exception = ImageContourMissing(context="image cluster extraction")

        assert "image cluster extraction requires at least one image contour" in str(
            exception
        )

    def test_world_descriptor_load_error_message(self):
        exception = WorldDescriptorLoadError(
            ros_package="robokudo",
            module_name="missing_world",
        )

        assert "robokudo.missing_world" in str(exception)

    def test_world_descriptor_bootstrap_error_message(self):
        exception = WorldDescriptorBootstrapError(
            operation="merge world descriptor into the current world"
        )

        assert "Failed to merge world descriptor into the current world" in str(
            exception
        )

    def test_get_scaled_color_image_raises_color_to_depth_ratio_missing(self):
        color_image = np.zeros((2, 2, 3), dtype=np.uint8)

        with pytest.raises(ColorToDepthRatioMissing):
            get_scaled_color_image_for_depth_image(_FakeCAS(), color_image)

    def test_get_scale_coordinates_raises_color_to_depth_ratio_missing(self):
        with pytest.raises(ColorToDepthRatioMissing):
            get_scale_coordinates(None, (1, 2))

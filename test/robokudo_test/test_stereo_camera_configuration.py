"""
Specific diagnostics for invalid stereo camera settings.
"""

from __future__ import annotations

# %% Imports

import numpy as np
import pytest

from robokudo.descriptors.camera_configs.config_stereo import StereoCameraConfig
from robokudo.descriptors.camera_configs.exceptions import (
    InvalidStereoCalibrationTolerance,
    InvalidStereoFrameAge,
    InvalidStereoHistoryDepth,
    InvalidStereoMaximumDepth,
    InvalidStereoSynchronizationQueueSize,
    StereoOpticalFrameMissing,
    StereoPoseSourcesConflict,
)

# %% Buffer limits


@pytest.mark.parametrize("history_depth", [0, -1])
def test_stereo_history_depth_has_specific_error(history_depth: int) -> None:
    """
    DDS history errors identify the rejected subscription depth.
    """
    with pytest.raises(InvalidStereoHistoryDepth) as error:
        StereoCameraConfig(history_depth=history_depth)
    assert error.value.history_depth == history_depth


@pytest.mark.parametrize("queue_size", [0, -1])
def test_stereo_synchronization_queue_has_specific_error(queue_size: int) -> None:
    """
    Filter queue errors identify the rejected buffering limit.
    """
    with pytest.raises(InvalidStereoSynchronizationQueueSize) as error:
        StereoCameraConfig(synchronization_queue_size=queue_size)
    assert error.value.synchronization_queue_size == queue_size


# %% Physical and timing limits


@pytest.fixture(params=[0.0, -1.0, np.nan, np.inf, -np.inf])
def invalid_positive_limit(request: pytest.FixtureRequest) -> float:
    """
    Provide nonpositive or nonfinite limits rejected by the input contract.
    """
    return request.param


def test_stereo_frame_age_has_specific_error(invalid_positive_limit: float) -> None:
    """
    Age errors retain the rejected acquisition lifetime in seconds.
    """
    with pytest.raises(InvalidStereoFrameAge) as error:
        StereoCameraConfig(max_frame_age=invalid_positive_limit)
    np.testing.assert_equal(error.value.max_frame_age, invalid_positive_limit)


def test_stereo_maximum_depth_has_specific_error(invalid_positive_limit: float) -> None:
    """
    Depth errors retain the rejected distance cutoff in meters.
    """
    with pytest.raises(InvalidStereoMaximumDepth) as error:
        StereoCameraConfig(max_depth=invalid_positive_limit)
    np.testing.assert_equal(error.value.max_depth, invalid_positive_limit)


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf, -np.inf])
def test_stereo_calibration_tolerance_has_specific_error(tolerance: float) -> None:
    """
    Calibration errors retain the rejected timestamp tolerance in seconds.
    """
    with pytest.raises(InvalidStereoCalibrationTolerance) as error:
        StereoCameraConfig(calibration_tolerance=tolerance)
    np.testing.assert_equal(error.value.calibration_tolerance, tolerance)


# %% Optical frame and pose


def test_stereo_missing_optical_frame_has_specific_error() -> None:
    """
    An absent optical frame is distinguished from conflicting pose sources.
    """
    with pytest.raises(StereoOpticalFrameMissing):
        StereoCameraConfig(tf_from="")


def test_stereo_conflicting_pose_sources_have_specific_error() -> None:
    """
    Enabling TF lookup and a fixed pose identifies the conflicting settings.
    """
    with pytest.raises(StereoPoseSourcesConflict):
        StereoCameraConfig(lookup_viewpoint=True, static_camera_transform_enabled=True)

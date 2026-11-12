"""
Invalid settings for ROS stereo camera acquisition.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from robokudo.io.exceptions import InvalidStereoCameraConfiguration

# %% Buffer limits


@dataclass(kw_only=True)
class InvalidStereoHistoryDepth(InvalidStereoCameraConfiguration):
    """
    DDS subscription history cannot retain any incoming messages.
    """

    history_depth: int
    """
    Rejected number of messages retained by each subscription.
    """

    def error_message(self) -> str:
        return (
            f"Stereo history_depth must be positive; received {self.history_depth!r}."
        )

    def suggest_correction(self) -> str:
        return "Set history_depth to an integer greater than zero."


@dataclass(kw_only=True)
class InvalidStereoSynchronizationQueueSize(InvalidStereoCameraConfiguration):
    """
    Synchronization queues cannot retain messages awaiting a match.
    """

    synchronization_queue_size: int
    """
    Rejected number of messages buffered per synchronizer input.
    """

    def error_message(self) -> str:
        return (
            "Stereo synchronization_queue_size must be positive; "
            f"received {self.synchronization_queue_size!r}."
        )

    def suggest_correction(self) -> str:
        return "Set synchronization_queue_size to an integer greater than zero."


# %% Physical and timing limits


@dataclass(kw_only=True)
class InvalidStereoMaximumDepth(InvalidStereoCameraConfiguration):
    """
    The distance cutoff is nonpositive or nonfinite.
    """

    max_depth: float
    """
    Rejected maximum valid depth in meters.
    """

    def error_message(self) -> str:
        return (
            "Stereo max_depth must be finite and positive; "
            f"received {self.max_depth!r} meters."
        )

    def suggest_correction(self) -> str:
        return "Set max_depth to a finite distance in meters greater than zero."


@dataclass(kw_only=True)
class InvalidStereoCalibrationTolerance(InvalidStereoCameraConfiguration):
    """
    Calibration timestamp tolerance is negative or nonfinite.
    """

    calibration_tolerance: float
    """
    Rejected maximum timestamp difference in seconds.
    """

    def error_message(self) -> str:
        return (
            "Stereo calibration_tolerance must be finite and nonnegative; "
            f"received {self.calibration_tolerance!r} seconds."
        )

    def suggest_correction(self) -> str:
        return (
            "Set calibration_tolerance to a finite nonnegative number of seconds; "
            "zero requires exact calibration timestamps."
        )


# %% Optical frame and pose


@dataclass(kw_only=True)
class StereoOpticalFrameMissing(InvalidStereoCameraConfiguration):
    """
    No optical frame identifies the aligned stereo inputs.
    """

    def error_message(self) -> str:
        return "Stereo tf_from is empty; the input optical frame is missing."

    def suggest_correction(self) -> str:
        return (
            "Set tf_from to the optical frame shared by color, depth and camera info."
        )


@dataclass(kw_only=True)
class StereoPoseSourcesConflict(InvalidStereoCameraConfiguration):
    """
    Both acquisition-time TF and a configured static pose are enabled.
    """

    def error_message(self) -> str:
        return (
            "Stereo lookup_viewpoint and static_camera_transform_enabled are both true."
        )

    def suggest_correction(self) -> str:
        return (
            "Disable lookup_viewpoint to use the configured static pose, or disable "
            "static_camera_transform_enabled to use acquisition-time TF."
        )

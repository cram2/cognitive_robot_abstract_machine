from __future__ import annotations

# %% Imports
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from krrood.adapters.json_serializer import to_json
from krrood_spatial_math.spatial_types import Point2
from robokudo.cas import CASViews
from semantic_digital_twin.datastructures.camera_model import (
    PinholeCameraModel,
    PlumbBobCameraDistortion,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution


# %% Recorded frames
@dataclass(frozen=True)
class RecordedFrame:
    """
    One RGB-D frame of a file recording, identified by the name its files share.
    """

    file_stem: str
    """
    Filename prefix and recording timestamp shared by the frame's files.
    """

    acquired_at: np.datetime64
    """
    Frame acquisition time, to the nanosecond.
    """


# %% Cup on counter recording
@dataclass(frozen=True)
class CupOnCounterRecording:
    """
    Two Kinect frames of a single cup standing on a kitchen counter, recorded in the
    file layout :class:`robokudo.io.file_reader_interface.FileReaderInterface` reads.
    """

    images_directory: Path = Path(__file__).parent / "cup_on_counter_images"
    """
    Directory holding the recorded color and depth images.
    """

    camera_frame: str = "head_mount_kinect_rgb_optical_frame"
    """
    Optical frame the images were taken in.
    """

    camera_model: PinholeCameraModel = field(
        default_factory=lambda: PinholeCameraModel(
            image_resolution=CameraResolution(width=1280, height=960),
            focal_length_x=1050.0,
            focal_length_y=1050.0,
            principal_point=Point2(x=639.5, y=479.5),
            distortion=PlumbBobCameraDistortion(),
        )
    )
    """
    Calibration of the color camera that took the images.
    """

    frames: tuple[RecordedFrame, ...] = (
        RecordedFrame(
            file_stem="rk_1669996652.648549",
            acquired_at=np.datetime64("2022-09-08T06:36:54.679565362", "ns"),
        ),
        RecordedFrame(
            file_stem="rk_1669996652.766856",
            acquired_at=np.datetime64("2022-09-08T06:36:55.071573884", "ns"),
        ),
    )
    """
    Recorded frames in playback order.
    """

    def write_to(self, directory: Path) -> Path:
        """
        Write the recording's images and camera observations into a directory.

        :param directory: Existing directory to write the recording into.
        :return: The directory, ready to be read by a file reader.
        """
        for image in self.images_directory.iterdir():
            shutil.copy(image, directory / image.name)
        for frame in self.frames:
            observation_path = (
                directory / f"{frame.file_stem}_{CASViews.CAMERA_OBSERVATION}.json"
            )
            observation_path.write_text(
                json.dumps(
                    {
                        "camera_frame": self.camera_frame,
                        "effective_camera_model": to_json(self.camera_model),
                        "timestamp_nanoseconds": int(
                            frame.acquired_at.astype(np.int64)
                        ),
                    }
                )
            )
        return directory

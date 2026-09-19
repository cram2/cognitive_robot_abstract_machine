"""
Tests for keeping a look at the scene in files and reading it back.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.montessori.perception.camera import (
    MILLIMETRES_PER_METRE,
    CameraIntrinsics,
)
from experiments.montessori.perception.captures import (
    CAPTURE_DIRECTORY,
    CapturePart,
    SceneCapture,
)
from experiments.montessori.perception.exceptions import CaptureIncomplete

from .dataset.montessori_capture_truths import CAPTURE_TRUTHS

# %% a made-up frame to write

CAPTURE_FORMAT = "rgb8; jpeg compressed bgr8"
"""
The ``format`` this camera publishes its compressed colour images under.
"""


def a_green_image() -> np.ndarray:
    """
    :return: A small, flat green colour image, in OpenCV's channel order.
    """
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 1] = 200
    return image


def a_sloping_depth_image() -> np.ndarray:
    """
    :return: A depth image in metres whose readings rise across the frame.
    """
    return np.linspace(0.5, 1.5, 24, dtype=np.float32).reshape(4, 6)


@pytest.fixture
def written_capture(tmp_path: Path) -> SceneCapture:
    """
    A capture of that frame, already written into a directory of its own.
    """
    capture = SceneCapture(
        name="four_by_six",
        recorded_from="a_recording",
        color_format=CAPTURE_FORMAT,
        intrinsics=CameraIntrinsics(
            focal_length_x=600.0,
            focal_length_y=601.0,
            principal_point_x=3.0,
            principal_point_y=2.0,
        ),
        reference_frame="map",
        reference_frame_T_camera=np.diag([1.0, -1.0, -1.0, 1.0]),
        directory=tmp_path,
    )
    capture.save(
        cv2.imencode(CapturePart.COLOR.value, a_green_image())[1].tobytes(),
        a_sloping_depth_image(),
    )
    return capture


# %% reading one back


def test_a_written_capture_reads_back_as_it_was_written(
    written_capture: SceneCapture, tmp_path: Path
) -> None:
    """
    Everything a capture records about the camera survives the files unchanged.
    """
    read_back = SceneCapture.load(written_capture.name, tmp_path)
    assert read_back.to_json() == written_capture.to_json()
    assert read_back.directory == tmp_path


def test_a_capture_keeps_the_colour_payload_byte_for_byte(
    written_capture: SceneCapture,
) -> None:
    """
    The colour file holds the camera's own compressed payload, not a re-encoding of it,
    so a capture and the live stream hand the detectors the same pixels.
    """
    assert written_capture.path_to(CapturePart.COLOR).read_bytes() == (
        cv2.imencode(CapturePart.COLOR.value, a_green_image())[1].tobytes()
    )


def test_depth_reads_back_to_the_nearest_millimetre(
    written_capture: SceneCapture,
) -> None:
    """
    Depth is kept in millimetres, so a reading comes back rounded to one.
    """
    written = a_sloping_depth_image()
    assert written_capture.to_frame().depth == pytest.approx(
        np.round(written * MILLIMETRES_PER_METRE) / MILLIMETRES_PER_METRE
    )


def test_a_frame_carries_the_camera_the_capture_recorded(
    written_capture: SceneCapture,
) -> None:
    """
    The frame a capture is read into sees, and stands, exactly where the camera did.
    """
    frame = written_capture.to_frame()
    assert frame.intrinsics == written_capture.intrinsics
    assert frame.reference_frame_T_camera == pytest.approx(
        written_capture.reference_frame_T_camera
    )


# %% what is missing


def test_a_capture_missing_a_file_is_refused(
    written_capture: SceneCapture, tmp_path: Path
) -> None:
    """
    A capture is only readable whole, and the failure says which file is gone.
    """
    written_capture.path_to(CapturePart.DEPTH).unlink()
    with pytest.raises(CaptureIncomplete) as failure:
        SceneCapture.load(written_capture.name, tmp_path)
    assert failure.value.missing_parts == [CapturePart.DEPTH.value]


# %% the captures this package ships


def test_every_capture_this_package_ships_is_described() -> None:
    """
    Each shipped capture has its contents written down, so a detection result can be
    measured against the scene rather than against an earlier run.
    """
    assert set(SceneCapture.names_in(CAPTURE_DIRECTORY)) == set(CAPTURE_TRUTHS)


def test_a_shipped_capture_reads_into_a_registered_frame() -> None:
    """
    The colour and depth images of a shipped capture were taken through one set of
    intrinsics, which is what lets a pixel name the same ray in both.
    """
    frame = SceneCapture.load(sorted(CAPTURE_TRUTHS)[0]).to_frame()
    assert frame.color.shape[:2] == frame.depth.shape[:2]

"""
Looks at the Montessori scene kept in the repository as files.

A capture is what the camera published for one moment -- the colour image byte for byte
as it came off the wire, the depth image registered onto it, and where the camera stood
-- written to disk so the pipeline can be run on real camera data with neither a camera
nor the rosbag it was recorded in. That is what lets a test measure detection against
the real table, and lets a reviewer look at the picture a result was measured on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
from typing_extensions import Any, Dict, List, Self

from experiments.montessori.perception.camera import (
    MILLIMETRES_PER_METRE,
    CameraIntrinsics,
    RgbdFrame,
    decode_compressed_color_image,
)
from experiments.montessori.perception.exceptions import CaptureIncomplete
from krrood.adapters.json_serializer import SubclassJSONSerializer

# %% where a capture lives

CAPTURE_DIRECTORY = Path(__file__).parent.parent / "resources" / "captures"
"""
Where the captures shipped with this package are kept.
"""


class CapturePart(StrEnum):
    """
    The three files one capture is written as, named by the suffix each takes.
    """

    COLOR = "_color.jpg"
    DEPTH = "_depth.png"
    CAMERA = "_camera.json"


class CaptureField(StrEnum):
    """
    The keys a capture's camera file is written under.
    """

    NAME = "name"
    RECORDED_FROM = "recorded_from"
    COLOR_FORMAT = "color_format"
    INTRINSIC_MATRIX = "intrinsic_matrix"
    REFERENCE_FRAME = "reference_frame"
    REFERENCE_FRAME_T_CAMERA = "reference_frame_T_camera"


def capture_path(name: str, part: CapturePart, directory: Path) -> Path:
    """
    Where one of a capture's files lies.

    :param name: The capture's name, which is the stem of all three of its files.
    :param part: Which file is wanted.
    :param directory: Directory the capture was written into.
    """
    return directory / f"{name}{part.value}"


# %% the capture


@dataclass(frozen=True)
class SceneCapture(SubclassJSONSerializer):
    """
    One look at the scene, and everything needed to place it in the world.

    The colour image is kept in the codec the camera published it in, so a capture and
    the live stream feed the detectors the same pixels, compression artefacts included.
    """

    name: str
    """
    What this look at the scene is called, and the stem of all three of its files.
    """

    recorded_from: str
    """
    Name of the rosbag the frame was taken out of.
    """

    color_format: str
    """
    The ``format`` field the camera published the colour image under, which says both
    what the payload is compressed as and what its pixels are ordered in.
    """

    intrinsics: CameraIntrinsics
    """
    The intrinsics the camera reported for both images.
    """

    reference_frame: str
    """
    Frame :attr:`reference_frame_T_camera` is given in, which is the frame detections
    made on this capture come out in.
    """

    reference_frame_T_camera: np.ndarray
    """
    Where the camera's optical frame stood, as a 4x4 homogeneous transformation.
    """

    directory: Path = field(default=CAPTURE_DIRECTORY)
    """
    Directory holding this capture's three files.
    """

    def path_to(self, part: CapturePart) -> Path:
        """
        :param part: Which of this capture's files is wanted.
        :return: Where that file lies.
        """
        return capture_path(self.name, part, self.directory)

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            CaptureField.NAME.value: self.name,
            CaptureField.RECORDED_FROM.value: self.recorded_from,
            CaptureField.COLOR_FORMAT.value: self.color_format,
            CaptureField.INTRINSIC_MATRIX.value: self.intrinsics.to_matrix().tolist(),
            CaptureField.REFERENCE_FRAME.value: self.reference_frame,
            CaptureField.REFERENCE_FRAME_T_CAMERA.value: (
                self.reference_frame_T_camera.tolist()
            ),
        }

    @classmethod
    def _from_json(
        cls, data: Dict[str, Any], directory: Path = CAPTURE_DIRECTORY
    ) -> Self:
        return cls(
            name=data[CaptureField.NAME.value],
            recorded_from=data[CaptureField.RECORDED_FROM.value],
            color_format=data[CaptureField.COLOR_FORMAT.value],
            intrinsics=CameraIntrinsics.from_camera_info_matrix(
                np.asarray(data[CaptureField.INTRINSIC_MATRIX.value], dtype=float)
            ),
            reference_frame=data[CaptureField.REFERENCE_FRAME.value],
            reference_frame_T_camera=np.asarray(
                data[CaptureField.REFERENCE_FRAME_T_CAMERA.value], dtype=float
            ),
            directory=directory,
        )

    @classmethod
    def load(cls, name: str, directory: Path = CAPTURE_DIRECTORY) -> Self:
        """
        Read a capture's camera file, leaving its images on disk until asked for.

        :param name: The capture's name.
        :param directory: Where its files lie.
        :raises CaptureIncomplete: If any of the capture's three files is missing.
        """
        missing = [
            part.value
            for part in CapturePart
            if not capture_path(name, part, directory).is_file()
        ]
        if missing:
            raise CaptureIncomplete(name, str(directory), missing)
        camera_file = capture_path(name, CapturePart.CAMERA, directory)
        return cls.from_json(json.loads(camera_file.read_text()), directory=directory)

    @classmethod
    def names_in(cls, directory: Path = CAPTURE_DIRECTORY) -> List[str]:
        """
        Every capture a directory holds, in a stable order.

        :param directory: Where to look.
        """
        suffix = CapturePart.CAMERA.value
        return sorted(
            path.name[: -len(suffix)] for path in directory.glob(f"*{suffix}")
        )

    def save(self, color_payload: bytes, depth: np.ndarray) -> None:
        """
        Write this capture's three files.

        :param color_payload: The colour image as the camera published it, compressed
            the way :attr:`color_format` says.
        :param depth: Depth in metres, zero where the sensor returned no reading.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path_to(CapturePart.COLOR).write_bytes(color_payload)
        cv2.imwrite(
            str(self.path_to(CapturePart.DEPTH)),
            np.round(depth * MILLIMETRES_PER_METRE).astype(np.uint16),
        )
        self.path_to(CapturePart.CAMERA).write_text(
            json.dumps(self.to_json(), indent=2) + "\n"
        )

    def to_frame(self) -> RgbdFrame:
        """
        Read the two images back into the frame a pipeline runs on.

        :raises UndecodableCompressedImage: If the colour file does not decode into
            pixels.
        """
        color = decode_compressed_color_image(
            self.path_to(CapturePart.COLOR).read_bytes(), self.color_format
        )
        millimetres = cv2.imread(
            str(self.path_to(CapturePart.DEPTH)), cv2.IMREAD_UNCHANGED
        )
        return RgbdFrame(
            color=color,
            depth=millimetres.astype(np.float32) / MILLIMETRES_PER_METRE,
            intrinsics=self.intrinsics,
            reference_frame_T_camera=self.reference_frame_T_camera,
        )

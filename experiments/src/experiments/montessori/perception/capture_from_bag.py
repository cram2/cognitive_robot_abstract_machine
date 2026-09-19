"""
Take one look at the scene out of a recorded rosbag and write it as a capture.

Run it as::

    python -m experiments.montessori.perception.capture_from_bag rosbags/<bag> \
        --camera-from rosbags/<bag carrying the transform tree>

A recording off this camera holds hundreds of frames of a scene that barely moves, so
one frame stands for the whole recording and the recording itself never has to be
committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.montessori.perception.captures import CAPTURE_DIRECTORY, SceneCapture
from experiments.montessori.perception.recordings import (
    REFERENCE_FRAME,
    RecordedCamera,
)

MIDWAY = 0.5
"""
How far into a recording a look is taken from unless another point is asked for.
"""


def write_capture(
    bag: Path,
    name: str,
    camera_bag: Path | None = None,
    reference_frame: str = REFERENCE_FRAME,
    at_fraction: float = MIDWAY,
    directory: Path = CAPTURE_DIRECTORY,
) -> SceneCapture:
    """
    Cut one frame out of a recording and write it as a capture.

    :param bag: Directory of the recording to read.
    :param name: What to call this look at the scene.
    :param camera_bag: Recording to read the camera's pose from, for a recording that
        carries no transform tree of its own.
    :param reference_frame: Frame to express the camera's pose in.
    :param at_fraction: How far into the recording to take the frame from.
    :param directory: Where to write the capture's three files.
    :return: The capture that was written.
    :raises NothingRecordedOnTopic: If the recording carries no colour image, no depth
        image or no camera calibration.
    """
    camera = RecordedCamera(
        bag=bag, reference_frame=reference_frame, camera_bag=camera_bag
    )
    look = camera.look_at(at_fraction)
    capture = SceneCapture(
        name=name,
        recorded_from=bag.name,
        color_format=look.color_format,
        intrinsics=camera.intrinsics,
        reference_frame=reference_frame,
        reference_frame_T_camera=camera.reference_frame_T_camera,
        directory=directory,
    )
    capture.save(look.color_payload, look.depth)
    return capture


def main() -> None:
    """
    Write one capture from the recording named on the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="directory of the recording to read")
    parser.add_argument(
        "--name",
        default=None,
        help="what to call the capture; defaults to the recording's own name",
    )
    parser.add_argument(
        "--camera-from",
        type=Path,
        default=None,
        help=(
            "recording to read the camera's pose from, for a recording that carries "
            "no transform tree of its own"
        ),
    )
    parser.add_argument(
        "--reference-frame",
        default=REFERENCE_FRAME,
        help="frame to express the camera's pose, and so the detections, in",
    )
    parser.add_argument(
        "--at",
        type=float,
        default=MIDWAY,
        help="how far into the recording to take the frame from",
    )
    parser.add_argument(
        "--into",
        type=Path,
        default=CAPTURE_DIRECTORY,
        help="directory to write the capture's three files into",
    )
    arguments = parser.parse_args()
    capture = write_capture(
        bag=arguments.bag,
        name=arguments.name or arguments.bag.name,
        camera_bag=arguments.camera_from,
        reference_frame=arguments.reference_frame,
        at_fraction=arguments.at,
        directory=arguments.into,
    )
    print(f"wrote {capture.name} to {capture.directory}")


if __name__ == "__main__":
    main()

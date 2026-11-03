"""
Interface of the model that estimates and tracks the 6D pose of an object from its CAD
model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy
from typing_extensions import TYPE_CHECKING

from robokudo.types.cv import Point2D, Rect

if TYPE_CHECKING:
    import numpy.typing as numpy_typing
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel


# %% Observations


@dataclass
class RGBDFrame:
    """
    A color and a depth image of the same resolution, taken by one camera.
    """

    rgb_image: numpy_typing.NDArray[numpy.uint8]
    """
    Color image in RGB channel order, at the resolution of :attr:`depth_image`.
    """

    depth_image: numpy_typing.NDArray[numpy.float32]
    """
    Depth image in meters; pixels without a measurement are zero.
    """

    camera_model: PinholeCameraModel
    """
    Calibration of the camera at the resolution of the images.
    """


@dataclass
class ObjectObservation:
    """
    One object in an RGB-D frame, marked by a mask.
    """

    frame: RGBDFrame
    """
    The frame the object is seen in.
    """

    mask: numpy_typing.NDArray[numpy.bool_]
    """
    Pixels of :attr:`frame` that show the object.
    """

    @cached_property
    def mask_region(self) -> Rect:
        """
        The smallest rectangle containing every pixel of :attr:`mask`.
        """
        left, top, width, height = cv2.boundingRect(self.mask.astype(numpy.uint8))
        return Rect(pos=Point2D(x=left, y=top), width=width, height=height)


# %% Pose estimation


@dataclass
class ObjectPoseEstimator(ABC):
    """
    Estimates and tracks the pose of an object in the camera frame, given the mesh of
    the object.

    Poses are 4x4 homogeneous transforms from the frame of the mesh to the camera frame.
    """

    @abstractmethod
    def load_meshes(self, mesh_files: list[Path]) -> None:
        """
        Load meshes so that selecting any of them with :meth:`select_mesh` is immediate.

        :param mesh_files: Mesh files in meters, each holding a single mesh.
        """

    @abstractmethod
    def select_mesh(self, mesh_file: Path) -> None:
        """
        Use a mesh for the following pose estimates and tracks.

        :param mesh_file: Mesh file in meters, holding a single mesh.
        """

    @abstractmethod
    def estimate_pose(
        self, observation: ObjectObservation
    ) -> numpy_typing.NDArray[numpy.float64] | None:
        """
        Estimate the pose of the selected mesh without prior knowledge.

        :param observation: The object to estimate the pose of.
        :return: The pose of the mesh, or ``None`` if the observation holds no depth
            measurement of the object.
        """

    @abstractmethod
    def track_pose(
        self,
        observation: ObjectObservation,
        previous_pose: numpy_typing.NDArray[numpy.float64],
    ) -> numpy_typing.NDArray[numpy.float64]:
        """
        Refine a known pose of the selected mesh to an observation of the object that
        moved a little since.

        :param observation: The object to track.
        :param previous_pose: The pose of the mesh before it moved.
        :return: The refined pose of the mesh.
        """

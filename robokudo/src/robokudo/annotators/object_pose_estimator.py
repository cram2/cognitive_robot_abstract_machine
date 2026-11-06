"""
Models that estimate and track object poses, with or without the mesh of the object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from krrood_spatial_math.spatial_types import Pose

    from robokudo.annotators.object_observation import ObjectObservation


# %% Object pose estimators


@dataclass
class ObjectPoseEstimator(ABC):
    """
    Estimates and tracks the pose of an object in the camera frame.
    """

    @abstractmethod
    def estimate_pose(self, object_observation: ObjectObservation) -> Pose | None:
        """
        Estimate the pose of an object without prior knowledge.

        :param object_observation: The object to estimate the pose of.
        :return: The pose of the object, or ``None`` if the observation holds no depth
            measurement of the object.
        """

    @abstractmethod
    def track_pose(
        self, object_observation: ObjectObservation, previous_pose: Pose
    ) -> Pose:
        """
        Refine a known pose of an object to an observation of the object that moved a
        little since.

        :param object_observation: The object to track.
        :param previous_pose: The pose of the object before it moved.
        :return: The refined pose of the object.
        """


@dataclass
class MeshBasedPoseEstimator(ObjectPoseEstimator, ABC):
    """
    An object pose estimator that needs the mesh of the object.

    Estimates and tracks refer to the selected mesh, whose frame is the frame of the
    object.
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

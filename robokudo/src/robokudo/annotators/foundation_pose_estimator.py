"""
FoundationPose running on the GPU, inside the RoboKudo process.

.. note::
    Needs the ROS package ``robokudo_foundation_pose_tensorrt`` of ``cram_ros2_packages``,
    built with a CUDA compiler and the ``tensorrt-cu12`` wheel of the active environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy
from krrood.ormatic.utils import classproperty
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
    Pose,
)
from robokudo_foundation_pose_tensorrt import FoundationPose, ensure_engines
from semantic_digital_twin.datastructures.camera_model import (
    CameraRange,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution

from robokudo.annotators.object_observation import (
    ColorAndDepthFrame,
    ObjectObservation,
)
from robokudo.annotators.object_pose_estimator import MeshBasedPoseEstimator

# %% Engine precision


class EnginePrecision(StrEnum):
    """
    Floating point precision the TensorRT engines compute in.
    """

    FP16 = "fp16"
    """
    Half precision, about three times faster than :attr:`FP32`.
    """

    FP32 = "fp32"
    """
    Single precision.
    """


# %% FoundationPose estimator


@dataclass
class FoundationPoseEstimator(MeshBasedPoseEstimator):
    """
    FoundationPose with its refine and score networks running as TensorRT engines.

    Missing engines are built on construction, once per GPU and TensorRT version, which
    takes a few minutes.
    """

    precision: EnginePrecision = EnginePrecision.FP16
    """
    Precision of the refine and score engines.
    """

    refine_iterations: int = 3
    """
    Refinement steps of a pose estimate; fewer are faster but less accurate.
    """

    track_iterations: int = 2
    """
    Refinement steps when tracking a pose.
    """

    usable_depth_range: CameraRange = field(
        default_factory=lambda: CameraRange(minimum_distance=0.1, maximum_distance=4.0)
    )
    """
    Depth measurements outside this range are ignored.
    """

    foundation_pose: FoundationPose = field(init=False, repr=False)
    """
    The estimator holding the engines and the loaded meshes on the GPU.
    """

    def __post_init__(self) -> None:
        """
        Load the engines of :attr:`precision`, building them first if they are missing.
        """
        refine_engine_file, score_engine_file = ensure_engines(precision=self.precision)
        self.foundation_pose = FoundationPose(
            refine_engine_file_path=refine_engine_file,
            score_engine_file_path=score_engine_file,
            refine_iterations=self.refine_iterations,
            min_depth=self.usable_depth_range.minimum_distance,
            max_depth=self.usable_depth_range.maximum_distance,
        )

    def load_meshes(self, mesh_files: list[Path]) -> None:
        """
        Like :meth:`MeshBasedPoseEstimator.load_meshes`, and also run a first estimate and
        track, since the first TensorRT inferences are slow.
        """
        for mesh_file in mesh_files:
            self.select_mesh(mesh_file)
        if not mesh_files:
            return
        warm_up_object_observation = self.warm_up_object_observation
        pose = self.estimate_pose(warm_up_object_observation)
        if pose is not None:
            self.track_pose(warm_up_object_observation, pose)

    def select_mesh(self, mesh_file: Path) -> None:
        self.foundation_pose.set_mesh(mesh_file_path=str(mesh_file))

    def estimate_pose(self, object_observation: ObjectObservation) -> Pose | None:
        frame = object_observation.frame
        result = self.foundation_pose.estimate(
            rgb=frame.color_image,
            depth=frame.depth_image,
            mask=object_observation.mask,
            cam_intrinsics=frame.camera_model.intrinsic_matrix,
        )
        if result is None:
            return None
        pose_matrix, _, _ = result
        return HomogeneousTransformationMatrix(data=pose_matrix).pose

    def track_pose(
        self, object_observation: ObjectObservation, previous_pose: Pose
    ) -> Pose:
        frame = object_observation.frame
        pose_matrix, _, _ = self.foundation_pose.track(
            rgb=frame.color_image,
            depth=frame.depth_image,
            mask=object_observation.mask,
            cam_intrinsics=frame.camera_model.intrinsic_matrix,
            previous_pose=previous_pose.to_np(),
            iterations=self.track_iterations,
        )
        return HomogeneousTransformationMatrix(data=pose_matrix).pose

    @classproperty
    def warm_up_object_observation(cls) -> ObjectObservation:
        """
        A square object in the center of a black VGA image of a plane half a meter away.
        """
        resolution = CameraResolution(width=640, height=480)
        focal_length = 500.0
        object_half_size = 40
        center_x, center_y = resolution.width // 2, resolution.height // 2
        mask = numpy.zeros((resolution.height, resolution.width), dtype=numpy.bool_)
        mask[
            center_y - object_half_size : center_y + object_half_size,
            center_x - object_half_size : center_x + object_half_size,
        ] = True
        frame = ColorAndDepthFrame(
            color_image=numpy.zeros(
                (resolution.height, resolution.width, 3), dtype=numpy.uint8
            ),
            depth_image=numpy.full(
                (resolution.height, resolution.width), 0.5, dtype=numpy.float32
            ),
            camera_model=PinholeCameraModel(
                image_resolution=resolution,
                focal_length_x=focal_length,
                focal_length_y=focal_length,
                principal_point=Point2(x=float(center_x), y=float(center_y)),
            ),
        )
        return ObjectObservation(frame=frame, mask=mask)

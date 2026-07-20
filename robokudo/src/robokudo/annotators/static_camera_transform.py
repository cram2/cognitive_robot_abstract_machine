"""Annotator for writing a configured static camera transform into the CAS.
Use this in scenarios where you do not have a world to camera transform published
or when you develop the perception system in isolation."""

from __future__ import annotations

from timeit import default_timer

from py_trees.common import Status
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from robokudo.annotators.core import BaseAnnotator
from robokudo.io.camera_interface import CameraInterface


class StaticCameraTransformAnnotator(BaseAnnotator):
    """Write a fixed camera-to-world transform to the current CAS."""

    class Descriptor(BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self) -> None:
                self.world_frame: str = "map"
                """Reference frame used for the camera-to-world transform."""

                self.camera_frame: str = "camera"
                """Camera frame used as the transform child frame."""

                self.world_T_camera: HomogeneousTransformationMatrix = (
                    HomogeneousTransformationMatrix()
                )
                """Camera pose relative to the configured world frame."""

        parameters = Parameters()

    def __init__(
        self,
        name: str = "StaticCameraTransform",
        descriptor: StaticCameraTransformAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name=name, descriptor=descriptor)

    def update(self) -> Status:
        start_timer = default_timer()

        cas = self.get_cas()
        params = self.descriptor.parameters

        CameraInterface.store_camera_to_world_transform_in_cas(
            cas=cas,
            world_frame=params.world_frame,
            camera_frame=params.camera_frame,
            world_T_camera=params.world_T_camera,
        )

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

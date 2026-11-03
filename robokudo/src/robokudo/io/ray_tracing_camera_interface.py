"""
Simulated RGB-D camera interface backed by the SemDT ray tracer.
"""

from __future__ import annotations

import time

from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
)
from robokudo.io.camera_interface import CameraInterface
from robokudo.io.ray_tracing_context import (
    RayTracingContext,
    RayTracingContextResolver,
)
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_rendering import RenderedRGBDFrame

# %% Semantic Digital Twin ray-tracer interface


class RayTracingCameraInterface(CameraInterface):
    """
    Render RGB-D camera data from a Semantic Digital Twin world.

    The interface orchestrates context resolution, rendering, and publication of one
    coherent semantic camera observation into the CAS.
    """

    def __init__(self, camera_config: RayTracingCameraConfiguration) -> None:
        """
        Initialize the camera interface.

        :param camera_config: Ray-tracer camera configuration descriptor.
        """
        super().__init__(camera_config)
        self.context_resolver = RayTracingContextResolver(camera_config)
        """
        Resolver for the configured world and semantic camera.
        """

    def has_new_data(self) -> bool:
        """
        Report whether rendered camera data is available.

        :return: Always ``True`` because simulated frames are rendered on demand.
        """
        return True

    def set_data(self, cas: CAS) -> None:
        """
        Render a simulated RGB-D frame and write it into the CAS.

        :param cas: CAS that receives rendered camera data and frame metadata.
        """
        context = self.context_resolver.resolve()
        frame = context.world.ray_tracer.render(
            context.camera, rgb_mode=self.camera_config.rgb_mode
        )
        camera_model = frame.camera_model

        timestamp_ns = time.time_ns()
        self._write_frame_to_cas(
            cas=cas,
            context=context,
            frame=frame,
            camera_model=camera_model,
            timestamp_ns=timestamp_ns,
        )

    def _write_frame_to_cas(
        self,
        cas: CAS,
        context: RayTracingContext,
        frame: RenderedRGBDFrame,
        camera_model: PinholeCameraModel,
        timestamp_ns: int,
    ) -> None:
        """
        Publish one rendered frame and its semantic camera state.

        :param cas: CAS receiving the frame.
        :param context: World and camera used to render the frame.
        :param frame: Aligned rendered image products.
        :param camera_model: Effective pinhole model used for rendering.
        :param timestamp_ns: Frame timestamp in nanoseconds since the epoch.
        """
        world = context.world
        camera = context.camera
        world_T_camera = camera.root.global_transform

        cas.set(CASViews.COLOR_IMAGE, frame.color_bgr.swapaxes(0, 1).copy())
        cas.set(CASViews.DEPTH_IMAGE, frame.depth_in_millimeters.swapaxes(0, 1).copy())
        cas.set(CASViews.COLOR_TO_DEPTH_RATIO, self.camera_config.color_to_depth_ratio)
        cas.set(CASViews.OBJECT_IMAGE, frame.segmentation.swapaxes(0, 1).copy())
        cas.set(CASViews.BODY_COLORS, frame.body_colors)
        cas.set_ref(CASViews.GROUND_TRUTH_WORLD_REFERENCE, world)
        cas.camera_observation = CameraObservation(
            camera=camera,
            effective_camera_model=camera_model,
            world_T_camera=world_T_camera,
            timestamp_nanoseconds=timestamp_ns,
        )

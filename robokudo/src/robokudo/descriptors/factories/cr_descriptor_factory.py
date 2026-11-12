"""
Construct collection readers for registered camera sources.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import Any
from krrood.ormatic.utils import classproperty
from krrood.patterns.called_positionally import called_positionally

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.descriptors.camera_configs.config_kinect_robot import KinectCameraConfig
from robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform import (
    KinectWoTfCameraConfig,
)
from robokudo.descriptors.camera_configs.config_realsense import RealsenseCameraConfig
from robokudo.descriptors.camera_configs.config_tiago import TiagoCameraConfig
from robokudo.descriptors.camera_configs.config_cv_camera_without_depth import (
    OpenCVCameraConfig,
)
from robokudo.descriptors.camera_configs.config_orbbec import OrbbecCameraConfig
from robokudo.descriptors.camera_configs.config_orbbec_wo_transform import (
    OrbbecCameraConfig as OrbbecWithoutTransformCameraConfig,
)
from robokudo.descriptors.camera_configs.config_filereader_playback import (
    FileReaderCameraConfig,
)
from robokudo.descriptors.camera_configs.config_hsr_ros2 import HsrRos2CameraConfig
from robokudo.descriptors.camera_configs.registry import CameraConfigRegistry
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
)
from robokudo.io.camera_interface import CameraInterface, KinectCameraInterface
from robokudo.io.camera_without_depth_interface import OpenCVCameraWithoutDepthInterface
from robokudo.io.file_reader_interface import RGBDFileReaderInterface
from robokudo.io.ray_tracing_camera_interface import (
    RayTracingCameraInterface,
)
from robokudo.io.storage_reader_interface import StorageReaderInterface
from robokudo.descriptors.camera_configs.config_stereo import StereoCameraConfig
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)
from robokudo.descriptors.camera_configs.config_zed import ZedCameraConfig
from robokudo.io.rectified_depth_camera_interface import RectifiedDepthCameraInterface


# %% Reader construction


@dataclass(kw_only=True)
class CollectionReaderDescriptorFactory:
    """
    Factory class for creating CollectionReader descriptors.
    """

    @classproperty
    def _camera_interface_types(cls) -> dict[str, type[CameraInterface]]:
        """
        Map registered sources to their acquisition implementations.
        """
        return {
            MongoCameraConfig.registry_name: StorageReaderInterface,
            KinectCameraConfig.registry_name: KinectCameraInterface,
            KinectWoTfCameraConfig.registry_name: KinectCameraInterface,
            RealsenseCameraConfig.registry_name: KinectCameraInterface,
            TiagoCameraConfig.registry_name: KinectCameraInterface,
            OpenCVCameraConfig.registry_name: OpenCVCameraWithoutDepthInterface,
            OrbbecCameraConfig.registry_name: KinectCameraInterface,
            OrbbecWithoutTransformCameraConfig.registry_name: KinectCameraInterface,
            FileReaderCameraConfig.registry_name: RGBDFileReaderInterface,
            RayTracingCameraConfiguration.registry_name: RayTracingCameraInterface,
            HsrRos2CameraConfig.registry_name: KinectCameraInterface,
            StereoCameraConfig.registry_name: RectifiedDepthCameraInterface,
            FoundationStereoCameraConfig.registry_name: RectifiedDepthCameraInterface,
            ZedCameraConfig.registry_name: RectifiedDepthCameraInterface,
        }

    @staticmethod
    @called_positionally
    def create_descriptor(
        camera: str, **kwargs: Any
    ) -> CollectionReaderAnnotator.Descriptor:
        """
        Create a CollectionReader descriptor for the specified camera.

        :param camera: The name of the camera to create a descriptor for.
        :param kwargs: Additional keyword arguments to pass to the camera configuration.
        :returns: A CollectionReader descriptor for the specified camera.
        :raises ValueError: If the given camera config name is not registered in the
            camera config registry.
        :raises TypeError: If the keyword arguments are invalid for the camera config
            class.
        """
        camera_config = CameraConfigRegistry.create_config(camera, **kwargs)
        return CollectionReaderAnnotator.Descriptor(
            camera_config=camera_config,
            camera_interface=CollectionReaderDescriptorFactory._camera_interface_types[
                camera
            ](camera_config=camera_config),
        )

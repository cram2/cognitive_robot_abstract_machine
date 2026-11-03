from __future__ import annotations

from typing_extensions import Any

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
from robokudo.io.camera_interface import KinectCameraInterface
from robokudo.io.camera_without_depth_interface import OpenCVCameraWithoutDepthInterface
from robokudo.io.file_reader_interface import RGBDFileReaderInterface
from robokudo.io.ray_tracing_camera_interface import (
    RayTracingCameraInterface,
)
from robokudo.io.storage_reader_interface import StorageReaderInterface


class CollectionReaderDescriptorFactory:
    """
    Factory class for creating CollectionReader descriptors.
    """

    _camera_interface_types = {
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
    }
    """
    Mapping of camera names to their corresponding camera interface types.
    """

    @staticmethod
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
            ](camera_config),
        )

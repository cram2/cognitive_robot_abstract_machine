from typing_extensions import Any

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.camera_configs.registry import CameraConfigRegistry
from robokudo.io.camera_interface import KinectCameraInterface
from robokudo.io.storage_reader_interface import StorageReaderInterface
from robokudo.io.camera_without_depth_interface import OpenCVCameraWithoutDepthInterface
from robokudo.io.file_reader_interface import RGBDFileReaderInterface


class CrDescriptorFactory:
    """Factory class for creating CollectionReader descriptors."""

    _camera_interface_types = {
        "mongo": StorageReaderInterface,
        "kinect": KinectCameraInterface,
        "kinect_wo_tf": KinectCameraInterface,
        "realsense": KinectCameraInterface,
        "tiago": KinectCameraInterface,
        "opencv": OpenCVCameraWithoutDepthInterface,
        "file_reader": RGBDFileReaderInterface,
    }
    """Mapping of camera names to their corresponding camera interface types."""

    @staticmethod
    def create_descriptor(
        camera: str, **kwargs: Any
    ) -> CollectionReaderAnnotator.Descriptor:
        """Create a CollectionReader descriptor for the specified camera.

        :param camera: The name of the camera to create a descriptor for.
        :param kwargs: Additional keyword arguments to pass to the camera configuration.
        :returns: A CollectionReader descriptor for the specified camera.
        :raises ValueError: If the given camera config name is not registered in the camera config registry.
        :raises TypeError: If the keyword arguments are invalid for the camera config class.
        """
        camera_config = CameraConfigRegistry.create_config(camera, **kwargs)
        return CollectionReaderAnnotator.Descriptor(
            camera_config=camera_config,
            camera_interface=CrDescriptorFactory._camera_interface_types[camera](
                camera_config
            ),
        )

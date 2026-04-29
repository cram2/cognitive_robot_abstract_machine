from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import WorldDescriptorComponent


@dataclass(slots=True)
class MongoCameraConfig(BaseCameraConfig, WorldDescriptorComponent):
    """Configuration class for MongoDB-based camera data playback.

    This class defines the configuration parameters for reading camera data from a
    MongoDB database, typically used for offline processing or testing with stored
    scene data.

    .. note::
        This configuration is used to read previously stored camera data from a
        MongoDB database, allowing for replay and analysis of recorded scenes.
    """

    registry_name: ClassVar[str] = "mongo"

    interface_type: str = "StorageReader"

    loop: bool = True
    """Flag to enable looping over database entries"""

    db_name: str = "rk_scenes"
    """Name of the MongoDB database to read from"""

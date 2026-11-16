from __future__ import annotations

# %% Imports
from krrood.ormatic.utils import classproperty
from dataclasses import dataclass
from enum import StrEnum


from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import WorldDescriptorComponent


class MongoReplayMode(StrEnum):
    """
    Choose which part of a recorded world playback restores.
    """

    SENSOR_CONTEXT = "sensor_context"
    """
    Restore only the camera and its required reference frames.
    """

    FULL_WORLD = "full_world"
    """
    Restore the complete recorded world for each frame.
    """


@dataclass(slots=True)
class MongoCameraConfig(BaseCameraConfig, WorldDescriptorComponent):
    """
    Configuration class for MongoDB-based camera data playback.

    This class defines the configuration parameters for reading camera data from a
    MongoDB database, typically used for offline processing or testing with stored
    scene data.

    .. note::
        This configuration is used to read previously stored camera data from a
        MongoDB database, allowing for replay and analysis of recorded scenes.
    """

    @classproperty
    def registry_name(cls) -> str:
        """
        Identifier selecting this camera configuration.
        """
        return "mongo"

    interface_type: str = "StorageReader"

    loop: bool = True
    """
    Flag to enable looping over database entries.
    """

    db_name: str = "rk_scenes"
    """
    Name of the MongoDB database to read from.
    """

    restore_annotations: bool = False
    """
    Whether to restore stored CAS annotations during playback.
    """

    replay_mode: MongoReplayMode = MongoReplayMode.SENSOR_CONTEXT
    """
    World content restored before decoding each recorded frame.
    """

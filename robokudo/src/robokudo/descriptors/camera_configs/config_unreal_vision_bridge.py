from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig


@dataclass(slots=True)
class UnrealVisionBridgeCameraConfig(BaseCameraConfig):
    """Configuration class for the Unreal Vision Bridge interface.

    This class defines the minimal configuration required for connecting to
    a camera interface in the Unreal Engine environment through the Vision Bridge.
    It is used for simulated camera data in Unreal Engine-based simulations.

    .. note::
        This is a minimal configuration that only specifies the interface type.
        Additional parameters may be required depending on the specific Unreal
        Engine simulation setup.
    """

    registry_name: ClassVar[str] = "unreal_vision_bridge"

    interface_type: str = "UnrealVisionBridge"

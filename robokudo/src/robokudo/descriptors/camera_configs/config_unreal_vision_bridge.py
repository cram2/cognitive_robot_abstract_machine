from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration class for the Unreal Vision Bridge interface.

    This class defines the minimal configuration required for connecting to
    a camera interface in the Unreal Engine environment through the Vision Bridge.
    It is used for simulated camera data in Unreal Engine-based simulations.

    :ivar interface_type: Type of camera interface, set to "UnrealVisionBridge"

    .. note::
        This is a minimal configuration that only specifies the interface type.
        Additional parameters may be required depending on the specific Unreal
        Engine simulation setup.
    """

    # camera
    interface_type: str = "UnrealVisionBridge"

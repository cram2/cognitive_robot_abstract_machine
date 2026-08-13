from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.config_orbbec import OrbbecCameraConfig


@dataclass(slots=True)
class OrbbecCameraConfig(OrbbecCameraConfig):
    """Configuration class for the OrbBec Astra camera without transform lookup."""

    registry_name: ClassVar[str] = "orbbec_wo_tf"

    lookup_viewpoint: bool = False

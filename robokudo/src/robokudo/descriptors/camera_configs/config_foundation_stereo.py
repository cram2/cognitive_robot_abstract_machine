"""
Topic defaults for an external FoundationStereo ROS depth producer.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass
from enum import StrEnum

from krrood.ormatic.utils import classproperty
from robokudo.descriptors.camera_configs.config_stereo import StereoCameraConfig

# %% FoundationStereo source


class FoundationStereoSource(StrEnum):
    """
    Registered name of the FoundationStereo topic preset.
    """

    FOUNDATION_STEREO = "foundation_stereo"


@dataclass(kw_only=True)
class FoundationStereoCameraConfig(StereoCameraConfig):
    """
    Use :class:`StereoCameraConfig` with Fast-FoundationStereo depth output.
    """

    topic_depth: str = "/ffs/depth"
    """
    Metric depth output from the external inference node.
    """

    @classproperty
    def registry_name(cls) -> str:
        """
        Select the FoundationStereo preset in the camera registry.
        """
        return FoundationStereoSource.FOUNDATION_STEREO

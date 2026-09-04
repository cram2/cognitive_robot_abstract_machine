from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.exceptions import InvalidCameraResolutionError


@dataclass
class CameraResolution:
    """
    Represents the pixel dimensions of a camera image.
    """

    width: int = field(default=512)
    """
    Number of pixels along the image width.
    """

    height: int = field(default=512)
    """
    Number of pixels along the image height.
    """

    def __post_init__(self) -> None:
        """
        Validates that the resolution can describe a camera image.
        """
        if self.width <= 0 or self.height <= 0:
            raise InvalidCameraResolutionError(
                width=self.width,
                height=self.height,
            )

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import ClassVar


@dataclass(kw_only=True)
class BaseCameraConfig:
    registry_name: ClassVar[str]
    """Name of the camera configuration in the registry. Must be unique."""

    interface_type: str
    """Type of camera interface"""

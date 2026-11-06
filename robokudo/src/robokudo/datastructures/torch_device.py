"""
Krrood serialization of PyTorch device selections without allocating tensors.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import torch
from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    ExternalClassJSONSerializer,
)
from krrood.utils import get_full_class_name
from typing_extensions import Any


# %% Device configuration
@dataclass(kw_only=True)
class TorchDeviceConfiguration:
    """
    Device selection that can be restored on a machine without that device.
    """

    device_identifier: str
    """
    PyTorch device name, including the optional device index.
    """


# %% Device serialization
@dataclass
class TorchDeviceJSONSerializer(ExternalClassJSONSerializer[torch.device]):
    """
    Adapt PyTorch devices through a declarative dataclass representation.
    """

    @classmethod
    def to_json(cls, instance: torch.device, **kwargs: Any) -> dict[str, Any]:
        """
        Preserve device identifiers in :meth:`ExternalClassJSONSerializer.to_json`.
        """
        result = DataclassJSONSerializer.to_json(
            TorchDeviceConfiguration(device_identifier=str(instance)), **kwargs
        )
        result[JSONField.TYPE] = get_full_class_name(type(instance))
        return result

    @classmethod
    def from_json(
        cls, data: dict[str, Any], clazz: type[torch.device], **kwargs: Any
    ) -> torch.device:
        """
        Restore a selection without hardware access; see
        :meth:`ExternalClassJSONSerializer.from_json`.
        """
        configuration = DataclassJSONSerializer.from_json(
            data, clazz=TorchDeviceConfiguration, **kwargs
        )
        return clazz(configuration.device_identifier)

"""
A class that holds data next to a JSON serializer for a class outside our control, which
only gives behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from krrood.adapters.json_serializer import ExternalClassJSONSerializer

# %% data


@dataclass(kw_only=True)
class SerializedRecord:
    """
    A value an interface stores.
    """

    label: str
    """
    The label of the record.
    """


class ValueOutsideOurControl:
    """
    A class whose inheritance a serializer cannot change, so it gets an external one.
    """


# %% behaviour


@dataclass(kw_only=True)
class ValueOutsideOurControlJSONSerializer(
    ExternalClassJSONSerializer[ValueOutsideOurControl]
):
    """
    A serializer, which describes nothing an interface stores.
    """

    @classmethod
    def to_json(
        cls, *, instance: ValueOutsideOurControl, **kwargs: Any
    ) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json(
        cls,
        *,
        data: dict[str, Any],
        clazz: type[ValueOutsideOurControl],
        **kwargs: Any,
    ) -> ValueOutsideOurControl:
        return ValueOutsideOurControl()

"""
JSON serializers for scipy sparse arrays.

Importing this module registers them, so a program that serializes sparse arrays imports
it and every other program does not need scipy.
"""

from __future__ import annotations

import enum

from scipy.sparse import coo_array, csr_array
from typing_extensions import Any

from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import (
    ExternalClassJSONSerializer,
    from_json,
    to_json,
)
from krrood.utils import get_full_class_name

# %% sparse arrays


class CoordinateSparseArrayJSONKey(enum.StrEnum):
    """
    The keys of the JSON a sparse array in coordinate format is serialized to.
    """

    DATA = "data"
    """
    The values of the stored entries.
    """

    ROWS = "rows"
    """
    The row of every stored entry.
    """

    COLUMNS = "columns"
    """
    The column of every stored entry.
    """

    SHAPE = "shape"
    """
    The shape of the dense array the sparse array describes.
    """


class CoordinateSparseArrayJSONSerializer(ExternalClassJSONSerializer[coo_array]):
    """
    External JSON serializer for scipy sparse arrays in coordinate format.

    Every stored entry is kept, including an explicitly stored zero.
    """

    @classmethod
    def to_json(cls, *, instance: coo_array, **kwargs) -> dict[str, Any]:
        """
        Write the stored entries; see :meth:`ExternalClassJSONSerializer.to_json`.
        """
        return {
            JSONField.TYPE: get_full_class_name(type(instance)),
            CoordinateSparseArrayJSONKey.DATA: to_json(instance.data, **kwargs),
            CoordinateSparseArrayJSONKey.ROWS: to_json(instance.row, **kwargs),
            CoordinateSparseArrayJSONKey.COLUMNS: to_json(instance.col, **kwargs),
            CoordinateSparseArrayJSONKey.SHAPE: list(instance.shape),
        }

    @classmethod
    def from_json(
        cls, *, data: dict[str, Any], clazz: type[coo_array], **kwargs
    ) -> coo_array:
        """
        Read the stored entries; see :meth:`ExternalClassJSONSerializer.from_json`.
        """
        return coo_array(
            (
                from_json(data[CoordinateSparseArrayJSONKey.DATA], **kwargs),
                (
                    from_json(data[CoordinateSparseArrayJSONKey.ROWS], **kwargs),
                    from_json(data[CoordinateSparseArrayJSONKey.COLUMNS], **kwargs),
                ),
            ),
            shape=tuple(data[CoordinateSparseArrayJSONKey.SHAPE]),
        )


class CompressedSparseRowArrayJSONSerializer(ExternalClassJSONSerializer[csr_array]):
    """
    External JSON serializer for scipy sparse arrays in compressed sparse row format.

    The array is written like a sparse array in coordinate format, through its stored
    entries.
    """

    @classmethod
    def to_json(cls, *, instance: csr_array, **kwargs) -> dict[str, Any]:
        """
        Write the stored entries; see :meth:`ExternalClassJSONSerializer.to_json`.
        """
        return {
            **CoordinateSparseArrayJSONSerializer.to_json(
                instance=instance.tocoo(), **kwargs
            ),
            JSONField.TYPE: get_full_class_name(type(instance)),
        }

    @classmethod
    def from_json(
        cls, *, data: dict[str, Any], clazz: type[csr_array], **kwargs
    ) -> csr_array:
        """
        Read the stored entries; see :meth:`ExternalClassJSONSerializer.from_json`.
        """
        return csr_array(
            CoordinateSparseArrayJSONSerializer.from_json(
                data=data, clazz=coo_array, **kwargs
            )
        )

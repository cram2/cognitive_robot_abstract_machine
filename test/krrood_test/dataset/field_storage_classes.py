"""
Dataclasses whose fields cover every way ORMatic can store a field.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from typing_extensions import Any, Dict, List, Type


@dataclass(kw_only=True)
class StorageLeaf:
    """
    A mapped class that other fields refer to.
    """

    number: int
    """
    A number.
    """


@dataclass(kw_only=True)
class UnmappedRecord:
    """
    A dataclass no table maps, which only the serializer for any dataclass writes.
    """

    number: int
    """
    A number.
    """


@dataclass(kw_only=True)
class FieldsOfEveryStorage:
    """
    Holds one field for each way ORMatic can store a field.
    """

    number: int
    """
    A builtin SQLAlchemy maps by itself.
    """

    kind: Type[StorageLeaf]
    """
    A class.
    """

    kinds: List[Type[StorageLeaf]]
    """
    A collection of classes.
    """

    leaf: StorageLeaf
    """
    A reference to an instance of a mapped class.
    """

    leaves: List[StorageLeaf]
    """
    A collection of instances of a mapped class.
    """

    day: datetime.date
    """
    A value of a type with a type mapping.
    """

    complex_number: complex
    """
    A builtin that needs a type mapping.
    """

    span: range
    """
    A value of a type with a JSON serializer.
    """

    numbers: List[int]
    """
    A collection of builtins.
    """

    days: List[datetime.date]
    """
    A collection of values of a type with a type mapping.
    """

    record: UnmappedRecord
    """
    A value of a dataclass no table maps.
    """

    records: List[UnmappedRecord]
    """
    A collection of values of a dataclass no table maps.
    """

    lookup: Dict[str, int]
    """
    A dictionary of builtins.
    """

    nested_numbers: list[list[float]]
    """
    A nested collection of builtins.
    """

    nested_leaves: list[list[StorageLeaf]]
    """
    A nested collection of instances of a mapped class.
    """

    leaves_by_name: dict[str, StorageLeaf]
    """
    A dictionary whose values are instances of a mapped class.
    """

    names_by_leaf: dict[StorageLeaf, str]
    """
    A dictionary whose keys are instances of a mapped class.
    """

    leaf_lists_by_name: dict[str, list[StorageLeaf]]
    """
    A dictionary whose values are collections of instances of a mapped class.
    """

    anything_by_name: dict[str, Any]
    """
    A dictionary whose values no rule knows how to store.
    """

    untyped_lookup: dict
    """
    A dictionary without key and value types.
    """

    anything: object
    """
    A value no rule knows how to store.
    """

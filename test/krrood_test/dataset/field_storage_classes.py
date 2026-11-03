"""
Dataclasses whose fields cover every way ORMatic can store a field.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from typing_extensions import Dict, List, Type


@dataclass
class StorageLeaf:
    """
    A mapped class that other fields refer to.
    """

    number: int
    """
    A number.
    """


@dataclass
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

    lookup: Dict[str, int]
    """
    A dictionary, which ORMatic cannot store.
    """

    anything: object
    """
    A value no rule knows how to store.
    """

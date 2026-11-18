"""
The kinds of OWL properties.
"""

from __future__ import annotations

from enum import StrEnum


class PropertyType(StrEnum):
    """
    The kind of an OWL property, by the kind of values it relates individuals to.
    """

    OBJECT_PROPERTY = "ObjectProperty"
    """
    Relates individuals to individuals.
    """

    DATA_PROPERTY = "DataProperty"
    """
    Relates individuals to literal values.
    """

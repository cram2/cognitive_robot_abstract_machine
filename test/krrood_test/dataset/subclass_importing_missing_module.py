"""
A subclass whose module imports a module that does not exist.

Mirrors a mapped datastructure whose module reaches for a generated ORM interface inside
a function body, which is absent for as long as the generator that writes it runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Any

from .base_with_late_bound_annotation import BaseWithLateBoundAnnotation


def load_mapping() -> Any:
    """
    Reach for the generated mapping the way a deferred ORM import does.

    :return: The mapping class from the generated interface.
    """
    from ungenerated_orm_interface import GeneratedMapping

    return GeneratedMapping


@dataclass
class SubclassImportingMissingModule(BaseWithLateBoundAnnotation):
    """
    Inherits the late-bound annotation, from a module holding the missing import.
    """

    label: str = ""
    """
    A field of this subclass, carrying no late-bound type of its own.
    """

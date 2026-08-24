"""
A module whose ``TYPE_CHECKING`` import targets a module that does not exist.

Mirrors a source file annotating a field with a generated ORM interface: the interface
module is absent until its generator has run, while the file itself still imports and
runs fine.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ungenerated_orm_interface import GeneratedMapping


@dataclass
class OwnerOfAnnotationFromMissingModule:
    """
    Owner of a field whose annotation lives in the missing module.
    """

    mapping: Optional[GeneratedMapping] = None
    """
    The annotation whose type only exists once the interface has been generated.
    """

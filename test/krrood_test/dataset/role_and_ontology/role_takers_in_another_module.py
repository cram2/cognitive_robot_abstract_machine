from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from .university_ontology_like_classes_without_descriptors import Person


@dataclass(eq=False)
class RoleTakerInAnotherModule:
    original_attribute: str
    attribute_with_annotation_from_role_module: Person

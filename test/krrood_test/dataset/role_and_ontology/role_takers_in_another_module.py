from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING
from krrood.patterns.role import HasRoles

if TYPE_CHECKING:
    from test.krrood_test.dataset.role_and_ontology.university_ontology_like_classes_without_descriptors import (
        PersonInRoleAndOntology,
    )


@dataclass(eq=False)
class RoleTakerInAnotherModule(HasRoles):
    original_attribute: str
    attribute_with_annotation_from_role_module: PersonInRoleAndOntology

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from .university_ontology_like_classes_without_descriptors import (
        Person,
        DelegateAsThirdRole,
    )

@dataclass(eq=False)
class RoleTakerInAnotherModuleRoleAttributes:
    introduced_attribute: str = field(init=False)
    same_module_annotated_introduced_attribute: DelegateAsThirdRole = field(init=False)

@dataclass(eq=False)
class RoleTakerInAnotherModuleMixin(RoleTakerInAnotherModuleRoleAttributes):
    original_attribute: str = field(init=False)
    attribute_with_annotation_from_role_module: Person = field(init=False)

@dataclass(eq=False)
class RoleTakerInAnotherModule(RoleTakerInAnotherModuleRoleAttributes):
    original_attribute: str
    attribute_with_annotation_from_role_module: Person

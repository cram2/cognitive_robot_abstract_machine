from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported for type checkers only, so the name is absent from the runtime namespace.
    from .annotation_only_referenced_type import TypeReferencedOnlyInAnnotations


@dataclass
class ClassWithTypeCheckingOnlyAnnotation:
    """A dataclass whose field type is available only under ``TYPE_CHECKING``.

    Reproduces the resolution scenario that previously raised ``CouldNotResolveType`` while building
    the class diagram for ``set_of`` translations.
    """

    annotation_only_field: TypeReferencedOnlyInAnnotations
